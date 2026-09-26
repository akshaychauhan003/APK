#!/usr/bin/env python3
"""
Local F_0.5 validator and threshold sweep tool.

Exactly replicates the competition's per-entity scoring + singleton rule.
Runs on training data with a train/val split to test threshold changes offline.

Usage:
    export PYTHONPATH=$(pwd)/src
    
    # Quick validation with current model:
    python scripts/validate_and_sweep.py --mode validate --sample-size 10000
    
    # Full threshold sweep:
    python scripts/validate_and_sweep.py --mode sweep --sample-size 20000
    
    # Both:
    python scripts/validate_and_sweep.py --mode all --sample-size 20000
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from business_entity_resolution.data import load_train
from business_entity_resolution.preprocessing import preprocess
from business_entity_resolution.blocking import generate_candidates
from business_entity_resolution.features import build_features
from business_entity_resolution.models.model import Classifier
from business_entity_resolution.models.train import create_labels
from business_entity_resolution.evaluation import split_train_val
from business_entity_resolution.features.feature_builder import FEATURE_COLS
from business_entity_resolution.config import (
    MODEL_DIR, S1_ID_COL, MATCHED_COL, ID_COL, COUNTRY_COL, F_BETA,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)


def f_beta(precision, recall, beta=F_BETA):
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def compute_macro_f05(pred_map, gt_map):
    """Compute macro P, R, F_0.5 exactly as the competition does.
    
    Handles the singleton rule: correctly predicting no-match scores 1.0.
    """
    precs, recs, f05s = [], [], []
    
    for s1_id, actual in gt_map.items():
        pred = pred_map.get(s1_id, set())
        tp = len(pred & actual)
        
        if not pred and not actual:
            # True singleton, correctly predicted empty → 1.0
            p = r = 1.0
        elif not pred:
            # Predicted empty but has matches → missed all
            p, r = 1.0, 0.0
        elif not actual:
            # True singleton but predicted something → precision 0
            p, r = 0.0, 1.0
        else:
            p = tp / len(pred)
            r = tp / len(actual)
        
        precs.append(p)
        recs.append(r)
        f05s.append(f_beta(p, r))
    
    n = len(f05s) or 1
    return {
        "macro_precision": sum(precs) / n,
        "macro_recall": sum(recs) / n,
        "macro_f0.5": sum(f05s) / n,
        "n_entities": n,
        "n_singletons": sum(1 for v in gt_map.values() if not v),
        "n_with_matches": sum(1 for v in gt_map.values() if v),
    }


def parse_id_list(s):
    s = str(s).strip()
    if s and s.lower() != "nan":
        return set(x.strip() for x in s.split(",") if x.strip())
    return set()


def prepare_data(sample_size=10000, val_split=0.20):
    """Load and prepare validation data."""
    log.info("Loading training data...")
    df_s1, df_s2, df_s3, df_gt = load_train()
    
    if sample_size and len(df_s1) > sample_size:
        log.info(f"Sampling {sample_size:,} S1 entities...")
        df_s1 = df_s1.sample(n=sample_size, random_state=42).reset_index(drop=True)
        df_gt = df_gt[df_gt[S1_ID_COL].isin(set(df_s1[ID_COL]))].copy()
    
    log.info("Preprocessing...")
    df_s1 = preprocess(df_s1)
    df_s2 = preprocess(df_s2)
    df_s3 = preprocess(df_s3)
    
    log.info("Splitting train/val...")
    _, val_s1, _, val_gt = split_train_val(df_s1, df_gt, val_size=val_split)
    log.info(f"Validation set: {len(val_s1):,} S1 entities")
    
    log.info("Generating candidates...")
    pairs = generate_candidates(val_s1, df_s2, df_s3)
    log.info(f"Candidate pairs: {len(pairs):,}")
    
    log.info("Building features...")
    df_cand = pd.concat([df_s2, df_s3], ignore_index=True)
    feats = build_features(pairs, val_s1, df_cand)
    log.info(f"Feature matrix: {feats.shape}")
    
    # Parse ground truth
    gt_map = val_gt.set_index(S1_ID_COL)[MATCHED_COL].apply(parse_id_list).to_dict()
    # Ensure all val S1 IDs are in gt_map (default to empty set)
    for sid in val_s1[ID_COL]:
        if sid not in gt_map:
            gt_map[sid] = set()
    
    return val_s1, feats, gt_map


def score_at_threshold(feats, gt_map, all_s1_ids, clf, feature_cols, threshold, margin=0.0):
    """Score predictions at a given threshold and return metrics."""
    if feats.empty:
        pred_map = {sid: set() for sid in all_s1_ids}
        return compute_macro_f05(pred_map, gt_map)
    
    probs = clf.predict_proba(feats[feature_cols])
    hits = feats[[S1_ID_COL, "candidate_entity_id"]].copy()
    hits["prob"] = probs
    
    above = hits[hits["prob"] >= threshold]
    
    # Apply margin-based singleton rejection if margin > 0
    if margin > 0 and not above.empty:
        grouped_max = hits.groupby(S1_ID_COL)["prob"].max().reset_index()
        grouped_max.columns = [S1_ID_COL, "top"]
        
        def second_best(g):
            if len(g) < 2: return 0.0
            return g.nlargest(2).iloc[-1]
        
        grouped_2nd = hits.groupby(S1_ID_COL)["prob"].apply(second_best).reset_index()
        grouped_2nd.columns = [S1_ID_COL, "second"]
        
        stats = grouped_max.merge(grouped_2nd, on=S1_ID_COL)
        stats["margin"] = stats["top"] - stats["second"]
        
        n_above = above.groupby(S1_ID_COL).size().reset_index(name="n")
        stats = stats.merge(n_above, on=S1_ID_COL, how="left")
        stats["n"] = stats["n"].fillna(0).astype(int)
        
        reject = stats[(stats["n"] == 1) & (stats["margin"] < margin) & (stats["top"] < threshold + 0.05)]
        reject_ids = set(reject[S1_ID_COL])
        if reject_ids:
            above = above[~above[S1_ID_COL].isin(reject_ids)]
    
    # Build pred_map
    pred_map = {}
    for sid in all_s1_ids:
        pred_map[sid] = set()
    
    if not above.empty:
        for sid, cid in zip(above[S1_ID_COL], above["candidate_entity_id"]):
            pred_map[sid].add(cid)
    
    return compute_macro_f05(pred_map, gt_map)


def run_validate(feats, gt_map, val_s1, threshold=0.90):
    """Validate current model at current threshold."""
    clf = Classifier.load(MODEL_DIR / "matching_model.pkl")
    
    if clf.feature_names:
        feature_cols = [c for c in clf.feature_names if c in feats.columns]
        for c in set(clf.feature_names) - set(feature_cols):
            feats[c] = 0.0
        feature_cols = list(clf.feature_names)
    else:
        feature_cols = [c for c in FEATURE_COLS if c in feats.columns]
    
    all_s1_ids = val_s1[ID_COL].values
    metrics = score_at_threshold(feats, gt_map, all_s1_ids, clf, feature_cols, threshold)
    
    print(f"\n{'='*60}")
    print(f"  VALIDATION RESULTS (threshold={threshold})")
    print(f"{'='*60}")
    print(f"  Entities:          {metrics['n_entities']:,}")
    print(f"  True singletons:   {metrics['n_singletons']:,}")
    print(f"  With matches:      {metrics['n_with_matches']:,}")
    print(f"  Macro Precision:   {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:      {metrics['macro_recall']:.4f}")
    print(f"  Macro F_0.5:       {metrics['macro_f0.5']:.4f}")
    print(f"{'='*60}\n")
    
    return metrics


def run_sweep(feats, gt_map, val_s1):
    """Run threshold sweep and report precision/recall/F_0.5 at each threshold."""
    clf = Classifier.load(MODEL_DIR / "matching_model.pkl")
    
    if clf.feature_names:
        feature_cols = [c for c in clf.feature_names if c in feats.columns]
        for c in set(clf.feature_names) - set(feature_cols):
            feats[c] = 0.0
        feature_cols = list(clf.feature_names)
    else:
        feature_cols = [c for c in FEATURE_COLS if c in feats.columns]
    
    all_s1_ids = val_s1[ID_COL].values
    
    thresholds = np.arange(0.50, 1.00, 0.02)
    margins = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    print(f"\n{'='*90}")
    print(f"  THRESHOLD SWEEP")
    print(f"{'='*90}")
    
    best_f05 = 0.0
    best_config = None
    
    results = []
    for margin in margins:
        print(f"\n  --- Margin = {margin} ---")
        print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F_0.5':>10}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        
        for t in thresholds:
            m = score_at_threshold(feats, gt_map, all_s1_ids, clf, feature_cols, t, margin=margin)
            results.append({"threshold": t, "margin": margin, **m})
            marker = " <<<" if m["macro_f0.5"] > best_f05 else ""
            print(f"  {t:>10.2f}  {m['macro_precision']:>10.4f}  {m['macro_recall']:>10.4f}  {m['macro_f0.5']:>10.4f}{marker}")
            
            if m["macro_f0.5"] > best_f05:
                best_f05 = m["macro_f0.5"]
                best_config = {"threshold": t, "margin": margin}
    
    print(f"\n{'='*90}")
    print(f"  BEST: threshold={best_config['threshold']:.2f}, margin={best_config['margin']:.2f}")
    print(f"  F_0.5 = {best_f05:.4f}")
    print(f"{'='*90}\n")
    
    return results, best_config


def main():
    p = argparse.ArgumentParser(description="Local F_0.5 validator and threshold sweep")
    p.add_argument("--mode", choices=["validate", "sweep", "all"], default="validate")
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--sample-size", type=int, default=10000)
    p.add_argument("--val-split", type=float, default=0.20)
    args = p.parse_args()
    
    val_s1, feats, gt_map = prepare_data(sample_size=args.sample_size, val_split=args.val_split)
    
    if args.mode in ("validate", "all"):
        run_validate(feats, gt_map, val_s1, threshold=args.threshold)
    
    if args.mode in ("sweep", "all"):
        run_sweep(feats, gt_map, val_s1)


if __name__ == "__main__":
    main()
