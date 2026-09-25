import logging

import numpy as np
import pandas as pd
from pathlib import Path

from .model import Classifier
from ..features.feature_builder import FEATURE_COLS
from ..config import MODEL_DIR, S1_ID_COL, MATCHED_COL

log = logging.getLogger(__name__)


def create_labels(df_features: pd.DataFrame, df_gt: pd.DataFrame) -> pd.Series:
    """Create binary labels (1=match, 0=non-match) for feature rows.
    
    Uses vectorized explode instead of iterrows for speed on large ground truth.
    """
    gt_clean = df_gt[df_gt[MATCHED_COL].str.strip().ne("")]
    if gt_clean.empty:
        return pd.Series(np.zeros(len(df_features), dtype=np.int8), name="label")

    # expand comma-separated matches into one row per (s1_id, candidate_id) pair
    gt_pairs = (
        gt_clean.assign(candidate_entity_id=gt_clean[MATCHED_COL].str.split(","))
        .explode("candidate_entity_id")
        .assign(candidate_entity_id=lambda x: x["candidate_entity_id"].str.strip())
        .query("candidate_entity_id != ''")
        [[S1_ID_COL, "candidate_entity_id"]]
    )
    gt_pairs["label"] = np.int8(1)

    merged = df_features[[S1_ID_COL, "candidate_entity_id"]].merge(
        gt_pairs, on=[S1_ID_COL, "candidate_entity_id"], how="left"
    )
    labels = merged["label"].fillna(0).astype(np.int8)

    pos, neg = int(labels.sum()), int((labels == 0).sum())
    log.info(f"  labels: {pos:,} pos / {neg:,} neg  ({neg // max(pos, 1)}:1 imbalance)")
    return labels


def train(
    df_features: pd.DataFrame,
    df_gt: pd.DataFrame,
    save_path: Path = MODEL_DIR / "matching_model.pkl",
) -> Classifier:
    if df_features.empty:
        raise ValueError("empty feature matrix — check blocking output")

    y = create_labels(df_features, df_gt)

    avail = [c for c in FEATURE_COLS if c in df_features.columns]
    if missing := set(FEATURE_COLS) - set(avail):
        log.warning(f"  missing feature columns (skipped): {missing}")

    X = df_features[avail]
    pos = int(y.sum())
    neg = int((y == 0).sum())
    spw = neg / max(pos, 1)
    log.info(f"  scale_pos_weight = {spw:.1f}")

    clf = Classifier(scale_pos_weight=spw)
    clf.fit(X, y)
    clf.save(save_path)
    return clf
