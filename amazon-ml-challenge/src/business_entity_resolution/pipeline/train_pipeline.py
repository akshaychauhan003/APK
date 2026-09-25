import logging

import pandas as pd

from ..data import load_train
from ..preprocessing import preprocess
from ..blocking import generate_candidates
from ..features import build_features
from ..models import train, predict
from ..evaluation import evaluate, split_train_val
from ..config import ID_COL, S1_ID_COL, MATCHED_COL, COUNTRY_COL, TRAIN_SAMPLE

log = logging.getLogger(__name__)

# how many S2/S3 negatives to sample per country during training
_CAND_PER_COUNTRY = 300_000


def _build_candidate_pool(df_s2, df_s3, df_gt, per_country):
    """Return a preprocessed S2+S3 pool that always includes all true-match records.
    
    Random sampling of S2/S3 almost never hits the true matches, so we guarantee
    positives and fill remaining slots with random negatives per country.
    """
    # collect all matched IDs from GT — vectorized, no iterrows
    true_ids = set(
        df_gt[MATCHED_COL]
        .str.split(",")
        .explode()
        .str.strip()
        .dropna()
        .pipe(lambda s: s[s.ne("")])
    )
    log.info(f"  guaranteed positives: {len(true_ids):,} unique matched IDs")

    combined = pd.concat([df_s2, df_s3], ignore_index=True)
    positives = combined[combined[ID_COL].isin(true_ids)]
    negatives = combined[~combined[ID_COL].isin(true_ids)]

    pos_by_country = positives.groupby(COUNTRY_COL)[ID_COL].count().to_dict()
    parts = [positives]
    for country, grp in negatives.groupby(COUNTRY_COL):
        slots = max(0, per_country - pos_by_country.get(country, 0))
        if slots:
            parts.append(grp.sample(n=min(slots, len(grp)), random_state=42))

    pool = pd.concat(parts, ignore_index=True)
    log.info(
        f"  candidate pool: {len(positives):,} pos + "
        f"{len(pool) - len(positives):,} neg = {len(pool):,} total"
    )
    return preprocess(pool)


def run_training(val_split: float = 0.20, sample_size: int = TRAIN_SAMPLE):
    log.info("loading train data...")
    df_s1, df_s2, df_s3, df_gt = load_train()
    log.info(f"  S1={len(df_s1):,}  S2={len(df_s2):,}  S3={len(df_s3):,}  GT={len(df_gt):,}")

    if sample_size and len(df_s1) > sample_size:
        log.info(f"  sampling {sample_size:,} S1 entities")
        df_s1 = df_s1.sample(n=sample_size, random_state=42).reset_index(drop=True)
        df_gt = df_gt[df_gt[S1_ID_COL].isin(set(df_s1[ID_COL]))].copy()

    log.info("preprocessing S1...")
    df_s1 = preprocess(df_s1)

    log.info("building candidate pool (S2+S3)...")
    df_pool = _build_candidate_pool(df_s2, df_s3, df_gt, _CAND_PER_COUNTRY)

    s2_pool = df_pool[df_pool[ID_COL].str.startswith("S2-")]
    s3_pool = df_pool[df_pool[ID_COL].str.startswith("S3-")]

    if val_split > 0:
        log.info(f"train/val split ({1-val_split:.0%} / {val_split:.0%})...")
        train_s1, val_s1, train_gt, val_gt = split_train_val(df_s1, df_gt, val_size=val_split)
    else:
        train_s1, train_gt = df_s1, df_gt
        val_s1 = val_gt = None

    log.info("blocking (train split)...")
    pairs_train = generate_candidates(train_s1, s2_pool, s3_pool)
    log.info(f"  {len(pairs_train):,} candidate pairs")

    if pairs_train.empty:
        log.error("no candidate pairs — check blocking config")
        return None, {}

    log.info("building features...")
    feats_train = build_features(pairs_train, train_s1, df_pool)

    log.info("training model...")
    clf = train(feats_train, train_gt)

    metrics = {}
    if val_s1 is not None:
        log.info("evaluating on validation split...")
        pairs_val = generate_candidates(val_s1, s2_pool, s3_pool)
        feats_val = build_features(pairs_val, val_s1, df_pool)
        preds     = predict(feats_val, all_s1_ids=val_s1[ID_COL])
        metrics   = evaluate(preds, val_gt)
        log.info(
            f"  P={metrics['macro_precision']:.4f}  "
            f"R={metrics['macro_recall']:.4f}  "
            f"F0.5={metrics['macro_f0.5']:.4f}"
        )

    return clf, metrics
