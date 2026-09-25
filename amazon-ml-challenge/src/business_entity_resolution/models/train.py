"""
Model training routine for entity matching model.

Handles class imbalance (few positives vs. many negatives in candidate pairs)
by computing class weights and passing them to the classifier.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

from .model import EntityResolutionModel
from ..config import MODEL_DIR, GROUND_TRUTH_S1_COL, GROUND_TRUTH_MATCHED_COL

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "name_jaccard",
    "name_first_word_match",
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_len_diff",
    "name_soundex_match",       # phonetic: catches Corp/Korp style typos
    "name_metaphone_match",     # phonetic: double-consonant variant matching
    "addr_jaccard",
    "postal_match",
    "country_match",
    "digit_jaccard",
    "addr_fuzz_ratio",
    "addr_token_set_ratio",
]


def create_training_labels(df_features: pd.DataFrame, df_gt: pd.DataFrame) -> pd.Series:
    """
    Map candidate pairs to binary labels (1 = match, 0 = non-match).

    Vectorized: builds a lookup set then uses pandas merge, avoiding iterrows().
    """
    # Build a flat set of (s1_id, cand_id) positive pairs from ground truth
    positive_pairs = set()
    for _, row in df_gt.iterrows():
        s1_id = row[GROUND_TRUTH_S1_COL]
        matched_str = str(row.get(GROUND_TRUTH_MATCHED_COL, "")).strip()
        if matched_str and matched_str.lower() not in ("nan", ""):
            for cid in matched_str.split(","):
                cid = cid.strip()
                if cid:
                    positive_pairs.add((s1_id, cid))

    # Vectorized label assignment
    s1_ids = df_features[GROUND_TRUTH_S1_COL].values
    cand_ids = df_features["candidate_entity_id"].values
    labels = np.array(
        [1 if (s1, c) in positive_pairs else 0 for s1, c in zip(s1_ids, cand_ids)],
        dtype=np.int8,
    )
    pos_count = labels.sum()
    neg_count = len(labels) - pos_count
    logger.info(
        f"  Labels: {pos_count:,} positives / {neg_count:,} negatives "
        f"(imbalance ratio = {neg_count / max(pos_count, 1):.1f}:1)"
    )
    return pd.Series(labels, name="label")


def train_matching_model(
    df_features: pd.DataFrame,
    df_gt: pd.DataFrame,
    save_path: Path = MODEL_DIR / "matching_model.pkl",
) -> EntityResolutionModel:
    """Train matching model with class-imbalance handling and save weights."""
    if df_features.empty:
        raise ValueError("Feature matrix is empty — check blocking stage output.")

    y = create_training_labels(df_features, df_gt)

    # Only keep feature columns that exist in the DataFrame
    available_cols = [c for c in FEATURE_COLS if c in df_features.columns]
    missing = set(FEATURE_COLS) - set(available_cols)
    if missing:
        logger.warning(f"  Missing feature columns (will be skipped): {missing}")

    X = df_features[available_cols]

    # Compute class weight: scale_pos_weight = neg / pos for imbalanced data
    pos = int(y.sum())
    neg = int((y == 0).sum())
    scale_pos_weight = neg / max(pos, 1)
    logger.info(f"  scale_pos_weight = {scale_pos_weight:.2f}")

    model = EntityResolutionModel(params={"scale_pos_weight": scale_pos_weight})
    model.fit(X, y)
    model.save(save_path)

    return model
