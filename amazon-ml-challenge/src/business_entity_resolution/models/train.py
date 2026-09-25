"""
Model training routine for entity matching model.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple

from .model import EntityResolutionModel
from ..config import MODEL_DIR, GROUND_TRUTH_S1_COL, GROUND_TRUTH_MATCHED_COL

FEATURE_COLS = [
    "name_jaccard",
    "name_first_word_match",
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_len_diff",
    "name_soundex_match",      # phonetic: catches Corp/Korp style typos
    "name_metaphone_match",    # phonetic: double-consonant variant matching
    "addr_jaccard",
    "postal_match",
    "country_match",
    "digit_jaccard",
    "addr_fuzz_ratio",
    "addr_token_set_ratio",
]


def create_training_labels(df_features: pd.DataFrame, df_gt: pd.DataFrame) -> pd.Series:
    """Map candidate pairs to binary labels (1 = match, 0 = non-match) using ground truth."""
    gt_map = {}
    for _, row in df_gt.iterrows():
        s1_id = row[GROUND_TRUTH_S1_COL]
        matched_str = str(row[GROUND_TRUTH_MATCHED_COL])
        matched_set = set(matched_str.split(",")) if matched_str else set()
        gt_map[s1_id] = matched_set

    labels = []
    for _, row in df_features.iterrows():
        s1_id = row[GROUND_TRUTH_S1_COL]
        cand_id = row["candidate_entity_id"]
        is_match = 1 if cand_id in gt_map.get(s1_id, set()) else 0
        labels.append(is_match)

    return pd.Series(labels, name="label")


def train_matching_model(
    df_features: pd.DataFrame,
    df_gt: pd.DataFrame,
    save_path: Path = MODEL_DIR / "matching_model.pkl",
) -> EntityResolutionModel:
    """Train matching model and save weights."""
    y = create_training_labels(df_features, df_gt)
    X = df_features[FEATURE_COLS]

    model = EntityResolutionModel()
    model.fit(X, y)
    model.save(save_path)

    return model
