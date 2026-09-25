"""
Inference module for applying trained entity resolution model on candidate pairs.
"""

import pandas as pd
from pathlib import Path
from .model import EntityResolutionModel
from .train import FEATURE_COLS
from ..config import (
    MODEL_DIR,
    CLASSIFICATION_THRESHOLD,
    GROUND_TRUTH_S1_COL,
    GROUND_TRUTH_MATCHED_COL,
)


def predict_matches(
    df_features: pd.DataFrame,
    all_s1_ids: pd.Series,
    threshold: float = CLASSIFICATION_THRESHOLD,
    model_path: Path = MODEL_DIR / "matching_model.pkl",
) -> pd.DataFrame:
    """Predict matches for candidate pairs and return formatted matching_results DataFrame."""
    if df_features.empty:
        df_results = pd.DataFrame({GROUND_TRUTH_S1_COL: all_s1_ids, GROUND_TRUTH_MATCHED_COL: ""})
        return df_results

    model = EntityResolutionModel.load(model_path)
    X = df_features[FEATURE_COLS]

    probs = model.predict_proba(X)
    df_features = df_features.copy()
    df_features["prob"] = probs
    df_features["is_match"] = (probs >= threshold).astype(int)

    # Filter positive matches
    matched_pairs = df_features[df_features["is_match"] == 1]

    # Group by S1 entity id
    grouped = matched_pairs.groupby(GROUND_TRUTH_S1_COL)["candidate_entity_id"].apply(lambda ids: ",".join(sorted(set(ids)))).reset_index()
    grouped.columns = [GROUND_TRUTH_S1_COL, GROUND_TRUTH_MATCHED_COL]

    # Combine with all S1 IDs to guarantee every S1 entity appears
    df_all_s1 = pd.DataFrame({GROUND_TRUTH_S1_COL: all_s1_ids})
    df_final = df_all_s1.merge(grouped, on=GROUND_TRUTH_S1_COL, how="left")
    df_final[GROUND_TRUTH_MATCHED_COL] = df_final[GROUND_TRUTH_MATCHED_COL].fillna("")

    return df_final
