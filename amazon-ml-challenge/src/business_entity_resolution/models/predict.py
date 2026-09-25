import logging
from pathlib import Path

import pandas as pd

from .model import Classifier
from ..features.feature_builder import FEATURE_COLS
from ..config import MODEL_DIR, THRESHOLD, S1_ID_COL, MATCHED_COL

log = logging.getLogger(__name__)


def predict(
    df_features: pd.DataFrame,
    all_s1_ids: pd.Series,
    threshold: float = THRESHOLD,
    model_path: Path = MODEL_DIR / "matching_model.pkl",
) -> pd.DataFrame:
    """Score candidate pairs and return one row per S1 entity with matched IDs."""
    base = pd.DataFrame({S1_ID_COL: all_s1_ids, MATCHED_COL: ""})
    if df_features.empty:
        return base

    clf = Classifier.load(model_path)

    missing = set(FEATURE_COLS) - set(df_features.columns)
    if missing:
        raise ValueError(
            f"feature matrix missing required columns: {sorted(missing)}. "
            "Re-run build_features or retrain the model."
        )

    probs = clf.predict_proba(df_features[FEATURE_COLS])
    hits = df_features[[S1_ID_COL, "candidate_entity_id"]].copy()
    hits["match"] = (probs >= threshold).astype(int)
    hits = hits[hits["match"] == 1]

    grouped = (
        hits.groupby(S1_ID_COL)["candidate_entity_id"]
        .apply(lambda ids: ",".join(sorted(set(ids))))
        .reset_index()
        .rename(columns={"candidate_entity_id": MATCHED_COL})
    )

    result = pd.DataFrame({S1_ID_COL: all_s1_ids}).merge(grouped, on=S1_ID_COL, how="left")
    result[MATCHED_COL] = result[MATCHED_COL].fillna("")

    n_matched = (result[MATCHED_COL] != "").sum()
    log.info(f"  {n_matched:,} matched, {len(result) - n_matched:,} singletons (threshold={threshold})")
    return result
