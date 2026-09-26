import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .model import Classifier
from ..features.feature_builder import FEATURE_COLS
from ..config import MODEL_DIR, THRESHOLD, S1_ID_COL, MATCHED_COL

log = logging.getLogger(__name__)

# Minimum margin the top candidate must have over the runner-up to be accepted.
# This directly targets the singleton rule: if no candidate is clearly better
# than the rest, predict empty (scoring 1.0 for true singletons instead of 0.0).
MARGIN_THRESHOLD = 0.10


def predict(
    df_features: pd.DataFrame,
    all_s1_ids: pd.Series,
    threshold: float = THRESHOLD,
    model_path: Path = MODEL_DIR / "matching_model.pkl",
) -> pd.DataFrame:
    """Score candidate pairs and return one row per S1 entity with matched IDs.
    
    Uses the model's stored feature_names for backward compatibility.
    Applies singleton rejection: only accepts matches when the top candidate
    score clears a margin over the runner-up.
    """
    base = pd.DataFrame({S1_ID_COL: all_s1_ids, MATCHED_COL: ""})
    if df_features.empty:
        return base

    clf = Classifier.load(model_path)

    # Use the model's trained feature set if available, otherwise fall back to FEATURE_COLS
    if clf.feature_names is not None:
        feature_cols = [c for c in clf.feature_names if c in df_features.columns]
        if missing := set(clf.feature_names) - set(feature_cols):
            log.warning(f"model expects features not in data (using 0.0): {sorted(missing)}")
            for col in missing:
                df_features[col] = 0.0
            feature_cols = list(clf.feature_names)
    else:
        feature_cols = [c for c in FEATURE_COLS if c in df_features.columns]
        missing = set(FEATURE_COLS) - set(feature_cols)
        if missing:
            log.warning(f"feature matrix missing columns (legacy model): {sorted(missing)}")
            for col in missing:
                df_features[col] = 0.0
            feature_cols = list(FEATURE_COLS)

    probs = clf.predict_proba(df_features[feature_cols])
    hits = df_features[[S1_ID_COL, "candidate_entity_id"]].copy()
    hits["prob"] = probs

    # Step 1: Filter by threshold
    hits_above = hits[hits["prob"] >= threshold].copy()

    # Step 2: Singleton rejection — for each S1 entity, check if the top
    # candidate has sufficient margin over the second-best candidate.
    # This prevents low-confidence matches that hurt F_0.5 via false positives.
    if not hits_above.empty:
        # Compute per-entity statistics for margin check
        entity_stats = hits.groupby(S1_ID_COL)["prob"].agg(["max", "count"]).reset_index()
        entity_stats.columns = [S1_ID_COL, "top_score", "n_candidates"]

        # Get second-best score per entity
        def second_best(group):
            if len(group) < 2:
                return 0.0
            return group.nlargest(2).iloc[-1]

        second_best_scores = hits.groupby(S1_ID_COL)["prob"].apply(second_best).reset_index()
        second_best_scores.columns = [S1_ID_COL, "second_score"]

        entity_stats = entity_stats.merge(second_best_scores, on=S1_ID_COL)
        entity_stats["margin"] = entity_stats["top_score"] - entity_stats["second_score"]

        # Only reject if the entity has exactly 1 match above threshold AND
        # the margin is too thin (ambiguous match)
        match_counts = hits_above.groupby(S1_ID_COL).size().reset_index(name="n_above")
        entity_stats = entity_stats.merge(match_counts, on=S1_ID_COL, how="left")
        entity_stats["n_above"] = entity_stats["n_above"].fillna(0).astype(int)

        # Entities to reject: single match with very low margin and moderate score
        reject_mask = (
            (entity_stats["n_above"] == 1) &
            (entity_stats["margin"] < MARGIN_THRESHOLD) &
            (entity_stats["top_score"] < threshold + 0.05)
        )
        reject_ids = set(entity_stats[reject_mask][S1_ID_COL])
        if reject_ids:
            log.info(f"  singleton rejection: dropping {len(reject_ids):,} ambiguous matches")
            hits_above = hits_above[~hits_above[S1_ID_COL].isin(reject_ids)]

    grouped = (
        hits_above.groupby(S1_ID_COL)["candidate_entity_id"]
        .apply(lambda ids: ",".join(sorted(set(ids))))
        .reset_index()
        .rename(columns={"candidate_entity_id": MATCHED_COL})
    )

    result = pd.DataFrame({S1_ID_COL: all_s1_ids}).merge(grouped, on=S1_ID_COL, how="left")
    result[MATCHED_COL] = result[MATCHED_COL].fillna("")

    n_matched = (result[MATCHED_COL] != "").sum()
    log.info(f"  {n_matched:,} matched, {len(result) - n_matched:,} singletons (threshold={threshold})")
    return result
