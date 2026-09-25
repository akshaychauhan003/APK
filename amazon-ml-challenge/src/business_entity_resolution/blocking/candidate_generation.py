"""
Candidate pair generation unifying exact and fuzzy blocking strategies.
"""

import logging
import pandas as pd

from .exact_blocking import generate_exact_blocks
from .fuzzy_blocking import generate_tfidf_candidates
from ..config import ENTITY_ID_COL, GROUND_TRUTH_S1_COL, BLOCKING_TOP_K

logger = logging.getLogger(__name__)

# Hard cap on total pairs to prevent downstream OOM.
# With 5k S1 and broad blocking this could reach millions.
MAX_TOTAL_PAIRS = 2_000_000


def generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    top_k: int = BLOCKING_TOP_K,
) -> pd.DataFrame:
    """
    Combine S2 and S3 candidates, apply blocking strategies, return candidate pairs DataFrame.

    Returns DataFrame with columns [GROUND_TRUTH_S1_COL, 'candidate_entity_id'].
    """
    logger.info(
        f"  Blocking: {len(df_s1):,} S1 vs "
        f"{len(df_s2):,} S2 + {len(df_s3):,} S3 candidates"
    )

    df_cand_all = pd.concat([df_s2, df_s3], ignore_index=True)
    df_cand_all = df_cand_all.drop_duplicates(subset=[ENTITY_ID_COL])
    logger.info(f"  Combined candidate pool: {len(df_cand_all):,}")

    # ── Exact blocking (vectorized merge) ─────────────────────────────────
    exact_pairs = generate_exact_blocks(df_s1, df_cand_all)

    # ── TF-IDF fuzzy blocking ─────────────────────────────────────────────
    fuzzy_pairs = generate_tfidf_candidates(df_s1, df_cand_all, top_k=top_k)

    # ── Combine ────────────────────────────────────────────────────────────
    all_pairs = exact_pairs | fuzzy_pairs
    logger.info(f"  Total unique candidate pairs: {len(all_pairs):,}")

    # Safety cap
    if len(all_pairs) > MAX_TOTAL_PAIRS:
        logger.warning(
            f"  Pair count ({len(all_pairs):,}) exceeds cap ({MAX_TOTAL_PAIRS:,}). "
            "Truncating — consider reducing BLOCKING_TOP_K or sample size."
        )
        all_pairs = set(list(all_pairs)[:MAX_TOTAL_PAIRS])

    df_pairs = pd.DataFrame(
        list(all_pairs),
        columns=[GROUND_TRUTH_S1_COL, "candidate_entity_id"],
    )
    return df_pairs


def format_candidate_pairs_tsv(df_pairs: pd.DataFrame, all_s1_ids: pd.Series) -> pd.DataFrame:
    """
    Format candidate pairs into TSV structure: [source1_entity_id, candidate_entity_ids].
    candidate_entity_ids is a comma-separated list (empty string for singletons).
    """
    if df_pairs.empty:
        full_df = pd.DataFrame({GROUND_TRUTH_S1_COL: all_s1_ids, "candidate_entity_ids": ""})
        return full_df

    grouped = (
        df_pairs.groupby(GROUND_TRUTH_S1_COL)["candidate_entity_id"]
        .apply(lambda ids: ",".join(sorted(set(ids.astype(str)))))
        .reset_index()
    )
    grouped.columns = [GROUND_TRUTH_S1_COL, "candidate_entity_ids"]

    # Ensure every S1 entity has exactly one row
    full_df = pd.DataFrame({GROUND_TRUTH_S1_COL: all_s1_ids})
    full_df = full_df.merge(grouped, on=GROUND_TRUTH_S1_COL, how="left")
    full_df["candidate_entity_ids"] = full_df["candidate_entity_ids"].fillna("")

    return full_df
