"""
Candidate pair generation unifying exact and fuzzy blocking strategies.
"""

from typing import Tuple, Dict, Set
import pandas as pd

from .exact_blocking import generate_exact_blocks
from .fuzzy_blocking import generate_tfidf_candidates
from ..config import ENTITY_ID_COL, GROUND_TRUTH_S1_COL, BLOCKING_TOP_K


def generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    top_k: int = BLOCKING_TOP_K,
) -> pd.DataFrame:
    """
    Combine S2 and S3 candidates, apply blocking strategies, and return candidate pairs DataFrame.
    """
    df_cand_all = pd.concat([df_s2, df_s3], ignore_index=True)

    exact_pairs = generate_exact_blocks(df_s1, df_cand_all)
    fuzzy_pairs = generate_tfidf_candidates(df_s1, df_cand_all, top_k=top_k)

    all_pairs = exact_pairs.union(fuzzy_pairs)

    # Convert to DataFrame
    df_pairs = pd.DataFrame(list(all_pairs), columns=[GROUND_TRUTH_S1_COL, "candidate_entity_id"])
    return df_pairs


def format_candidate_pairs_tsv(df_pairs: pd.DataFrame, all_s1_ids: pd.Series) -> pd.DataFrame:
    """
    Format candidate pairs into TSV structure with columns [source1_entity_id, candidate_entity_ids].
    """
    grouped = df_pairs.groupby(GROUND_TRUTH_S1_COL)["candidate_entity_id"].apply(lambda ids: ",".join(sorted(set(ids)))).reset_index()
    grouped.columns = [GROUND_TRUTH_S1_COL, "candidate_entity_ids"]

    # Reindex to ensure all S1 entities exist in output
    full_df = pd.DataFrame({GROUND_TRUTH_S1_COL: all_s1_ids})
    full_df = full_df.merge(grouped, on=GROUND_TRUTH_S1_COL, how="left")
    full_df["candidate_entity_ids"] = full_df["candidate_entity_ids"].fillna("")

    return full_df
