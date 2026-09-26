import logging
from typing import Optional
from collections import Counter

import pandas as pd

from .exact_blocking import generate_exact_blocks, precompute_blocking_keys
from .fuzzy_blocking import generate_tfidf_candidates
from ..config import ID_COL, S1_ID_COL, BLOCKING_TOP_K, CAND_COL

log = logging.getLogger(__name__)

MAX_PAIRS = 10_000_000


def generate_candidates(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    top_k: int = BLOCKING_TOP_K,
    precomputed_cand: Optional[pd.DataFrame] = None,
    gram_counts: Optional[Counter] = None,
    df_cand_deduped: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run exact + TF-IDF blocking and return a pairs DataFrame [s1_id, candidate_entity_id].
    
    If df_cand_deduped is provided, skip the concat+dedup step (saves ~5 min per call).
    If precomputed_cand is provided, it's passed to exact blocking to skip candidate key computation.
    If gram_counts is provided, it's used for rarest 3-gram computation on S1.
    """
    if df_cand_deduped is not None:
        df_cand = df_cand_deduped
    else:
        df_cand = pd.concat([df_s2, df_s3], ignore_index=True).drop_duplicates(subset=[ID_COL])
    log.info(f"blocking: {len(df_s1):,} S1 vs {len(df_cand):,} candidates")

    exact = generate_exact_blocks(df_s1, df_cand, precomputed_cand=precomputed_cand, gram_counts=gram_counts)
    fuzzy = generate_tfidf_candidates(df_s1, df_cand, top_k=top_k)

    all_pairs = exact | fuzzy
    log.info(f"total candidate pairs: {len(all_pairs):,}")

    if len(all_pairs) > MAX_PAIRS:
        log.warning(f"pair cap hit ({MAX_PAIRS:,}) — truncating")
        all_pairs = set(list(all_pairs)[:MAX_PAIRS])

    return pd.DataFrame(list(all_pairs), columns=[S1_ID_COL, "candidate_entity_id"])


def format_candidate_tsv(df_pairs: pd.DataFrame, all_s1_ids: pd.Series) -> pd.DataFrame:
    """Collapse pairs into one row per S1 entity with comma-separated candidate list."""
    base = pd.DataFrame({S1_ID_COL: all_s1_ids})
    if df_pairs.empty:
        base[CAND_COL] = ""
        return base

    grouped = (
        df_pairs.groupby(S1_ID_COL)["candidate_entity_id"]
        .apply(lambda ids: ",".join(sorted(set(ids.astype(str)))))
        .reset_index()
        .rename(columns={"candidate_entity_id": CAND_COL})
    )
    result = base.merge(grouped, on=S1_ID_COL, how="left")
    result[CAND_COL] = result[CAND_COL].fillna("")
    return result
