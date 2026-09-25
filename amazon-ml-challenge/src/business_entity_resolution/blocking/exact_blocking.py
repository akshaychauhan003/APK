"""
Exact rule-based blocking key generator.
Vectorized for large-scale datasets (millions of rows).
"""

from typing import Dict, List, Set, Tuple
import pandas as pd
from ..config import ENTITY_ID_COL, COUNTRY_COL


def _get_first_word_series(s: pd.Series) -> pd.Series:
    """Return the first whitespace-token of each string."""
    return s.str.strip().str.split().str[0].fillna("")


def generate_exact_blocks(df_s1: pd.DataFrame, df_candidates: pd.DataFrame) -> Set[Tuple[str, str]]:
    """
    Generate candidate pairs (S1_id, candidate_id) using exact blocking keys.

    Keys used:
      - (country, first_word_of_clean_name)
      - (country, postal_code)

    Returns a set of (s1_id, cand_id) tuples.
    """
    candidate_pairs: Set[Tuple[str, str]] = set()

    # ── Index candidates ──────────────────────────────────────────────────
    cand = df_candidates[[ENTITY_ID_COL, COUNTRY_COL, "clean_name", "postal_code"]].copy()
    cand["country_lc"] = cand[COUNTRY_COL].str.strip().str.lower()
    cand["first_word"] = _get_first_word_series(cand["clean_name"])

    # Name block index: key → list of candidate ids
    name_block: Dict[Tuple[str, str], List[str]] = {}
    for row in cand.itertuples(index=False):
        if row.first_word:
            key = (row.country_lc, row.first_word)
            name_block.setdefault(key, []).append(row.entity_id)

    # Postal block index
    postal_block: Dict[Tuple[str, str], List[str]] = {}
    for row in cand.itertuples(index=False):
        if row.postal_code:
            key = (row.country_lc, row.postal_code)
            postal_block.setdefault(key, []).append(row.entity_id)

    # ── Match S1 against indexes ───────────────────────────────────────────
    s1 = df_s1[[ENTITY_ID_COL, COUNTRY_COL, "clean_name", "postal_code"]].copy()
    s1["country_lc"] = s1[COUNTRY_COL].str.strip().str.lower()
    s1["first_word"] = _get_first_word_series(s1["clean_name"])

    for row in s1.itertuples(index=False):
        s1_id = row.entity_id
        country = row.country_lc

        if row.first_word:
            for cid in name_block.get((country, row.first_word), []):
                candidate_pairs.add((s1_id, cid))

        if row.postal_code:
            for cid in postal_block.get((country, row.postal_code), []):
                candidate_pairs.add((s1_id, cid))

    return candidate_pairs
