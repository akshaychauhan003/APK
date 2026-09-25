"""
Exact rule-based blocking key generator.
Fully vectorized using pandas merge for production-scale datasets (10M+ rows).
"""

import logging
from typing import Set, Tuple
import pandas as pd

from ..config import ENTITY_ID_COL, COUNTRY_COL

logger = logging.getLogger(__name__)

# Generic first words that match so many businesses they're useless as blocking keys.
# Blocking on "the" or "global" would generate millions of false pairs.
_NAME_BLOCK_STOPWORDS = {
    "the", "a", "an", "new", "old", "national", "global", "international",
    "american", "general", "united", "first", "best", "top", "prime",
    "great", "good", "super", "mega", "metro", "city", "state", "central",
    "north", "south", "east", "west", "royal", "golden", "blue", "green",
    "red", "white", "black", "star", "sun", "sky", "land", "home", "world",
    "modern", "standard", "professional", "premium", "elite", "alpha",
    "omega", "apex", "delta", "sigma", "shri", "sri", "m", "s", "r", "k",
}


def _get_first_word_series(s: pd.Series) -> pd.Series:
    """Return the first whitespace-token of each string."""
    return s.str.strip().str.split().str[0].fillna("")


def generate_exact_blocks(df_s1: pd.DataFrame, df_candidates: pd.DataFrame) -> Set[Tuple[str, str]]:
    """
    Generate candidate pairs (S1_id, candidate_id) using exact blocking keys.
    Fully vectorized with pandas merge — no Python loops over rows.

    Keys:
      - (country_lc, first_word_of_clean_name)
      - (country_lc, postal_code)  [only when postal_code is non-empty]
    """
    # ── Prepare S1 keys ──────────────────────────────────────────────────
    s1 = df_s1[[ENTITY_ID_COL, COUNTRY_COL, "clean_name", "postal_code"]].copy()
    s1 = s1.rename(columns={ENTITY_ID_COL: "s1_id"})
    s1["country_lc"] = s1[COUNTRY_COL].str.strip().str.lower()
    s1["first_word"] = _get_first_word_series(s1["clean_name"])

    # ── Prepare candidate keys ────────────────────────────────────────────
    cand = df_candidates[[ENTITY_ID_COL, COUNTRY_COL, "clean_name", "postal_code"]].copy()
    cand = cand.rename(columns={ENTITY_ID_COL: "cand_id"})
    cand["country_lc"] = cand[COUNTRY_COL].str.strip().str.lower()
    cand["first_word"] = _get_first_word_series(cand["clean_name"])

    pairs_list = []

    # ── Key 1: (country, first_word) — name block ─────────────────────────
    # Filter out stopwords to avoid pair explosion on generic names
    s1_name = s1[
        (s1["first_word"] != "") & (~s1["first_word"].isin(_NAME_BLOCK_STOPWORDS))
    ][["s1_id", "country_lc", "first_word"]]
    cand_name = cand[
        (cand["first_word"] != "") & (~cand["first_word"].isin(_NAME_BLOCK_STOPWORDS))
    ][["cand_id", "country_lc", "first_word"]]
    if not s1_name.empty and not cand_name.empty:
        merged_name = s1_name.merge(cand_name, on=["country_lc", "first_word"], how="inner")
        pairs_list.append(merged_name[["s1_id", "cand_id"]])
        logger.info(f"  Exact-name block: {len(merged_name):,} pairs")

    # ── Key 2: (country, postal_code) — merge on postal block ─────────────
    s1_post = s1[s1["postal_code"] != ""][["s1_id", "country_lc", "postal_code"]]
    cand_post = cand[cand["postal_code"] != ""][["cand_id", "country_lc", "postal_code"]]
    if not s1_post.empty and not cand_post.empty:
        merged_post = s1_post.merge(cand_post, on=["country_lc", "postal_code"], how="inner")
        pairs_list.append(merged_post[["s1_id", "cand_id"]])
        logger.info(f"  Exact-postal block: {len(merged_post):,} pairs")

    if not pairs_list:
        return set()

    all_pairs = pd.concat(pairs_list, ignore_index=True).drop_duplicates()
    result = set(zip(all_pairs["s1_id"], all_pairs["cand_id"]))
    logger.info(f"  Exact blocking total unique pairs: {len(result):,}")
    return result
