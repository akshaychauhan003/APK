"""
Exact rule-based blocking key generator.
"""

from typing import Dict, List, Set, Tuple
import pandas as pd
from ..config import ENTITY_ID_COL, COUNTRY_COL


def get_first_word(text: str) -> str:
    """Return the first word of string or empty."""
    parts = text.strip().split()
    return parts[0] if parts else ""


def generate_exact_blocks(df_s1: pd.DataFrame, df_candidates: pd.DataFrame) -> Set[Tuple[str, str]]:
    """
    Generate candidate pairs (S1_id, candidate_id) using exact blocking keys.
    """
    candidate_pairs: Set[Tuple[str, str]] = set()

    # Index candidates by exact keys
    name_blocks: Dict[Tuple[str, str], List[str]] = {}
    postal_blocks: Dict[Tuple[str, str], List[str]] = {}

    for _, row in df_candidates.iterrows():
        cand_id = row[ENTITY_ID_COL]
        country = row.get(COUNTRY_COL, "").lower()
        first_word = get_first_word(row.get("clean_name", ""))
        postal = row.get("postal_code", "")

        if first_word:
            key = (country, first_word)
            name_blocks.setdefault(key, []).append(cand_id)

        if postal:
            key = (country, postal)
            postal_blocks.setdefault(key, []).append(cand_id)

    # Match S1 records against blocks
    for _, row in df_s1.iterrows():
        s1_id = row[ENTITY_ID_COL]
        country = row.get(COUNTRY_COL, "").lower()
        first_word = get_first_word(row.get("clean_name", ""))
        postal = row.get("postal_code", "")

        if first_word:
            key = (country, first_word)
            for cand_id in name_blocks.get(key, []):
                candidate_pairs.add((s1_id, cand_id))

        if postal:
            key = (country, postal)
            for cand_id in postal_blocks.get(key, []):
                candidate_pairs.add((s1_id, cand_id))

    return candidate_pairs
