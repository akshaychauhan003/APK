"""
Feature builder module combining name and address feature extractors.

Vectorized implementation for production-scale datasets (millions of pairs).
"""

import pandas as pd
import numpy as np
from typing import List

from .name_features import compute_name_features
from .address_features import compute_address_features
from ..config import ENTITY_ID_COL, COUNTRY_COL, GROUND_TRUTH_S1_COL

import logging
logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as _rfuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def _safe_div(a, b):
    """Element-wise safe division; returns 0 where b == 0."""
    result = np.where(b > 0, a / b, 0.0)
    return result.astype(float)


def _jaccard_tokens(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    """Compute token-level Jaccard similarity for two string Series (vectorized)."""
    results = np.zeros(len(s1), dtype=float)
    for i, (a, b) in enumerate(zip(s1, s2)):
        ta = set(a.split()) if a else set()
        tb = set(b.split()) if b else set()
        u = ta | tb
        results[i] = len(ta & tb) / len(u) if u else 0.0
    return results


def _first_word_match(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    """Binary: whether the first token of each string matches."""
    def fw(s):
        parts = s.split()
        return parts[0] if parts else ""
    return np.array([
        1.0 if fw(a) and fw(b) and fw(a) == fw(b) else 0.0
        for a, b in zip(s1, s2)
    ], dtype=float)


def _fuzz_ratio_series(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    if HAS_RAPIDFUZZ:
        return np.array([_rfuzz.ratio(a, b) / 100.0 for a, b in zip(s1, s2)], dtype=float)
    from difflib import SequenceMatcher
    return np.array([SequenceMatcher(None, a, b).ratio() for a, b in zip(s1, s2)], dtype=float)


def _fuzz_token_sort_series(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    if HAS_RAPIDFUZZ:
        return np.array([_rfuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1, s2)], dtype=float)
    return _jaccard_tokens(s1, s2)  # fallback


def _fuzz_token_set_series(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    if HAS_RAPIDFUZZ:
        return np.array([_rfuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1, s2)], dtype=float)
    return _jaccard_tokens(s1, s2)  # fallback


def _digit_jaccard(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    import re
    results = np.zeros(len(s1), dtype=float)
    for i, (a, b) in enumerate(zip(s1, s2)):
        da = set(re.findall(r"\d+", a))
        db = set(re.findall(r"\d+", b))
        u = da | db
        results[i] = len(da & db) / len(u) if u else 1.0
    return results


def _postal_match_series(p1: pd.Series, p2: pd.Series) -> np.ndarray:
    results = np.zeros(len(p1), dtype=float)
    for i, (a, b) in enumerate(zip(p1, p2)):
        if a and b:
            results[i] = 1.0 if a == b else 0.0
        else:
            results[i] = 0.5  # one missing — give partial credit
    return results


def _soundex_series(s: pd.Series) -> pd.Series:
    try:
        import jellyfish
        def sdx(text):
            first = text.split()[0] if text.split() else ""
            try:
                return jellyfish.soundex(first) if first else ""
            except Exception:
                return ""
        return s.apply(sdx)
    except ImportError:
        return pd.Series([""] * len(s), index=s.index)


def _metaphone_series(s: pd.Series) -> pd.Series:
    try:
        import jellyfish
        def mp(text):
            first = text.split()[0] if text.split() else ""
            try:
                return jellyfish.metaphone(first) if first else ""
            except Exception:
                return ""
        return s.apply(mp)
    except ImportError:
        return pd.Series([""] * len(s), index=s.index)


def build_feature_matrix(
    df_pairs: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build feature matrix for all candidate pairs in df_pairs.

    Vectorized: avoids iterrows() for large-scale datasets.
    """
    if df_pairs.empty:
        return pd.DataFrame()

    logger.info(f"  Building features for {len(df_pairs):,} candidate pairs...")

    # Build lookup dicts once
    s1_dict = df_s1.set_index(ENTITY_ID_COL).to_dict(orient="index")
    cand_dict = df_candidates.set_index(ENTITY_ID_COL).to_dict(orient="index")

    # Pull columns for both sides of each pair
    s1_ids = df_pairs[GROUND_TRUTH_S1_COL].values
    cand_ids = df_pairs["candidate_entity_id"].values

    def get_col(lookup, ids, col):
        return [lookup.get(i, {}).get(col, "") for i in ids]

    n1 = pd.Series(get_col(s1_dict, s1_ids, "clean_name"))
    n2 = pd.Series(get_col(cand_dict, cand_ids, "clean_name"))
    a1 = pd.Series(get_col(s1_dict, s1_ids, "clean_address"))
    a2 = pd.Series(get_col(cand_dict, cand_ids, "clean_address"))
    p1 = pd.Series(get_col(s1_dict, s1_ids, "postal_code"))
    p2 = pd.Series(get_col(cand_dict, cand_ids, "postal_code"))
    c1 = pd.Series(get_col(s1_dict, s1_ids, COUNTRY_COL)).str.strip().str.lower()
    c2 = pd.Series(get_col(cand_dict, cand_ids, COUNTRY_COL)).str.strip().str.lower()

    # ── Name features ──────────────────────────────────────────────────────
    name_jaccard        = _jaccard_tokens(n1, n2)
    name_first_word     = _first_word_match(n1, n2)
    name_fuzz_ratio     = _fuzz_ratio_series(n1, n2)
    name_token_sort     = _fuzz_token_sort_series(n1, n2)
    name_token_set      = _fuzz_token_set_series(n1, n2)
    len1 = n1.str.len().values.astype(float)
    len2 = n2.str.len().values.astype(float)
    max_len = np.maximum(len1, len2)
    name_len_diff       = _safe_div(np.abs(len1 - len2), max_len)

    sdx1 = _soundex_series(n1)
    sdx2 = _soundex_series(n2)
    name_soundex_match  = np.array([
        1.0 if (a and b and a == b) else 0.0
        for a, b in zip(sdx1, sdx2)
    ], dtype=float)

    mp1 = _metaphone_series(n1)
    mp2 = _metaphone_series(n2)
    name_metaphone_match = np.array([
        1.0 if (a and b and a == b) else 0.0
        for a, b in zip(mp1, mp2)
    ], dtype=float)

    # ── Address features ───────────────────────────────────────────────────
    addr_jaccard        = _jaccard_tokens(a1, a2)
    postal_match        = _postal_match_series(p1, p2)
    country_match       = (c1.values == c2.values).astype(float)
    digit_jaccard       = _digit_jaccard(a1, a2)
    addr_fuzz_ratio     = _fuzz_ratio_series(a1, a2)
    addr_token_set      = _fuzz_token_set_series(a1, a2)

    df_feats = pd.DataFrame({
        GROUND_TRUTH_S1_COL:       s1_ids,
        "candidate_entity_id":     cand_ids,
        # name features
        "name_jaccard":            name_jaccard,
        "name_first_word_match":   name_first_word,
        "name_fuzz_ratio":         name_fuzz_ratio,
        "name_token_sort_ratio":   name_token_sort,
        "name_token_set_ratio":    name_token_set,
        "name_len_diff":           name_len_diff,
        "name_soundex_match":      name_soundex_match,
        "name_metaphone_match":    name_metaphone_match,
        # address features
        "addr_jaccard":            addr_jaccard,
        "postal_match":            postal_match,
        "country_match":           country_match,
        "digit_jaccard":           digit_jaccard,
        "addr_fuzz_ratio":         addr_fuzz_ratio,
        "addr_token_set_ratio":    addr_token_set,
    })

    logger.info(f"  Feature matrix built: {df_feats.shape}")
    return df_feats
