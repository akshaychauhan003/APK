import re
import logging

import numpy as np
import pandas as pd

from ..config import ID_COL, COUNTRY_COL, S1_ID_COL

log = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as _fuzz
    _HAS_RF = True
except ImportError:
    from difflib import SequenceMatcher
    _HAS_RF = False

try:
    import jellyfish as _jf
    _HAS_JF = True
except ImportError:
    _HAS_JF = False


# ---- similarity helpers ----

def _jaccard(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    out = np.zeros(len(s1), dtype=float)
    for i, (a, b) in enumerate(zip(s1, s2)):
        ta = set(a.split()) if a else set()
        tb = set(b.split()) if b else set()
        u = ta | tb
        out[i] = len(ta & tb) / len(u) if u else 0.0
    return out


def _fuzz_ratio(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    if _HAS_RF:
        return np.array([_fuzz.ratio(a, b) / 100.0 for a, b in zip(s1, s2)], dtype=float)
    return np.array([SequenceMatcher(None, a, b).ratio() for a, b in zip(s1, s2)], dtype=float)


def _token_sort(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    if _HAS_RF:
        return np.array([_fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1, s2)], dtype=float)
    return _jaccard(s1, s2)


def _token_set(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    if _HAS_RF:
        return np.array([_fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1, s2)], dtype=float)
    return _jaccard(s1, s2)


def _digit_jaccard(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    out = np.zeros(len(s1), dtype=float)
    for i, (a, b) in enumerate(zip(s1, s2)):
        da = set(re.findall(r"\d+", a))
        db = set(re.findall(r"\d+", b))
        u = da | db
        out[i] = len(da & db) / len(u) if u else 1.0
    return out


def _first_word_match(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    def fw(s):
        p = s.split()
        return p[0] if p else ""
    return np.array([1.0 if fw(a) and fw(a) == fw(b) else 0.0 for a, b in zip(s1, s2)], dtype=float)


def _soundex(s: pd.Series) -> pd.Series:
    if not _HAS_JF:
        return pd.Series([""] * len(s), index=s.index)
    def _sdx(text):
        w = text.split()[0] if text.split() else ""
        try:
            return _jf.soundex(w) if w else ""
        except Exception:
            return ""
    return s.apply(_sdx)


def _metaphone(s: pd.Series) -> pd.Series:
    if not _HAS_JF:
        return pd.Series([""] * len(s), index=s.index)
    def _mp(text):
        w = text.split()[0] if text.split() else ""
        try:
            return _jf.metaphone(w) if w else ""
        except Exception:
            return ""
    return s.apply(_mp)


def _match_arr(a: pd.Series, b: pd.Series) -> np.ndarray:
    return np.array([1.0 if (x and y and x == y) else 0.0 for x, y in zip(a, b)], dtype=float)


# ---- main builder ----

FEATURE_COLS = [
    "name_jaccard", "name_first_word", "name_fuzz", "name_token_sort", "name_token_set",
    "name_len_diff", "name_soundex", "name_metaphone",
    "addr_jaccard", "postal_match", "country_match", "digit_jaccard",
    "addr_fuzz", "addr_token_set",
]


def build_features(df_pairs: pd.DataFrame, df_s1: pd.DataFrame, df_cand: pd.DataFrame) -> pd.DataFrame:
    """Build 14-feature similarity matrix for all candidate pairs."""
    if df_pairs.empty:
        return pd.DataFrame()

    log.info(f"building features for {len(df_pairs):,} pairs")

    s1_lu   = df_s1.set_index(ID_COL).to_dict(orient="index")
    cand_lu = df_cand.set_index(ID_COL).to_dict(orient="index")

    s1_ids   = df_pairs[S1_ID_COL].values
    cand_ids = df_pairs["candidate_entity_id"].values

    def col(lu, ids, field):
        return [lu.get(i, {}).get(field, "") for i in ids]

    n1 = pd.Series(col(s1_lu, s1_ids, "clean_name"))
    n2 = pd.Series(col(cand_lu, cand_ids, "clean_name"))
    a1 = pd.Series(col(s1_lu, s1_ids, "clean_addr"))
    a2 = pd.Series(col(cand_lu, cand_ids, "clean_addr"))
    p1 = pd.Series(col(s1_lu, s1_ids, "postal"))
    p2 = pd.Series(col(cand_lu, cand_ids, "postal"))
    c1 = pd.Series(col(s1_lu, s1_ids, COUNTRY_COL)).str.strip().str.lower()
    c2 = pd.Series(col(cand_lu, cand_ids, COUNTRY_COL)).str.strip().str.lower()

    # postal: 1.0 both match, 0.5 one missing, 0.0 mismatch
    postal = np.array([
        1.0 if (x and y and x == y) else (0.5 if not x or not y else 0.0)
        for x, y in zip(p1, p2)
    ], dtype=float)

    len1    = n1.str.len().values.astype(float)
    len2    = n2.str.len().values.astype(float)
    max_len = np.maximum(len1, len2)
    len_diff = np.where(max_len > 0, np.abs(len1 - len2) / max_len, 0.0)

    sdx1 = _soundex(n1);   sdx2 = _soundex(n2)
    mp1  = _metaphone(n1); mp2  = _metaphone(n2)

    df_out = pd.DataFrame({
        S1_ID_COL:               s1_ids,
        "candidate_entity_id":   cand_ids,
        "name_jaccard":          _jaccard(n1, n2),
        "name_first_word":       _first_word_match(n1, n2),
        "name_fuzz":             _fuzz_ratio(n1, n2),
        "name_token_sort":       _token_sort(n1, n2),
        "name_token_set":        _token_set(n1, n2),
        "name_len_diff":         len_diff,
        "name_soundex":          _match_arr(sdx1, sdx2),
        "name_metaphone":        _match_arr(mp1, mp2),
        "addr_jaccard":          _jaccard(a1, a2),
        "postal_match":          postal,
        "country_match":         (c1.values == c2.values).astype(float),
        "digit_jaccard":         _digit_jaccard(a1, a2),
        "addr_fuzz":             _fuzz_ratio(a1, a2),
        "addr_token_set":        _token_set(a1, a2),
    })

    log.info(f"feature matrix: {df_out.shape}")
    return df_out
