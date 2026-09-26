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


def _name_token_features(s1: pd.Series, s2: pd.Series):
    n = len(s1)
    contains = np.zeros(n, dtype=float)
    sorted_exact = np.zeros(n, dtype=float)
    overlap_ratio = np.zeros(n, dtype=float)
    token_diff = np.zeros(n, dtype=float)
    
    for i, (a, b) in enumerate(zip(s1, s2)):
        wa = a.split() if a else []
        wb = b.split() if b else []
        
        ta = set(wa)
        tb = set(wb)
        
        la = len(ta)
        lb = len(tb)
        intersect = ta & tb
        li = len(intersect)
        
        if la > 0 and lb > 0:
            if ta.issubset(tb) or tb.issubset(ta):
                contains[i] = 1.0
            
            if ta == tb:
                sorted_exact[i] = 1.0
                
            overlap_ratio[i] = li / min(la, lb)
            
        max_tok = max(len(wa), len(wb))
        if max_tok > 0:
            token_diff[i] = abs(len(wa) - len(wb)) / max_tok
            
    return contains, sorted_exact, overlap_ratio, token_diff


def _first_word_match(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    out = np.empty(len(s1), dtype=float)
    for i, (a, b) in enumerate(zip(s1, s2)):
        wa = a.split()[0] if a and a.split() else ""
        wb = b.split()[0] if b and b.split() else ""
        out[i] = 1.0 if (wa and wa == wb) else 0.0
    return out


def _name_fuzz(s1: pd.Series, s2: pd.Series):
    """ratio, token_sort, token_set in a single pass instead of 3 separate loops."""
    n = len(s1)
    ratio = np.empty(n, dtype=float)
    tsort = np.empty(n, dtype=float)
    tset  = np.empty(n, dtype=float)
    if _HAS_RF:
        for i, (a, b) in enumerate(zip(s1, s2)):
            ratio[i] = _fuzz.ratio(a, b) / 100.0
            tsort[i] = _fuzz.token_sort_ratio(a, b) / 100.0
            tset[i]  = _fuzz.token_set_ratio(a, b) / 100.0
    else:
        for i, (a, b) in enumerate(zip(s1, s2)):
            ratio[i] = SequenceMatcher(None, a, b).ratio()
            ta = set(a.split()) if a else set()
            tb = set(b.split()) if b else set()
            u = ta | tb
            j = len(ta & tb) / len(u) if u else 0.0
            tsort[i] = j
            tset[i]  = j
    return ratio, tsort, tset


def _addr_fuzz(s1: pd.Series, s2: pd.Series):
    """ratio and token_set in a single pass."""
    n = len(s1)
    ratio = np.empty(n, dtype=float)
    tset  = np.empty(n, dtype=float)
    if _HAS_RF:
        for i, (a, b) in enumerate(zip(s1, s2)):
            ratio[i] = _fuzz.ratio(a, b) / 100.0
            tset[i]  = _fuzz.token_set_ratio(a, b) / 100.0
    else:
        for i, (a, b) in enumerate(zip(s1, s2)):
            ratio[i] = SequenceMatcher(None, a, b).ratio()
            ta = set(a.split()) if a else set()
            tb = set(b.split()) if b else set()
            u = ta | tb
            tset[i] = len(ta & tb) / len(u) if u else 0.0
    return ratio, tset


def _digit_features(s1: pd.Series, s2: pd.Series):
    n = len(s1)
    jacc = np.zeros(n, dtype=float)
    exact = np.zeros(n, dtype=float)
    for i, (a, b) in enumerate(zip(s1, s2)):
        da = set(re.findall(r"\d+", a or ""))
        db = set(re.findall(r"\d+", b or ""))
        u = da | db
        if u:
            jacc[i] = len(da & db) / len(u)
            if da == db:
                exact[i] = 1.0
        else:
            jacc[i] = 1.0
            exact[i] = 1.0
    return jacc, exact


def _soundex(s: pd.Series) -> pd.Series:
    if not _HAS_JF:
        return pd.Series([""] * len(s), index=s.index)
    def _sdx(text):
        w = text.split()[0] if text and text.split() else ""
        try:
            return _jf.soundex(w) if w else ""
        except Exception:
            return ""
    return s.apply(_sdx)


def _metaphone(s: pd.Series) -> pd.Series:
    if not _HAS_JF:
        return pd.Series([""] * len(s), index=s.index)
    def _mp(text):
        w = text.split()[0] if text and text.split() else ""
        try:
            return _jf.metaphone(w) if w else ""
        except Exception:
            return ""
    return s.apply(_mp)


def _jaro_winkler(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    if not _HAS_JF:
        return np.zeros(len(s1), dtype=float)
    out = np.zeros(len(s1), dtype=float)
    for i, (a, b) in enumerate(zip(s1, s2)):
        out[i] = _jf.jaro_winkler_similarity(a or "", b or "")
    return out


def _prefix_match(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    out = np.zeros(len(s1), dtype=float)
    for i, (a, b) in enumerate(zip(s1, s2)):
        a = a or ""
        b = b or ""
        max_len = max(len(a), len(b))
        if max_len == 0:
            continue
        c = 0
        for ca, cb in zip(a, b):
            if ca == cb:
                c += 1
            else:
                break
        out[i] = c / max_len
    return out


def _char_jaccard(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    out = np.zeros(len(s1), dtype=float)
    for i, (a, b) in enumerate(zip(s1, s2)):
        a = a or ""
        b = b or ""
        bg_a = set(a[j:j+2] for j in range(len(a)-1))
        bg_b = set(b[j:j+2] for j in range(len(b)-1))
        u = bg_a | bg_b
        out[i] = len(bg_a & bg_b) / len(u) if u else 0.0
    return out


def _match_arr(a: pd.Series, b: pd.Series) -> np.ndarray:
    return np.array([1.0 if (x and y and x == y) else 0.0 for x, y in zip(a, b)], dtype=float)


# ---- main builder ----

FEATURE_COLS = [
    "name_jaccard", "name_first_word", "name_fuzz", "name_token_sort", "name_token_set",
    "name_len_diff", "name_soundex", "name_metaphone",
    "addr_jaccard", "postal_match", "country_match", "digit_jaccard",
    "addr_fuzz", "addr_token_set",
    "name_jaro_winkler", "name_contains", "name_sorted_tokens_exact", "name_prefix_match",
    "name_char_jaccard", "addr_jaro_winkler", "name_token_overlap_ratio",
    "name_length_ratio", "addr_digit_exact", "num_token_diff"
]


def build_features(df_pairs: pd.DataFrame, df_s1: pd.DataFrame, df_cand: pd.DataFrame) -> pd.DataFrame:
    """Build similarity features for candidate pairs."""
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
    
    min_len = np.minimum(len1, len2)
    len_ratio = np.where(max_len > 0, min_len / max_len, 0.0)

    sdx1 = _soundex(n1);   sdx2 = _soundex(n2)
    mp1  = _metaphone(n1); mp2  = _metaphone(n2)

    # single pass per field group avoids redundant iteration over pairs
    n_fuzz, n_tsort, n_tset = _name_fuzz(n1, n2)
    a_fuzz, a_tset = _addr_fuzz(a1, a2)
    
    n_contains, n_sorted_exact, n_overlap_ratio, n_token_diff = _name_token_features(n1, n2)
    a_digit_jacc, a_digit_exact = _digit_features(a1, a2)

    df_out = pd.DataFrame({
        S1_ID_COL:               s1_ids,
        "candidate_entity_id":   cand_ids,
        "name_jaccard":          _jaccard(n1, n2),
        "name_first_word":       _first_word_match(n1, n2),
        "name_fuzz":             n_fuzz,
        "name_token_sort":       n_tsort,
        "name_token_set":        n_tset,
        "name_len_diff":         len_diff,
        "name_soundex":          _match_arr(sdx1, sdx2),
        "name_metaphone":        _match_arr(mp1, mp2),
        "addr_jaccard":          _jaccard(a1, a2),
        "postal_match":          postal,
        "country_match":         (c1.values == c2.values).astype(float),
        "digit_jaccard":         a_digit_jacc,
        "addr_fuzz":             a_fuzz,
        "addr_token_set":        a_tset,
        "name_jaro_winkler":     _jaro_winkler(n1, n2),
        "name_contains":         n_contains,
        "name_sorted_tokens_exact": n_sorted_exact,
        "name_prefix_match":     _prefix_match(n1, n2),
        "name_char_jaccard":     _char_jaccard(n1, n2),
        "addr_jaro_winkler":     _jaro_winkler(a1, a2),
        "name_token_overlap_ratio": n_overlap_ratio,
        "name_length_ratio":     len_ratio,
        "addr_digit_exact":      a_digit_exact,
        "num_token_diff":        n_token_diff,
    })

    log.info(f"feature matrix: {df_out.shape}")
    return df_out
