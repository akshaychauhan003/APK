import logging
import re
from typing import Set, Tuple, Optional
from collections import Counter

import pandas as pd
try:
    import jellyfish
    _HAS_JF = True
except ImportError:
    _HAS_JF = False

from ..config import ID_COL, COUNTRY_COL

log = logging.getLogger(__name__)

# first words that are too common to use as blocking keys — would create cartesian explosions
_STOPWORDS = {
    "the", "a", "an", "new", "old", "national", "global", "international",
    "american", "general", "united", "first", "best", "top", "prime",
    "great", "good", "super", "mega", "metro", "city", "state", "central",
    "north", "south", "east", "west", "royal", "golden", "blue", "green",
    "red", "white", "black", "star", "sun", "sky", "land", "home", "world",
    "modern", "standard", "professional", "premium", "elite", "alpha",
    "omega", "apex", "delta", "sigma", "shri", "sri", "m", "s", "r", "k",
}


def _get_sorted_2word(text):
    tokens = str(text).strip().split()
    if len(tokens) >= 2:
        return "".join(sorted(tokens[:2]))
    elif len(tokens) == 1:
        return tokens[0]
    return ""

def _get_sorted_numbers(text):
    nums = re.findall(r'\d+', str(text))
    return "".join(sorted(nums))


def precompute_blocking_keys(df: pd.DataFrame, role: str = "cand"):
    """Pre-compute all blocking keys for a DataFrame. Call once for candidates.
    
    Returns:
        If role == "cand": (DataFrame, gram_counts Counter)
        If role != "cand": DataFrame
    """
    id_col_name = "cand_id" if role == "cand" else "s1_id"
    out = df[[ID_COL, COUNTRY_COL, "clean_name", "clean_addr", "postal"]].copy()
    out = out.rename(columns={ID_COL: id_col_name})
    out["country_lc"] = out[COUNTRY_COL].str.strip().str.lower()

    out["first_word"] = out["clean_name"].str.strip().str.split().str[0].fillna("")
    out["sorted_2word"] = out["clean_name"].apply(_get_sorted_2word)
    out["sorted_nums"] = out["clean_addr"].apply(_get_sorted_numbers)

    if _HAS_JF:
        out["soundex"] = out["first_word"].apply(lambda x: jellyfish.soundex(str(x)) if x else "")
    else:
        out["soundex"] = ""

    gram_counts = Counter()
    if role == "cand":
        for text in out["clean_name"].dropna().values:
            t = str(text).replace(" ", "")
            for i in range(len(t)-2):
                gram_counts[t[i:i+3]] += 1
        
        def get_rarest_3gram(text):
            t = str(text).replace(" ", "")
            grams = [t[i:i+3] for i in range(len(t)-2)]
            if not grams:
                return ""
            return min(grams, key=lambda g: gram_counts.get(g, 0))
        
        out["rarest_3gram"] = out["clean_name"].apply(get_rarest_3gram)
        return out, gram_counts
    
    return out


def _compute_s1_rarest_3gram(s1_df: pd.DataFrame, gram_counts: Counter) -> pd.Series:
    """Compute rarest 3-gram for S1 using candidate corpus statistics."""
    def get_rarest(text):
        t = str(text).replace(" ", "")
        grams = [t[i:i+3] for i in range(len(t)-2)]
        if not grams:
            return ""
        return min(grams, key=lambda g: gram_counts.get(g, 0))
    return s1_df["clean_name"].apply(get_rarest)


def generate_exact_blocks(
    df_s1: pd.DataFrame, df_cand: pd.DataFrame,
    precomputed_cand: Optional[pd.DataFrame] = None,
    gram_counts: Optional[Counter] = None,
) -> Set[Tuple[str, str]]:
    """Return (s1_id, cand_id) pairs from exact key matching.
    
    If precomputed_cand is provided, skip candidate key computation (much faster for chunked inference).
    If gram_counts is provided, use it for rarest 3-gram computation on S1.
    """
    # S1 keys — always computed (small)
    s1 = df_s1[[ID_COL, COUNTRY_COL, "clean_name", "clean_addr", "postal"]].copy()
    s1 = s1.rename(columns={ID_COL: "s1_id"})
    s1["country_lc"] = s1[COUNTRY_COL].str.strip().str.lower()
    s1["first_word"] = s1["clean_name"].str.strip().str.split().str[0].fillna("")
    s1["sorted_2word"] = s1["clean_name"].apply(_get_sorted_2word)
    s1["sorted_nums"] = s1["clean_addr"].apply(_get_sorted_numbers)
    if _HAS_JF:
        s1["soundex"] = s1["first_word"].apply(lambda x: jellyfish.soundex(str(x)) if x else "")
    else:
        s1["soundex"] = ""
    
    # Candidate keys — use precomputed if available
    if precomputed_cand is not None:
        cand = precomputed_cand
        if gram_counts is None:
            gram_counts = Counter()
    else:
        cand = df_cand[[ID_COL, COUNTRY_COL, "clean_name", "clean_addr", "postal"]].copy()
        cand = cand.rename(columns={ID_COL: "cand_id"})
        cand["country_lc"] = cand[COUNTRY_COL].str.strip().str.lower()
        cand["first_word"] = cand["clean_name"].str.strip().str.split().str[0].fillna("")
        cand["sorted_2word"] = cand["clean_name"].apply(_get_sorted_2word)
        cand["sorted_nums"] = cand["clean_addr"].apply(_get_sorted_numbers)
        if _HAS_JF:
            cand["soundex"] = cand["first_word"].apply(lambda x: jellyfish.soundex(str(x)) if x else "")
        else:
            cand["soundex"] = ""
        
        gram_counts = Counter()
        for text in cand["clean_name"].dropna().values:
            t = str(text).replace(" ", "")
            for i in range(len(t)-2):
                gram_counts[t[i:i+3]] += 1
        
        def get_rarest_3gram_cand(text):
            t = str(text).replace(" ", "")
            grams = [t[i:i+3] for i in range(len(t)-2)]
            if not grams:
                return ""
            return min(grams, key=lambda g: gram_counts.get(g, 0))
        
        cand["rarest_3gram"] = cand["clean_name"].apply(get_rarest_3gram_cand)
    
    # S1 rarest_3gram uses candidate corpus stats
    s1["rarest_3gram"] = _compute_s1_rarest_3gram(s1, gram_counts)
    
    parts = []

    def block_on_key(key_col, s1_df, cand_df, max_block_size, name):
        s1_sub = s1_df[s1_df[key_col] != ""]
        cd_sub = cand_df[cand_df[key_col] != ""]
        if s1_sub.empty or cd_sub.empty:
            return
        
        freq = cd_sub.groupby(["country_lc", key_col]).size()
        valid = freq[freq <= max_block_size].reset_index()[["country_lc", key_col]]
        
        s1_merged = s1_sub.merge(valid, on=["country_lc", key_col], how="inner")
        cd_merged = cd_sub.merge(valid, on=["country_lc", key_col], how="inner")
        
        if not s1_merged.empty and not cd_merged.empty:
            m = s1_merged.merge(cd_merged, on=["country_lc", key_col], how="inner")
            parts.append(m[["s1_id", "cand_id"]])
            log.info(f"  {name} block: {len(m):,} pairs")

    # key 1: country + first word of cleaned name
    s1_n = s1[~s1["first_word"].isin(_STOPWORDS)]
    cd_n = cand[~cand["first_word"].isin(_STOPWORDS)]
    block_on_key("first_word", s1_n, cd_n, 800, "exact-name")

    # key 2: country + postal code
    block_on_key("postal", s1, cand, 800, "exact-postal")

    # key 3: Sorted 2-word prefix
    s1_2w = s1[~s1["sorted_2word"].isin(_STOPWORDS)]
    cd_2w = cand[~cand["sorted_2word"].isin(_STOPWORDS)]
    block_on_key("sorted_2word", s1_2w, cd_2w, 800, "sorted-2word")

    # key 4: Numeric tokens
    block_on_key("sorted_nums", s1, cand, 800, "sorted-nums")

    # key 5: Soundex of first word
    block_on_key("soundex", s1_n, cd_n, 800, "soundex")

    # key 6: Rarest 3-gram
    block_on_key("rarest_3gram", s1, cand, 800, "rarest-3gram")

    if not parts:
        return set()

    combined = pd.concat(parts, ignore_index=True).drop_duplicates()
    result = set(zip(combined["s1_id"], combined["cand_id"]))
    log.info(f"  exact blocking: {len(result):,} unique pairs")
    return result
