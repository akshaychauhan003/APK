import logging
from typing import Set, Tuple

import pandas as pd

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


def generate_exact_blocks(
    df_s1: pd.DataFrame, df_cand: pd.DataFrame
) -> Set[Tuple[str, str]]:
    """Return (s1_id, cand_id) pairs from exact key matching on name+country and postal+country."""
    s1 = df_s1[[ID_COL, COUNTRY_COL, "clean_name", "postal"]].copy()
    s1 = s1.rename(columns={ID_COL: "s1_id"})
    s1["country_lc"] = s1[COUNTRY_COL].str.strip().str.lower()
    s1["first_word"] = s1["clean_name"].str.strip().str.split().str[0].fillna("")

    cand = df_cand[[ID_COL, COUNTRY_COL, "clean_name", "postal"]].copy()
    cand = cand.rename(columns={ID_COL: "cand_id"})
    cand["country_lc"] = cand[COUNTRY_COL].str.strip().str.lower()
    cand["first_word"] = cand["clean_name"].str.strip().str.split().str[0].fillna("")

    parts = []

    # key 1: country + first word of cleaned name
    s1_n = s1[(s1["first_word"] != "") & (~s1["first_word"].isin(_STOPWORDS))]
    cd_n = cand[(cand["first_word"] != "") & (~cand["first_word"].isin(_STOPWORDS))]

    if not cd_n.empty:
        # drop blocks that are too large (generic word shared by >500 candidates)
        freq = cd_n.groupby(["country_lc", "first_word"]).size()
        valid = freq[freq <= 500].reset_index()[["country_lc", "first_word"]]
        s1_n  = s1_n.merge(valid, on=["country_lc", "first_word"], how="inner")
        cd_n  = cd_n.merge(valid, on=["country_lc", "first_word"], how="inner")

    if not s1_n.empty and not cd_n.empty:
        m = s1_n.merge(cd_n, on=["country_lc", "first_word"], how="inner")
        parts.append(m[["s1_id", "cand_id"]])
        log.info(f"  exact-name block: {len(m):,} pairs")

    # key 2: country + postal code
    s1_p  = s1[s1["postal"] != ""][["s1_id", "country_lc", "postal"]]
    cd_p  = cand[cand["postal"] != ""][["cand_id", "country_lc", "postal"]]
    if not s1_p.empty and not cd_p.empty:
        m = s1_p.merge(cd_p, on=["country_lc", "postal"], how="inner")
        parts.append(m[["s1_id", "cand_id"]])
        log.info(f"  exact-postal block: {len(m):,} pairs")

    if not parts:
        return set()

    combined = pd.concat(parts, ignore_index=True).drop_duplicates()
    result = set(zip(combined["s1_id"], combined["cand_id"]))
    log.info(f"  exact blocking: {len(result):,} unique pairs")
    return result
