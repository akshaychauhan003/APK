import re
import unicodedata
import pandas as pd
import logging

from ..config import NAME_COL, ADDR_COL

log = logging.getLogger(__name__)

_LEGAL = re.compile(
    r"\b(corp(?:oration)?|inc(?:orporated)?|ltd|limited|pvt|private"
    r"|llc|llp|co|company|enterprises?|services?|solutions?|group"
    r"|technologies|tech|sarl|sas|gmbh|ag|plc)\b",
    re.IGNORECASE,
)

_ABBREVS = [
    (r"\brd\b", "road"),   (r"\bst\b", "street"), (r"\bave\b", "avenue"),
    (r"\bblvd\b", "boulevard"), (r"\bhwy\b", "highway"), (r"\bapt\b", "apartment"),
    (r"\bste\b", "suite"), (r"\bfl\b", "floor"),   (r"\bnr\b", "near"),
    (r"\bopp\b", "opposite"), (r"\bbldg\b", "building"), (r"\bdist\b", "district"),
    (r"\bctr\b", "center"), (r"\bintl\b", "international"), (r"\bno\b", "number"),
]


def _to_ascii(s: pd.Series) -> pd.Series:
    return s.apply(
        lambda x: unicodedata.normalize("NFKD", x).encode("ASCII", "ignore").decode()
        if isinstance(x, str) else ""
    )


def _clean_names(s: pd.Series) -> pd.Series:
    out = _to_ascii(s.fillna("")).str.lower()
    out = out.str.replace(r"&", " and ", regex=True)
    out = out.str.replace(_LEGAL, " ", regex=True)
    out = out.str.replace(r"[^\w\s]", " ", regex=True)
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def _clean_addrs(s: pd.Series) -> pd.Series:
    out = _to_ascii(s.fillna("")).str.lower()
    for pat, repl in _ABBREVS:
        out = out.str.replace(pat, repl, regex=True)
    out = out.str.replace(r"[^\w\s]", " ", regex=True)
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def _extract_postal(s: pd.Series) -> pd.Series:
    return s.fillna("").str.extract(r"\b(\d{5,6})\b", expand=False).fillna("")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"preprocessing {len(df):,} records")
    out = df.copy()
    out["clean_name"] = _clean_names(out[NAME_COL])
    out["clean_addr"] = _clean_addrs(out[ADDR_COL])
    out["postal"]     = _extract_postal(out[ADDR_COL])
    return out
