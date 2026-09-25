"""
Master preprocessing orchestrator for full entity datasets.
Uses vectorized pandas operations for large-scale data (millions of rows).
"""

import re
import pandas as pd
import unicodedata
from ..config import NAME_COL, ADDRESS_COL

import logging
logger = logging.getLogger(__name__)

# ── Address abbreviation expansion (vectorized via str.replace) ────────────────
ADDRESS_ABBREVS = [
    (r'\brd\b', 'road'),
    (r'\bst\b', 'street'),
    (r'\bave\b', 'avenue'),
    (r'\bblvd\b', 'boulevard'),
    (r'\bhwy\b', 'highway'),
    (r'\bapt\b', 'apartment'),
    (r'\bste\b', 'suite'),
    (r'\bfl\b', 'floor'),
    (r'\bnr\b', 'near'),
    (r'\bopp\b', 'opposite'),
    (r'\bbldg\b', 'building'),
    (r'\bdist\b', 'district'),
    (r'\bctr\b', 'center'),
    (r'\bintl\b', 'international'),
    (r'\bno\b', 'number'),
]

# ── Legal suffix removal (name normalization) ─────────────────────────────────
LEGAL_SUFFIXES_PATTERN = re.compile(
    r'\b(corp(?:oration)?|inc(?:orporated)?|ltd|limited|pvt|private'
    r'|llc|llp|co|company|enterprises?|services?|solutions?|group'
    r'|technologies|tech|sarl|sas|gmbh|ag|plc)\b',
    re.IGNORECASE
)


def _normalize_unicode(s: pd.Series) -> pd.Series:
    """Strip Unicode to ASCII (handles Hindi/French chars gracefully)."""
    return s.apply(
        lambda x: unicodedata.normalize("NFKD", x).encode("ASCII", "ignore").decode("utf-8")
        if isinstance(x, str) else ""
    )


def _normalize_name_series(s: pd.Series) -> pd.Series:
    """Vectorized business name normalization."""
    out = _normalize_unicode(s.fillna(""))
    out = out.str.lower()
    out = out.str.replace(r'&', ' and ', regex=True)
    out = out.str.replace(LEGAL_SUFFIXES_PATTERN, ' ', regex=True)
    out = out.str.replace(r'[^\w\s]', ' ', regex=True)
    out = out.str.replace(r'\s+', ' ', regex=True).str.strip()
    return out


def _normalize_address_series(s: pd.Series) -> pd.Series:
    """Vectorized address normalization."""
    out = _normalize_unicode(s.fillna(""))
    out = out.str.lower()
    for pattern, replacement in ADDRESS_ABBREVS:
        out = out.str.replace(pattern, replacement, regex=True)
    out = out.str.replace(r'[^\w\s]', ' ', regex=True)
    out = out.str.replace(r'\s+', ' ', regex=True).str.strip()
    return out


def _extract_postal_series(s: pd.Series) -> pd.Series:
    """Extract 5-6 digit postal/PIN code from address string (vectorized)."""
    # str.extract returns the first match group
    extracted = s.fillna("").str.extract(r'\b(\d{5,6})\b', expand=False)
    return extracted.fillna("")


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned name, cleaned address, and extracted postal code columns.

    Fully vectorized — no iterrows(). Handles millions of rows efficiently.
    """
    logger.info(f"  Preprocessing {len(df):,} records...")
    df_clean = df.copy()

    df_clean["clean_name"]    = _normalize_name_series(df_clean[NAME_COL])
    df_clean["clean_address"] = _normalize_address_series(df_clean[ADDRESS_COL])
    df_clean["postal_code"]   = _extract_postal_series(df_clean[ADDRESS_COL])

    logger.info(f"  Preprocessing done: {len(df_clean):,} records")
    return df_clean
