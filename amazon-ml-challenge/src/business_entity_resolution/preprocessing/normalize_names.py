"""
Name normalization utilities for business entity resolution.
"""

import re
import unicodedata

# Common legal suffixes pattern
LEGAL_SUFFIXES = [
    r"\bcorp(oration)?\b",
    r"\binc(orporated)?\b",
    r"\bltd\b",
    r"\blimited\b",
    r"\bpvt\b",
    r"\bprivate\b",
    r"\bllc\b",
    r"\bllp\b",
    r"\bco\b",
    r"\bcompany\b",
    r"\benterprises?\b",
    r"\bservices?\b",
    r"\bsolutions?\b",
    r"\bgroup\b",
    r"\btechnologies\b",
    r"\btech\b",
]

LEGAL_SUFFIX_RE = re.compile(r"|".join(LEGAL_SUFFIXES), re.IGNORECASE)


def normalize_business_name(name: str) -> str:
    """Normalize a business name by removing noise, legal suffixes, and punctuation."""
    if not name or not isinstance(name, str):
        return ""

    # Unicode normalization
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("utf-8")
    name = name.lower()

    # Replace ampersands
    name = re.sub(r"&", " and ", name)

    # Remove legal suffixes
    name = LEGAL_SUFFIX_RE.sub(" ", name)

    # Remove punctuation & special characters
    name = re.sub(r"[^\w\s]", " ", name)

    # Normalize spaces
    tokens = name.strip().split()
    return " ".join(tokens)
