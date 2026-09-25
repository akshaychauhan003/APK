"""
Address similarity feature computation.
"""

import re
from typing import Dict
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def compute_address_features(
    addr1: str,
    addr2: str,
    postal1: str,
    postal2: str,
    country1: str,
    country2: str,
) -> Dict[str, float]:
    """Compute similarity features between two business addresses and metadata."""
    a1 = addr1 or ""
    a2 = addr2 or ""

    tokens1 = set(a1.split())
    tokens2 = set(a2.split())

    # Token Jaccard
    union = tokens1.union(tokens2)
    jaccard = len(tokens1.intersection(tokens2)) / len(union) if union else 0.0

    # Postal Match
    p1 = postal1 or ""
    p2 = postal2 or ""
    postal_match = 1.0 if (p1 and p2 and p1 == p2) else (0.5 if not p1 or not p2 else 0.0)

    # Country Match
    c1 = (country1 or "").strip().lower()
    c2 = (country2 or "").strip().lower()
    country_match = 1.0 if c1 == c2 else 0.0

    # Digit match ratio
    digits1 = set(re.findall(r"\d+", a1))
    digits2 = set(re.findall(r"\d+", a2))
    digit_union = digits1.union(digits2)
    digit_jaccard = len(digits1.intersection(digits2)) / len(digit_union) if digit_union else 1.0

    # Fuzz ratio
    if HAS_RAPIDFUZZ:
        fuzz_ratio = fuzz.ratio(a1, a2) / 100.0
        token_set_ratio = fuzz.token_set_ratio(a1, a2) / 100.0
    else:
        fuzz_ratio = SequenceMatcher(None, a1, a2).ratio()
        token_set_ratio = jaccard

    return {
        "addr_jaccard": jaccard,
        "postal_match": postal_match,
        "country_match": country_match,
        "digit_jaccard": digit_jaccard,
        "addr_fuzz_ratio": fuzz_ratio,
        "addr_token_set_ratio": token_set_ratio,
    }
