"""
Name similarity feature computation.
Includes string edit distance, token-based similarity, and phonetic matching.
"""

from typing import Dict
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

try:
    import jellyfish
    HAS_JELLYFISH = True
except ImportError:
    HAS_JELLYFISH = False


def _soundex(text: str) -> str:
    """Return Soundex code for first word of text, or empty string."""
    if not text:
        return ""
    first_word = text.split()[0] if text.split() else ""
    if not first_word:
        return ""
    if HAS_JELLYFISH:
        try:
            return jellyfish.soundex(first_word)
        except Exception:
            return ""
    return ""


def _metaphone(text: str) -> str:
    """Return Metaphone code for first word of text, or empty string."""
    if not text or not HAS_JELLYFISH:
        return ""
    first_word = text.split()[0] if text.split() else ""
    if not first_word:
        return ""
    try:
        return jellyfish.metaphone(first_word)
    except Exception:
        return ""


def compute_name_features(name1: str, name2: str) -> Dict[str, float]:
    """Compute string and token similarity features between two business names."""
    n1 = name1 or ""
    n2 = name2 or ""

    tokens1 = set(n1.split())
    tokens2 = set(n2.split())

    # Jaccard Token Similarity
    union = tokens1.union(tokens2)
    jaccard = len(tokens1.intersection(tokens2)) / len(union) if union else 0.0

    # First Word Match (exact)
    w1 = n1.split()[0] if n1.split() else ""
    w2 = n2.split()[0] if n2.split() else ""
    first_word_match = 1.0 if (w1 and w2 and w1 == w2) else 0.0

    # RapidFuzz / Standard Edit Distances
    if HAS_RAPIDFUZZ:
        ratio = fuzz.ratio(n1, n2) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(n1, n2) / 100.0
        token_set_ratio = fuzz.token_set_ratio(n1, n2) / 100.0
    else:
        ratio = SequenceMatcher(None, n1, n2).ratio()
        token_sort_ratio = jaccard
        token_set_ratio = jaccard

    # Length Diff Ratio
    max_len = max(len(n1), len(n2))
    len_diff = abs(len(n1) - len(n2)) / max_len if max_len > 0 else 0.0

    # Phonetic Matching (catches typos that SOUND the same, e.g. "Corp" vs "Korp")
    sdx1, sdx2 = _soundex(n1), _soundex(n2)
    soundex_match = 1.0 if (sdx1 and sdx2 and sdx1 == sdx2) else 0.0

    mphone1, mphone2 = _metaphone(n1), _metaphone(n2)
    metaphone_match = 1.0 if (mphone1 and mphone2 and mphone1 == mphone2) else 0.0

    return {
        "name_jaccard": jaccard,
        "name_first_word_match": first_word_match,
        "name_fuzz_ratio": ratio,
        "name_token_sort_ratio": token_sort_ratio,
        "name_token_set_ratio": token_set_ratio,
        "name_len_diff": len_diff,
        "name_soundex_match": soundex_match,
        "name_metaphone_match": metaphone_match,
    }
