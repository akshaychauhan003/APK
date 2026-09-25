"""
Blocking and candidate pair generation modules.
"""

from .exact_blocking import generate_exact_blocks
from .fuzzy_blocking import generate_tfidf_candidates
from .candidate_generation import generate_candidate_pairs

__all__ = [
    "generate_exact_blocks",
    "generate_tfidf_candidates",
    "generate_candidate_pairs",
]
