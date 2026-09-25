"""
Feature extraction modules for entity similarity metrics.
"""

from .name_features import compute_name_features
from .address_features import compute_address_features
from .feature_builder import build_feature_matrix

__all__ = [
    "compute_name_features",
    "compute_address_features",
    "build_feature_matrix",
]
