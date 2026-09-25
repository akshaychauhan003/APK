"""
Model definitions, training, and inference utilities.
"""

from .model import EntityResolutionModel
from .train import train_matching_model
from .predict import predict_matches

__all__ = [
    "EntityResolutionModel",
    "train_matching_model",
    "predict_matches",
]
