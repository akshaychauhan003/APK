"""
Entity resolution binary matching classifier wrapper.
"""

import joblib
from pathlib import Path
from typing import Any

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import GradientBoostingClassifier
from ..config import RANDOM_STATE


class EntityResolutionModel:
    """Classifier model wrapper for predicting entity matches."""

    def __init__(self, params: Any = None):
        if HAS_LIGHTGBM:
            default_params = {
                "n_estimators": 150,
                "learning_rate": 0.05,
                "max_depth": 6,
                "random_state": RANDOM_STATE,
                "verbose": -1,
            }
            if params:
                default_params.update(params)
            self.model = lgb.LGBMClassifier(**default_params)
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=RANDOM_STATE,
            )

    def fit(self, X, y):
        """Fit the binary classification model."""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """Predict match probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path):
        """Save model to disk."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path):
        """Load model from disk."""
        instance = cls()
        instance.model = joblib.load(path)
        return instance
