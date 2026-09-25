"""
Entity resolution binary matching classifier wrapper.

Priority order for gradient boosting backend:
  1. LightGBM  (fastest, if libomp available)
  2. XGBoost   (fast, if libomp available)
  3. sklearn HistGradientBoostingClassifier  (no native deps, near-LightGBM speed)
  4. sklearn GradientBoostingClassifier      (pure Python fallback)
"""

import joblib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 1. LightGBM ──────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    logger.info("Backend: LightGBM")
except Exception:
    HAS_LIGHTGBM = False

# ── 2. XGBoost ───────────────────────────────────────────────────────────────
if not HAS_LIGHTGBM:
    try:
        import xgboost as xgb
        HAS_XGBOOST = True
        logger.info("Backend: XGBoost (LightGBM unavailable)")
    except Exception:
        HAS_XGBOOST = False
else:
    HAS_XGBOOST = False

# ── 3+4. scikit-learn ─────────────────────────────────────────────────────────
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from ..config import RANDOM_STATE


class EntityResolutionModel:
    """Classifier model wrapper for predicting entity matches.

    Automatically selects the best available gradient-boosting backend:
    LightGBM → XGBoost → HistGradientBoosting → GradientBoosting.
    """

    def __init__(self, params: Any = None):
        params = params or {}
        # Extract scale_pos_weight for downstream translation
        spw = params.pop("scale_pos_weight", None)

        if HAS_LIGHTGBM:
            default_params = {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 7,
                "num_leaves": 63,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": RANDOM_STATE,
                "verbose": -1,
                "n_jobs": -1,
            }
            if spw is not None:
                default_params["scale_pos_weight"] = spw
            default_params.update(params)
            self.model = lgb.LGBMClassifier(**default_params)
            self._backend = "lightgbm"

        elif HAS_XGBOOST:
            default_params = {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 7,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": RANDOM_STATE,
                "eval_metric": "logloss",
                "verbosity": 0,
                "n_jobs": -1,
            }
            if spw is not None:
                default_params["scale_pos_weight"] = spw
            default_params.update(params)
            self.model = xgb.XGBClassifier(**default_params)
            self._backend = "xgboost"

        else:
            # HistGradientBoostingClassifier: sklearn's fast GBDT.
            # Translate scale_pos_weight → class_weight dict.
            class_weight = None
            if spw is not None:
                class_weight = {0: 1.0, 1: float(spw)}
            self.model = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=7,
                min_samples_leaf=20,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
            )
            self._backend = "hist_gbm"
            logger.warning(
                "LightGBM and XGBoost unavailable (missing libomp). "
                "Falling back to sklearn HistGradientBoostingClassifier. "
                "Install `brew install libomp` for better performance."
            )

    def fit(self, X, y):
        """Fit the binary classification model."""
        logger.info(f"Training EntityResolutionModel with backend={self._backend}, X.shape={X.shape}")
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """Predict match probabilities (class=1)."""
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "backend": self._backend}, path)
        logger.info(f"Model saved → {path}  (backend={self._backend})")

    @classmethod
    def load(cls, path: Path):
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls.__new__(cls)
        if isinstance(data, dict):
            instance.model = data["model"]
            instance._backend = data.get("backend", "unknown")
        else:
            # Legacy pkl that stored the raw estimator directly
            instance.model = data
            instance._backend = "legacy"
        logger.info(f"Model loaded from {path}  (backend={instance._backend})")
        return instance
