import logging
import joblib
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    _BACKEND = "lgbm"
except Exception:
    try:
        import xgboost as xgb
        _BACKEND = "xgb"
    except Exception:
        _BACKEND = "hgbm"
        log.warning(
            "LightGBM and XGBoost unavailable. "
            "Falling back to sklearn HistGradientBoostingClassifier. "
            "Run `brew install libomp` on Mac to enable LightGBM."
        )

from sklearn.ensemble import HistGradientBoostingClassifier
from ..config import RANDOM_STATE


class Classifier:
    """Picks best available GBDT: LightGBM > XGBoost > sklearn HistGBM."""

    def __init__(self, scale_pos_weight=None):
        spw = scale_pos_weight

        if _BACKEND == "lgbm":
            params = dict(
                n_estimators=500, learning_rate=0.05, max_depth=7,
                num_leaves=63, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
            )
            if spw is not None:
                params["scale_pos_weight"] = spw
            self._clf = lgb.LGBMClassifier(**params)

        elif _BACKEND == "xgb":
            params = dict(
                n_estimators=500, learning_rate=0.05, max_depth=7,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=RANDOM_STATE, eval_metric="logloss",
                verbosity=0, n_jobs=-1,
            )
            if spw is not None:
                params["scale_pos_weight"] = spw
            self._clf = xgb.XGBClassifier(**params)

        else:
            cw = {0: 1.0, 1: float(spw)} if spw is not None else None
            self._clf = HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.05, max_depth=7,
                min_samples_leaf=20, class_weight=cw, random_state=RANDOM_STATE,
            )

        self.backend = _BACKEND
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names or list(X.columns)
        log.info(f"training Classifier (backend={self.backend}, shape={X.shape}, features={len(self.feature_names)})")
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._clf.predict_proba(X)[:, 1]

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "clf": self._clf,
            "backend": self.backend,
            "feature_names": self.feature_names,
        }, path)
        log.info(f"model saved → {path}")

    @classmethod
    def load(cls, path: Path):
        data = joblib.load(path)
        obj = cls.__new__(cls)
        if isinstance(data, dict):
            obj._clf = data["clf"]
            obj.backend = data.get("backend", "unknown")
            obj.feature_names = data.get("feature_names", None)
        else:
            obj._clf = data
            obj.backend = "legacy"
            obj.feature_names = None
        log.info(f"model loaded from {path} (backend={obj.backend}, features={obj.feature_names})")
        return obj
