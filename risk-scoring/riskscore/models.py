"""Обёртки: селектор колонок, калиброванная модель PD (единый сериализуемый артефакт)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PD_MIN, PD_MAX = 0.002, 0.95        # границы прогноза: PD не бывает ровно 0 или 1


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=()):
        self.columns = list(columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class CalibratedPD:
    """pipeline (сырая заявка -> необработанный скор) + калибратор (Platt/isotonic) -> PD.

    Детерминирован: никакого случайного состояния при инференсе.
    """

    def __init__(self, pipeline, calibrator=None, method="none", version="v1", meta=None):
        self.pipeline, self.calibrator, self.method = pipeline, calibrator, method
        self.version, self.meta = version, meta or {}

    def raw_score(self, raw: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(raw)[:, 1]

    def calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        if self.method == "platt":
            p = self.calibrator.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            p = self.calibrator.predict(p_raw)
        else:
            p = p_raw
        return np.clip(p, PD_MIN, PD_MAX)

    def predict_pd(self, raw: pd.DataFrame) -> np.ndarray:
        return self.calibrate(self.raw_score(raw))
