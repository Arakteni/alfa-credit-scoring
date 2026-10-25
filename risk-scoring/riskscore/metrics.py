"""Метрики качества скоринговой модели: различающая способность, точность вероятностей, стабильность."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def calibration_table(y, p, n_bins=10) -> pd.DataFrame:
    """Квантильные корзины по прогнозу: средний прогноз vs фактическая доля дефолтов."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    d["bin"] = pd.qcut(d.p.rank(method="first"), n_bins, labels=False)
    t = d.groupby("bin").agg(n=("y", "size"), mean_pd=("p", "mean"), observed=("y", "mean")).reset_index(drop=True)
    return t


def ece(y, p, n_bins=10) -> float:
    t = calibration_table(y, p, n_bins)
    return float((t.n * (t.mean_pd - t.observed).abs()).sum() / t.n.sum())


def psi(expected, actual, n_bins=10, eps=1e-4) -> float:
    """Population Stability Index по квантильным корзинам распределения `expected`."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, n_bins + 1)[1:-1]))
    pe = np.bincount(np.searchsorted(edges, e, side="left"), minlength=len(edges) + 1) / len(e)
    pa = np.bincount(np.searchsorted(edges, a, side="left"), minlength=len(edges) + 1) / len(a)
    pe, pa = np.clip(pe, eps, None), np.clip(pa, eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def summary(y, p, name="") -> dict:
    return dict(name=name, n=int(len(y)), default_rate=float(np.mean(y)), auc=float(roc_auc_score(y, p)),
                gini=gini(y, p), ks=ks(y, p), brier=brier(y, p), ece=ece(y, p), mean_pd=float(np.mean(p)))
