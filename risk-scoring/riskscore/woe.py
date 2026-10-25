"""WOE-кодирование с монотонным биннингом (основа скор-карты).

WOE здесь = ln( доля плохих в корзине / доля хороших в корзине ):
положительный WOE -> корзина рискованнее среднего. Пропуск (NaN) - отдельная корзина.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

OTHER, MISSING = "__OTHER__", "__MISSING__"


class WOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=(), cat_cols=(), directions=None, max_bins=8, min_bin_frac=0.05,
                 min_bin_count=30, n_init_bins=20, alpha=0.5, min_missing=30):
        self.num_cols, self.cat_cols, self.directions = list(num_cols), list(cat_cols), directions
        self.max_bins, self.min_bin_frac, self.min_bin_count = max_bins, min_bin_frac, min_bin_count
        self.n_init_bins, self.alpha, self.min_missing = n_init_bins, alpha, min_missing

    # ------------------------------------------------------------------ подбор корзин
    @staticmethod
    def _stats(b, y, k):
        return np.bincount(b, minlength=k), np.bincount(b, weights=y, minlength=k)

    def _edges_num(self, v, y, direction):
        uniq = np.unique(v)
        discrete = len(uniq) <= 6
        if discrete:
            edges = list((uniq[:-1] + uniq[1:]) / 2)
            min_n = self.min_bin_count
        else:
            edges = list(np.unique(np.quantile(v, np.linspace(0, 1, self.n_init_bins + 1)[1:-1])))
            min_n = max(self.min_bin_frac * len(v), self.min_bin_count)
        while edges:
            b = np.searchsorted(edges, v, side="left")
            n, bad = self._stats(b, y, len(edges) + 1)
            rate = bad / np.maximum(n, 1)
            small = np.where(n < min_n)[0]
            if small.size:                                   # 1) слишком маленькая корзина -> слить с ближайшей по доле плохих
                i = small[np.argmin(n[small])]
                if i == 0: j = 0
                elif i == len(n) - 1: j = len(n) - 2
                else: j = i - 1 if abs(rate[i] - rate[i - 1]) <= abs(rate[i] - rate[i + 1]) else i
                edges.pop(j); continue
            if direction != 0:                               # 2) монотонность (ожидаемый знак влияния на риск)
                viol = np.where(direction * np.diff(rate) < 0)[0]
                if viol.size:
                    edges.pop(viol[0]); continue
            if len(n) > self.max_bins:                       # 3) ограничение числа корзин
                edges.pop(int(np.argmin(np.abs(np.diff(rate))))); continue
            break
        return np.array(edges, float)

    def _woe_from_counts(self, n, bad, tot_n, tot_bad):
        k = len(n); a = self.alpha
        good, tot_good = n - bad, tot_n - tot_bad
        db = (bad + a) / (tot_bad + a * k)
        dg = (good + a) / (tot_good + a * k)
        woe = np.log(db / dg)
        return woe, (db - dg) * woe

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        tot_n, tot_bad = len(y), y.sum()
        dirs = self.directions or {}
        self.bins_ = {}
        for c in self.num_cols:
            x = pd.to_numeric(X[c], errors="coerce").to_numpy(float)
            m = ~np.isnan(x)
            edges = self._edges_num(x[m], y[m], dirs.get(c, 0)) if m.sum() > 0 else np.array([])
            b = np.searchsorted(edges, x[m], side="left")
            n, bad = self._stats(b, y[m], len(edges) + 1)
            labels = [self._label(edges, i) for i in range(len(edges) + 1)]
            has_miss = (~m).sum() >= self.min_missing
            if has_miss:
                n = np.append(n, (~m).sum()); bad = np.append(bad, y[~m].sum()); labels.append(MISSING)
            woe, iv = self._woe_from_counts(n, bad, tot_n, tot_bad)
            self.bins_[c] = dict(kind="num", edges=edges, woe=woe[:len(edges) + 1], miss_woe=float(woe[-1]) if has_miss else 0.0,
                                 table=pd.DataFrame(dict(bin=labels, n=n.astype(int), bad=bad.astype(int), bad_rate=bad / np.maximum(n, 1), woe=woe, iv=iv)))
        for c in self.cat_cols:
            x = X[c].astype(object).where(X[c].notna(), MISSING).astype(str).to_numpy()
            cats, cnt = np.unique(x, return_counts=True)
            keep = [k for k, v in zip(cats, cnt) if v >= max(self.min_bin_frac * len(x) * .5, self.min_bin_count) or k == MISSING and v >= self.min_missing]
            lab = np.where(np.isin(x, keep), x, OTHER)
            uc = list(np.unique(lab))
            n = np.array([(lab == k).sum() for k in uc]); bad = np.array([y[lab == k].sum() for k in uc])
            woe, iv = self._woe_from_counts(n, bad, tot_n, tot_bad)
            self.bins_[c] = dict(kind="cat", mapping=dict(zip(uc, woe)), has_other=OTHER in uc,
                                 table=pd.DataFrame(dict(bin=uc, n=n.astype(int), bad=bad.astype(int), bad_rate=bad / np.maximum(n, 1), woe=woe, iv=iv)))
        self.iv_ = pd.Series({c: b["table"]["iv"].sum() for c, b in self.bins_.items()}).sort_values(ascending=False)
        return self

    @staticmethod
    def _label(edges, i):
        lo = "-inf" if i == 0 else f"{edges[i - 1]:.4g}"
        hi = "+inf" if i == len(edges) else f"{edges[i]:.4g}"
        return f"({lo}; {hi}]"

    # ------------------------------------------------------------------ применение
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c, b in self.bins_.items():
            if b["kind"] == "num":
                x = pd.to_numeric(X[c], errors="coerce").to_numpy(float)
                w = np.full(len(x), b["miss_woe"])
                m = ~np.isnan(x)
                if len(b["woe"]):
                    w[m] = b["woe"][np.searchsorted(b["edges"], x[m], side="left")]
                out[c] = w
            else:
                x = X[c].astype(object).where(X[c].notna(), MISSING).astype(str)
                mp = b["mapping"]
                out[c] = x.map(lambda v: mp.get(v, mp.get(OTHER, 0.0))).to_numpy(float)
        return pd.DataFrame(out, index=X.index)

    def bin_index(self, X: pd.DataFrame, col: str) -> np.ndarray:
        """Индекс корзины (для скор-карты)."""
        b = self.bins_[col]; x = X[col]
        if b["kind"] == "num":
            v = pd.to_numeric(x, errors="coerce").to_numpy(float)
            idx = np.full(len(v), len(b["woe"]))               # последняя строка таблицы = MISSING
            m = ~np.isnan(v)
            idx[m] = np.searchsorted(b["edges"], v[m], side="left")
            return idx
        labels = list(b["table"]["bin"])
        x = x.astype(object).where(x.notna(), MISSING).astype(str)
        return x.map(lambda v: labels.index(v) if v in labels else (labels.index(OTHER) if OTHER in labels else 0)).to_numpy()
