"""Движок скоринга: оценка риска + рекомендация + флаги политики + объяснение. Детерминирован.

Данные и модель синтетические. Движок НЕ принимает кредитное решение (это зона credit-core).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_META, compute_features

# --- продуктовые параметры рекомендации (конфигурация, а не пороги одобрения)
PRODUCT_CONFIG = {
    #                  база ставки, макс. срок, k * средний месячный оборот, потолок лимита, мин. лимит
    "OVERDRAFT":       dict(rate=0.205, max_term=12, k_turnover=0.35, cap=15_000_000, min_limit=100_000),
    "WORKING_CAPITAL": dict(rate=0.190, max_term=36, k_turnover=1.50, cap=40_000_000, min_limit=100_000),
    "INVESTMENT":      dict(rate=0.175, max_term=60, k_turnover=3.00, cap=50_000_000, min_limit=300_000),
    "EXPRESS":         dict(rate=0.240, max_term=24, k_turnover=0.50, cap=3_000_000, min_limit=100_000),
}
DEFAULT_PRODUCT = "WORKING_CAPITAL"

INDUSTRY_RU = {
    "retail_food": "розничная торговля продуктами", "catering": "общественное питание", "construction": "строительство",
    "transport": "транспорт и логистика", "it_services": "IT-услуги", "manufacturing": "производство",
    "wholesale": "оптовая торговля", "agriculture": "сельское хозяйство", "personal_serv": "бытовые услуги",
    "health_edu": "медицина и образование", "real_estate": "аренда недвижимости",
}

# код: (важность, описание). Флаги НЕ являются решением: их интерпретирует credit-core.
FLAG_DEFS = {
    "LIQUIDATION_IN_PROGRESS": ("CRITICAL", "Компания находится в процессе ликвидации"),
    "ACCOUNT_BLOCKED": ("CRITICAL", "Есть блокировка расчётных счетов"),
    "DELINQUENCY_90_PLUS": ("CRITICAL", "В кредитной истории есть просрочка 90+ дней"),
    "DISQUALIFIED_DIRECTOR": ("CRITICAL", "Среди руководителей есть дисквалифицированное лицо"),
    "BANKRUPTCY_SIGNS": ("CRITICAL", "Обнаружены сведения о процедуре банкротства"),
    "TAX_DEBT_SIGNIFICANT": ("WARNING", "Значимая задолженность по налогам"),
    "CURRENT_OVERDUE": ("WARNING", "Есть текущая просроченная задолженность по кредитам"),
    "DEFAULT_IN_HISTORY": ("WARNING", "В кредитной истории есть дефолты"),
    "NO_CREDIT_HISTORY": ("WARNING", "Кредитная история отсутствует: оценка менее надёжна"),
    "MASS_ADDRESS": ("WARNING", "Адрес регистрации помечен как массовый"),
    "MASS_DIRECTOR": ("WARNING", "Руководитель числится в большом числе компаний"),
    "YOUNG_BUSINESS": ("WARNING", "Бизнесу менее 6 месяцев"),
    "NEGATIVE_EQUITY": ("WARNING", "Отрицательный собственный капитал"),
    "HIGH_DEBT_BURDEN": ("WARNING", "Совокупные платежи превышают месячную EBITDA (DSCR < 1)"),
    "FREQUENT_INQUIRIES": ("INFO", "Много кредитных запросов за последние 3 месяца"),
    "TURNOVER_MISMATCH": ("INFO", "Обороты по счёту заметно расходятся с выручкой из отчётности"),
}
# признаки, значения которых показываем в процентах
_PCT = {"net_margin", "gross_margin", "ebitda_margin", "roa", "equity_ratio", "revenue_growth_yoy", "revenue_growth_last2q", "revenue_cv",
        "turnover_cv", "turnover_trend", "revenue_trend", "new_payment_to_ebitda", "existing_payment_to_ebitda", "payment_burden",
        "liabilities_to_revenue", "debt_to_revenue", "balance_to_turnover", "tax_debt_ratio", "turnover_to_revenue", "amount_to_revenue",
        "own_contribution_pct", "collateral_cover"}
_DAYS = {"payables_days", "receivables_days", "max_overdue_days"}
_MONTHS = {"company_age_months", "history_months", "bank_tenure_months"}


def _fmt(code: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "нет данных"
    if isinstance(v, str):
        return INDUSTRY_RU.get(v, v)
    if code in _PCT:
        return f"{v * 100:.1f}%"
    if code in _DAYS:
        return f"{v:.0f} дн."
    if code in _MONTHS:
        return f"{v:.0f} мес."
    if code.endswith("_log"):
        return f"{np.exp(v):,.0f}".replace(",", " ")
    if float(v).is_integer():
        return f"{v:.0f}"
    return f"{v:.2f}"


class ScoringEngine:
    def __init__(self, models_dir: str | Path, model_file: str = "risk_model_woe_logit_v1.joblib"):
        d = Path(models_dir)
        self.model = joblib.load(d / model_file)
        pipe = self.model.pipeline
        self.clip, self.enc, self.sel = pipe.named_steps["clip"], pipe.named_steps["woe"], pipe.named_steps["select"]
        self.tail = pipe[2:]                              # WOE -> отбор -> логрегрессия
        self.card = json.loads((d / "scorecard_v1.json").read_text(encoding="utf-8"))
        self.scale = json.loads((d / "rating_scale_v1.json").read_text(encoding="utf-8"))["grades"]
        self.grades = [g["grade"] for g in self.scale]
        self.upper = [g["pd_to"] for g in self.scale]
        self.segment = {g["grade"]: g["segment"] for g in self.scale}
        self.version = self.model.version
        self.cols = list(self.card["features"])
        self.pts = {c: np.array(v["points"]) for c, v in self.card["features"].items()}
        self.ref = np.array([self.card["features"][c]["ref_points"] for c in self.cols])

    # ------------------------------------------------------------------ вспомогательное
    def _pd_and_points(self, Xc: pd.DataFrame):
        p = self.model.calibrate(self.tail.predict_proba(Xc)[:, 1])
        pts = np.column_stack([self.pts[c][np.minimum(self.enc.bin_index(Xc, c), len(self.pts[c]) - 1)] for c in self.cols])
        return p, pts

    def grade_of(self, p) -> np.ndarray:
        return np.array(self.grades)[np.searchsorted(self.upper, p, side="right").clip(0, len(self.grades) - 1)]

    def _flags(self, raw: pd.DataFrame, X: pd.DataFrame) -> list[list[dict]]:
        rev_ltm = np.exp(X["revenue_ltm_log"].to_numpy())
        tax_debt = pd.to_numeric(raw.get("fns_tax_debt", pd.Series(0.0, index=raw.index)), errors="coerce").fillna(0).to_numpy()
        cond = {
            "LIQUIDATION_IN_PROGRESS": X.is_liquidating.eq(1),
            "ACCOUNT_BLOCKED": X.account_blocked.eq(1),
            "DELINQUENCY_90_PLUS": X.overdue_90_cnt.gt(0) | X.max_overdue_days.ge(90),
            "DISQUALIFIED_DIRECTOR": X.disqualified.eq(1),
            "BANKRUPTCY_SIGNS": X.bankruptcy.eq(1),
            "TAX_DEBT_SIGNIFICANT": pd.Series(tax_debt >= np.maximum(100_000, 0.02 * rev_ltm), index=X.index),
            "CURRENT_OVERDUE": X.current_overdue_flag.eq(1),
            "DEFAULT_IN_HISTORY": X.defaults_cnt.gt(0),
            "NO_CREDIT_HISTORY": X.has_credit_history.eq(0),
            "MASS_ADDRESS": X.mass_address.eq(1),
            "MASS_DIRECTOR": X.mass_director.eq(1),
            "YOUNG_BUSINESS": X.company_age_months.lt(6),
            "NEGATIVE_EQUITY": X.negative_equity.eq(1),
            "HIGH_DEBT_BURDEN": X.payment_burden.gt(1.0),
            "FREQUENT_INQUIRIES": X.inquiries_3m.ge(4),
            "TURNOVER_MISMATCH": X.turnover_to_revenue.lt(0.5) | X.turnover_to_revenue.gt(1.6),
        }
        M = np.column_stack([cond[k].fillna(False).to_numpy(bool) for k in FLAG_DEFS])
        keys = list(FLAG_DEFS)
        return [[dict(code=keys[j], severity=FLAG_DEFS[keys[j]][0], description=FLAG_DEFS[keys[j]][1]) for j in np.where(r)[0]] for r in M]

    def _offer(self, raw: pd.DataFrame, X: pd.DataFrame, p: np.ndarray):
        n = len(raw)
        prod = raw.get("product_code", pd.Series(DEFAULT_PRODUCT, index=raw.index)).astype(object)
        known = prod.isin(PRODUCT_CONFIG).to_numpy()
        prod_eff = prod.where(known, DEFAULT_PRODUCT)
        cfg = lambda k: prod_eff.map(lambda x: PRODUCT_CONFIG[x][k]).to_numpy(float)
        ebitda_m = pd.to_numeric(raw[[f"ebitda_q{k}" for k in range(5, 9)]].stack(future_stack=True).groupby(level=0).sum(min_count=4), errors="coerce").reindex(raw.index).to_numpy() / 12 \
            if all(f"ebitda_q{k}" in raw for k in range(5, 9)) else np.full(n, np.nan)
        existing = np.nan_to_num(pd.to_numeric(raw.get("bki_monthly_payment", pd.Series(np.nan, index=raw.index)), errors="coerce").to_numpy(float), nan=0.0)
        turn = np.exp(X["turnover_avg_log"].to_numpy())
        cover = np.nan_to_num(X["collateral_cover"].to_numpy(), nan=0.0)
        term_req = pd.to_numeric(raw.get("term_months", pd.Series(np.nan, index=raw.index)), errors="coerce").to_numpy(float)

        share = np.clip(0.65 - 1.5 * p, 0.0, 0.6)
        rate = cfg("rate") + np.minimum(0.005 + 0.45 * p, 0.12)
        max_term = np.maximum(np.round(cfg("max_term") * np.clip(1 - 1.2 * p, 0.4, 1.0) / 6) * 6, 6)
        max_term = np.minimum(max_term, cfg("max_term"))
        term_ref = np.where(np.isnan(term_req), max_term, np.minimum(np.maximum(term_req, 1), max_term))
        capacity = np.maximum(0.0, share * np.nan_to_num(ebitda_m, nan=0.0) - existing)
        r = rate / 12
        limit_cf = capacity * (1 - (1 + r) ** (-term_ref)) / r
        limit_t = cfg("k_turnover") * turn * (1 + 0.25 * np.minimum(cover, 1.0))
        limit = np.minimum.reduce([limit_cf, limit_t, cfg("cap")])
        limit = np.floor(limit / 10_000) * 10_000
        limit = np.where((limit < cfg("min_limit")) | ~(np.nan_to_num(ebitda_m, nan=-1) > 0), 0.0, limit)
        return limit, rate, max_term, known, capacity

    def _reasons(self, X: pd.DataFrame, Xc: pd.DataFrame, pts: np.ndarray, n_up=4, n_down=2) -> list[list[dict]]:
        contrib = pts - self.ref                          # баллы относительно среднего заёмщика (минус = хуже)
        out = []
        for i in range(len(X)):
            c = contrib[i]
            up = [j for j in np.argsort(c) if c[j] < -0.5][:n_up]            # повышают риск
            down = [j for j in np.argsort(-c) if c[j] > 0.5][:n_down]        # снижают риск
            rest = [j for j in np.argsort(-np.abs(c)) if j not in up + down]
            chosen = up + down
            chosen += rest[: max(0, 5 - len(chosen))]
            chosen.sort(key=lambda j: -abs(c[j]))
            rs = []
            for j in chosen:
                code = self.cols[j]; v = X[code].iloc[i]
                risk_up = c[j] < 0
                rs.append(dict(
                    featureCode=code, featureLabel=FEATURE_META[code]["label"],
                    direction="INCREASES_RISK" if risk_up else "DECREASES_RISK", contribution=round(float(c[j]), 1),
                    explanation=f"{FEATURE_META[code]['label']}: {_fmt(code, v)}. "
                                f"{'Повышает' if risk_up else 'Снижает'} оценку риска ({c[j]:+.0f} баллов к скору относительно среднего заёмщика)."))
            out.append(rs)
        return out

    # ------------------------------------------------------------------ публичный API
    def score_frame(self, raw: pd.DataFrame, calculated_at: str | None = None) -> list[dict]:
        t0 = time.perf_counter()
        raw = raw.reset_index(drop=True)
        X = compute_features(raw)
        Xc = self.clip.transform(X)
        p, pts = self._pd_and_points(Xc)
        grade = self.grade_of(p)
        score = np.clip(np.round(self.card["base_points"] + pts.sum(1)), self.card["score_min"], self.card["score_max"])
        limit, rate, max_term, known, capacity = self._offer(raw, X, p)
        flags, reasons = self._flags(raw, X), self._reasons(X, Xc, pts)
        clipped = self.clip.clipped_mask(X)
        key_inputs = ["revenue_q8", "ebitda_q8", "total_assets_q8", "requested_amount", "term_months", "application_date", "reg_date"]
        now = calculated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        res = []
        for i in range(len(raw)):
            w = [f"FEATURE_CLIPPED:{c}" for c in clipped.columns[clipped.iloc[i].to_numpy(bool)]]
            w += [f"INPUT_MISSING:{k}" for k in key_inputs if k not in raw or pd.isna(raw[k].iloc[i])]
            if not known[i]:
                w.append(f"UNKNOWN_PRODUCT:{raw['product_code'].iloc[i] if 'product_code' in raw else None}->{DEFAULT_PRODUCT}")
            res.append(dict(
                pd=round(float(p[i]), 5), riskScore=int(score[i]), ratingGrade=str(grade[i]), riskSegment=self.segment[str(grade[i])],
                recommendedLimit=int(limit[i]), recommendedRate=round(float(rate[i]), 4), maxTermMonths=int(max_term[i]),
                policyFlags=flags[i], reasons=reasons[i], warnings=w, modelVersion=self.version, calculatedAt=now, latencyMs=0.0))
        per_row = (time.perf_counter() - t0) * 1000 / max(len(raw), 1)
        for r in res:
            r["latencyMs"] = round(per_row, 2)
        return res

    def score_one(self, raw_row: dict, calculated_at: str | None = None) -> dict:
        return self.score_frame(pd.DataFrame([raw_row]), calculated_at)[0]

    def suggest_conditions(self, raw_row: dict, factors=(1.0, .9, .8, .7, .6, .5, .4, .3, .2)) -> list[dict]:
        """What-if: при каких сумме/сроке рекомендованный лимит покрывает запрос и нет критических флагов."""
        prod = raw_row.get("product_code", DEFAULT_PRODUCT)
        prod = prod if prod in PRODUCT_CONFIG else DEFAULT_PRODUCT
        max_t = PRODUCT_CONFIG[prod]["max_term"]
        terms = sorted({t for t in (6, 12, 18, 24, 36, 48, 60) if t <= max_t} | {int(raw_row.get("term_months", 12))})
        rows = [dict(raw_row, requested_amount=round(float(raw_row["requested_amount"]) * f, -4), term_months=t) for t in terms for f in factors]
        res = self.score_frame(pd.DataFrame(rows), calculated_at="1970-01-01T00:00:00Z")
        out = []
        for t in terms:
            ok = [(r, rw) for r, rw in zip(res, rows) if rw["term_months"] == t and r["recommendedLimit"] >= rw["requested_amount"] > 0
                  and not any(f["severity"] == "CRITICAL" for f in r["policyFlags"])]
            if ok:
                r, rw = max(ok, key=lambda z: z[1]["requested_amount"])
                out.append(dict(termMonths=t, maxAmount=rw["requested_amount"], pd=r["pd"], ratingGrade=r["ratingGrade"], recommendedRate=r["recommendedRate"]))
        return out

    def model_info(self, metrics_path: str | Path | None = None) -> dict:
        info = dict(modelVersion=self.version, trainedAt=self.model.meta.get("trained_at"), datasetVersion=self.model.meta.get("dataset_version"),
                    features=[dict(code=c, label=FEATURE_META[c]["label"], group=FEATURE_META[c]["group"]) for c in self.cols],
                    allFeaturesCount=len(FEATURE_META), ratingScale=self.scale, scoreScale=dict(base=self.card["base_score"], pdAtBase=self.card["pd_ref"], pdo=self.card["pdo"], min=self.card["score_min"], max=self.card["score_max"]),
                    note="Модель обучена на синтетических данных; метрики не переносятся на реальный портфель.")
        if metrics_path and Path(metrics_path).exists():
            m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
            t = m["metrics"]["WOE-LR + скор-карта (основная)|test"]
            info["metrics"] = dict(gini=t["gini"], ks=t["ks"], brier=t["brier"], auc=t["auc"], psiScoreTrainTest=m["psi"]["score_train_test"], sample="test")
        return info
