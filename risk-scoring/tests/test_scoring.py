"""Тесты пайплайна признаков, движка скоринга и генератора синтетики (все данные синтетические)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "risk-scoring"), str(ROOT)]
from data.generate import is_valid_inn, is_valid_ogrn, is_synthetic_id          # noqa: E402
from riskscore.engine import ScoringEngine, PRODUCT_CONFIG, FLAG_DEFS             # noqa: E402
from riskscore.features import FEATURE_META, FEATURE_COLUMNS, compute_features    # noqa: E402

STR = {"inn": str, "ogrn": str, "okved": str, "region_code": str}


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(ROOT / "data/generated/applications_raw.csv", dtype=STR).iloc[:600]


@pytest.fixture(scope="module")
def eng():
    return ScoringEngine(ROOT / "risk-scoring/models")


def _strip(r):
    return {k: v for k, v in r.items() if k not in ("latencyMs", "calculatedAt")}


# ------------------------------------------------------------------ данные и признаки
def test_inn_ogrn_checksums_and_synthetic_range(raw):
    assert raw.inn.map(is_valid_inn).all() and raw.ogrn.map(is_valid_ogrn).all()
    assert raw.inn.map(is_synthetic_id).all()
    assert raw.inn.is_unique


def test_balance_identity(raw):
    for k in range(1, 9):
        assert (raw[f"total_assets_q{k}"] - raw[f"total_liabilities_q{k}"] - raw[f"equity_q{k}"]).abs().max() <= 2


def test_at_least_25_features_in_5_groups():
    assert len(FEATURE_COLUMNS) >= 25
    assert len({m["group"] for m in FEATURE_META.values()}) == 5


def test_features_deterministic(raw):
    assert compute_features(raw).equals(compute_features(raw))


def test_missing_columns_and_garbage_do_not_crash(raw):
    bad = raw.iloc[:3].drop(columns=["turnover_credit_m5", "balance_end_m3", "revenue_q1"]).copy()
    bad.loc[bad.index[0], "application_date"] = "не дата"
    bad.loc[bad.index[1], "requested_amount"] = "abc"
    X = compute_features(bad)
    assert X.shape == (3, len(FEATURE_COLUMNS)) and not np.isinf(X.select_dtypes("number").to_numpy()).any()


def test_no_credit_history_is_nan_not_zero(raw):
    nh = raw[raw.bki_has_history == 0].iloc[:50]
    X = compute_features(nh)
    assert (X.has_credit_history == 0).all() and X.overdue_90_cnt.isna().all() and X.open_loans.isna().all()


# ------------------------------------------------------------------ контракт ответа
def test_response_contract_fields(eng, raw):
    r = eng.score_one(raw.iloc[0].to_dict())
    for k in ("pd", "riskScore", "ratingGrade", "riskSegment", "recommendedLimit", "recommendedRate", "maxTermMonths",
              "policyFlags", "reasons", "modelVersion", "calculatedAt", "latencyMs", "warnings"):
        assert k in r
    assert 0 < r["pd"] < 1 and 300 <= r["riskScore"] <= 900


def test_reasons_at_least_5_with_required_fields(eng, raw):
    for i in range(0, 600, 60):
        rs = eng.score_one(raw.iloc[i].to_dict())["reasons"]
        assert len(rs) >= 5
        for x in rs:
            assert {"featureCode", "featureLabel", "direction", "contribution", "explanation"} <= set(x)
            assert x["direction"] in ("INCREASES_RISK", "DECREASES_RISK") and len(x["explanation"]) > 10


def test_deterministic_inference(eng, raw):
    a, b = eng.score_one(raw.iloc[7].to_dict()), eng.score_one(raw.iloc[7].to_dict())
    assert _strip(a) == _strip(b)


def test_batch_equals_single(eng, raw):
    batch = eng.score_frame(raw.iloc[:25])
    for i in (0, 11, 24):
        assert _strip(batch[i]) == _strip(eng.score_one(raw.iloc[i].to_dict()))


def test_sum_of_points_matches_riskscore(eng, raw):
    X = eng.clip.transform(compute_features(raw.iloc[:200]))
    p, pts = eng._pd_and_points(X)
    score = np.clip(np.round(eng.card["base_points"] + pts.sum(1)), 300, 900)
    direct = np.clip(np.round(600 - eng.card["factor"] * (np.log(p / (1 - p)) - np.log(0.05 / 0.95))), 300, 900)
    inside = (p > 0.0021) & (p < 0.949)
    assert np.abs(score - direct)[inside].max() <= 1


def test_rating_scale_is_monotone_in_pd(eng):
    ps = np.linspace(0.002, 0.95, 400)
    order = {g: i for i, g in enumerate(eng.grades)}
    idx = [order[g] for g in eng.grade_of(ps)]
    assert idx == sorted(idx) and len(set(idx)) == len(eng.grades) and len(eng.grades) >= 7


# ------------------------------------------------------------------ монотонность и бизнес-логика
def test_pd_not_decreasing_with_larger_amount(eng, raw):
    base = raw.iloc[:200]
    p0 = np.array([r["pd"] for r in eng.score_frame(base)])
    z = base.copy(); z["requested_amount"] = z.requested_amount * 2
    p1 = np.array([r["pd"] for r in eng.score_frame(z)])
    assert (p1 >= p0 - 1e-9).all()


def test_pd_not_decreasing_with_more_existing_debt_payment(eng, raw):
    base = raw[raw.bki_has_history == 1].iloc[:200]
    p0 = np.array([r["pd"] for r in eng.score_frame(base)])
    z = base.copy(); z["bki_monthly_payment"] = z.bki_monthly_payment * 2 + 20_000
    p1 = np.array([r["pd"] for r in eng.score_frame(z)])
    assert (p1 >= p0 - 1e-9).all()


def test_pd_not_decreasing_with_delinquency(eng, raw):
    base = raw[raw.bki_has_history == 1].iloc[:300]
    p0 = np.array([r["pd"] for r in eng.score_frame(base)])
    z = base.copy(); z["bki_overdue_90_cnt"] = 2; z["bki_overdue_60_cnt"] = 2; z["bki_overdue_30_cnt"] = 3; z["bki_max_overdue_days"] = 150
    p1 = np.array([r["pd"] for r in eng.score_frame(z)])
    assert (p1 >= p0 - 1e-9).all() and p1.mean() > p0.mean() + 0.01


def test_pd_not_decreasing_with_blocking(eng, raw):
    base = raw.iloc[:200]
    p0 = np.array([r["pd"] for r in eng.score_frame(base)])
    z = base.copy(); z["fns_account_blocked"] = 1
    p1 = np.array([r["pd"] for r in eng.score_frame(z)])
    assert (p1 >= p0 - 1e-9).all() and p1.mean() > p0.mean()


def test_same_profile_different_products_gives_different_recommendations(eng, raw):
    base = raw.iloc[3].to_dict()
    outs = {p: eng.score_one(dict(base, product_code=p)) for p in PRODUCT_CONFIG}
    assert len({(o["recommendedLimit"], o["recommendedRate"], o["maxTermMonths"]) for o in outs.values()}) == len(PRODUCT_CONFIG)


def test_limit_capped_by_product_and_non_negative(eng, raw):
    for r, (_, rw) in zip(eng.score_frame(raw), raw.iterrows()):
        assert 0 <= r["recommendedLimit"] <= PRODUCT_CONFIG[rw.product_code]["cap"]


def test_policy_flags_set_for_scenarios(eng, raw):
    base = raw.iloc[2].to_dict()
    cases = {"LIQUIDATION_IN_PROGRESS": dict(status="LIQUIDATING"), "ACCOUNT_BLOCKED": dict(fns_account_blocked=1),
             "MASS_ADDRESS": dict(fns_mass_address=1), "DISQUALIFIED_DIRECTOR": dict(fns_disqualified=1),
             "DELINQUENCY_90_PLUS": dict(bki_has_history=1, bki_overdue_90_cnt=1, bki_max_overdue_days=120)}
    for code, patch in cases.items():
        codes = {f["code"] for f in eng.score_one(dict(base, **patch))["policyFlags"]}
        assert code in codes and FLAG_DEFS[code][0] in ("CRITICAL", "WARNING")
    assert "NO_CREDIT_HISTORY" in {f["code"] for f in eng.score_one(dict(base, bki_has_history=0))["policyFlags"]}


def test_scoring_does_not_decide(eng, raw):
    """Граница ответственности: в ответе нет полей решения об одобрении."""
    r = eng.score_one(raw.iloc[0].to_dict())
    assert not {"decision", "approved", "status", "approve"} & set(r)


# ------------------------------------------------------------------ устойчивость
def test_extreme_and_missing_input_gives_warnings_not_exception(eng, raw):
    bad = raw.iloc[10].to_dict()
    bad.update(revenue_q8=1e15, requested_amount=np.nan, product_code="UNKNOWN_X", term_months=-3)
    r = eng.score_one(bad)
    assert 0 < r["pd"] < 1 and any(w.startswith("UNKNOWN_PRODUCT") for w in r["warnings"]) and any(w.startswith("INPUT_MISSING") for w in r["warnings"])


def test_empty_minimal_input_does_not_crash(eng):
    r = eng.score_one({"requested_amount": 1_000_000, "term_months": 12, "product_code": "EXPRESS"})
    assert r["recommendedLimit"] == 0 and len(r["warnings"]) > 0


def test_model_info(eng):
    info = eng.model_info(ROOT / "docs/ml/metrics.json")
    assert info["modelVersion"] == eng.version and len(info["ratingScale"]) >= 7 and "metrics" in info
