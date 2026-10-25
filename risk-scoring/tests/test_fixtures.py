"""Фикстуры: единые ИНН, маркировка синтетики, round-trip и регрессия ожидаемых результатов."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "risk-scoring"), str(ROOT)]
from data import fixtures as FX                       # noqa: E402
from data.generate import is_valid_inn                # noqa: E402
from riskscore.engine import ScoringEngine            # noqa: E402

FIX = ROOT / "data/fixtures"
TS = "2026-10-26T12:00:00Z"


@pytest.fixture(scope="module")
def companies():
    return json.loads((FIX / "companies.json").read_text(encoding="utf-8"))["companies"]


def test_at_least_60_demo_companies_and_12_scenarios(companies):
    assert len(companies) >= 60
    assert len([c for c in companies if c["demo"]["scenario"] != "TYPICAL"]) >= 12


def test_all_fixture_files_marked_synthetic_and_share_same_inns(companies):
    inns = {c["company"]["inn"] for c in companies}
    assert len(inns) == len(companies) and all(is_valid_inn(i) and i.startswith("00") for i in inns)
    for p in FIX.glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["_synthetic"] is True and "СИНТЕТИЧЕСКИЕ" in d["_notice"]
        if "byInn" in d:
            assert set(d["byInn"]) == inns


def test_fixture_roundtrip_and_expected_results_reproduce(companies):
    eng = ScoringEngine(ROOT / "risk-scoring/models")
    for c in companies:
        row = FX.fixture_to_row(c)
        again = FX.row_to_fixture(row)
        assert again["company"] == c["company"] and again["financials"] == c["financials"] and again["creditHistory"] == c["creditHistory"]
        out = eng.score_one(row, TS)
        e = c["expected"]
        assert (out["pd"], out["ratingGrade"], out["recommendedLimit"]) == (e["pd"], e["ratingGrade"], e["recommendedLimit"])
        assert [f["code"] for f in out["policyFlags"]] == e["policyFlags"]


def test_no_credit_history_case_has_null_bki_fields(companies):
    c = next(c for c in companies if c["demo"]["scenario"] == "FLAG_NO_CREDIT_HISTORY")
    assert c["creditHistory"]["hasHistory"] == 0 and c["creditHistory"]["openLoans"] is None
