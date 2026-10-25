"""Конвертация «плоская строка заявки» <-> вложенный JSON-фикстура (формат для моков и сидирования БД).
Все данные синтетические."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

NOTICE = ("СИНТЕТИЧЕСКИЕ ДАННЫЕ: сгенерированы командой хакатона; к реальным лицам и организациям не относятся. "
          "ИНН/ОГРН из несуществующего диапазона (код региона 00).")

COMPANY = [("id", "company_id"), ("inn", "inn"), ("ogrn", "ogrn"), ("name", "company_name"), ("legalForm", "legal_form"), ("regDate", "reg_date"),
           ("okved", "okved"), ("industry", "industry"), ("regionCode", "region_code"), ("regionName", "region_name"), ("regionTier", "region_tier"),
           ("employees", "employees"), ("taxRegime", "tax_regime"), ("status", "status")]
FIN = [("revenue", "revenue"), ("costOfSales", "cost_of_sales"), ("ebitda", "ebitda"), ("netProfit", "net_profit"), ("totalAssets", "total_assets"),
       ("totalLiabilities", "total_liabilities"), ("equity", "equity"), ("shortTermLiabilities", "short_term_liabilities"),
       ("currentAssets", "current_assets"), ("receivables", "receivables"), ("payables", "payables")]
TURN = [("creditTurnover", "turnover_credit_m"), ("debitTurnover", "turnover_debit_m"), ("endBalance", "balance_end_m")]
TURN_SCALAR = [("bankTenureMonths", "bank_tenure_months"), ("counterpartiesCnt", "counterparties_cnt")]
BKI = [("hasHistory", "bki_has_history"), ("historyMonths", "bki_history_months"), ("openLoans", "bki_open_loans"), ("closedLoans", "bki_closed_loans"),
       ("openAmount", "bki_open_amount"), ("outstandingDebt", "bki_outstanding_debt"), ("monthlyPayment", "bki_monthly_payment"),
       ("closedAmount", "bki_closed_amount"), ("overdue30", "bki_overdue_30_cnt"), ("overdue60", "bki_overdue_60_cnt"), ("overdue90", "bki_overdue_90_cnt"),
       ("maxOverdueDays", "bki_max_overdue_days"), ("currentOverdueAmount", "bki_current_overdue_amount"), ("defaultsCnt", "bki_defaults_cnt"),
       ("inquiries3m", "bki_inquiries_3m"), ("inquiries12m", "bki_inquiries_12m")]
FNS = [("taxDebt", "fns_tax_debt"), ("accountBlocked", "fns_account_blocked"), ("massAddress", "fns_mass_address"), ("massDirector", "fns_mass_director"),
       ("disqualified", "fns_disqualified"), ("bankruptcy", "fns_bankruptcy")]
APP = [("applicationDate", "application_date"), ("productCode", "product_code"), ("requestedAmount", "requested_amount"), ("termMonths", "term_months"),
       ("indicativeRate", "indicative_rate"), ("purpose", "purpose"), ("ownContributionPct", "own_contribution_pct"),
       ("collateralType", "collateral_type"), ("collateralValue", "collateral_value")]
_INT_COLS = {"employees", "region_tier", "term_months", "bki_has_history", "bki_open_loans", "bki_closed_loans", "bki_overdue_30_cnt", "bki_overdue_60_cnt",
             "bki_overdue_90_cnt", "bki_max_overdue_days", "bki_defaults_cnt", "bki_inquiries_3m", "bki_inquiries_12m", "fns_account_blocked", "fns_mass_address",
             "fns_mass_director", "fns_disqualified", "fns_bankruptcy", "bank_tenure_months", "counterparties_cnt", "bki_history_months"}


def _py(col: str, v):
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return int(v) if v.is_integer() and (col in _INT_COLS or col.rsplit("_q", 1)[0] in {c for _, c in FIN}) else v
    return v


def quarter_labels(app_date: str) -> list[str]:
    d = pd.Timestamp(app_date)
    y, q = d.year, (d.month - 1) // 3          # квартал даты заявки; последний полный = предыдущий
    labels = []
    for _ in range(8):
        q -= 1
        if q < 0:
            q, y = 3, y - 1
        labels.append(f"{y}-Q{q + 1}")
    return labels[::-1]


def month_labels(app_date: str) -> list[str]:
    d = pd.Timestamp(app_date).to_period("M")
    return [str(d - i) for i in range(12, 0, -1)]


def row_to_fixture(r: dict) -> dict:
    g = lambda c: _py(c, r.get(c))
    ql, ml = quarter_labels(r["application_date"]), month_labels(r["application_date"])
    fin = [dict(quarter=ql[k], **{a: g(f"{c}_q{k + 1}") for a, c in FIN}) for k in range(8)]
    months = [dict(month=ml[k], **{a: g(f"{c}{k + 1}") for a, c in TURN}) for k in range(12)]
    return dict(
        _synthetic=True,
        company={a: g(c) for a, c in COMPANY},
        financials=fin,
        accountTurnover=dict(**{a: g(c) for a, c in TURN_SCALAR}, months=months),
        creditHistory={a: g(c) for a, c in BKI},
        fns={a: g(c) for a, c in FNS},
        application={a: g(c) for a, c in APP},
    )


def fixture_to_row(f: dict) -> dict:
    nan = lambda v: np.nan if v is None else v
    r = {c: nan(f["company"][a]) for a, c in COMPANY}
    for k, q in enumerate(f["financials"]):
        r.update({f"{c}_q{k + 1}": nan(q[a]) for a, c in FIN})
    at = f["accountTurnover"]
    r.update({c: nan(at[a]) for a, c in TURN_SCALAR})
    for k, m in enumerate(at["months"]):
        r.update({f"{c}{k + 1}": nan(m[a]) for a, c in TURN})
    r.update({c: nan(f["creditHistory"][a]) for a, c in BKI})
    r.update({c: nan(f["fns"][a]) for a, c in FNS})
    r.update({c: nan(f["application"][a]) for a, c in APP})
    return r
