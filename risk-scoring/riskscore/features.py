"""Единый расчёт признаков (обучение == инференс) + защита от выбросов.

Вход: «сырая» заявка (одна строка = одна компания; формат data/generated/applications_raw.csv).
Выход: DataFrame признаков в фиксированном порядке FEATURE_COLUMNS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

G1, G2, G3, G4, G5 = ("1. Профиль и проверки", "2. Финансы", "3. Обороты по счёту",
                      "4. Кредитная история", "5. Параметры заявки")

# имя: (группа, тип, подпись, описание, источник, направление влияния на риск при росте значения)
# тип: num - число, bin - 0/1, cat - категория. направление: '+' риск растёт, '-' снижается, '~' немонотонно
_META = {
    # --- 1. профиль и проверки
    "company_age_months": (G1, "num", "Возраст бизнеса, мес.", "Месяцев с даты регистрации до даты заявки", "ЕГРЮЛ/ЕГРИП", "-"),
    "is_ip": (G1, "bin", "Форма: ИП", "1 - индивидуальный предприниматель, 0 - ООО", "ЕГРЮЛ/ЕГРИП", "~"),
    "industry": (G1, "cat", "Отрасль", "Отрасль по ОКВЭД (11 групп)", "ЕГРЮЛ/ЕГРИП", "~"),
    "region_tier": (G1, "num", "Уровень региона", "1 - столичные регионы, 2 - крупные, 3 - прочие", "ЕГРЮЛ/ЕГРИП", "~"),
    "employees_log": (G1, "num", "Численность персонала (log)", "ln(1 + численность)", "ФНС / заявка", "-"),
    "tax_regime": (G1, "cat", "Налоговый режим", "УСН доходы / УСН доходы-расходы / ОСН / ПСН", "ФНС", "~"),
    "tax_debt_ratio": (G1, "num", "Налоговый долг к месячной выручке", "Задолженность по налогам / средняя месячная выручка", "ФНС", "+"),
    "account_blocked": (G1, "bin", "Блокировка счетов", "Есть решение о блокировке счетов", "ФНС", "+"),
    "mass_address": (G1, "bin", "Массовый адрес", "Адрес регистрации помечен как массовый", "ФНС «Прозрачный бизнес»", "+"),
    "mass_director": (G1, "bin", "Массовый руководитель", "Руководитель числится в большом числе компаний", "ФНС «Прозрачный бизнес»", "+"),
    "disqualified": (G1, "bin", "Дисквалификация руководителя", "Есть дисквалифицированное лицо среди руководителей", "ФНС", "+"),
    "bankruptcy": (G1, "bin", "Признаки банкротства", "Сведения о процедуре банкротства", "ФНС / ЕФРСБ", "+"),
    "is_liquidating": (G1, "bin", "Идёт ликвидация", "Статус «в процессе ликвидации»", "ЕГРЮЛ", "+"),
    # --- 2. финансы
    "revenue_ltm_log": (G2, "num", "Выручка за 12 мес. (log)", "ln(выручка последних 4 кварталов)", "Отчётность", "-"),
    "revenue_growth_yoy": (G2, "num", "Рост выручки г/г", "Выручка кв.5-8 / выручка кв.1-4 - 1", "Отчётность", "-"),
    "revenue_growth_last2q": (G2, "num", "Динамика выручки за 2 кв.", "Выручка кв.7-8 / кв.5-6 - 1", "Отчётность", "-"),
    "revenue_cv": (G2, "num", "Волатильность выручки", "Коэффициент вариации выручки за 8 кварталов", "Отчётность", "+"),
    "revenue_trend": (G2, "num", "Тренд выручки", "Наклон линейного тренда выручки / средняя выручка", "Отчётность", "-"),
    "gross_margin": (G2, "num", "Валовая рентабельность", "(Выручка - себестоимость) / выручка, 4 кв.", "Отчётность", "-"),
    "ebitda_margin": (G2, "num", "Рентабельность по EBITDA", "EBITDA / выручка, 4 кв.", "Отчётность", "-"),
    "net_margin": (G2, "num", "Рентабельность по чистой прибыли", "Чистая прибыль / выручка, 4 кв.", "Отчётность", "-"),
    "roa": (G2, "num", "Рентабельность активов", "Чистая прибыль 4 кв. / активы на конец периода", "Отчётность", "-"),
    "current_ratio": (G2, "num", "Текущая ликвидность", "Оборотные активы / краткосрочные обязательства", "Отчётность", "-"),
    "equity_ratio": (G2, "num", "Коэффициент автономии", "Собственный капитал / активы", "Отчётность", "-"),
    "negative_equity": (G2, "bin", "Отрицательный капитал", "Собственный капитал < 0", "Отчётность", "+"),
    "liabilities_to_revenue": (G2, "num", "Обязательства к выручке", "Обязательства / выручка за 12 мес.", "Отчётность", "+"),
    "liabilities_to_ebitda": (G2, "num", "Обязательства к EBITDA", "Обязательства / EBITDA за 12 мес. (EBITDA<=0 -> 50)", "Отчётность", "+"),
    "receivables_days": (G2, "num", "Оборачиваемость дебиторки, дн.", "Дебиторка / выручка последнего квартала * 91", "Отчётность", "+"),
    "payables_days": (G2, "num", "Оборачиваемость кредиторки, дн.", "Кредиторка / выручка последнего квартала * 91", "Отчётность", "+"),
    "loss_quarters": (G2, "num", "Убыточных кварталов из 8", "Число кварталов с чистым убытком", "Отчётность", "+"),
    # --- 3. обороты
    "turnover_avg_log": (G3, "num", "Средний месячный оборот (log)", "ln(среднемесячные поступления на счёт за 12 мес.)", "Банк", "-"),
    "turnover_trend": (G3, "num", "Динамика оборотов", "Среднее поступлений за посл. 3 мес. / первые 3 мес. - 1", "Банк", "-"),
    "turnover_cv": (G3, "num", "Волатильность оборотов", "Коэффициент вариации месячных поступлений", "Банк", "+"),
    "turnover_to_revenue": (G3, "num", "Оборот к выручке", "Поступления на счёт за 12 мес. / выручка за 12 мес.", "Банк / отчётность", "~"),
    "counterparties_log": (G3, "num", "Число контрагентов (log)", "ln(число уникальных контрагентов)", "Банк", "-"),
    "bank_tenure_months": (G3, "num", "Срок обслуживания в банке, мес.", "Месяцев с открытия счёта", "Банк", "-"),
    "balance_to_turnover": (G3, "num", "Остаток к обороту", "Средний остаток на конец месяца / средний месячный оборот", "Банк", "-"),
    "neg_balance_months": (G3, "num", "Месяцев с отрицательным остатком", "Число месяцев из 12 с остатком < 0", "Банк", "+"),
    "debit_credit_ratio": (G3, "num", "Списания / поступления", "Дебетовые обороты / кредитовые обороты, 12 мес.", "Банк", "~"),
    # --- 4. кредитная история
    "has_credit_history": (G4, "bin", "Есть кредитная история", "0 - в БКИ нет сведений (тонкое дело), 1 - есть", "БКИ", "-"),
    "history_months": (G4, "num", "Длина кредитной истории, мес.", "Месяцев с первого кредита", "БКИ", "-"),
    "open_loans": (G4, "num", "Открытых кредитов", "Число действующих обязательств", "БКИ", "~"),
    "closed_loans": (G4, "num", "Закрытых кредитов", "Число закрытых обязательств", "БКИ", "-"),
    "debt_to_revenue": (G4, "num", "Долг к выручке", "Текущая задолженность по БКИ / выручка за 12 мес.", "БКИ / отчётность", "+"),
    "existing_payment_to_ebitda": (G4, "num", "Текущие платежи к EBITDA", "Ежемесячные платежи по действующим кредитам / EBITDA в месяц", "БКИ / отчётность", "+"),
    "overdue_30_cnt": (G4, "num", "Просрочек 30+", "Число просрочек от 30 дней", "БКИ", "+"),
    "overdue_60_cnt": (G4, "num", "Просрочек 60+", "Число просрочек от 60 дней", "БКИ", "+"),
    "overdue_90_cnt": (G4, "num", "Просрочек 90+", "Число просрочек от 90 дней", "БКИ", "+"),
    "max_overdue_days": (G4, "num", "Макс. длительность просрочки, дн.", "Максимальная просрочка за историю", "БКИ", "+"),
    "delinquency_severity": (G4, "num", "Тяжесть просрочек", "0 - просрочек нет, 1 - до 59 дн., 2 - 60-89 дн., 3 - 90+ дн.; нет истории - пропуск", "БКИ", "+"),
    "current_overdue_flag": (G4, "bin", "Текущая просроченная задолженность", "Есть непогашенная просрочка сейчас", "БКИ", "+"),
    "defaults_cnt": (G4, "num", "Дефолтов в истории", "Число случаев списания/дефолта", "БКИ", "+"),
    "inquiries_3m": (G4, "num", "Запросов в БКИ за 3 мес.", "Число кредитных запросов за 3 месяца", "БКИ", "+"),
    "inquiries_12m": (G4, "num", "Запросов в БКИ за 12 мес.", "Число кредитных запросов за 12 месяцев", "БКИ", "+"),
    # --- 5. параметры заявки
    "product_code": (G5, "cat", "Продукт", "OVERDRAFT / WORKING_CAPITAL / INVESTMENT / EXPRESS", "Заявка", "~"),
    "requested_amount_log": (G5, "num", "Запрошенная сумма (log)", "ln(сумма заявки)", "Заявка", "+"),
    "term_months": (G5, "num", "Срок, мес.", "Запрашиваемый срок", "Заявка", "~"),
    "amount_to_revenue": (G5, "num", "Сумма к выручке", "Сумма заявки / выручка за 12 мес.", "Заявка / отчётность", "+"),
    "amount_to_turnover": (G5, "num", "Сумма к среднему обороту", "Сумма заявки / среднемесячный оборот", "Заявка / банк", "+"),
    "new_payment_to_ebitda": (G5, "num", "Новый платёж к EBITDA", "Аннуитетный платёж по заявке / EBITDA в месяц", "Заявка / отчётность", "+"),
    "payment_burden": (G5, "num", "Совокупная нагрузка", "(Текущие платежи + новый платёж) / EBITDA в месяц", "Заявка / БКИ / отчётность", "+"),
    "dscr": (G5, "num", "DSCR", "EBITDA в месяц / (текущие платежи + новый платёж)", "Заявка / БКИ / отчётность", "-"),
    "own_contribution_pct": (G5, "num", "Собственный взнос", "Доля собственных средств в проекте", "Заявка", "-"),
    "collateral_cover": (G5, "num", "Покрытие залогом", "Стоимость обеспечения / сумма заявки (0 - без обеспечения)", "Заявка", "-"),
}
FEATURE_META = {k: dict(group=v[0], kind=v[1], label=v[2], desc=v[3], source=v[4], direction=v[5]) for k, v in _META.items()}
FEATURE_COLUMNS = list(_META)
CAT_FEATURES = [k for k, m in FEATURE_META.items() if m["kind"] == "cat"]
BIN_FEATURES = [k for k, m in FEATURE_META.items() if m["kind"] == "bin"]
NUM_FEATURES = [k for k, m in FEATURE_META.items() if m["kind"] == "num"]
CLIP_FEATURES = NUM_FEATURES  # клиппинг применяем только к числовым признакам

_Q = [f"q{k}" for k in range(1, 9)]
_M = [f"m{k}" for k in range(1, 13)]


def annuity_payment(amount, rate, term):
    """Аннуитетный платёж. Защищён от нулевых/некорректных rate и term."""
    amount = np.asarray(amount, float)
    rate = np.nan_to_num(np.asarray(rate, float), nan=0.0)
    term = np.maximum(np.nan_to_num(np.asarray(term, float), nan=12.0), 1.0)
    r = rate / 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        pay = np.where(r > 1e-9, amount * r / (1.0 - (1.0 + r) ** (-term)), amount / term)
    return pay


def _col(raw: pd.DataFrame, name: str) -> pd.Series:
    """Числовая колонка; отсутствующая/нечисловая -> NaN (сервис не должен падать)."""
    if name in raw.columns:
        return pd.to_numeric(raw[name], errors="coerce").astype(float)
    return pd.Series(np.nan, index=raw.index, dtype=float)


def _mat(raw: pd.DataFrame, prefix: str, suffixes: list[str]) -> np.ndarray:
    return np.column_stack([_col(raw, f"{prefix}_{s}").to_numpy() for s in suffixes])


def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 1e-9, a / b, np.nan)


def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Единая функция расчёта признаков. Чистая (без состояния), детерминированная."""
    ix = raw.index
    nat = pd.Series(pd.NaT, index=ix, dtype="datetime64[ns]")
    app = pd.to_datetime(raw.get("application_date", nat), errors="coerce", format="ISO8601")
    reg = pd.to_datetime(raw.get("reg_date", nat), errors="coerce", format="ISO8601")
    f = pd.DataFrame(index=ix)

    REV, COGS, EBITDA, NET = (_mat(raw, n, _Q) for n in ("revenue", "cost_of_sales", "ebitda", "net_profit"))
    ASSETS, LIAB, EQ = (_mat(raw, n, _Q)[:, 7] for n in ("total_assets", "total_liabilities", "equity"))
    CA, STL = _mat(raw, "current_assets", _Q)[:, 7], _mat(raw, "short_term_liabilities", _Q)[:, 7]
    REC, PAY = _mat(raw, "receivables", _Q)[:, 7], _mat(raw, "payables", _Q)[:, 7]
    CR, DR, BAL = (_mat(raw, n, _M) for n in ("turnover_credit", "turnover_debit", "balance_end"))

    rev_ltm, rev_prev = REV[:, 4:].sum(1), REV[:, :4].sum(1)
    ebitda_ltm, net_ltm, cogs_ltm = EBITDA[:, 4:].sum(1), NET[:, 4:].sum(1), COGS[:, 4:].sum(1)
    ebitda_m = ebitda_ltm / 12.0
    rev_month = rev_ltm / 12.0
    cr_avg = CR.mean(1)
    amount, term, rate = _col(raw, "requested_amount").to_numpy(), _col(raw, "term_months").to_numpy(), _col(raw, "indicative_rate").to_numpy()
    new_pay = annuity_payment(amount, rate, term)
    ex_pay = np.nan_to_num(_col(raw, "bki_monthly_payment").to_numpy(), nan=0.0)   # нет данных -> известных обязательств нет
    tot_pay = ex_pay + new_pay
    has_hist = _col(raw, "bki_has_history").to_numpy()
    K = np.arange(8) - 3.5
    revm = REV.mean(1)

    # 1. профиль и проверки
    f["company_age_months"] = ((app - reg).dt.days / 30.44).to_numpy()
    f["is_ip"] = (raw.get("legal_form", pd.Series("", index=ix)).astype(str) == "IP").astype(float).to_numpy()
    f["industry"] = raw.get("industry", pd.Series(np.nan, index=ix)).astype(object).to_numpy()
    f["region_tier"] = _col(raw, "region_tier").to_numpy()
    f["employees_log"] = np.log1p(np.maximum(_col(raw, "employees").to_numpy(), 0))
    f["tax_regime"] = raw.get("tax_regime", pd.Series(np.nan, index=ix)).astype(object).to_numpy()
    f["tax_debt_ratio"] = _div(np.nan_to_num(_col(raw, "fns_tax_debt").to_numpy(), nan=0.0), rev_month)
    for out, src in (("account_blocked", "fns_account_blocked"), ("mass_address", "fns_mass_address"),
                     ("mass_director", "fns_mass_director"), ("disqualified", "fns_disqualified"),
                     ("bankruptcy", "fns_bankruptcy")):
        f[out] = np.nan_to_num(_col(raw, src).to_numpy(), nan=0.0)
    f["is_liquidating"] = (raw.get("status", pd.Series("ACTIVE", index=ix)).astype(str) == "LIQUIDATING").astype(float).to_numpy()

    # 2. финансы
    f["revenue_ltm_log"] = np.log(np.maximum(rev_ltm, 1.0))
    f["revenue_growth_yoy"] = _div(rev_ltm, rev_prev) - 1
    f["revenue_growth_last2q"] = _div(REV[:, 6:].sum(1), REV[:, 4:6].sum(1)) - 1
    f["revenue_cv"] = _div(REV.std(1), revm)
    f["revenue_trend"] = _div((K * (REV - revm[:, None])).sum(1) / (K ** 2).sum(), revm)
    f["gross_margin"] = _div(rev_ltm - cogs_ltm, rev_ltm)
    f["ebitda_margin"] = _div(ebitda_ltm, rev_ltm)
    f["net_margin"] = _div(net_ltm, rev_ltm)
    f["roa"] = _div(net_ltm, ASSETS)
    f["current_ratio"] = _div(CA, STL)
    f["equity_ratio"] = _div(EQ, ASSETS)
    f["negative_equity"] = (EQ < 0).astype(float)
    f["liabilities_to_revenue"] = _div(LIAB, rev_ltm)
    f["liabilities_to_ebitda"] = np.where(ebitda_ltm > 0, np.minimum(_div(LIAB, ebitda_ltm), 50.0), 50.0)
    f["receivables_days"] = _div(REC, REV[:, 7]) * 91
    f["payables_days"] = _div(PAY, REV[:, 7]) * 91
    f["loss_quarters"] = (NET < 0).sum(1).astype(float)

    # 3. обороты
    f["turnover_avg_log"] = np.log(np.maximum(cr_avg, 1.0))
    f["turnover_trend"] = _div(CR[:, 9:].mean(1), CR[:, :3].mean(1)) - 1
    f["turnover_cv"] = _div(CR.std(1), cr_avg)
    f["turnover_to_revenue"] = _div(CR.sum(1), rev_ltm)
    f["counterparties_log"] = np.log(np.maximum(_col(raw, "counterparties_cnt").to_numpy(), 1.0))
    f["bank_tenure_months"] = _col(raw, "bank_tenure_months").to_numpy()
    f["balance_to_turnover"] = _div(BAL.mean(1), cr_avg)
    f["neg_balance_months"] = (BAL < 0).sum(1).astype(float)
    f["debit_credit_ratio"] = _div(DR.sum(1), CR.sum(1))

    # 4. кредитная история (при отсутствии истории поля = NaN, а не 0)
    f["has_credit_history"] = np.nan_to_num(has_hist, nan=0.0)
    f["history_months"] = _col(raw, "bki_history_months").to_numpy()
    f["open_loans"] = _col(raw, "bki_open_loans").to_numpy()
    f["closed_loans"] = _col(raw, "bki_closed_loans").to_numpy()
    f["debt_to_revenue"] = _div(_col(raw, "bki_outstanding_debt").to_numpy(), rev_ltm)
    f["existing_payment_to_ebitda"] = np.where(has_hist == 1, np.where(ebitda_m > 0, np.minimum(ex_pay / np.maximum(ebitda_m, 1.0), 10.0), 10.0), np.nan)
    f["overdue_30_cnt"] = _col(raw, "bki_overdue_30_cnt").to_numpy()
    f["overdue_60_cnt"] = _col(raw, "bki_overdue_60_cnt").to_numpy()
    f["overdue_90_cnt"] = _col(raw, "bki_overdue_90_cnt").to_numpy()
    f["max_overdue_days"] = _col(raw, "bki_max_overdue_days").to_numpy()
    md = f["max_overdue_days"].to_numpy()
    f["delinquency_severity"] = np.where(np.isnan(md), np.nan, np.select([md >= 90, md >= 60, md > 0], [3.0, 2.0, 1.0], default=0.0))
    cur = _col(raw, "bki_current_overdue_amount").to_numpy()
    f["current_overdue_flag"] = np.where(np.isnan(cur), np.nan, (cur > 0).astype(float))
    f["defaults_cnt"] = _col(raw, "bki_defaults_cnt").to_numpy()
    f["inquiries_3m"] = _col(raw, "bki_inquiries_3m").to_numpy()
    f["inquiries_12m"] = _col(raw, "bki_inquiries_12m").to_numpy()

    # 5. параметры заявки
    f["product_code"] = raw.get("product_code", pd.Series(np.nan, index=ix)).astype(object).to_numpy()
    f["requested_amount_log"] = np.log(np.maximum(amount, 1.0))
    f["term_months"] = term
    f["amount_to_revenue"] = _div(amount, rev_ltm)
    f["amount_to_turnover"] = _div(amount, cr_avg)
    f["new_payment_to_ebitda"] = np.where(ebitda_m > 0, np.minimum(new_pay / np.maximum(ebitda_m, 1.0), 10.0), 10.0)
    f["payment_burden"] = np.where(ebitda_m > 0, np.minimum(tot_pay / np.maximum(ebitda_m, 1.0), 10.0), 10.0)
    f["dscr"] = np.where(tot_pay > 0, np.clip(ebitda_m / np.maximum(tot_pay, 1.0), -5.0, 10.0), 10.0)
    f["own_contribution_pct"] = np.nan_to_num(_col(raw, "own_contribution_pct").to_numpy(), nan=0.0)
    f["collateral_cover"] = np.nan_to_num(_div(_col(raw, "collateral_value").to_numpy(), amount), nan=0.0)

    f = f[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return f


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Клиппинг числовых признаков по обученным квантилям (устойчивость к выбросам на инференсе)."""

    def __init__(self, lower=0.005, upper=0.995, columns=None):
        self.lower, self.upper, self.columns = lower, upper, columns

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns or CLIP_FEATURES
        self.bounds_ = {c: (float(X[c].quantile(self.lower)), float(X[c].quantile(self.upper))) for c in cols if c in X}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = pd.to_numeric(X[c], errors="coerce").clip(lo, hi)
        return X

    def clipped_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        """True там, где значение вышло за обученный диапазон (для warnings[] в ответе API)."""
        out = {}
        for c, (lo, hi) in self.bounds_.items():
            v = pd.to_numeric(X[c], errors="coerce")
            out[c] = (v < lo) | (v > hi)
        return pd.DataFrame(out, index=X.index)
