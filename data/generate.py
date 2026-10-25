"""Генератор СИНТЕТИЧЕСКИХ данных для прототипа «Умный кредитный кабинет».

Все данные вымышлены и сгенерированы командой. ИНН/ОГРН имеют корректные контрольные
суммы, но код региона = «00» (не существует) -> реальных лиц и организаций не бывает.

Запуск:  python -m data.generate --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_VERSION = "synth-v1.0"
SYNTH_NOTICE = ("СИНТЕТИЧЕСКИЕ ДАННЫЕ: сгенерированы командой хакатона; к реальным лицам "
                "и организациям не относятся. ИНН/ОГРН - из несуществующего диапазона (регион 00).")
DEMO_DATE = "2026-10-26"
TRAIN_DATE_FROM, TRAIN_DATE_TO = "2023-01-01", "2025-09-30"   # исход 12 мес. успевает наблюдаться
N_TRAIN, N_DEMO_POOL = 12000, 600
SIGNAL = 0.55   # сила наблюдаемых факторов в порождающей модели (подобрана так, чтобы Gini был реалистичным)


# ----------------------------------------------------------------------------- ИНН / ОГРН
_W10 = [2, 4, 10, 3, 5, 9, 4, 6, 8]
_W11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
_W12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]


def _ctrl(digits: str, weights: list[int]) -> int:
    return sum(int(d) * w for d, w in zip(digits, weights)) % 11 % 10


def make_inn10(i: int) -> str:
    """ИНН юрлица (10 цифр): '00' + 7 цифр + контрольная."""
    body = "00" + f"{(i * 7919 + 13) % 10**7:07d}"
    return body + str(_ctrl(body, _W10))


def make_inn12(i: int) -> str:
    """ИНН ИП (12 цифр): '00' + 8 цифр + 2 контрольные."""
    body = "00" + f"{(i * 7919 + 13) % 10**8:08d}"
    c11 = str(_ctrl(body, _W11))
    c12 = str(_ctrl(body + c11, _W12))
    return body + c11 + c12


def make_ogrn(i: int, year: int) -> str:
    """ОГРН (13 цифр): '1' + год(2) + регион '00' + 7 цифр + контрольная (mod 11)."""
    body = "1" + f"{year % 100:02d}" + "00" + f"{(i * 104729 + 7) % 10**7:07d}"
    return body + str(int(body) % 11 % 10)


def make_ogrnip(i: int, year: int) -> str:
    """ОГРНИП (15 цифр): '3' + год(2) + регион '00' + 9 цифр + контрольная (mod 13)."""
    body = "3" + f"{year % 100:02d}" + "00" + f"{(i * 104729 + 7) % 10**9:09d}"
    return body + str(int(body) % 13 % 10)


def is_valid_inn(inn: str) -> bool:
    if not inn.isdigit():
        return False
    if len(inn) == 10:
        return _ctrl(inn[:9], _W10) == int(inn[9])
    if len(inn) == 12:
        return _ctrl(inn[:10], _W11) == int(inn[10]) and _ctrl(inn[:11], _W12) == int(inn[11])
    return False


def is_valid_ogrn(ogrn: str) -> bool:
    if not ogrn.isdigit():
        return False
    if len(ogrn) == 13:
        return int(ogrn[:12]) % 11 % 10 == int(ogrn[12])
    if len(ogrn) == 15:
        return int(ogrn[:14]) % 13 % 10 == int(ogrn[14])
    return False


def is_synthetic_id(inn: str) -> bool:
    """Синтетика = регион «00» в ИНН (реальных ИНН с кодом 00 не бывает)."""
    return inn[:2] == "00"


# ----------------------------------------------------------------------------- справочники
# (код, [ОКВЭД], валовая маржа, EBITDA-маржа, сдвиг риска (логит), амплитуда сезонности,
#  фаза, дни дебиторки, дни кредиторки, дни запасов, вес в выборке)
INDUSTRIES = [
    ("retail_food",    ["47.11", "47.19", "47.25"], .22, .06,  .00, .05, 0.0,  4, 20, 25, .16),
    ("catering",       ["56.10", "56.29"],          .60, .10,  .25, .10, 1.0,  3, 15, 10, .10),
    ("construction",   ["41.20", "43.21", "43.39"], .22, .08,  .30, .25, 0.5, 45, 40, 20, .11),
    ("transport",      ["49.41", "52.29"],          .25, .14,  .10, .08, 0.0, 30, 25,  5, .09),
    ("it_services",    ["62.01", "62.02", "63.11"], .60, .22, -.35, .03, 0.0, 40, 15,  0, .09),
    ("manufacturing",  ["25.62", "10.71", "16.10"], .30, .13,  .00, .10, 0.8, 40, 35, 45, .11),
    ("wholesale",      ["46.90", "46.43", "46.71"], .14, .05,  .05, .06, 0.0, 35, 30, 40, .13),
    ("agriculture",    ["01.11", "01.41", "01.13"], .30, .16,  .15, .40, 2.0, 30, 30, 60, .05),
    ("personal_serv",  ["96.02", "96.04", "95.11"], .70, .20, -.10, .05, 0.0,  5, 10,  5, .08),
    ("health_edu",     ["86.21", "85.41", "86.23"], .65, .18, -.25, .05, 0.0, 15, 15,  5, .05),
    ("real_estate",    ["68.20", "68.31"],          .80, .45, -.10, .03, 0.0, 10, 10,  0, .03),
]
IND_NAMES = [x[0] for x in INDUSTRIES]
IND_W = np.array([x[10] for x in INDUSTRIES])
IND_W = IND_W / IND_W.sum()

# (код региона, название, уровень 1..3, вес)
REGIONS = [
    ("77", "Москва", 1, .14), ("78", "Санкт-Петербург", 1, .08), ("50", "Московская область", 1, .09),
    ("23", "Краснодарский край", 2, .07), ("66", "Свердловская область", 2, .06),
    ("16", "Республика Татарстан", 2, .06), ("52", "Нижегородская область", 2, .05),
    ("54", "Новосибирская область", 2, .05), ("63", "Самарская область", 2, .04),
    ("61", "Ростовская область", 2, .05), ("02", "Республика Башкортостан", 2, .04),
    ("74", "Челябинская область", 3, .04), ("24", "Красноярский край", 3, .04),
    ("59", "Пермский край", 3, .04), ("36", "Воронежская область", 3, .03),
    ("38", "Иркутская область", 3, .03), ("42", "Кемеровская область", 3, .03),
]
REG_W = np.array([r[3] for r in REGIONS]); REG_W = REG_W / REG_W.sum()

# индикативные параметры продуктов (допущение; реальные условия задаёт credit-core)
PRODUCTS = {
    "OVERDRAFT":       dict(w=.20, rate=.205, terms=[6, 12],            mult=(0.10, 0.40), cap=15e6),
    "WORKING_CAPITAL": dict(w=.35, rate=.190, terms=[6, 12, 18, 24, 36], mult=(0.15, 0.80), cap=40e6),
    "INVESTMENT":      dict(w=.25, rate=.175, terms=[24, 36, 48, 60],    mult=(0.5, 2.5), cap=50e6),
    "EXPRESS":         dict(w=.20, rate=.240, terms=[6, 12, 18, 24],     mult=(0.03, 0.25), cap=3e6),
}
PURPOSES = {
    "OVERDRAFT": ["кассовый разрыв", "оплата поставщикам"],
    "WORKING_CAPITAL": ["пополнение оборотных средств", "закупка товара/сырья", "расширение/новая точка"],
    "INVESTMENT": ["покупка оборудования", "покупка транспорта", "ремонт/модернизация", "рефинансирование"],
    "EXPRESS": ["срочные расходы", "оплата поставщикам", "налоговые платежи"],
}
NAME_WORDS = ["Вектор", "Ромашка", "Орион", "Кедр", "Север", "Горизонт", "Мастер", "Прогресс", "Дельта",
              "Волна", "Исток", "Зенит", "Базис", "Лидер", "Маяк", "Родник", "Ладога", "Атлас", "Берег", "Факел"]
TAX_REGIMES = ["USN_INCOME", "USN_INCOME_EXPENSE", "OSN", "PSN"]


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def annuity(P, rate, n):
    r = rate / 12.0
    return P * r / (1.0 - (1.0 + r) ** (-n))


# ----------------------------------------------------------------------------- генерация
def _bki(rng, n, q, age_m, fin_debt_last, ind_risk, rev_month):
    """Кредитная история (БКИ). Для компаний без истории - NaN (а не 0!)."""
    p_none = np.clip(0.30 - 0.0035 * age_m + 0.04, 0.03, 0.5)
    has = rng.random(n) >= p_none
    hist_m = np.where(has, np.minimum(age_m, rng.uniform(6, 120, n)), np.nan)

    n_closed = rng.poisson(0.8 + np.nan_to_num(hist_m) / 28.0).astype(float)
    lam_open = 0.4 + fin_debt_last / 3.0e6
    n_open = np.minimum(rng.poisson(np.clip(lam_open, 0, 6)), 8).astype(float)
    n_open = np.where(has & (fin_debt_last > 1e5), np.maximum(n_open, 1), n_open)
    n_open = np.where(has & (fin_debt_last <= 1e5), np.minimum(n_open, 1) * (rng.random(n) < .4), n_open)

    outstanding = np.where(has, fin_debt_last * rng.uniform(0.4, 1.0, n), 0.0) * (n_open > 0)
    outstanding = np.round(outstanding, -3)
    rem_term = rng.uniform(12, 72, n)
    exist_pay = outstanding * (1.0 / rem_term + 0.18 / 12.0)
    open_amount = np.round(outstanding / rng.uniform(0.5, 0.9, n), -3)
    closed_amount = np.round(n_closed * rev_month * rng.uniform(0.3, 2.0, n), -3)

    # просрочки: чем хуже q, тем выше интенсивность
    n_all = n_open + n_closed + 1
    lam30 = np.exp(-2.1 - 0.75 * q + 0.35 * ind_risk + 0.30 * np.log(n_all))
    o30 = np.minimum(rng.poisson(lam30), 6)
    o60 = rng.binomial(o30, 0.35)
    o90 = rng.binomial(o60, 0.38)
    max_days = np.where(o90 > 0, rng.integers(91, 301, n),
               np.where(o60 > 0, rng.integers(61, 90, n),
               np.where(o30 > 0, rng.integers(30, 60, n), 0))).astype(float)
    cur_od = np.where(((o30 > 0) & (rng.random(n) < 0.25)) | ((o90 > 0) & (rng.random(n) < 0.45)),
                      outstanding * rng.uniform(0.01, 0.15, n), 0.0)
    cur_od = np.round(cur_od, -2)
    defaults = ((o90 > 0) & (rng.random(n) < 0.18)).astype(float)

    stress = np.clip(-q, 0, 3)
    inq3 = np.minimum(rng.poisson(0.5 + 0.9 * stress), 10).astype(float)
    inq12 = inq3 + np.minimum(rng.poisson(1.0 + 0.8 * stress), 15)

    out = {
        "bki_has_history": has.astype(int),
        "bki_history_months": hist_m,
        "bki_open_loans": n_open, "bki_closed_loans": n_closed,
        "bki_open_amount": open_amount, "bki_outstanding_debt": outstanding,
        "bki_monthly_payment": np.round(exist_pay, -2), "bki_closed_amount": closed_amount,
        "bki_overdue_30_cnt": o30.astype(float), "bki_overdue_60_cnt": o60.astype(float),
        "bki_overdue_90_cnt": o90.astype(float), "bki_max_overdue_days": max_days,
        "bki_current_overdue_amount": cur_od, "bki_defaults_cnt": defaults,
    }
    for k in out:  # нет истории -> все поля пропущены (кроме флага)
        if k != "bki_has_history":
            out[k] = np.where(has, out[k], np.nan)
    out["bki_inquiries_3m"], out["bki_inquiries_12m"] = inq3, inq12   # запросы известны и без истории
    return out


def generate_population(n: int, seed: int, id_start: int, date_from: str, date_to: str,
                        with_target: bool = True, target_rate: float = 0.10) -> pd.DataFrame:
    """Возвращает широкую таблицу: 1 строка = 1 компания-заявка."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    K = np.arange(1, 9)

    # ---------- скрытые переменные и профиль
    q = rng.normal(size=n)
    ind = rng.choice(len(INDUSTRIES), n, p=IND_W)
    gm0, em0, irisk, amp, phase, rec_d, pay_d, inv_d = (np.array([INDUSTRIES[j][c] for j in ind])
                                                         for c in (2, 3, 4, 5, 6, 7, 8, 9))
    is_ip = rng.random(n) < 0.38
    d0 = np.datetime64(date_from)
    span = (np.datetime64(date_to) - d0).astype(int)
    app_date = d0 + rng.integers(0, span + 1, n).astype("timedelta64[D]")
    app_month = (app_date.astype("datetime64[M]").astype(int) % 12) + 1
    app_year = app_date.astype("datetime64[Y]").astype(int) + 1970

    age_m = np.clip(np.exp(rng.normal(np.log(42), 0.9, n) + 0.15 * q), 3, 360).round()
    reg_date = app_date - (age_m * 30.4).astype(int).astype("timedelta64[D]")
    reg_year = reg_date.astype("datetime64[Y]").astype(int) + 1970

    rev_annual = np.exp(rng.normal(np.log(24e6), 0.95, n)) * np.where(is_ip, 0.55, 1.0) * (age_m / 42) ** 0.1
    rev_annual = np.clip(rev_annual, 2e6, 8e8)
    emp = np.round(rev_annual / np.exp(rng.normal(np.log(4.5e6), 0.5, n)))
    emp = np.clip(np.where(is_ip, emp - 1, np.maximum(emp, 1)), 0, 250)

    reg_idx = rng.choice(len(REGIONS), n, p=REG_W)
    p_reg = np.where(is_ip[:, None], [.50, .38, .05, .07], [.30, .25, .43, .02])
    cum = p_reg.cumsum(1)
    tax_i = (rng.random(n)[:, None] > cum).sum(1)
    tax_i = np.where(rev_annual > 3e8, 2, tax_i)
    tax_i = np.clip(tax_i, 0, 3)

    liquidating = rng.random(n) < 0.010

    # ---------- выручка по 8 кварталам (K=1 самый старый, K=8 последний полный)
    base_q = rev_annual / 4
    g = rng.normal(0.008 + 0.012 * q, 0.025)
    last_qoy = ((app_month - 1) // 3 - 1) % 4
    qoy = (last_qoy[:, None] - (8 - K)[None, :]) % 4
    season = 1 + amp[:, None] * np.sin(2 * np.pi * qoy / 4 + phase[:, None])
    shock = rng.random(n) < _sig(-1.9 - 0.9 * q)
    shock_start = rng.integers(6, 9, n)
    shock_depth = rng.uniform(0.15, 0.5, n)
    shock_f = 1 - shock_depth[:, None] * ((K[None, :] >= shock_start[:, None]) & shock[:, None])
    vol = np.clip(0.05 * np.exp(-0.30 * q + rng.normal(0, .35, n)), 0.01, 0.35)
    rev = base_q[:, None] * np.exp(g[:, None] * (K[None, :] - 4.5)) * season * shock_f
    rev *= np.exp(rng.normal(0, 1, (n, 8)) * vol[:, None])

    # ---------- P&L
    gm = np.clip(gm0 + 0.03 * q + rng.normal(0, .03, n), 0.05, 0.90)
    em_t = np.clip(em0 + 0.03 * q + rng.normal(0, .025, n), -0.05, 0.50)
    fixed = np.maximum(base_q * (gm - em_t), 0.02 * base_q)[:, None] * (1 + 0.01 * (K[None, :] - 4.5))
    gp = rev * np.clip(gm[:, None] + rng.normal(0, .01, (n, 8)), 0.03, 0.92)
    cogs = rev - gp
    ebitda = gp - fixed
    dep = 0.012 * rev

    # ---------- баланс: прокатка капитала
    assets0 = rev_annual * np.exp(rng.normal(np.log(0.65), 0.35, n))
    autonomy0 = np.clip(0.30 + 0.10 * q + rng.normal(0, .12, n), -0.2, 0.85)
    equity_prev, liab_prev = autonomy0 * assets0, assets0 * (1 - autonomy0)
    p_none_pre = None  # доля финдолга задаётся ниже, после определения наличия истории
    fd_share = rng.uniform(0.1, 0.5, n)
    # компании без кредитной истории почти не имеют банковского долга
    age_for_hist = age_m
    p_none = np.clip(0.30 - 0.0035 * age_for_hist + 0.04, 0.03, 0.5)
    no_hist_hint = rng.random(n) < p_none
    fd_share = np.where(no_hist_hint, rng.uniform(0.0, 0.08, n), fd_share)
    div_rate = rng.uniform(0, 0.6, n)
    st_share = rng.uniform(0.2, 0.7, n)

    names = ("revenue", "cost_of_sales", "ebitda", "net_profit", "total_assets", "total_liabilities",
             "equity", "short_term_liabilities", "current_assets", "receivables", "payables")
    A = {k: np.zeros((n, 8)) for k in names}
    fin_debt_hist = np.zeros((n, 8))
    prev_rev = rev[:, 0] / np.exp(g)
    for k in range(8):
        r_k = rev[:, k]
        interest = fin_debt_hist[:, k - 1] * 0.17 / 4 if k > 0 else liab_prev * fd_share * 0.17 / 4
        pbt = ebitda[:, k] - dep[:, k] - interest
        tax = np.select(
            [tax_i == 0, tax_i == 1, tax_i == 2],
            [0.045 * r_k, np.maximum(0.15 * pbt, 0.01 * r_k), 0.20 * np.maximum(pbt, 0)],
            default=0.02 * r_k)
        net = pbt - tax
        equity = equity_prev + net - div_rate * np.maximum(net, 0)
        liab = liab_prev * np.exp(0.6 * np.log(r_k / prev_rev) + rng.normal(0, .04, n)) + 0.8 * np.maximum(-net, 0)
        liab = np.maximum(liab, -equity + 0.08 * r_k)
        assets = equity + liab
        payables = np.minimum(0.7 * liab, r_k * pay_d / 91 * np.exp(rng.normal(0, .15, n)))
        fd = np.minimum(fd_share * liab, liab - payables)
        other = liab - payables - fd
        stl = payables + 0.8 * other + st_share * fd
        recv = np.minimum(0.6 * assets, r_k * rec_d / 91 * np.exp(rng.normal(0, .2, n)))
        inv = np.minimum(0.5 * assets, cogs[:, k] * inv_d / 91)
        cash = r_k * 0.02 * np.exp(rng.normal(0, .5, n))
        ca = np.minimum(0.95 * assets, recv + inv + cash)
        for nm, v in (("revenue", r_k), ("cost_of_sales", cogs[:, k]), ("ebitda", ebitda[:, k]),
                      ("net_profit", net), ("total_assets", assets), ("total_liabilities", liab),
                      ("equity", equity), ("short_term_liabilities", stl), ("current_assets", ca),
                      ("receivables", recv), ("payables", payables)):
            A[nm][:, k] = v
        fin_debt_hist[:, k] = fd
        equity_prev, liab_prev, prev_rev = equity, liab, r_k

    # ---------- обороты по счёту, 12 месяцев (M=1 самый старый)
    M = np.arange(12)
    q_of_m = 4 + M // 3
    mon_rev = rev[:, q_of_m] / 3 * np.exp(rng.normal(0, 1, (n, 12)) * (0.10 + vol[:, None]))
    ratio_c = np.exp(rng.normal(0, 0.2, n))
    credit = mon_rev * ratio_c[:, None] * np.exp(rng.normal(0, .06, (n, 12)))
    debit = credit * rng.uniform(0.88, 1.03, n)[:, None] * np.exp(rng.normal(0, .05, (n, 12)))
    bal_level = credit.mean(1) * np.clip(0.15 + 0.08 * q + rng.normal(0, .08, n), 0.005, 0.6)
    sd_bal = 0.5 + 0.3 * np.clip(-q, 0, 3)
    end_bal = bal_level[:, None] * (1 + rng.normal(0, 1, (n, 12)) * sd_bal[:, None])
    counterparties = np.maximum(1, np.round(np.exp(rng.normal(np.log(14) + .35 * np.log(rev_annual / 24e6), .55, n))))
    tenure = np.clip(np.round(age_m * rng.uniform(0.25, 1.0, n)), 1, age_m)

    # ---------- БКИ
    fin_debt_last = fin_debt_hist[:, 7]
    bki = _bki(rng, n, q, age_m, fin_debt_last, irisk, rev[:, 7] / 3)

    # ---------- ФНС / проверки
    rev_ltm = rev[:, 4:].sum(1)
    tax_flag = rng.random(n) < _sig(-2.6 - 0.7 * q + irisk)
    tax_debt = np.where(tax_flag, np.round(rev_ltm / 12 * rng.uniform(0.01, 0.5, n), -2), 0.0)
    blocked = rng.random(n) < _sig(-4.3 - 0.4 * q + 1.0 * tax_flag)
    mass_addr = rng.random(n) < 0.02
    mass_dir = rng.random(n) < 0.015
    disq = rng.random(n) < 0.004
    bankr = rng.random(n) < _sig(-5.6 - 0.8 * q)

    # ---------- заявка
    prod_names = list(PRODUCTS)
    prod = rng.choice(prod_names, n, p=[PRODUCTS[p]["w"] for p in prod_names])
    mon_rev_last = rev[:, 7] / 3
    amount = np.zeros(n); term = np.zeros(n); rate = np.zeros(n); purpose = np.empty(n, dtype=object)
    for p in prod_names:
        m = prod == p
        cfg = PRODUCTS[p]
        lo, hi = cfg["mult"]
        amount[m] = np.clip(mon_rev_last[m] * rng.uniform(lo, hi, m.sum()), 1e5, cfg["cap"])
        term[m] = rng.choice(cfg["terms"], m.sum())
        rate[m] = cfg["rate"]
        purpose[m] = rng.choice(PURPOSES[p], m.sum())
    amount = np.round(amount, -4)
    own_contrib = np.where(prod == "INVESTMENT", np.round(rng.uniform(0, 0.4, n), 2), 0.0)
    p_coll = np.select([prod == "INVESTMENT", prod == "WORKING_CAPITAL"], [.75, .35], default=.10)
    has_coll = rng.random(n) < p_coll
    coll_type = np.where(has_coll, rng.choice(["REAL_ESTATE", "VEHICLE", "EQUIPMENT", "GUARANTEE"], n,
                                              p=[.30, .20, .25, .25]), "NONE")
    coll_cover = np.where(has_coll, np.round(rng.uniform(0.5, 1.5, n), 2), 0.0)
    coll_value = np.round(amount * coll_cover, -4)
    new_pay = annuity(amount, rate, term)

    # ---------- порождающая модель риска (DGP) для default_flag
    ebitda_ltm = A["ebitda"][:, 4:].sum(1)
    ebitda_m = ebitda_ltm / 12
    exist_pay = np.nan_to_num(bki["bki_monthly_payment"])
    burden = np.where(ebitda_m > 0, (exist_pay + new_pay) / np.maximum(ebitda_m, 1.0), 3.0)
    rev_prev = rev[:, :4].sum(1)
    growth = rev_ltm / rev_prev - 1
    cv = rev.std(1) / rev.mean(1)
    em_ltm = ebitda_ltm / rev_ltm
    eq_ratio = A["equity"][:, 7] / A["total_assets"][:, 7]
    cur_ratio = A["current_assets"][:, 7] / np.maximum(A["short_term_liabilities"][:, 7], 1.0)
    loss_q = (A["net_profit"] < 0).sum(1)
    tax_ratio = tax_debt / np.maximum(rev_ltm / 12, 1.0)
    nh = bki["bki_has_history"] == 0
    o30, o60, o90 = (np.nan_to_num(bki[f"bki_overdue_{x}_cnt"]) for x in (30, 60, 90))
    cur_od = np.nan_to_num(bki["bki_current_overdue_amount"])
    closed = np.nan_to_num(bki["bki_closed_loans"])
    year_drift = 0.15 * (app_year - 2024)
    eps = rng.normal(0, 1.5, n)
    obs = (1.1 * np.clip(burden, 0, 2.5)
             + 0.9 * (o90 > 0) + 0.35 * np.minimum(o60, 3) + 0.20 * np.minimum(o30, 4)
             + 0.8 * (cur_od > 0)
             + 1.5 * blocked + 0.6 * mass_addr + 0.5 * mass_dir + 1.2 * disq + 1.8 * bankr
             + 1.5 * np.minimum(tax_ratio, 1.0)
             + 2.0 * (np.clip(cv, 0, .6) - 0.12)
             - 1.2 * np.clip(growth, -0.5, 0.5)
             - 0.45 * (np.log(age_m) - np.log(36))
             + 0.8 * (eq_ratio < 0) - 1.0 * np.clip(eq_ratio, -0.3, 0.6)
             - 0.4 * np.clip(cur_ratio - 1, -1, 1.5)
             + 0.18 * np.minimum(bki["bki_inquiries_3m"], 6)
             + irisk + 0.25 * nh - 0.3 * ((closed >= 3) & (o30 == 0))
             - 0.9 * np.minimum(coll_cover, 1.2) * has_coll
             + 0.25 * loss_q - 3.0 * np.clip(em_ltm - 0.10, -0.3, 0.3)
             )
    logit = SIGNAL * obs + year_drift - 0.35 * q + eps
    # свободный член подбирается так, чтобы средняя доля дефолтов ~ target_rate
    lo, hi = -15.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sig(logit + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    b0 = (lo + hi) / 2
    true_pd = _sig(logit + b0)                       # PD, из которой реально «выбрасывается» исход
    oracle_pd = _sig(SIGNAL * obs + year_drift + b0)    # PD только по наблюдаемой части (без скрытых q и шума)
    default_flag = (rng.random(n) < true_pd).astype(int) if with_target else np.full(n, np.nan)

    # ---------- сборка таблицы
    gid = id_start + idx
    inn = np.array([make_inn12(int(i)) if ip else make_inn10(int(i)) for i, ip in zip(gid, is_ip)])
    ogrn = np.array([make_ogrnip(int(i), int(y)) if ip else make_ogrn(int(i), int(y))
                     for i, ip, y in zip(gid, is_ip, reg_year)])
    w1 = np.array(NAME_WORDS)[rng.integers(0, len(NAME_WORDS), n)]
    w2 = np.array(NAME_WORDS)[rng.integers(0, len(NAME_WORDS), n)]
    name = np.array([f"ИП Тестовый-{i:06d} (синт.)" if ip else f"ООО «{a}-{b} Синт-{i:06d}»"
                     for i, ip, a, b in zip(gid, is_ip, w1, w2)])
    okved = np.array([INDUSTRIES[j][1][rng.integers(0, len(INDUSTRIES[j][1]))] for j in ind])

    cols = {
        "company_id": np.array([f"C{i:06d}" for i in gid]), "inn": inn, "ogrn": ogrn, "company_name": name,
        "legal_form": np.where(is_ip, "IP", "OOO"),
        "reg_date": reg_date.astype("datetime64[D]").astype(str), "application_date": app_date.astype("datetime64[D]").astype(str),
        "okved": okved, "industry": np.array(IND_NAMES)[ind],
        "region_code": np.array([REGIONS[j][0] for j in reg_idx]),
        "region_name": np.array([REGIONS[j][1] for j in reg_idx]),
        "region_tier": np.array([REGIONS[j][2] for j in reg_idx]),
        "employees": emp.astype(int), "tax_regime": np.array(TAX_REGIMES)[tax_i],
        "status": np.where(liquidating, "LIQUIDATING", "ACTIVE"),
        "bank_tenure_months": tenure.astype(int), "counterparties_cnt": counterparties.astype(int),
    }
    for nm in names:
        for k in range(8):
            cols[f"{nm}_q{k + 1}"] = np.round(A[nm][:, k], 0)
    for k in range(12):
        cols[f"turnover_credit_m{k + 1}"] = np.round(credit[:, k], 0)
        cols[f"turnover_debit_m{k + 1}"] = np.round(debit[:, k], 0)
        cols[f"balance_end_m{k + 1}"] = np.round(end_bal[:, k], 0)
    cols.update(bki)
    cols.update({
        "fns_tax_debt": tax_debt, "fns_account_blocked": blocked.astype(int),
        "fns_mass_address": mass_addr.astype(int), "fns_mass_director": mass_dir.astype(int),
        "fns_disqualified": disq.astype(int), "fns_bankruptcy": bankr.astype(int),
        "product_code": prod, "requested_amount": amount, "term_months": term.astype(int),
        "indicative_rate": rate, "purpose": purpose, "own_contribution_pct": own_contrib,
        "collateral_type": coll_type, "collateral_value": coll_value,
        "default_flag": default_flag, "oracle_pd": np.round(oracle_pd, 5),
    })
    df = pd.DataFrame(cols)
    for c in df.columns:  # счётчики БКИ - целые с пропусками
        if c.startswith("bki_") and c not in ("bki_history_months",) and "amount" not in c and "payment" not in c \
                and "debt" not in c:
            df[c] = df[c].astype("Int64")
    return df


# ----------------------------------------------------------------------------- демо-сценарии
def apply_scenario(df: pd.DataFrame, row: int, scenario: str) -> None:
    """Жёстко задаёт признаки строки `row` под сценарий (флаги политики)."""
    c = df.columns
    rev_ltm = sum(df.loc[row, f"revenue_q{k}"] for k in range(5, 9))
    if scenario == "LIQUIDATION":
        df.loc[row, "status"] = "LIQUIDATING"
    elif scenario == "ACCOUNT_BLOCKED":
        df.loc[row, "fns_account_blocked"] = 1
    elif scenario == "TAX_DEBT":
        df.loc[row, "fns_tax_debt"] = round(0.06 * rev_ltm, -2)
    elif scenario == "DELINQUENCY_90":
        out = df.loc[row, "bki_outstanding_debt"]
        out = out if pd.notna(out) and out > 0 else 2_000_000.0
        df.loc[row, ["bki_has_history", "bki_open_loans", "bki_outstanding_debt"]] = [1, 2, out]
        df.loc[row, ["bki_overdue_30_cnt", "bki_overdue_60_cnt", "bki_overdue_90_cnt"]] = [3, 2, 1]
        df.loc[row, "bki_max_overdue_days"] = 142
        df.loc[row, "bki_current_overdue_amount"] = round(out * 0.08, -2)
        df.loc[row, "bki_history_months"] = max(24, df.loc[row, "bki_history_months"] if pd.notna(df.loc[row, "bki_history_months"]) else 24)
    elif scenario == "NO_HISTORY":
        for col in c:
            if col.startswith("bki_") and col not in ("bki_inquiries_3m", "bki_inquiries_12m"):
                df.loc[row, col] = np.nan
        df.loc[row, "bki_has_history"] = 0
    elif scenario == "MASS_ADDRESS":
        df.loc[row, "fns_mass_address"] = 1
    elif scenario == "MASS_DIRECTOR":
        df.loc[row, "fns_mass_director"] = 1
    elif scenario == "DISQUALIFIED":
        df.loc[row, "fns_disqualified"] = 1
    elif scenario == "BANKRUPTCY":
        df.loc[row, "fns_bankruptcy"] = 1
    elif scenario == "YOUNG_BUSINESS":
        d = pd.Timestamp(df.loc[row, "application_date"]) - pd.Timedelta(days=120)
        df.loc[row, "reg_date"] = d.strftime("%Y-%m-%d")
        df.loc[row, "bank_tenure_months"] = 3
    elif scenario == "NEGATIVE_EQUITY":
        a = df.loc[row, "total_assets_q8"]
        df.loc[row, "equity_q8"] = -0.1 * a
        df.loc[row, "total_liabilities_q8"] = 1.1 * a
    else:
        raise ValueError(scenario)


# ----------------------------------------------------------------------------- запись
def _write_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False, float_format="%.4f", lineterminator="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(seed: int = 42, out_dir: str = "data/generated") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = generate_population(N_TRAIN, seed, id_start=1, date_from=TRAIN_DATE_FROM, date_to=TRAIN_DATE_TO)
    pool = generate_population(N_DEMO_POOL, seed + 1000, id_start=900_001, date_from=DEMO_DATE,
                               date_to=DEMO_DATE, with_target=False)
    manifest = {
        "dataset_version": DATASET_VERSION, "seed": seed, "notice": SYNTH_NOTICE,
        "train_rows": len(train), "train_default_rate": round(float(train["default_flag"].mean()), 4),
        "demo_pool_rows": len(pool), "demo_date": DEMO_DATE,
        "files": {
            "applications_raw.csv": _write_csv(train, out / "applications_raw.csv"),
            "demo_pool_raw.csv": _write_csv(pool, out / "demo_pool_raw.csv"),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/generated")
    a = ap.parse_args()
    m = main(a.seed, a.out)
    print(json.dumps(m, ensure_ascii=False, indent=2))
