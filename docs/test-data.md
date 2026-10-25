# Тестовые данные: демо-кейсы (ИНН → ожидаемое поведение)

> **Все данные синтетические**: сгенерированы командой, к реальным лицам и организациям не относятся; ИНН/ОГРН имеют валидные контрольные суммы, но лежат в несуществующем диапазоне (регион `00`).

Версия датасета: `synth-v1.0`, seed 42, дата заявок демо-набора 2026-10-26, модель `v1.0.0-woe-logit`. Файлы: `data/fixtures/*.json` (по источнику и общий `companies.json`), `companies_flat.csv`, `demo_cases.csv`.

**Как читать колонку «Референсное решение».** Это иллюстрация того, как `credit-core` может интерпретировать оценку (пороги — конфигурация Backend). Скоринг отдаёт `pd`, рейтинг, лимит и флаги; решение принимает не он. Иллюстративные правила: любой `CRITICAL`-флаг → отказ (hard-stop); рейтинг C/D или нулевой лимит → отказ; `WARNING`-флаг или рейтинг B/CCC/CC → ручное рассмотрение; лимит ≥ запроса → автоодобрение; лимит ≥ 30% запроса → контр-оферта.

## Сценарные кейсы

| Кейс | ИНН | Сценарий | Продукт | Запрос, ₽ | Срок | PD | Рейтинг | Лимит, ₽ | Флаги политики | Референсное решение |
|---|---|---|---|---|---|---|---|---|---|---|
| DC01 | `0092935763` | AUTO_APPROVE_PRIME<br>_Сильный заёмщик: рейтинг A и выше  флагов нет  лимит ≥ 120% запроса — автоодобрение_ | WORKING_CAPITAL | 220 000 | 18 | 0.9% | AAA | 630 000 | — | **AUTO_APPROVE** |
| DC02 | `0008932140` | AUTO_APPROVE_STANDARD<br>_Типичный заёмщик (BBB/BB): лимит покрывает запрос  флагов нет_ | WORKING_CAPITAL | 630 000 | 24 | 4.7% | BBB | 850 000 | — | **AUTO_APPROVE** |
| DC03 | `002899265408` | COUNTER_OFFER_LIMIT<br>_Рекомендованный лимит ниже запрошенной суммы — контр-оферта по сумме/сроку_ | OVERDRAFT | 180 000 | 12 | 1.2% | AAA | 130 000 | — | **COUNTER_OFFER** |
| DC04 | `0006160498` | MANUAL_REVIEW_GREY_ZONE<br>_Серая зона по рейтингу (B/CCC) без флагов → ручное рассмотрение андеррайтером_ | WORKING_CAPITAL | 390 000 | 24 | 12.4% | B | 340 000 | — | **MANUAL_REVIEW** |
| DC05 | `002958657907` | REJECT_HIGH_RISK<br>_Высокий PD (рейтинг C/D) без стоп-флагов — отказ по риску_ | OVERDRAFT | 100 000 | 6 | 28.2% | C | 0 | HIGH_DEBT_BURDEN | **REJECT_RISK** |
| DC06 | `002924606206` | FLAG_NO_CREDIT_HISTORY<br>_Нет кредитной истории (тонкое дело) — оценка менее надёжна  нужен ручной разбор_ | WORKING_CAPITAL | 1 120 000 | 36 | 1.2% | AAA | 5 840 000 | NO_CREDIT_HISTORY | **MANUAL_REVIEW** |
| DC07 | `0010199184` | FLAG_LIQUIDATION_IN_PROGRESS<br>_Компания в процессе ликвидации — hard-stop_ | INVESTMENT | 2 150 000 | 36 | 1.1% | AAA | 10 550 000 | LIQUIDATION_IN_PROGRESS | **REJECT_HARD_STOP** |
| DC08 | `0088580317` | FLAG_ACCOUNT_BLOCKED<br>_Блокировка расчётных счетов — hard-stop_ | EXPRESS | 250 000 | 12 | 1.2% | AAA | 2 250 000 | ACCOUNT_BLOCKED | **REJECT_HARD_STOP** |
| DC09 | `0083749721` | FLAG_TAX_DEBT_SIGNIFICANT<br>_Значимый налоговый долг (6% годовой выручки) — требуется разбор_ | EXPRESS | 450 000 | 24 | 1.2% | AAA | 1 720 000 | TAX_DEBT_SIGNIFICANT | **MANUAL_REVIEW** |
| DC10 | `003062396855` | FLAG_DELINQUENCY_90_PLUS<br>_Просрочка 90+ дней в кредитной истории — hard-stop_ | EXPRESS | 120 000 | 24 | 6.2% | BBB | 870 000 | DELINQUENCY_90_PLUS;CURRENT_OVERDUE | **REJECT_HARD_STOP** |
| DC11 | `0001884232` | FLAG_MASS_ADDRESS<br>_Массовый адрес регистрации — риск «технической» компании_ | EXPRESS | 870 000 | 12 | 1.2% | AAA | 2 260 000 | MASS_ADDRESS | **MANUAL_REVIEW** |
| DC12 | `002940444249` | FLAG_DISQUALIFIED_DIRECTOR<br>_Дисквалифицированный руководитель — hard-stop_ | WORKING_CAPITAL | 100 000 | 24 | 1.2% | AAA | 430 000 | DISQUALIFIED_DIRECTOR | **REJECT_HARD_STOP** |
| DC13 | `002923814316` | FLAG_YOUNG_BUSINESS<br>_Бизнесу ~4 месяца — недостаточно истории_ | INVESTMENT | 1 300 000 | 24 | 1.5% | AAA | 3 580 000 | YOUNG_BUSINESS | **MANUAL_REVIEW** |
| DC14 | `003132875910` | FLAG_BANKRUPTCY_SIGNS<br>_Признаки банкротства — hard-stop_ | INVESTMENT | 2 160 000 | 24 | 1.3% | AAA | 5 960 000 | BANKRUPTCY_SIGNS | **REJECT_HARD_STOP** |
| DC15 | `002900057388` | FLAG_NEGATIVE_EQUITY<br>_Отрицательный собственный капитал_ | EXPRESS | 560 000 | 12 | 4.4% | A | 1 660 000 | NEGATIVE_EQUITY | **MANUAL_REVIEW** |

## Типичные компании (по 5 на рейтинговую градацию)

| Кейс | ИНН | Продукт | Запрос, ₽ | PD | Рейтинг | Лимит, ₽ | Флаги | Референсное решение |
|---|---|---|---|---|---|---|---|---|
| T01 | `0002834516` | WORKING_CAPITAL | 4 050 000 | 1.1% | AAA | 6 120 000 | LIQUIDATION_IN_PROGRESS | REJECT_HARD_STOP |
| T02 | `002854127133` | WORKING_CAPITAL | 150 000 | 1.5% | AAA | 840 000 | — | AUTO_APPROVE |
| T03 | `003091697180` | EXPRESS | 210 000 | 1.7% | AAA | 300 000 | NO_CREDIT_HISTORY | MANUAL_REVIEW |
| T04 | `003014882862` | OVERDRAFT | 160 000 | 1.8% | AAA | 340 000 | — | AUTO_APPROVE |
| T05 | `002960241737` | WORKING_CAPITAL | 990 000 | 1.9% | AAA | 2 080 000 | — | AUTO_APPROVE |
| T06 | `0017167907` | INVESTMENT | 5 980 000 | 2.0% | AA | 12 610 000 | — | AUTO_APPROVE |
| T07 | `002868381348` | WORKING_CAPITAL | 350 000 | 2.1% | AA | 540 000 | — | AUTO_APPROVE |
| T08 | `0018514134` | WORKING_CAPITAL | 1 340 000 | 2.3% | AA | 2 880 000 | NO_CREDIT_HISTORY | MANUAL_REVIEW |
| T09 | `002819283524` | WORKING_CAPITAL | 990 000 | 2.5% | AA | 1 980 000 | — | AUTO_APPROVE |
| T10 | `0094598753` | INVESTMENT | 4 180 000 | 2.8% | AA | 3 630 000 | — | COUNTER_OFFER |
| T11 | `0007744295` | OVERDRAFT | 720 000 | 3.0% | A | 1 140 000 | — | AUTO_APPROVE |
| T12 | `003103575616` | INVESTMENT | 250 000 | 3.3% | A | 0 | HIGH_DEBT_BURDEN | REJECT_NO_CAPACITY |
| T13 | `0075513962` | INVESTMENT | 50 000 000 | 3.4% | A | 50 000 000 | — | AUTO_APPROVE |
| T14 | `0090164113` | EXPRESS | 100 000 | 3.7% | A | 550 000 | — | AUTO_APPROVE |
| T15 | `0078523180` | EXPRESS | 140 000 | 4.0% | A | 610 000 | — | AUTO_APPROVE |
| T16 | `0072742315` | WORKING_CAPITAL | 260 000 | 4.5% | BBB | 210 000 | — | COUNTER_OFFER |
| T17 | `0004418310` | INVESTMENT | 4 350 000 | 4.9% | BBB | 660 000 | HIGH_DEBT_BURDEN | MANUAL_REVIEW |
| T18 | `0080344558` | OVERDRAFT | 1 320 000 | 5.1% | BBB | 1 350 000 | NO_CREDIT_HISTORY | MANUAL_REVIEW |
| T19 | `003098824215` | INVESTMENT | 220 000 | 5.6% | BBB | 490 000 | NO_CREDIT_HISTORY;FREQUENT_INQUIRIES | MANUAL_REVIEW |
| T20 | `003128124566` | INVESTMENT | 3 970 000 | 6.0% | BBB | 0 | HIGH_DEBT_BURDEN | REJECT_NO_CAPACITY |
| T21 | `0002042616` | WORKING_CAPITAL | 690 000 | 6.5% | BB | 530 000 | — | COUNTER_OFFER |
| T22 | `0084779197` | OVERDRAFT | 100 000 | 7.1% | BB | 0 | NO_CREDIT_HISTORY;HIGH_DEBT_BURDEN | REJECT_NO_CAPACITY |
| T23 | `0082403495` | WORKING_CAPITAL | 2 490 000 | 7.5% | BB | 0 | HIGH_DEBT_BURDEN | REJECT_NO_CAPACITY |
| T24 | `0096103364` | OVERDRAFT | 100 000 | 7.9% | BB | 0 | HIGH_DEBT_BURDEN | REJECT_NO_CAPACITY |
| T25 | `003073483411` | WORKING_CAPITAL | 1 060 000 | 8.3% | BB | 190 000 | — | REJECT_NO_CAPACITY |
| T26 | `002974495957` | OVERDRAFT | 790 000 | 9.2% | B | 0 | — | REJECT_NO_CAPACITY |
| T27 | `0016217626` | EXPRESS | 100 000 | 9.8% | B | 160 000 | TAX_DEBT_SIGNIFICANT | MANUAL_REVIEW |
| T28 | `002731382623` | EXPRESS | 100 000 | 10.6% | B | 0 | HIGH_DEBT_BURDEN | REJECT_NO_CAPACITY |
| T29 | `0010753518` | EXPRESS | 380 000 | 11.1% | B | 0 | HIGH_DEBT_BURDEN | REJECT_NO_CAPACITY |
| T30 | `003158216743` | EXPRESS | 100 000 | 11.3% | B | 0 | — | REJECT_NO_CAPACITY |
| T31 | `003174054793` | WORKING_CAPITAL | 210 000 | 12.6% | CCC | 100 000 | NO_CREDIT_HISTORY | MANUAL_REVIEW |
| T32 | `002977663543` | EXPRESS | 100 000 | 13.0% | CCC | 0 | HIGH_DEBT_BURDEN | REJECT_NO_CAPACITY |
| T33 | `002862838067` | WORKING_CAPITAL | 2 640 000 | 14.1% | CCC | 1 510 000 | NO_CREDIT_HISTORY | MANUAL_REVIEW |
| T34 | `0098795826` | WORKING_CAPITAL | 2 590 000 | 15.1% | CCC | 0 | CURRENT_OVERDUE;HIGH_DEBT_BURDEN;FREQUENT_INQUIRIES | REJECT_NO_CAPACITY |
| T35 | `0013841924` | INVESTMENT | 2 740 000 | 17.0% | CCC | 0 | HIGH_DEBT_BURDEN | REJECT_NO_CAPACITY |
| T36 | `0009328090` | EXPRESS | 100 000 | 18.2% | CC | 0 | HIGH_DEBT_BURDEN | REJECT_NO_CAPACITY |
| T37 | `0098558254` | OVERDRAFT | 120 000 | 19.6% | CC | 0 | HIGH_DEBT_BURDEN | REJECT_NO_CAPACITY |
| T38 | `002880259841` | WORKING_CAPITAL | 110 000 | 20.7% | CC | 0 | DELINQUENCY_90_PLUS;HIGH_DEBT_BURDEN | REJECT_HARD_STOP |
| T39 | `002763850514` | OVERDRAFT | 780 000 | 22.4% | CC | 0 | HIGH_DEBT_BURDEN;TURNOVER_MISMATCH | REJECT_NO_CAPACITY |
| T40 | `0015346530` | INVESTMENT | 1 660 000 | 24.1% | CC | 0 | HIGH_DEBT_BURDEN | REJECT_NO_CAPACITY |
| T41 | `0013921111` | EXPRESS | 100 000 | 28.4% | C | 0 | HIGH_DEBT_BURDEN | REJECT_RISK |
| T42 | `002900849231` | WORKING_CAPITAL | 100 000 | 30.4% | C | 0 | HIGH_DEBT_BURDEN | REJECT_RISK |
| T43 | `0000696388` | OVERDRAFT | 150 000 | 33.0% | C | 0 | DELINQUENCY_90_PLUS;CURRENT_OVERDUE;HIGH_DEBT_BURDEN | REJECT_HARD_STOP |
| T44 | `0093410906` | EXPRESS | 100 000 | 36.5% | C | 0 | NEGATIVE_EQUITY;HIGH_DEBT_BURDEN;TURNOVER_MISMATCH | REJECT_RISK |
| T45 | `0071792031` | WORKING_CAPITAL | 330 000 | 37.7% | C | 0 | NEGATIVE_EQUITY;HIGH_DEBT_BURDEN | REJECT_RISK |
| T46 | `003097240489` | INVESTMENT | 380 000 | 43.9% | D | 0 | CURRENT_OVERDUE;HIGH_DEBT_BURDEN | REJECT_RISK |
| T47 | `0014713019` | WORKING_CAPITAL | 320 000 | 45.2% | D | 0 | NEGATIVE_EQUITY;HIGH_DEBT_BURDEN | REJECT_RISK |
| T48 | `0077335338` | WORKING_CAPITAL | 1 460 000 | 46.7% | D | 0 | NEGATIVE_EQUITY;HIGH_DEBT_BURDEN | REJECT_RISK |
| T49 | `0004972649` | WORKING_CAPITAL | 270 000 | 52.9% | D | 0 | HIGH_DEBT_BURDEN | REJECT_RISK |
| T50 | `002865213779` | INVESTMENT | 1 930 000 | 61.6% | D | 0 | DELINQUENCY_90_PLUS;NEGATIVE_EQUITY;HIGH_DEBT_BURDEN | REJECT_HARD_STOP |
