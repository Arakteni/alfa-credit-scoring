# Воспроизведение DS-части (все данные синтетические). Запуск: make data | train | report | test | all
PY ?= python3
NB = $(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900

data:    ## генератор синтетики, побайтово воспроизводим
	$(PY) -m data.generate --seed 42

train:   ## признаки -> модели -> скор-карта и рейтинги (артефакты в risk-scoring/models)
	$(NB) notebooks/01_data_generation.ipynb
	$(NB) notebooks/02_features.ipynb
	$(NB) notebooks/03_modeling.ipynb
	$(NB) notebooks/04_scorecard_rating.ipynb

report:  ## метрики, графики, фикстуры, документация
	$(NB) notebooks/05_validation.ipynb
	$(NB) notebooks/06_scoring_engine.ipynb
	$(NB) notebooks/07_demo_cases_fixtures.ipynb
	$(NB) notebooks/08_documentation.ipynb

test:
	$(PY) -m pytest -q risk-scoring/tests

all: train report test
.PHONY: data train report test all
