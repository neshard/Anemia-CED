# Anemia-CED
Evaluating XGBoost with SHAP for Questionnaire-Based Risk Screening of Anemia and Chronic Energy Deficiency in Pregnant Women: A Case Study in North Lombok

Undergraduate thesis pipeline (Informatics, Telkom University). Two targets are estimated from questionnaire answers only:

* **Hemoglobin** (regression), then flagged as anemia below 11.0 g/dL.
* **Chronic energy deficiency (CED) risk** (binary classification, recorded MUAC of 23 cm or less).

Hemoglobin and MUAC never enter a feature matrix. Imputation, feature selection, tuning and decision thresholds are fitted inside the folds of a nested repeated cross-validation. XGBoost is evaluated against pre-specified baselines and compared with GBM, LightGBM, CatBoost and AdaBoost under an identical protocol. SHAP (TreeExplainer) explains the models, and a Streamlit prototype shows the result to a health cadre.

The hypothesis verdicts are summarised in [results/report.md](results/report.md), and every reported number is in the aggregate tables under [results/tables](results/tables) and [results/exploratory](results/exploratory). All results are internal validation on one regional dataset of fewer than 420 women.

## Data availability

The real questionnaire data are **not** in this repository. They describe identifiable pregnant women and are kept by the researcher under the study's ethics conditions. Only aggregate results are published here.

[data/synthetic/synthetic_dataset.csv](data/synthetic/synthetic_dataset.csv) contains 400 **fictitious** records in the exact format of the Google Forms export, so that the code and the tests run without the real data. Numbers produced from it are meaningless. It is produced by [scripts/make_synthetic_dataset.py](scripts/make_synthetic_dataset.py) (`python scripts/make_synthetic_dataset.py`, defaults reproduce the file byte for byte; `--n`, `--seed` and `--out` make other sizes).

## Setup

Python 3.11 on Windows, macOS or Linux.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

All library versions are pinned in [requirements.txt](requirements.txt) (for example numpy 1.26.4, scikit-learn 1.5.2, xgboost 2.1.3, lightgbm 4.5.0, catboost 1.2.7, shap 0.46.0, optuna 4.1.0). pyarrow is pinned to 18.1.0 because later versions refuse NumPy 1.26.

## Check the code without the real data

```bash
python -m pytest                                        # about 300 tests, about 2 minutes
python scripts/run_smoke_test.py <empty output folder>  # scripts 00, 02 to 09 and 12 to 14 on the synthetic file
```

The test suite covers the leakage guard (Hb and MUAC refused at fit and prediction time), a fit spy proving no fit sees evaluation rows, identical folds and feature sets across all models, two-run reproducibility, the SHAP additivity check, the model selection rule and its edge cases, and the dashboard (including that it refuses a model without a valid explainer). Tests that need the real model files are skipped when `artifacts/` is empty.

The smoke test writes everything into the folder you give it and never touches `results/` or `artifacts/`.

## Run the full analysis (real data)

Put the export at `data/raw/dataset.csv` (path, separator, encoding and expected row counts are in [config/config.yaml](config/config.yaml)). Run every script from the project root, in this order:

| Step | Script | What it does |
|---|---|---|
| 1 | `experiments/00_prepare_data.py` | Cleaning, sample sizes (asserted: 396 Hb, 410 CED), stratification keys |
| 2 | `experiments/01_eda.py` | Aggregate description of the cleaned data |
| 3 | `experiments/02_feature_selection_exploratory.py` | One exploratory feature ranking (never used to build a model) |
| 4 | `experiments/03_canaries.py` | Leakage canaries (label permutation, fit spy) for all boosting models. Stops on failure |
| 5 | `experiments/04_main_hb.py` | Nested 5 x 5 CV for every Hb model (about 10 minutes with 8 parallel folds) |
| 6 | `experiments/05_main_ced.py` | The same for CED (about 20 minutes) |
| 7 | `experiments/06_exploratory.py` | Exploratory variants (SMOGN, SMOTE-NC, class weighting, direct anemia classifier) |
| 8 | `experiments/07_sensitivity.py` | Sensitivity analyses (XGBoost and baselines) |
| 9 | `experiments/08_evaluation_and_report.py` | Hypothesis verdicts, metrics with bootstrap CIs, `results/report.md` (about 35 minutes) |
| 10 | `experiments/12_boosting_comparison.py` | XGBoost against GBM, LightGBM, CatBoost and AdaBoost, figures |
| 11 | `experiments/13_shap_boosting.py` | SHAP for every boosting model, eligibility, stability, agreement between algorithms |
| 12 | `experiments/14_select_final_model.py` | Applies the locked selection rule, writes `artifacts/model_selection.json` |
| 13 | `experiments/09_final_models.py` | Refits the selected models on all eligible records for the dashboard |
| 14 | `experiments/10_calibration_and_intervals.py` | CED calibration and Hb interval coverage for the selected models |
| 15 | `experiments/11_shap_analysis.py` | SHAP of the selected models and the dashboard explainers |

Every outer fold is checkpointed under a hash of the configuration, the hypotheses, the raw data and the source code, so an interrupted run resumes where it stopped and a changed setup never reuses stale results. Parallelism is set by `runtime.n_jobs` and does not change any number.

## Dashboard

After steps 13 to 15 the trained models are in `artifacts/` (not in this repository, because they are produced from the real data).

```bash
streamlit run app/Skrining.py
```

Open http://localhost:8501. The app is served on localhost only. Sessions are stored in a local SQLite file (`data/app/sessions.sqlite`) with anonymous codes, never names. The models to load are read from `artifacts/final_models.json` and [config/dashboard.yaml](config/dashboard.yaml), never hard-coded.

## Pre-specified protocol and deviations

* [config/hypotheses.yaml](config/hypotheses.yaml): the four hypotheses of the proposal (Table 1.1), locked on 2026-10-17. All four were not supported.
* [config/model_selection.yaml](config/model_selection.yaml): the rule that picks the dashboard model per target. **Version 2 (2026-10-28) is a deviation decided after the results were known**: version 1 preferred XGBoost; version 2 selects the eligible boosting model with the best point estimate (GBM for Hb, XGBoost for CED). Both selections are reported side by side.
* [config/config.yaml](config/config.yaml): every path, seed, threshold rule and Optuna search space.
* [CHANGELOG.md](CHANGELOG.md): every change to criteria, models or data handling with its date and reason, including the version 2 deviation and what was known when each rule was written.

## Repository layout

```
app/                 Streamlit dashboard (presentation layer)
config/              configuration, hypotheses, selection rule, dashboard and narrative settings
data/synthetic/      fictitious data in the raw export format
experiments/         pipeline scripts 00 to 14
scripts/             smoke test, de-identified copy, exploratory audits
src/                 engine: cleaning, labels, splits, preprocessing, feature selection, models,
                     nested CV, statistics, hypotheses, model selection; src/postmodel/ for SHAP,
                     calibration, inference and SQLite
tests/               automated tests
results/tables/      aggregate result tables (c canaries, d evaluation, m boosting comparison)
results/exploratory/ pre-specified exploratory tables (x) and SHAP figures
results/figures/     boosting comparison figures
reports/figures/     descriptive figures
```

Result table prefixes: `c00` canaries, `d01` to `d13` hypotheses, metrics and sensitivity analyses, `m00` to `m11` boosting comparison, SHAP and model selection, `x01` to `x22` exploratory analyses.
