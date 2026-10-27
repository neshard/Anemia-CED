# %% [markdown]
# # 04 Main nested run: hemoglobin (XGBRegressor), baselines mean-only and Ridge, exploratory Decision Tree and Random Forest
# Every model runs through the identical protocol: repeated stratified 5 x 5
# outer CV on all eligible records, inner 3-fold CV for Optuna tuning (same
# trial budget for every tuned model) and for the decision thresholds,
# per-fold feature selection, no resampling. Fold results are checkpointed,
# so a crash resumes instead of restarting. Metrics and verdicts are computed
# by 08_evaluation_and_report.py.
#
# Run from the project root:  python experiments/04_main_hb.py

# %%
from __future__ import annotations

from _common import start

from src.cleaning import clean_dataset
from src.models import HB_MODELS
from src.nested_cv import build_data
from src.runs import main_run, run_and_save

config, _ = start("04_main_hb")
data = build_data(config, clean_dataset(config).frame, "hb")
print(f"hb: {len(data.y)} eligible records, strata {data.meta['cell_counts']}")

# %% All models, identical folds and budget
for spec in HB_MODELS.values():
    result = run_and_save(main_run(spec, config), data, config)
    print(f"{spec.name}: {len(result.folds)} outer folds done, {len(result.oof)} out-of-fold predictions")
