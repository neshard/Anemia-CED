# %% [markdown]
# # 04 Train the CED classifier (XGBClassifier)
# Runs once per feature setup in config run_setups. Tunes on separate tuning folds, evaluates with 5 x 5 stratified repeated CV
# and GroupKFold by health centre, then refits on the whole hold-out training
# set and saves the pipeline to artifacts/. The hold-out test set is not used.
#
# Run from the project root:  python experiments/04_train_ced.py

# %%
from __future__ import annotations

import pandas as pd
from _common import start

from src.models import CED_MODELS
from src.training import cv_summary_row, load_training_data, run_model, save_run
from src.utils import save_table

config, _ = start()
TARGET = "ced_model"
MODELS = CED_MODELS
SUMMARY_NAME = "c04_cv_summary_ced_model_xgboost"
METRICS = ["auc_roc", "pr_auc", "brier", "accuracy", "precision", "recall", "f1", "specificity"]

# %% Train every configured feature setup (primary first)
rows = []
for setup in config["run_setups"]:
    data = load_training_data(config, TARGET, setup)
    print(f"[{setup}] {TARGET}: {len(data.y)} training rows, input columns {list(data.X.columns)}")
    resampler = config["models"]["primary_resampling"][TARGET]
    run = run_model(MODELS["xgboost"], data, config, resampler)
    save_run(run, data, config)
    rows.append(cv_summary_row(run, data, METRICS))
    print(f"[{setup}] tuned parameters: {run.params}")

# %% Summary
summary = pd.DataFrame(rows)
save_table(summary, SUMMARY_NAME, config)
print(summary.T.to_string(header=False))
print("Dummy comparison and statistical tests are in experiments/07_evaluation_and_stats.py")
