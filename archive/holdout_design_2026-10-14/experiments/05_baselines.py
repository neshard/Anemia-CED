# %% [markdown]
# # 05 Baselines and direct anemia classifiers
# Runs once per feature setup in config run_setups.
# Every baseline uses the same pipeline (imputation, resampling), the same
# evaluation folds and the same tuning folds as the XGBoost models. Dummy
# models are not tuned.
#
# The direct classifiers predict anemia without going through Hb:
# * direct_xgb_classifier: binary Hb < 11
# * direct_xgb_who: the WHO classes present in the training data
# They test whether regression followed by thresholding is better than
# classifying directly.
#
# Run from the project root:  python experiments/05_baselines.py

# %%
from __future__ import annotations

import pandas as pd
from _common import start

from src.models import CED_MODELS, DIRECT_ANEMIA_MODELS, HB_MODELS
from src.training import cv_summary_row, load_training_data, run_model, save_run
from src.utils import save_table

config, _ = start()
REG_METRICS = ["mae", "rmse", "r2", "mae_hb_below_11", "sensitivity", "specificity", "auc_roc", "pr_auc"]
CLF_METRICS = ["auc_roc", "pr_auc", "brier", "accuracy", "precision", "recall", "f1", "specificity"]
WHO_METRICS = ["macro_f1", "macro_precision", "macro_recall", "ovr_auc", "auc_roc", "sensitivity", "specificity"]
rows = []

for setup in config["run_setups"]:
    # --- Hb baselines
    data = load_training_data(config, "hb_model", setup)
    resampler = config["models"]["primary_resampling"]["hb_model"]
    for name, spec in HB_MODELS.items():
        if name == "xgboost":
            continue
        run = run_model(spec, data, config, resampler)
        save_run(run, data, config)
        rows.append(cv_summary_row(run, data, REG_METRICS))
        print(f"[{setup}] hb_model {name}: CV MAE {run.cv.mean('mae'):.3f}")

    # --- CED baselines
    data = load_training_data(config, "ced_model", setup)
    resampler = config["models"]["primary_resampling"]["ced_model"]
    for name, spec in CED_MODELS.items():
        if name == "xgboost":
            continue
        run = run_model(spec, data, config, resampler)
        save_run(run, data, config)
        rows.append(cv_summary_row(run, data, CLF_METRICS))
        print(f"[{setup}] ced_model {name}: CV AUC {run.cv.mean('auc_roc'):.3f}")

    # --- Direct anemia classifiers
    resampler = config["models"]["direct_comparator_resampling"]
    for target, name, metrics in (("anemia_binary", "direct_xgb_classifier", CLF_METRICS),
                                  ("anemia_who", "direct_xgb_who", WHO_METRICS)):
        data = load_training_data(config, target, setup)
        run = run_model(DIRECT_ANEMIA_MODELS[name], data, config, resampler)
        save_run(run, data, config)
        rows.append(cv_summary_row(run, data, metrics))
        print(f"[{setup}] {target} {name}: CV AUC {run.cv.mean('auc_roc'):.3f} (classes {data.class_names})")

# %% Summary table
summary = pd.DataFrame(rows)
save_table(summary, "c05_cv_summary_baselines_and_direct", config)
print(summary.to_string(index=False))
