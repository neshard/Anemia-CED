# %% [markdown]
# # 06 Resampling ablations
# Runs for each setup in config ablation_setups. hypotheses.yaml ablation_tuning decides
# the hyperparameters: "retune" gives every variant its own Optuna search with the same
# trial budget (tuning.n_trials.xgb), "reuse" keeps the parameters tuned in 03 and 04.
#
# * CED: no resampling, SMOTENC, scale_pos_weight (computed per training fold)
# * Hb: no resampling, SMOGN, SMOGN without under-sampling of normal values
#
# For SMOGN the table reports the share of synthetic rows that came from
# SMOTER interpolation versus the Gaussian noise branch.
#
# Run from the project root:  python experiments/06_ablations.py

# %%
from __future__ import annotations

import pandas as pd
from _common import start

from src.models import CED_MODELS, CED_SCALE_POS_WEIGHT, HB_MODELS
from src.training import cv_summary_row, load_params, load_training_data, run_model, run_tag, save_run
from src.utils import save_table

config, hyp = start()
assert hyp is not None
RETUNE = hyp["ablation_tuning"] == "retune"
print(f"ablation_tuning = {hyp['ablation_tuning']}")
primary = config["models"]["primary_resampling"]
RESAMPLE_COLS = ["n_synthetic", "n_interpolation", "n_gaussian_noise", "n_removed_by_undersampling",
                 "share_interpolation", "share_gaussian_noise"]

hb_rows, ced_rows = [], []
for setup in config["ablation_setups"]:
    # --- Hb ablation
    data = load_training_data(config, "hb_model", setup)
    params = None if RETUNE else load_params(config, run_tag(setup, "hb_model", "xgboost", primary["hb_model"]))
    for variant in ("none", "smogn", "smogn_no_undersampling"):
        run = run_model(HB_MODELS["xgboost"], data, config, variant, params=params, variant=f"ablation_{variant}")
        save_run(run, data, config, save_model=False)
        row = cv_summary_row(run, data, ["mae", "rmse", "r2", "mae_hb_below_11", "sensitivity", "auc_roc"])
        for col in RESAMPLE_COLS:
            if col in run.cv.folds.columns:
                row[f"mean_{col}"] = round(float(pd.to_numeric(run.cv.folds[col]).mean()), 4)
        hb_rows.append(row)

    # --- CED ablation
    data = load_training_data(config, "ced_model", setup)
    params = None if RETUNE else load_params(config, run_tag(setup, "ced_model", "xgboost", primary["ced_model"]))
    for variant, spec, resampler in (
        ("none", CED_MODELS["xgboost"], "none"),
        ("smotenc", CED_MODELS["xgboost"], "smotenc"),
        ("scale_pos_weight", CED_SCALE_POS_WEIGHT, "none"),
    ):
        run = run_model(spec, data, config, resampler, params=params, variant=f"ablation_{variant}")
        run.name = "xgboost"
        save_run(run, data, config, save_model=False)
        row = cv_summary_row(run, data, ["auc_roc", "pr_auc", "brier", "recall", "specificity", "f1"])
        if "n_synthetic" in run.cv.folds.columns:
            row["mean_n_synthetic"] = round(float(pd.to_numeric(run.cv.folds["n_synthetic"]).mean()), 1)
        ced_rows.append(row)

# %% Tables
hb_table = pd.DataFrame(hb_rows)
save_table(hb_table, "c06_ablation_hb_resampling", config)
print(hb_table.T.to_string(header=False))
ced_table = pd.DataFrame(ced_rows)
save_table(ced_table, "c07_ablation_ced_resampling", config)
print(ced_table.T.to_string(header=False))
print("Note: Brier scores after SMOTENC reflect a shifted base rate. Calibration is examined in Phase D.")
