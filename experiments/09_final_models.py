# %% [markdown]
# # 09 Final models (not used for any reported metric)
# The dashboard list of each model is the set of features selected in at
# least half of the 25 outer folds of the main run. The final model is tuned
# on all eligible records (396 Hb, 410 CED) by the same procedure (inner
# 3-fold CV, same trial budget) using that fixed list. Thresholds and the Hb
# interval half-width come from the inner-fold predictions of that tuning,
# never from predictions of the final model on its own training data.
# Reported metrics come only from the outer CV with per-fold selection.
#
# Run from the project root:  python experiments/09_final_models.py

# %%
from __future__ import annotations

import json

import joblib
import numpy as np
from _common import start

from src.cleaning import clean_dataset
from src.feature_selection import selection_frequency
from src.models import CED_MODELS, HB_MODELS
from src.nested_cv import (
    build_data,
    conformal_half_width,
    cutoff_flagging_share,
    make_pipeline_factory,
    sensitivity_threshold,
    tune_inner,
    youden_threshold,
)
from src.runs import load_folds, main_run
from src.utils import resolve_path

config, _ = start("09_final_models")
df = clean_dataset(config).frame
fs, th = config["feature_selection"], config["thresholds"]
artifacts = resolve_path(config, "artifacts_dir")
artifacts.mkdir(parents=True, exist_ok=True)
summary = {}

# %% Fit and save
for target, spec in (("hb", HB_MODELS["xgboost"]), ("ced", CED_MODELS["xgboost"])):
    folds = load_folds(config, f"main__{target}__xgboost__main")
    freq = selection_frequency([s.split("|") for s in folds["selected_features"]], config["features"]["candidate_pool"],
                               float(fs["dashboard_min_fold_share"]))
    features = freq.loc[freq["in_dashboard_list"], "feature"].tolist()
    data = build_data(config, df, target, analysis="final")
    run = main_run(spec, config, feature_mode="fixed", fixed_features=tuple(features), variant="final")
    all_rows = np.arange(len(data.y))
    params, inner_pred, _ = tune_inner(run, data, config, all_rows, int(config["project"]["random_seed"]))
    pipe = make_pipeline_factory(run, config)(params).fit(data.X, data.y)
    record = {"model": f"{target}_xgboost", "features": features, "params": params, "n_records": int(len(data.y)),
              "note": "Final model on all eligible records with a fixed feature list. Not used for any reported metric."}
    if target == "hb":
        record["hb_cutoff"] = cutoff_flagging_share(inner_pred, data.y_binary.astype(bool), float(th["hb_flag_share_of_cases"]))
        record["interval_half_width"] = conformal_half_width(np.abs(data.y - inner_pred), float(config["intervals"]["nominal"]))
    else:
        record["threshold_youden"] = youden_threshold(inner_pred, data.y.astype(bool))
        for t in th["ced_extra_sensitivity_targets"]:
            record[f"threshold_sens_{int(round(t * 100))}"] = sensitivity_threshold(inner_pred, data.y.astype(bool), t)
    joblib.dump(pipe, artifacts / f"final_{target}_xgboost.joblib")
    summary[target] = record
    print(json.dumps(record, indent=2, default=str))

(artifacts / "final_models.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
print(f"Saved to {artifacts}")
