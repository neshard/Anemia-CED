# %% [markdown]
# # 09 Final models (not used for any reported metric)
# The algorithm per target comes from artifacts/model_selection.json (written by 14 with the
# locked rule in config/model_selection.yaml). Until that file exists the default of
# config/dashboard.yaml (XGBoost) is used.
# The dashboard list of each model is the set of features selected in at least half of the
# 25 outer folds of that model's main run. The final model is tuned on all eligible records
# (396 Hb, 410 CED) by the same procedure (inner 3-fold CV, same trial budget) using that
# fixed list. Thresholds and the Hb interval half-width come from the inner-fold predictions
# of that tuning, never from predictions of the final model on its own training data.
# Reported metrics come only from the outer CV with per-fold selection.
#
# Run from the project root:  python experiments/09_final_models.py

# %%
from __future__ import annotations

import json

import joblib
import numpy as np
import yaml
from _common import ROOT, start

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
from src.postmodel.explain import ExplainerNotSupportedError, additivity_error, tree_explainer
from src.runs import load_folds, main_run
from src.utils import library_versions, resolve_path

config, _ = start("09_final_models")
df = clean_dataset(config).frame
fs, th = config["feature_selection"], config["thresholds"]
artifacts = resolve_path(config, "artifacts_dir")
artifacts.mkdir(parents=True, exist_ok=True)
dashboard_cfg = yaml.safe_load((ROOT / "config" / "dashboard.yaml").read_text(encoding="utf-8"))
selection_path = artifacts / "model_selection.json"
selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else None
if selection is None:
    print(f"No {selection_path.name} yet: using the default algorithm '{dashboard_cfg['default_algorithm']}'.")
summary = {}

# %% Fit and save
for target, models in (("hb", HB_MODELS), ("ced", CED_MODELS)):
    algorithm = selection[target]["algorithm"] if selection else dashboard_cfg["default_algorithm"]
    spec = models[algorithm]
    folds = load_folds(config, f"main__{target}__{algorithm}__main")
    freq = selection_frequency([s.split("|") for s in folds["selected_features"]], config["features"]["candidate_pool"],
                               float(fs["dashboard_min_fold_share"]))
    features = freq.loc[freq["in_dashboard_list"], "feature"].tolist()
    data = build_data(config, df, target, analysis="final")
    run = main_run(spec, config, feature_mode="fixed", fixed_features=tuple(features), variant="final")
    all_rows = np.arange(len(data.y))
    params, inner_pred, _ = tune_inner(run, data, config, all_rows, int(config["project"]["random_seed"]))
    pipe = make_pipeline_factory(run, config)(params).fit(data.X, data.y)
    record = {"model": f"{target}_{algorithm}", "features": features, "params": params, "n_records": int(len(data.y)),
              "note": "Final model on all eligible records with a fixed feature list. Not used for any reported metric."}
    if target == "hb":
        record["hb_cutoff"] = cutoff_flagging_share(inner_pred, data.y_binary.astype(bool), float(th["hb_flag_share_of_cases"]))
        record["interval_half_width"] = conformal_half_width(np.abs(data.y - inner_pred), float(config["intervals"]["nominal"]))
    else:
        record["threshold_youden"] = youden_threshold(inner_pred, data.y.astype(bool))
        for t in th["ced_extra_sensitivity_targets"]:
            record[f"threshold_sens_{int(round(t * 100))}"] = sensitivity_threshold(inner_pred, data.y.astype(bool), t)
    try:
        add_err = additivity_error(pipe, data.X, tree_explainer(pipe))
        explainer_type = "shap.TreeExplainer"
    except ExplainerNotSupportedError:
        add_err, explainer_type = float("nan"), "none"
    tol = float(dashboard_cfg["additivity_tolerance"])
    record.update({
        "algorithm": algorithm,
        "artifact": f"artifacts/final_{target}_{algorithm}.joblib",
        "explainer": f"artifacts/explainer_{target}.joblib",
        "explainer_type": explainer_type,
        "shap_additivity_max_error": add_err,
        "shap_additivity_tolerance": tol,
        "shap_additivity_passed": bool(explainer_type != "none" and add_err <= tol),
        "seed": int(config["project"]["random_seed"]),
        "library_versions": library_versions(),
        "selection": selection[target] if selection else {"algorithm": algorithm, "reason": "default before selection"},
    })
    if target == "hb":
        gate = selection["hb"].get("explanation_gate", {}) if selection else {}
        record["hb_explanation_enabled"] = bool(gate.get("hb_explanation_enabled", False))
    joblib.dump(pipe, ROOT / record["artifact"])
    summary[target] = record
    print(json.dumps(record, indent=2, default=str))
    if not record["shap_additivity_passed"]:
        raise SystemExit(f"{target} final model ({algorithm}) failed the SHAP check. The dashboard would refuse it.")

(artifacts / "final_models.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
print(f"Saved to {artifacts}")
