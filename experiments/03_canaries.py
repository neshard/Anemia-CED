# %% [markdown]
# # 03 Leakage and sanity canaries (run before any headline result)
# 1. Label permutation: Hb and CED labels are shuffled and the full nested
#    pipeline runs on a reduced tuning budget. AUC must be near 0.50 and Hb R2
#    near or below 0. A clearly higher value means leakage and stops the run.
# 2. Fit spy: the real pipeline runs with a recorder as its first step. Every
#    fit (imputer, selector, encoder, scaler, model, and every tuning fit) must
#    see training rows only, and inner fits only their inner training split.
# Canary 3 (results that look too good) runs in 08_evaluation_and_report.py.
#
# Run from the project root:  python experiments/03_canaries.py

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import start

from src.cleaning import clean_dataset
from src.models import CED_MODELS, HB_MODELS
from src.nested_cv import build_data
from src.protocol_stats import ced_metrics, hb_metrics
from src.runs import main_run, run_and_save, save_result_table

config, _ = start("03_canaries")
can = config["canaries"]
df = clean_dataset(config).frame
rows: list[dict] = []

# %% Canary 1: label permutation
rng = np.random.default_rng(int(can["permutation_seed"]))
for target, spec, metric_fn in (("hb", HB_MODELS["xgboost"], hb_metrics), ("ced", CED_MODELS["xgboost"], ced_metrics)):
    data = build_data(config, df, target, analysis="canary_permutation")
    perm = rng.permutation(len(data.y))
    data.y = data.y[perm]
    if data.y_binary is not None:
        data.y_binary = data.y_binary[perm]
    run = main_run(spec, config, n_trials=int(can["n_trials"]), n_repeats=int(can["n_repeats"]), variant="permuted")
    result = run_and_save(run, data, config)
    m = metric_fn(result.oof)
    if target == "hb":
        ok = m["r2"] <= float(can["max_hb_r2"]) and m["auc_roc"] <= float(can["max_auc"])
        rows.append({"canary": "label_permutation_hb", "r2": m["r2"], "auc_roc": m["auc_roc"],
                     "limit": f"r2 <= {can['max_hb_r2']}, auc <= {can['max_auc']}", "passed": ok})
    else:
        ok = m["auc_roc"] <= float(can["max_auc"])
        rows.append({"canary": "label_permutation_ced", "auc_roc": m["auc_roc"],
                     "limit": f"auc <= {can['max_auc']}", "passed": ok})
    print(rows[-1])

# %% Canary 2: fit spy on the real pipeline (one repetition, reduced budget)
for target, spec in (("hb", HB_MODELS["xgboost"]), ("ced", CED_MODELS["xgboost"])):
    data = build_data(config, df, target, analysis="canary_spy")
    run = main_run(spec, config, n_trials=int(can["n_trials"]), n_repeats=1, variant="spy")
    result = run_and_save(run, data, config, spy=True)
    n_checked = sum(r["spy"]["n_fits_checked"] for r in result.folds)
    thresholds_ok = all(r["threshold_source"] == "inner_out_of_fold" for r in result.folds)
    rows.append({"canary": f"fit_spy_{target}", "fits_checked": n_checked, "thresholds_from_inner_only": thresholds_ok,
                 "passed": thresholds_ok and n_checked > 0})
    print(rows[-1])

table = pd.DataFrame(rows)
save_result_table(table, "c00_canaries", config)
print(table.to_string(index=False))
if not table["passed"].all():
    raise SystemExit("CANARY FAILED. Possible leakage. Stop and investigate before any headline result.")
print("All canaries passed.")
