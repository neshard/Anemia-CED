# %% [markdown]
# # x04 Anthropometric baselines for CED (EXPLORATORY, POST HOC)
# Supervisor question of 2026-10-28, asked after all results were known: how much of the CED
# signal is plain body size? Three Logistic Regression models use only
#   (a) pre-pregnancy weight, (b) pre-pregnancy BMI, (c) pre-pregnancy weight + height,
# and are compared with the main XGBoost model (and the main Logistic Regression on the
# per-fold selected features). No verdict. Nothing in results/tables changes.
#
# Protocol identical to the main run: the same outer 5 x 5 folds (checked against the stored
# main out-of-fold predictions), inner 3-fold CV with 30 Optuna trials for C and the Youden
# threshold, standardisation and median imputation inside the folds, and the paired bootstrap
# of 08 for CED (2,000 resamples per repetition, seed 142), so the XGBoost and Logistic
# Regression numbers equal d04. BMI = weight / (height in m)^2 is a fixed formula per woman,
# so computing it before the split cannot leak; a missing weight or height gives a missing
# BMI that is imputed inside the folds.
#
# Run from the project root:  python scripts/exploratory/x04_ced_anthropometric_baselines.py

# %%
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cleaning import clean_dataset  # noqa: E402
from src.evaluation import corrected_resampled_ttest  # noqa: E402
from src.models import CED_MODELS  # noqa: E402
from src.nested_cv import build_data, fold_table, run_nested  # noqa: E402
from src.protocol_stats import ced_metrics, paired_bootstrap, paired_table, summary_table  # noqa: E402
from src.runs import main_run  # noqa: E402
from src.utils import assert_no_identifiers, load_config, load_hypotheses, set_global_seeds  # noqa: E402

LABEL = "exploratory, post hoc"
config = load_config()
set_global_seeds(int(config["project"]["random_seed"]))
hyp = load_hypotheses(config)
OUT = ROOT / "results" / "exploratory"
RUNS = OUT / "x04_anthropometric_runs"  # checkpoints and out-of-fold files, git-ignored
B, SEED = int(hyp["n_bootstrap"]), int(config["project"]["random_seed"]) + 100  # the CED seed of 08
VARIANTS = {"lr_weight": ("prepreg_weight",), "lr_bmi": ("prepreg_bmi",), "lr_weight_height": ("prepreg_weight", "height")}
METRICS = ["auc_roc", "pr_auc", "brier", "sensitivity", "specificity", "ppv", "npv"]


def save(df: pd.DataFrame, name: str) -> None:
    assert_no_identifiers(df, config)
    df.insert(0, "label", LABEL)
    path = OUT / f"{name}_exploratory_post_hoc.csv"
    df.to_csv(path, index=False)
    print(f"written {path.relative_to(ROOT)}")


# %% Data: the CED cohort of the main run plus BMI
cfg = copy.deepcopy(config)
cfg["features"]["numeric"] = [*cfg["features"]["numeric"], "prepreg_bmi"]  # median imputation inside the folds
cfg["paths"]["results_dir"] = str(RUNS)
data = build_data(cfg, clean_dataset(cfg).frame, "ced", analysis="posthoc_anthropometric")
data.X = data.X.assign(prepreg_bmi=data.X["prepreg_weight"] / (data.X["height"] / 100) ** 2)
print(f"CED cohort {len(data.y)}, BMI missing for {int(data.X['prepreg_bmi'].isna().sum())} women, "
      f"median BMI {data.X['prepreg_bmi'].median():.1f}")

# %% Runs (same folds and tuning budget as the main run)
main_oof = {m: pd.read_csv(ROOT / "results" / "oof" / f"main__ced__{m}__main.csv") for m in ("xgboost", "logistic_regression")}
frames = dict(main_oof)
fold_rows = []
for variant, feats in VARIANTS.items():
    run = main_run(CED_MODELS["logistic_regression"], cfg, feature_mode="fixed", fixed_features=feats, variant=variant)
    result = run_nested(run, data, cfg, label=f"posthoc_anthropometric__ced__{variant}")
    oof = result.oof
    ref = main_oof["xgboost"]
    same = (oof.sort_values(["repeat", "anon_id"])[["repeat", "fold", "anon_id"]].reset_index(drop=True)
            .equals(ref.sort_values(["repeat", "anon_id"])[["repeat", "fold", "anon_id"]].reset_index(drop=True)))
    if not same:
        raise SystemExit(f"{variant}: outer folds differ from the main run. Stop.")
    frames[variant] = oof
    oof.to_csv(RUNS / f"oof_{variant}.csv", index=False)
    fold_rows.append(fold_table(result))
    print(f"{variant}: 25 outer folds, identical to the main folds")

# %% Paired bootstrap (same women, same seed as 08) and corrected t-test
rb = paired_bootstrap(frames, ced_metrics, B, SEED)
names = ["xgboost", "logistic_regression", *VARIANTS]
summary = summary_table(rb, names, METRICS, 0.95)
summary["features"] = summary["model"].map({"xgboost": "per-fold selected (7)", "logistic_regression": "per-fold selected (7)",
                                            "lr_weight": "pre-pregnancy weight", "lr_bmi": "pre-pregnancy BMI",
                                            "lr_weight_height": "pre-pregnancy weight + height"})
save(summary.round(4), "x26_ced_anthropometric_baselines_metrics")

diffs = []
folds_main = pd.read_csv(ROOT / "results" / "folds" / "main__ced__xgboost__main.csv")


def fold_auc(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["repeat", "fold"]).apply(lambda b: ced_metrics(b)["auc_roc"], include_groups=False)


for metric, quantity in (("auc_roc", "difference"), ("pr_auc", "difference"), ("brier", "reduction")):
    table = paired_table(rb, "xgboost", ["logistic_regression", *VARIANTS], metric, quantity, 0.95)
    if metric == "auc_roc":
        ref = fold_auc(frames["xgboost"])
        tt = [corrected_resampled_ttest((ref - fold_auc(frames[v])).to_numpy(), folds_main["n_train"].mean(),
                                        folds_main["n_test"].mean()) for v in table["versus"]]
        table["t_corrected"] = [t["t"] for t in tt]
        table["p_corrected_approximate"] = [t["p"] for t in tt]
    diffs.append(table)
diffs = pd.concat(diffs, ignore_index=True)
diffs["note"] = ("XGBoost minus comparator for AUC and PR-AUC, comparator minus XGBoost for Brier: positive favours "
                 "XGBoost. Post hoc, no verdict.")
save(diffs.round(4), "x27_ced_anthropometric_baselines_paired_differences")

params = pd.concat(fold_rows, ignore_index=True)[["variant", "params", "threshold_youden"]]
print(summary[summary["metric"].isin(["auc_roc", "pr_auc", "brier"])][["model", "metric", "point", "ci_low", "ci_high"]]
      .round(3).to_string(index=False))
print(diffs[["versus", "metric", "quantity", "point", "ci_low", "ci_high"]].round(3).to_string(index=False))
print("median tuned C per variant:",
      params.assign(C=params["params"].map(lambda s: float(eval(s)["C"]))).groupby("variant")["C"].median().round(4).to_dict())
