# %% [markdown]
# # x02 Near-duplicate sensitivity analysis (EXPLORATORY, POST HOC)
# Question: do the conclusions change if records that are probably the same woman
# entered twice are removed? The locked verdicts of P1, P2, S1 and S2 stay with the
# main analysis of 2026-10-18. This analysis gives no verdict.
#
# Duplicate rules, fixed before running (audit finding 11):
# * strict: two records identical on all 21 compared fields
#   (12 core fields: age, height, pre-pregnancy weight, ANC visits, number of children,
#   children under five, education, income, egg frequency, food taboo, Hb, MUAC,
#   plus the 9 HFIAS items).
# * loose: two records that differ in at most 1 of those 21 fields.
# In every cluster of duplicates the first record is kept and the others are removed.
#
# Protocol: identical to the main run (nested 5 x 5 CV, inner 3-fold, 30 Optuna trials,
# per-fold selection, thresholds from inner folds, 2,000 bootstrap resamples per
# repetition). Models: XGBoost with the confirmatory baselines (Hb: mean-only Dummy and
# Ridge, CED: class-proportion Dummy and Logistic Regression). Folds differ from the main
# run because the sample differs.
#
# Data: data/raw/dataset_deidentified.csv (cleaned data identical to the original).
# Every output, checkpoint and log goes to results/exploratory. Nothing in results/tables,
# src/ or experiments/ is changed.
#
# Run from the project root:  python scripts/exploratory/x02_dedup_sensitivity.py

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
from src.models import CED_MODELS, HB_MODELS  # noqa: E402
from src.nested_cv import build_data  # noqa: E402
from src.protocol_stats import ced_metrics, hb_metrics, paired_bootstrap, paired_table, summary_table  # noqa: E402
from src.runs import main_run, run_and_save  # noqa: E402
from src.utils import load_config  # noqa: E402

LABEL = "exploratory, post hoc"
OUT = ROOT / "results" / "exploratory"
base = load_config()
config = copy.deepcopy(base)
config["paths"]["raw_data"] = str(ROOT / "data" / "raw" / "dataset_deidentified.csv")
config["paths"]["results_dir"] = str(OUT / "x02_dedup_runs")
config["data"]["expected_counts"] = None  # counts are asserted below for the full data instead
seed = int(config["project"]["random_seed"])
B = 2000
CORE = ["age", "height", "prepreg_weight", "anc_visits", "n_children", "n_under5", "education_mother", "income",
        "egg_freq", "food_taboo", "hb", "muac"]
HFIAS = [f"hfias_q{i}" for i in range(1, 10)]
written: list[str] = []


def save(df: pd.DataFrame, name: str) -> None:
    if "anon_id" in df.columns or "respondent_code" in df.columns:
        raise ValueError("Identifier column in an exploratory table")
    path = OUT / f"{name}_exploratory_post_hoc.csv"
    df.insert(0, "label", LABEL)
    df.to_csv(path, index=False)
    written.append(str(path.relative_to(ROOT)))


# %% Rebuild the deduplicated raw frame exactly as src/cleaning.load_raw does, so row i is anon_id P{i+1:04d}
raw = pd.read_csv(config["paths"]["raw_data"], sep=config["data"]["csv_sep"], encoding=config["data"]["csv_encoding"],
                  dtype=str)
mapped = {k: v for k, v in config["columns"].items() if k != "respondent_code"}
frame = raw[list(mapped.values())].copy()
frame.columns = list(mapped)
frame = frame.loc[~frame.duplicated(keep="first")].reset_index(drop=True)
anon = [f"P{i:04d}" for i in range(1, len(frame) + 1)]

full = clean_dataset(config).frame
if list(full["anon_id"]) != [a for a in anon if a in set(full["anon_id"])]:
    raise SystemExit("Anonymous IDs do not line up with the deduplicated raw frame. Stopping.")
assert len(frame) == 419, "expected 419 records after the protocol deduplication"

values = frame[CORE + HFIAS].astype(str).apply(lambda s: s.str.strip()).to_numpy()


def clusters(max_differences: int) -> list[list[int]]:
    """Groups of rows linked by pairs that differ in at most ``max_differences`` of the 21 fields."""
    parent = list(range(len(values)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(values)):
        diff = (values[i + 1:] != values[i]).sum(axis=1)
        for off in np.where(diff <= max_differences)[0]:
            a, b = find(i), find(i + 1 + int(off))
            parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(len(values)):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


rules = {"strict": 0, "loose": 1}
removed: dict[str, set[str]] = {}
rule_rows = []
for rule, k in rules.items():
    groups = clusters(k)
    drop = {anon[i] for g in groups for i in g[1:]}
    removed[rule] = drop
    rule_rows.append({"rule": rule, "max_differing_fields_of_21": k, "duplicate_clusters": len(groups),
                      "records_removed": len(drop),
                      "removed_from_hb_model": int(full[full["anon_id"].isin(drop)]["hb_target"].notna().sum()),
                      "removed_from_ced_model": int(full[full["anon_id"].isin(drop)]["ced"].notna().sum())})
save(pd.DataFrame(rule_rows), "x08_dedup_rules")
print(pd.DataFrame(rule_rows).to_string(index=False))

# %% Rerun the protocol for each rule
HB_SET = ["xgboost", "dummy_mean", "ridge"]
CED_SET = ["xgboost", "dummy_prior", "logistic_regression"]
metric_rows, diff_rows = [], []
for rule, drop in removed.items():
    df = full[~full["anon_id"].isin(drop)].reset_index(drop=True)
    for target, names, specs, fn, bseed in (("hb", HB_SET, HB_MODELS, hb_metrics, seed),
                                            ("ced", CED_SET, CED_MODELS, ced_metrics, seed + 100)):
        data = build_data(config, df, target, analysis=f"dedup_{rule}")
        frames = {}
        for name in names:
            result = run_and_save(main_run(specs[name], config), data, config)
            frames[name] = result.oof
        rb = paired_bootstrap(frames, fn, B, bseed)
        metrics = (["mae", "rmse", "r2", "sensitivity", "specificity", "auc_roc", "referral_rate"] if target == "hb"
                   else ["auc_roc", "pr_auc", "brier", "sensitivity", "specificity", "referral_rate"])
        s = summary_table(rb, names, metrics, 0.95)
        s.insert(0, "n", len(data.y))
        s.insert(0, "target", target)
        s.insert(0, "rule", rule)
        metric_rows.append(s)
        if target == "hb":
            d = paired_table(rb, "xgboost", ["dummy_mean", "ridge"], "mae", "reduction", 0.95)
        else:
            d = paired_table(rb, "xgboost", ["dummy_prior", "logistic_regression"], "auc_roc", "difference", 0.95)
        d.insert(0, "target", target)
        d.insert(0, "rule", rule)
        diff_rows.append(d)

# %% Compare with the locked main analysis (read from results/tables, never modified)
main_hb = pd.read_csv(ROOT / "results" / "tables" / "d03_hb_metrics.csv")
main_ced = pd.read_csv(ROOT / "results" / "tables" / "d04_ced_metrics.csv")
main_cmp = pd.read_csv(ROOT / "results" / "tables" / "d06_baseline_comparisons.csv")
main = pd.concat([main_hb.assign(target="hb"), main_ced.assign(target="ced")])
main = main[main["model"].isin(HB_SET + CED_SET)][["target", "model", "metric", "point", "ci_low", "ci_high"]]
main.insert(0, "rule", "main (no extra removal, locked)")
metrics_all = pd.concat([main, *[m[["rule", "target", "model", "metric", "point", "ci_low", "ci_high"]] for m in metric_rows]],
                        ignore_index=True)
save(metrics_all.round(4), "x09_dedup_metrics_vs_main")
mc = main_cmp[main_cmp["versus"].isin(["dummy_mean", "ridge", "dummy_prior", "logistic_regression"])
              & main_cmp["metric"].isin(["mae", "auc_roc"])
              & ~((main_cmp["metric"] == "auc_roc") & main_cmp["versus"].isin(["dummy_mean", "ridge"]))]
mc = mc.assign(rule="main (no extra removal, locked)", target=np.where(mc["metric"] == "mae", "hb", "ced"))
diffs = pd.concat([mc[["rule", "target", "versus", "metric", "quantity", "point", "ci_low", "ci_high"]],
                   *[d[["rule", "target", "versus", "metric", "quantity", "point", "ci_low", "ci_high"]] for d in diff_rows]],
                  ignore_index=True)
save(diffs.round(4), "x10_dedup_paired_differences_vs_main")
print(diffs.round(3).to_string(index=False))
print("Written:")
for w in written:
    print(f"  {w}")
