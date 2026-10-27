# %% [markdown]
# # 08 Evaluation, verdicts and report
# Reads only the out-of-fold predictions and fold tables written by 03 to 07.
# Nothing is fitted here. Criteria come from config/hypotheses.yaml and are
# not changed after seeing results.
#
# Run from the project root:  python experiments/08_evaluation_and_report.py

# %%
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from _common import start
from sklearn.metrics import roc_auc_score

from src.cleaning import clean_dataset
from src.evaluation import corrected_resampled_ttest
from src.feature_selection import selection_frequency
from src.hypotheses import decide
from src.nested_cv import build_data
from src.protocol_stats import (
    ced_metrics,
    confusion,
    hb_metrics,
    paired_bootstrap,
    paired_table,
    summary_table,
    three_class_metrics,
)
from src.runs import load_folds, load_oof, results_dir, save_result_table
from src.utils import resolve_path

config, hyp = start("08_evaluation_and_report")
assert hyp is not None
ev = config["evaluation"]
B, LEVEL = int(hyp["n_bootstrap"]), float(ev["ci_level"])
SEED = int(config["project"]["random_seed"])
SENS = [int(round(t * 100)) for t in config["thresholds"]["ced_extra_sensitivity_targets"]]
HB_ALL = ["xgboost", "dummy_mean", "ridge", "decision_tree", "random_forest"]
CED_ALL = ["xgboost", "dummy_prior", "logistic_regression", "decision_tree", "random_forest"]
HB_METRICS = ["mae", "rmse", "r2", "sensitivity", "specificity", "ppv", "npv", "auc_roc", "pr_auc", "referral_rate",
              "interval_coverage", "interval_mean_width"]
CED_METRICS = ["auc_roc", "pr_auc", "brier", "sensitivity", "specificity", "ppv", "npv", "accuracy", "f1", "referral_rate"]
written: list[str] = []
warnings: list[str] = []


def save(df: pd.DataFrame, name: str) -> None:
    written.append(str(save_result_table(df, name, config).relative_to(resolve_path(config, "results_dir").parent)))


def ced_all_thresholds(df: pd.DataFrame) -> dict[str, float]:
    out = ced_metrics(df, "flag")
    for s in SENS:
        out.update({f"sens{s}_{k}": v for k, v in ced_metrics(df, f"flag_sens_{s}").items()
                    if k in ("sensitivity", "specificity", "ppv", "npv", "referral_rate")})
    return out


def oofs(analysis: str, target: str, models: list[str], variant: str = "main") -> dict[str, pd.DataFrame]:
    return {m: load_oof(config, f"{analysis}__{target}__{m}__{variant}") for m in models}


def rep_point_mean(df: pd.DataFrame, fn, metric: str) -> float:
    return float(np.mean([fn(block)[metric] for _, block in df.groupby("repeat")]))


# %% Canary 3: results that look too good are investigated before any report
hb_main = oofs("main", "hb", HB_ALL)
ced_main = oofs("main", "ced", CED_ALL)
too = ev["too_good"]
ced_auc = rep_point_mean(ced_main["xgboost"], ced_metrics, "auc_roc")
hb_r2 = rep_point_mean(hb_main["xgboost"], hb_metrics, "r2")
canary3 = pd.DataFrame([{"canary": "too_good_ced_auc", "value": ced_auc, "limit": too["ced_auc"], "passed": ced_auc <= too["ced_auc"]},
                        {"canary": "too_good_hb_r2", "value": hb_r2, "limit": too["hb_r2"], "passed": hb_r2 <= too["hb_r2"]}])
save(canary3, "c00b_canary_too_good")
print(canary3.to_string(index=False))
if not canary3["passed"].all():
    raise SystemExit("A headline result exceeds the too-good limit. Treat as possible leakage. Investigate and report before continuing.")

# %% Paired bootstrap per repetition (same resampled women for every model)
rb_hb = paired_bootstrap(hb_main, hb_metrics, B, SEED)
rb_ced = paired_bootstrap(ced_main, ced_all_thresholds, B, SEED + 100)

# %% Hypothesis verdicts (Holm across P1 and P2 by CI level)
verdicts, components = decide(hyp, {"hb": rb_hb, "ced": rb_ced})
save(verdicts, "d01_hypothesis_verdicts")
save(components, "d02_hypothesis_components")
print(verdicts[["hypothesis", "role", "verdict", "ci_level_used", "holm_order", "hypothesis_p_value"]].to_string(index=False))
print(components[["hypothesis", "component", "criterion", "point", "ci_low", "ci_high", "ci_level", "p_one_sided", "outcome"]]
      .round(4).to_string(index=False))

# %% Headline metrics with CIs, ranges and confusion matrices
hb_summary = summary_table(rb_hb, HB_ALL, HB_METRICS, LEVEL)
ced_metric_list = CED_METRICS + [f"sens{s}_{k}" for s in SENS for k in ("sensitivity", "specificity", "ppv", "npv", "referral_rate")]
ced_summary = summary_table(rb_ced, CED_ALL, ced_metric_list, LEVEL)
save(hb_summary, "d03_hb_metrics")
save(ced_summary, "d04_ced_metrics")
cm_rows = []
for target, frames, truth_col, flags in (("hb", hb_main, "y_binary", ["flag"]),
                                         ("ced", ced_main, "y_true", ["flag"] + [f"flag_sens_{s}" for s in SENS])):
    for model, df in frames.items():
        for flag in flags:
            per_rep = [confusion(b[truth_col].to_numpy(), b[flag].to_numpy()) for _, b in df.groupby("repeat")]
            cm_rows.append({"target": target, "model": model, "threshold": flag,
                            **{k: float(np.mean([c[k] for c in per_rep])) for k in ("tp", "fp", "tn", "fn")},
                            "n": int(len(df) / df["repeat"].nunique())})
cm = pd.DataFrame(cm_rows)
cm["referral_burden_per_100"] = (100 * (cm["tp"] + cm["fp"]) / cm["n"]).round(1)
cm["missed_cases_per_100"] = (100 * cm["fn"] / cm["n"]).round(1)
save(cm, "d05_confusion_matrices_mean_over_repetitions")

# %% Baseline comparisons (paired bootstrap decides, corrected t only approximate)
cmp_rows = [
    paired_table(rb_hb, "xgboost", ["dummy_mean", "ridge"], "mae", "reduction", LEVEL).assign(role="confirmatory baseline"),
    paired_table(rb_hb, "xgboost", ["decision_tree", "random_forest"], "mae", "reduction", LEVEL).assign(role="exploratory"),
    paired_table(rb_hb, "xgboost", ["dummy_mean", "ridge"], "auc_roc", "difference", LEVEL).assign(role="confirmatory baseline"),
    paired_table(rb_ced, "xgboost", ["dummy_prior", "logistic_regression"], "auc_roc", "difference", LEVEL).assign(role="confirmatory baseline"),
    paired_table(rb_ced, "xgboost", ["decision_tree", "random_forest"], "auc_roc", "difference", LEVEL).assign(role="exploratory"),
    paired_table(rb_ced, "xgboost", ["dummy_prior", "logistic_regression"], "brier", "difference", LEVEL).assign(role="confirmatory baseline"),
]
comparisons = pd.concat(cmp_rows, ignore_index=True)


def fold_metric(df: pd.DataFrame, fn, metric: str) -> pd.Series:
    return df.groupby(["repeat", "fold"]).apply(lambda b: fn(b)[metric], include_groups=False)


approx = []
for target, frames, fn, metric, sign in (("hb", hb_main, hb_metrics, "mae", -1), ("ced", ced_main, ced_metrics, "auc_roc", 1)):
    ref = fold_metric(frames["xgboost"], fn, metric)
    folds = load_folds(config, f"main__{target}__xgboost__main")
    for other in [m for m in frames if m != "xgboost"]:
        diff = sign * (ref - fold_metric(frames[other], fn, metric))
        t = corrected_resampled_ttest(diff.to_numpy(), folds["n_train"].mean(), folds["n_test"].mean())
        approx.append({"target": target, "metric": metric, "versus": other, "mean_fold_improvement": float(diff.mean()),
                       "t_corrected": t["t"], "p_corrected_approximate": t["p"]})
comparisons = comparisons.merge(pd.DataFrame(approx).rename(columns={"versus": "versus_t"}), how="left",
                                left_on=["versus", "metric"], right_on=["versus_t", "metric"]).drop(columns=["versus_t"])
save(comparisons, "d06_baseline_comparisons")

hb_vs_dummy = comparisons[(comparisons["versus"] == "dummy_mean") & (comparisons["metric"] == "mae")].iloc[0]
ced_vs_dummy = comparisons[(comparisons["versus"] == "dummy_prior") & (comparisons["metric"] == "auc_roc")].iloc[0]
if hb_vs_dummy["ci_low"] <= 0:
    warnings.append(f"WARNING: the Hb XGBoost model is not better than the mean-only Dummy (MAE reduction "
                    f"{hb_vs_dummy['point']:.3f} g/dL, 95% CI {hb_vs_dummy['ci_low']:.3f} to {hb_vs_dummy['ci_high']:.3f}).")
if ced_vs_dummy["ci_low"] <= 0:
    warnings.append(f"WARNING: the CED XGBoost model is not better than the class-proportion Dummy (AUC difference "
                    f"{ced_vs_dummy['point']:.3f}, 95% CI {ced_vs_dummy['ci_low']:.3f} to {ced_vs_dummy['ci_high']:.3f}).")

# %% Feature selection frequency over the 25 outer folds and the dashboard lists
fs = config["feature_selection"]
freq_rows, dashboard = [], {}
for target in ("hb", "ced"):
    folds = load_folds(config, f"main__{target}__xgboost__main")
    selected = [s.split("|") for s in folds["selected_features"]]
    table = selection_frequency(selected, config["features"]["candidate_pool"], float(fs["dashboard_min_fold_share"]))
    table.insert(0, "target", target)
    freq_rows.append(table)
    dashboard[target] = table.loc[table["in_dashboard_list"], "feature"].tolist()
freq = pd.concat(freq_rows, ignore_index=True)
save(freq, "d07_feature_selection_frequency")

# %% Exploratory comparisons (no verdicts)
explo = []
smogn = paired_bootstrap({"xgboost": hb_main["xgboost"], "xgboost_smogn": load_oof(config, "main__hb__xgboost__smogn")},
                         hb_metrics, B, SEED + 200)
explo.append(paired_table(smogn, "xgboost_smogn", ["xgboost"], "mae", "reduction", LEVEL))
explo.append(paired_table(smogn, "xgboost_smogn", ["xgboost"], "sensitivity", "difference", LEVEL))
ced_var = {"xgboost": ced_main["xgboost"], "xgboost_smotenc": load_oof(config, "main__ced__xgboost__smotenc"),
           "xgboost_class_weighting": load_oof(config, "main__ced__xgboost_class_weighted__class_weighting")}
rb_var = paired_bootstrap(ced_var, ced_metrics, B, SEED + 300)
for variant in ("xgboost_smotenc", "xgboost_class_weighting"):
    for metric in ("auc_roc", "pr_auc", "brier"):
        explo.append(paired_table(rb_var, variant, ["xgboost"], metric, "difference", LEVEL))
regression = hb_main["xgboost"].assign(score=lambda d: -d["pred"], truth=lambda d: d["y_binary"])
direct = load_oof(config, "main__anemia_binary__direct_xgb_classifier__direct").assign(score=lambda d: d["proba"],
                                                                                       truth=lambda d: d["y_true"])


def auc_only(d: pd.DataFrame) -> dict[str, float]:
    return {"auc_roc": float(roc_auc_score(d["truth"], d["score"]))}


rb_dir = paired_bootstrap({"regression_then_threshold": regression, "direct_classifier": direct}, auc_only, B, SEED + 400)
explo.append(paired_table(rb_dir, "regression_then_threshold", ["direct_classifier"], "auc_roc", "difference", LEVEL))
exploratory = pd.concat(explo, ignore_index=True)
exploratory.insert(0, "note", "EXPLORATORY, no verdict")
save(exploratory, "d08_exploratory_comparisons")
save(hb_summary[(hb_summary["model"] == "xgboost") & hb_summary["metric"].str.startswith("interval")]
     .assign(note="split-conformal, 90 percent nominal, calibrated on inner-fold predictions. May be slightly conservative."),
     "d09_conformal_intervals")
bins = np.linspace(0, 1, 11)
rel = []
for rep, block in ced_main["xgboost"].groupby("repeat"):
    block = block.assign(bin=pd.cut(block["proba"], bins, include_lowest=True))
    agg = block.groupby("bin", observed=True).agg(mean_predicted=("proba", "mean"), observed_rate=("y_true", "mean"),
                                                  n=("y_true", "size")).reset_index()
    agg["repeat"] = rep
    rel.append(agg)
reliability = pd.concat(rel).groupby("bin", observed=True)[["mean_predicted", "observed_rate", "n"]].mean().reset_index()
reliability["bin"] = reliability["bin"].astype(str)
save(reliability, "d10_ced_reliability_curve")

# %% Sensitivity analyses (no verdicts)
df_clean = clean_dataset(config).frame
sens_rows, sens_meta = [], []
SENS_ANALYSES = [("sens_ced_le22", "ced"), ("sens_ced_le24", "ced"), ("sens_trimester", "hb"), ("sens_gap_le90", "hb"),
                 ("sens_gap_le30", "hb"), ("sens_grouped", "hb"), ("sens_grouped", "ced"), ("sens_all11", "hb"),
                 ("sens_all11", "ced"), ("sens_fixed_preliminary", "hb"), ("sens_fixed_preliminary", "ced")]
build_args = {"sens_ced_le22": {"ced_cutoff_cm": 22}, "sens_ced_le24": {"ced_cutoff_cm": 24},
              "sens_trimester": {"trimester": True}, "sens_gap_le90": {"max_gap_days": 90}, "sens_gap_le30": {"max_gap_days": 30}}
for analysis, target in SENS_ANALYSES:
    models = ["xgboost", "dummy_mean", "ridge"] if target == "hb" else ["xgboost", "dummy_prior", "logistic_regression"]
    frames = oofs(analysis, target, models)
    fn = hb_metrics if target == "hb" else ced_metrics
    rb = paired_bootstrap(frames, fn, B, SEED + 500)
    metrics = ["mae", "rmse", "r2", "sensitivity", "specificity", "auc_roc"] if target == "hb" else ["auc_roc", "pr_auc", "brier", "sensitivity", "specificity"]
    s = summary_table(rb, models, metrics, LEVEL)
    s.insert(0, "target", target)
    s.insert(0, "analysis", analysis)
    sens_rows.append(s)
    data = build_data(config, df_clean, target, analysis=analysis, **build_args.get(analysis, {}))
    sens_meta.append({"analysis": analysis, "target": target, **{k: v for k, v in data.meta.items() if k != "cell_counts"}})
    if analysis == "sens_grouped":
        x = frames["xgboost"]
        per_group = []
        for group, block in x.groupby("group"):
            m = fn(block)
            per_group.append({"target": target, "subdistrict": group, "n": len(block),
                              "cases": int(block["y_binary" if target == "hb" else "y_true"].sum()),
                              **{k: m.get(k) for k in metrics}})
        save(pd.DataFrame(per_group), f"d12_grouped_cv_per_subdistrict_{target}")
three = paired_bootstrap({"xgboost": hb_main["xgboost"]}, three_class_metrics, B, SEED + 600)
three_tab = summary_table(three, ["xgboost"], ["macro_f1", "macro_precision", "macro_recall"], LEVEL)
three_tab.insert(0, "analysis", "sens_three_class")
sens_rows.append(three_tab)
save(pd.concat(sens_rows, ignore_index=True), "d11_sensitivity_analyses")
save(pd.DataFrame(sens_meta), "d13_sensitivity_sample_sizes")

# %% Report
v = verdicts.set_index("hypothesis")
lines = [f"# Results report ({datetime.now().strftime('%Y-%m-%d %H:%M')})", ""]
lines += [f"**{w}**" for w in warnings] + ([""] if warnings else [])
lines += [
    "All results are internal validation on one regional dataset of fewer than 420 women. They show whether the "
    "approach is feasible in this dataset. They do not show that it generalizes to other regions or that it is clinically useful.",
    "",
    "Reported metrics come only from the outer cross-validation with per-fold feature selection. The final models "
    "(script 09) use a fixed feature list (selected in at least half of the 25 outer folds) and are not used for any reported metric.",
    "",
    "## Hypotheses",
    "",
    "| Hypothesis | Role | Verdict | CI level | Holm order | p-value |",
    "|---|---|---|---|---|---|",
]
for hid, row in v.iterrows():
    order = "" if pd.isna(row["holm_order"]) else int(row["holm_order"])
    lines.append(f"| {hid} | {row['role']} | {row['verdict']} | {row['ci_level_used']} | {order} | {row['hypothesis_p_value']:.4f} |")
lines += ["", "| Hypothesis | Component | Criterion | Point | CI | Outcome |", "|---|---|---|---|---|---|"]
for _, c in components.iterrows():
    lines.append(f"| {c['hypothesis']} | {c['component']} | {c['criterion']} ({c['rule']}) | {c['point']:.3f} | "
                 f"{c['ci_low']:.3f} to {c['ci_high']:.3f} ({c['ci_level']:.1%}) | {c['outcome']} |")
lines += ["", "## Dashboard feature lists (selected in at least half of the 25 outer folds)", ""]
for target, feats in dashboard.items():
    lines.append(f"* {target}: {', '.join(feats)}")
lines += ["", "## Files", ""] + [f"* {w}" for w in written]
report_path = results_dir(config, "") / "report.md"
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"Report written to {report_path}")
