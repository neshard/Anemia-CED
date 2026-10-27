# %% [markdown]
# # 12 Boosting comparison (supervisor revision 2026-10-27, descriptive, no verdict)
# XGBoost against GBM, LightGBM, CatBoost and AdaBoost on both targets. Every model ran
# through the identical protocol in 04/05 (same outer and inner folds, preprocessing,
# per-fold feature set, 30 Optuna TPE trials, threshold rules, no resampling).
# Reads only out-of-fold predictions and fold tables. Nothing is fitted here.
#
# * m00: canary 3 (too good) for the comparators.
# * m01/m02: one row per model, metric means over the 5 repetitions with 95 percent CIs.
#   Bootstrap seeds: paired_bootstrap_seed in config/model_selection.yaml (equal to those of 08,
#   so the XGBoost rows match d03/d04 exactly).
# * m03: paired differences XGBoost minus each comparator (same resampled women), the
#   corrected resampled t-test, and Holm across the four comparisons per target on the
#   primary metric when comparison.holm_correction is true.
# * m04: all pairwise differences on the primary metric (input of the selection in 14).
# * m05: wall-clock times (descriptive only) and the tree-node inference cost proxy.
# * Figures: forest plot of the paired differences, boxplot over repetitions.
#
# Run from the project root:  python experiments/12_boosting_comparison.py

# %%
from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import ROOT, start  # noqa: E402

from src.evaluation import METRIC_DIRECTION, corrected_resampled_ttest, holm_adjust  # noqa: E402
from src.model_selection import load_rule  # noqa: E402
from src.protocol_stats import (  # noqa: E402
    ced_metrics,
    hb_metrics,
    paired_bootstrap,
    paired_table,
    quantity_samples,
    summary_table,
)
from src.runs import load_folds, load_oof, save_result_table  # noqa: E402

config, hyp = start("12_boosting_comparison")
assert hyp is not None
cmp_cfg = config["comparison"]
B, LEVEL, SEED = int(hyp["n_bootstrap"]), float(config["evaluation"]["ci_level"]), int(config["project"]["random_seed"])
CANDS = list(cmp_cfg["candidates"])
REF = cmp_cfg["reference"]
OTHERS = [c for c in CANDS if c != REF]
HOLM = bool(cmp_cfg["holm_correction"])
# Bootstrap seeds come from the locked selection rule, so the selection CIs (m04) are reproducible.
BOOT_SEED = {t: int(s) for t, s in load_rule(require_locked=not config["project"].get("smoke_test"))["paired_bootstrap_seed"].items()}
TARGETS = {
    "hb": {"fn": hb_metrics, "seed": BOOT_SEED["hb"], "primary": "mae",
           "metrics": ["mae", "rmse", "r2", "sensitivity", "specificity", "ppv", "npv", "auc_roc", "pr_auc"],
           "forest": ["mae", "rmse", "auc_roc"], "box": ["mae", "auc_roc"]},
    "ced": {"fn": ced_metrics, "seed": BOOT_SEED["ced"], "primary": "auc_roc",
            "metrics": ["sensitivity", "specificity", "ppv", "npv", "auc_roc", "pr_auc", "brier"],
            "forest": ["auc_roc", "pr_auc", "brier"], "box": ["auc_roc", "brier"]},
}
LABELS = {"xgboost": "XGBoost", "gbm": "GBM", "lightgbm": "LightGBM", "catboost": "CatBoost", "adaboost": "AdaBoost"}
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
NOTE = "Descriptive comparison, no verdict. Hypotheses P1, P2, S1 and S2 concern XGBoost only."
written: list[str] = []


def save(df: pd.DataFrame, name: str) -> None:
    written.append(str(save_result_table(df, name, config).relative_to(ROOT)))


def quantity(metric: str) -> str:
    """Errors (lower is better) are reported as reductions, so a positive value always favours XGBoost."""
    return "reduction" if METRIC_DIRECTION[metric] < 0 else "difference"


def two_sided_p(boots: list[np.ndarray]) -> float:
    """Bootstrap two-sided p for a zero difference, consistent with the repetition-averaged CI.

    The CI averages each quantile over the repetitions, so the draws of every repetition are
    sorted and averaged position by position (quantile averaging) and the p-value is read from
    that averaged distribution: (count in the smaller tail + 1) / (B + 1), doubled. A 95 percent
    CI that contains 0 then always has p >= 0.05. Averaging per-repetition p-values instead would
    not be: when repetitions disagree in sign, each can be small while the averaged CI contains 0.
    """
    n = min(int(np.isfinite(np.asarray(b, dtype=float)).sum()) for b in boots)
    probs = (np.arange(n) + 0.5) / n
    avg = np.mean([np.nanquantile(np.asarray(b, dtype=float), probs) for b in boots], axis=0)
    tail = min(int((avg <= 0).sum()), int((avg >= 0).sum()))
    return float(min(1.0, 2 * (tail + 1) / (n + 1)))


def fold_metric(df: pd.DataFrame, fn, metric: str) -> pd.Series:
    return df.groupby(["repeat", "fold"]).apply(lambda b: fn(b)[metric], include_groups=False)


def style(ax) -> None:  # type: ignore[no-untyped-def]
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# %% Load, canary 3, bootstrap
too = config["evaluation"]["too_good"]
canary, wide_rows, diff_rows, pair_rows, time_rows = [], [], [], [], []
boots_by_target = {}
for target, t in TARGETS.items():
    frames = {m: load_oof(config, f"main__{target}__{m}__main") for m in CANDS}
    folds = {m: load_folds(config, f"main__{target}__{m}__main") for m in CANDS}
    for m in OTHERS:
        key, limit = ("r2", too["hb_r2"]) if target == "hb" else ("auc_roc", too["ced_auc"])
        value = float(np.mean([t["fn"](b)[key] for _, b in frames[m].groupby("repeat")]))
        canary.append({"canary": f"too_good_{target}_{key}", "model": m, "value": value, "limit": limit,
                       "passed": value <= limit})
    rb = paired_bootstrap(frames, t["fn"], B, t["seed"])
    boots_by_target[target] = (rb, frames)

    # one row per model, mean and CI for every metric
    long = summary_table(rb, CANDS, t["metrics"], LEVEL)
    for m in CANDS:
        row = {"target": target, "model": m}
        for _, r in long[long["model"] == m].iterrows():
            row.update({r["metric"]: r["point"], f"{r['metric']}_ci_low": r["ci_low"],
                        f"{r['metric']}_ci_high": r["ci_high"], f"{r['metric']}_range": f"{r['point_min']:.4f} to {r['point_max']:.4f}"})
        wide_rows.append(row)

    # XGBoost minus each comparator, plus the corrected resampled t-test on fold metrics
    for metric in t["metrics"]:
        q = quantity(metric)
        table = paired_table(rb, REF, OTHERS, metric, q, LEVEL)
        ref_fold = fold_metric(frames[REF], t["fn"], metric)
        sign = -1 if q == "reduction" else 1
        extra = []
        for other in OTHERS:
            _, bts = quantity_samples(rb, REF, metric, other, q)
            diff = sign * (ref_fold - fold_metric(frames[other], t["fn"], metric))
            tt = corrected_resampled_ttest(diff.to_numpy(), folds[REF]["n_train"].mean(), folds[REF]["n_test"].mean())
            extra.append({"p_bootstrap_two_sided": two_sided_p(bts), "mean_fold_difference": float(diff.mean()),
                          "t_corrected": tt["t"], "p_corrected_approximate": tt["p"]})
        table = pd.concat([table, pd.DataFrame(extra)], axis=1)
        table.insert(0, "target", target)
        table["primary_metric"] = metric == t["primary"]
        if metric == t["primary"] and HOLM:
            table["p_bootstrap_holm"] = holm_adjust(table["p_bootstrap_two_sided"])
            table["p_corrected_holm"] = holm_adjust(table["p_corrected_approximate"])
        else:
            table["p_bootstrap_holm"] = np.nan
            table["p_corrected_holm"] = np.nan
        diff_rows.append(table)

    # every ordered pair on the primary metric (metric(a) - metric(b)), for the selection rule in 14
    for a, b in itertools.permutations(CANDS, 2):
        pts, bts = quantity_samples(rb, a, t["primary"], b, "difference")
        alpha = (1 - 0.95) / 2
        pair_rows.append({"target": target, "model_a": a, "model_b": b, "metric": t["primary"],
                          "bootstrap_seed": t["seed"], "n_bootstrap": B,
                          "difference_a_minus_b": float(np.mean(pts)),
                          "ci_low_95": float(np.mean([np.nanquantile(x, alpha) for x in bts])),
                          "ci_high_95": float(np.mean([np.nanquantile(x, 1 - alpha) for x in bts]))})

    for m in CANDS:
        f = folds[m]
        time_rows.append({"target": target, "model": m,
                          "median_tuning_seconds_per_fold": float(f["tuning_seconds"].median()),
                          "median_fit_seconds_per_fold": float(f["fit_seconds"].median()),
                          "median_predict_ms_per_row": float(1000 * f["predict_seconds_per_row"].median()),
                          "iqr_predict_ms_per_row": float(1000 * (f["predict_seconds_per_row"].quantile(0.75)
                                                                  - f["predict_seconds_per_row"].quantile(0.25))),
                          "median_tree_nodes": float(f["n_tree_nodes"].median()),
                          "min_tree_nodes": float(f["n_tree_nodes"].min()),
                          "max_tree_nodes": float(f["n_tree_nodes"].max())})

canary_table = pd.DataFrame(canary)
save(canary_table, "m00_canary_too_good_boosting_comparators")
print(canary_table.to_string(index=False))
if not canary_table["passed"].all():
    raise SystemExit("A comparator exceeds the too-good limit. Treat as possible leakage and investigate.")

wide = pd.DataFrame(wide_rows)
wide["note"] = NOTE
save(wide[wide["target"] == "hb"].dropna(axis=1, how="all"), "m01_hb_boosting_metrics")
save(wide[wide["target"] == "ced"].dropna(axis=1, how="all"), "m02_ced_boosting_metrics")
diffs = pd.concat(diff_rows, ignore_index=True)
diffs["holm_correction"] = HOLM
diffs["note"] = NOTE + " Positive values favour XGBoost (reduction for error metrics, difference otherwise)."
save(diffs, "m03_paired_differences_xgboost_vs_comparators")
save(pd.DataFrame(pair_rows), "m04_pairwise_primary_metric_differences")
times = pd.DataFrame(time_rows)
times["note"] = ("Wall-clock times with 8 outer folds in parallel, one thread per fit. Descriptive only, never used "
                 "for model selection. Tree nodes are the deterministic inference cost proxy.")
save(times, "m05_training_inference_cost")

# %% Figures
for target, t in TARGETS.items():
    rb, frames = boots_by_target[target]
    sub = diffs[diffs["target"] == target]
    fig, axes = plt.subplots(1, len(t["forest"]), figsize=(3.2 * len(t["forest"]), 2.8), sharey=True)
    for ax, metric in zip(axes, t["forest"], strict=True):
        d = sub[sub["metric"] == metric].iloc[::-1]
        y = np.arange(len(d))
        ax.axvline(0, color="#898781", lw=1, ls="--")
        ax.errorbar(d["point"], y, xerr=[d["point"] - d["ci_low"], d["ci_high"] - d["point"]], fmt="o", color="#3987e5",
                    ecolor="#3987e5", ms=6, lw=2, capsize=0)
        ax.set_yticks(y, [f"vs {LABELS[v]}" for v in d["versus"]])
        word = "reduction" if quantity(metric) == "reduction" else "difference"
        ax.set_title(f"{metric.upper().replace('_', '-')} {word}", loc="left", fontsize=9)
        style(ax)
    fig.suptitle(f"{'Hb' if target == 'hb' else 'CED'}: XGBoost minus comparator, 95% paired bootstrap CI "
                 "(positive favours XGBoost)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"m_fig01_forest_paired_differences_{target}.png", dpi=300, facecolor="white")
    plt.close(fig)
    written.append(f"results/figures/m_fig01_forest_paired_differences_{target}.png")

    fig, axes = plt.subplots(1, len(t["box"]), figsize=(4.2 * len(t["box"]), 3.0))
    for ax, metric in zip(axes, t["box"], strict=True):
        per_rep = [[rb.points[(m, r)][metric] for r in rb.repeats] for m in CANDS]
        ax.boxplot(per_rep, widths=0.5, medianprops={"color": "#3987e5", "lw": 2},
                   boxprops={"color": "#52514e"}, whiskerprops={"color": "#52514e"}, capprops={"color": "#52514e"})
        for i, vals in enumerate(per_rep, start=1):
            ax.scatter(np.full(len(vals), i), vals, s=14, color="#52514e", zorder=3)
        ax.set_xticks(range(1, len(CANDS) + 1), [LABELS[m] for m in CANDS], fontsize=8)
        ax.set_title(f"{metric.upper().replace('_', '-')} per repetition (pooled out-of-fold)", loc="left", fontsize=9)
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / f"m_fig02_boxplot_repetitions_{target}.png", dpi=300, facecolor="white")
    plt.close(fig)
    written.append(f"results/figures/m_fig02_boxplot_repetitions_{target}.png")

print(wide.drop(columns="note").round(4).to_string(index=False))
print(diffs[diffs["primary_metric"]][["target", "versus", "metric", "quantity", "point", "ci_low", "ci_high",
                                      "p_bootstrap_two_sided", "p_bootstrap_holm", "p_corrected_approximate"]]
      .round(4).to_string(index=False))
print(times.drop(columns="note").round(3).to_string(index=False))
print("Written:", *written, sep="\n  ")
