# %% [markdown]
# # 07 Hold-out evaluation and statistical comparison
# This is the ONLY script that loads the hold-out test set, and it can do so
# once: ``claim_holdout`` writes a record to artifacts/ and refuses a second
# claim. All feature setups are evaluated under that single claim. Pipelines,
# the H3 cut-off and the H8 interval were all fixed on the training set by
# 03 to 06. Nothing is refitted or tuned here.
#
# Decisions follow config/hypotheses.yaml through src/hypotheses.py:
# * decision_setup: the feature setup whose results decide each hypothesis.
#   Other setups get the same tables, labelled secondary.
# * holm_family: decision family. The other family is reported as a sensitivity check.
# * pass_rule, min_meaningful_difference, ci_level, evaluation, min_auc,
#   auc_ci_lower_min, min_sensitivity, min_specificity, coverage_min and coverage_max
#   are applied exactly as named. success_criterion text is not read.
#
# Run from the project root:  python experiments/07_evaluation_and_stats.py

# %%
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from _common import start

from src.evaluation import (
    METRIC_DIRECTION,
    binary_metrics,
    bootstrap_metrics,
    dummy_closeness_notes,
    multiclass_metrics,
    paired_comparison,
    plot_confusion_matrix,
    regression_metrics,
    save_figure,
    screening_metrics,
    who_metrics_from_hb,
)
from src.hypotheses import adjust_pvalues, comparison_plan, decide, pooled_improvement
from src.intervals import ConformalInterval, empirical_coverage
from src.models import CED_MODELS, HB_MODELS
from src.training import artifact_path, claim_holdout, load_run_record, load_test_data, run_tag
from src.utils import resolve_path, save_table

config, hyp = start()
assert hyp is not None
SETUPS = list(config["run_setups"])
DECISION = str(hyp["decision_setup"])
primary = config["models"]["primary_resampling"]
direct_rs = config["models"]["direct_comparator_resampling"]
ev = config["evaluation"]
seed = int(config["project"]["random_seed"])
figures = resolve_path(config, "reports_figures")
oof_dir = resolve_path(config, "artifacts_dir") / "oof"
notes: list[dict[str, str]] = []

# %% Load CV fold tables written by 03 to 06
fold_files = sorted((resolve_path(config, "reports_tables") / "cv_folds").glob("*__stratified.csv"))
folds = pd.concat([pd.read_csv(f) for f in fold_files], ignore_index=True)


def fold_scores(setup: str, target: str, model: str, variant: str) -> pd.DataFrame:
    sub = folds[(folds["setup"] == setup) & (folds["target"] == target) & (folds["model"] == model)
                & (folds["variant"] == variant)]
    if sub.empty:
        raise KeyError(f"No CV folds for {setup} {target} {model} {variant}. Run scripts 03 to 06 first.")
    return sub.sort_values("fold")


def oof_of(setup: str, ref: tuple[str, str, str]) -> pd.DataFrame:
    return pd.read_csv(oof_dir / f"{run_tag(setup, *ref)}__stratified.csv")


# %% Paired CV comparisons for every setup (training data only)
plan = comparison_plan(hyp, config)
p_col = "p_corrected_t" if hyp["paired_test"] == "corrected_resampled_t" else "p_wilcoxon"
rows = []
for setup in SETUPS:
    for c in plan:
        if c.kind == "ablation" and setup not in config["ablation_setups"]:
            continue
        ref_f, cmp_f = fold_scores(setup, *c.reference), fold_scores(setup, *c.comparator)
        if not np.array_equal(ref_f["n_val"].to_numpy(), cmp_f["n_val"].to_numpy()):
            raise RuntimeError(f"{setup} {c.hypothesis}: fold sizes differ, so the folds are not paired")
        res = paired_comparison(ref_f, cmp_f, c.metric, level=c.ci_level)
        decision_improvement = res["mean_improvement"]
        if c.evaluation == "pooled_out_of_fold":
            decision_improvement = pooled_improvement(oof_of(setup, c.reference), oof_of(setup, c.comparator),
                                                      c.metric, METRIC_DIRECTION[c.metric])
        rows.append({"setup": setup, "role": "decision" if setup == DECISION else "secondary",
                     "hypothesis": c.hypothesis, "kind": c.kind, "evaluation": c.evaluation,
                     "reference": "/".join(c.reference), "comparator": "/".join(c.comparator),
                     "ci_level": c.ci_level, **res, "paired_dz": res["cohens_dz"],
                     "decision_improvement": decision_improvement})
comparisons = adjust_pvalues(pd.DataFrame(rows).drop(columns=["cohens_dz"]), hyp, p_col)
comparisons["p_used"] = p_col
save_table(comparisons.round(5), "c08_paired_comparisons_cv", config)
print(comparisons[["setup", "hypothesis", "comparator", "metric", "decision_improvement", "p_adjusted",
                   "p_adjusted_sensitivity", "paired_dz"]].round(4).to_string(index=False))

# %% Hold-out test set: claimed ONCE for all setups
token = claim_holdout(config, purpose=f"07_evaluation_and_stats for setups {SETUPS}, decision setup {DECISION}")
thr = float(ev["anemia_threshold"])
cthr = float(config["models"]["ced_threshold"])
nb, lvl = int(ev["n_bootstrap"]), float(ev["ci_level"])
hb_metric_names = ["mae", "rmse", "r2", "sensitivity", "specificity", "ppv", "npv", "auc_roc", "pr_auc"]
ced_metric_names = ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc", "pr_auc", "brier"]
hb_tables, ced_tables, who_rows, direct_tables, op_rows = [], [], [], [], []
test_cache: dict[tuple[str, str], object] = {}


def test_data(target: str, setup: str, class_names: list[str] | None = None):  # type: ignore[no-untyped-def]
    """Hold-out rows, loaded under the single claim and cached for the rest of the script."""
    key = (target, setup)
    if key not in test_cache:
        test_cache[key] = load_test_data(config, target, token, setup, class_names)
    return test_cache[key]


for setup in SETUPS:
    # --- Hb regressor and baselines
    hb_test = test_data("hb_model", setup)
    hb_preds = {}
    for name in HB_MODELS:
        pipe = joblib.load(artifact_path(config, run_tag(setup, "hb_model", name, primary["hb_model"])))
        hb_preds[name] = pipe.predict(hb_test.X)
        table = bootstrap_metrics(lambda t, p: regression_metrics(t, p, thr), hb_test.y, hb_preds[name],
                                  hb_metric_names, nb, lvl, seed)
        table.insert(0, "model", name)
        table.insert(0, "setup", setup)
        hb_tables.append(table)

    # --- H3 operating point and H8 interval, both fixed on training out-of-fold predictions
    record = load_run_record(config, run_tag(setup, "hb_model", "xgboost", primary["hb_model"]))
    cut = float(record["operating_point"]["cutoff"])
    truth = hb_test.y < thr
    screen = screening_metrics(truth, hb_preds["xgboost"] < cut, -hb_preds["xgboost"])
    interval = ConformalInterval(**{k: record["interval"][k] for k in ("half_width", "nominal", "n_calibration", "method")})
    lo, hi = interval.predict(hb_preds["xgboost"])
    op_rows.append({"setup": setup, "cutoff_g_dl": cut, **{k: record["operating_point"][k] for k in ("oof_sensitivity", "oof_specificity")},
                    "holdout_sensitivity": screen["sensitivity"], "holdout_specificity": screen["specificity"],
                    "holdout_ppv": screen["ppv"], "holdout_npv": screen["npv"], "n_holdout": len(truth),
                    "n_holdout_anemic": int(truth.sum()), "interval_half_width": interval.half_width,
                    "interval_nominal": interval.nominal, "holdout_coverage": empirical_coverage(hb_test.y, lo, hi)})

    # --- WHO classes from predicted Hb (XGBoost)
    for merged, label in ((False, "4class"), (True, "3class_moderate_severe_merged")):
        res = who_metrics_from_hb(hb_test.true_class, hb_preds["xgboost"], merge_severe=merged)
        cm = pd.DataFrame(res["confusion_matrix"], index=[f"true_{c}" for c in res["labels"]],
                          columns=[f"pred_{c}" for c in res["labels"]])
        save_table(cm.reset_index(), f"c10_test_who_confusion_{label}__{setup}", config)
        save_figure(plot_confusion_matrix(res["confusion_matrix"], res["labels"],
                                          f"WHO anemia class from predicted Hb ({label}, {setup})"),
                    f"c10_test_who_confusion_{label}__{setup}", figures)
        who_rows.append({"setup": setup, "version": label,
                         **{k: v for k, v in res.items() if k not in ("labels", "confusion_matrix", "support")},
                         **{f"n_true_{k}": v for k, v in res["support"].items()}})
        if setup == DECISION:
            for cls, n in res["support"].items():
                if 0 < n < int(ev["min_cases_warning"]):
                    notes.append({"topic": f"WHO {label}", "note": f"Only {n} test cases are '{cls}'. Per-class "
                                  f"metrics for '{cls}' are unreliable and its CI is very wide or undefined."})

    # --- CED classifier and baselines
    ced_test = test_data("ced_model", setup)
    ced_probas = {}
    for name in CED_MODELS:
        pipe = joblib.load(artifact_path(config, run_tag(setup, "ced_model", name, primary["ced_model"])))
        ced_probas[name] = pipe.predict_proba(ced_test.X)[:, 1]
        table = bootstrap_metrics(lambda t, p: binary_metrics(t, p, cthr), ced_test.y, ced_probas[name],
                                  ced_metric_names, nb, lvl, seed)
        table.insert(0, "model", name)
        table.insert(0, "setup", setup)
        ced_tables.append(table)
    pred = (ced_probas["xgboost"] >= cthr).astype(int)
    cm = np.array([[((ced_test.y == i) & (pred == j)).sum() for j in (0, 1)] for i in (0, 1)])
    save_table(pd.DataFrame(cm, index=["true_no_risk", "true_ced_risk"], columns=["pred_no_risk", "pred_ced_risk"]).reset_index(),
               f"c12_test_ced_confusion__{setup}", config)
    save_figure(plot_confusion_matrix(cm, ["No CED risk", "CED risk"], f"CED risk, XGBoost, threshold {cthr} ({setup})"),
                f"c12_test_ced_confusion__{setup}", figures)

    # --- Direct anemia classifiers (descriptive. H4 is decided on CV folds as specified)
    bin_test = test_data("anemia_binary", setup)
    pipe = joblib.load(artifact_path(config, run_tag(setup, "anemia_binary", "direct_xgb_classifier", direct_rs)))
    t = bootstrap_metrics(lambda a, b: binary_metrics(a, b, 0.5), bin_test.y, pipe.predict_proba(bin_test.X)[:, 1],
                          ["sensitivity", "specificity", "auc_roc", "pr_auc"], nb, lvl, seed)
    t.insert(0, "model", "direct_xgb_classifier (binary)")
    t.insert(0, "setup", setup)
    direct_tables.append(t)
    who_tag = run_tag(setup, "anemia_who", "direct_xgb_who", direct_rs)
    class_names = load_run_record(config, who_tag)["class_names"]
    who_test = test_data("anemia_who", setup, class_names)
    pipe = joblib.load(artifact_path(config, who_tag))
    t = bootstrap_metrics(lambda a, b: multiclass_metrics(a, b, class_names), who_test.y, pipe.predict_proba(who_test.X),
                          ["macro_f1", "macro_recall", "ovr_auc", "auc_roc"], nb, lvl, seed)
    t.insert(0, "model", f"direct_xgb_who ({', '.join(class_names)})")
    t.insert(0, "setup", setup)
    direct_tables.append(t)

hb_test_table = pd.concat(hb_tables, ignore_index=True)
ced_test_table = pd.concat(ced_tables, ignore_index=True)
op_table = pd.DataFrame(op_rows)
save_table(hb_test_table.round(4), "c09_test_hb_regression_and_screening", config)
save_table(pd.DataFrame(who_rows).round(4), "c11_test_who_class_metrics", config)
save_table(ced_test_table.round(4), "c12_test_ced_classification", config)
save_table(op_table.round(4), "c16_test_operating_point_and_interval_coverage", config)
reg_bin = hb_test_table[(hb_test_table["model"] == "xgboost")
                        & hb_test_table["metric"].isin(["sensitivity", "specificity", "auc_roc", "pr_auc"])].copy()
reg_bin["model"] = "xgboost regression then Hb < 11 (fixed 11.0 cut-off)"
save_table(pd.concat([reg_bin, *direct_tables], ignore_index=True).round(4), "c13_test_regression_vs_direct", config)

# %% Hypothesis verdicts (decision setup), plus the sensitivity family


def test_value(table: pd.DataFrame, setup: str, model: str, metric: str, col: str = "value") -> float:
    return float(table[(table["setup"] == setup) & (table["model"] == model) & (table["metric"] == metric)][col].iloc[0])


op = op_table.set_index("setup").loc[DECISION]
holdout_values = {
    "H3_anemia_screening": {"cutoff": float(op["cutoff_g_dl"]), "sensitivity": float(op["holdout_sensitivity"]),
                            "specificity": float(op["holdout_specificity"]),
                            "auc_ci_low": test_value(hb_test_table, DECISION, "xgboost", "auc_roc", "ci_low")},
    "H5_ced_classification": {"auc": test_value(ced_test_table, DECISION, "xgboost", "auc_roc"),
                              "auc_ci_low": test_value(ced_test_table, DECISION, "xgboost", "auc_roc", "ci_low")},
    "H8_hb_interval_coverage": {"coverage": float(op["holdout_coverage"]), "half_width": float(op["interval_half_width"])},
}
decision_rows = comparisons[comparisons["setup"] == DECISION]
verdicts = decide(hyp, decision_rows, holdout_values, p_column="p_adjusted")
sensitivity = decide(hyp, decision_rows, holdout_values, p_column="p_adjusted_sensitivity")
verdicts["supported_under_sensitivity_family"] = sensitivity["supported"].to_numpy()
verdicts.insert(0, "setup", DECISION)
save_table(verdicts, "c14_hypothesis_verdicts", config)
print(verdicts[["hypothesis", "role", "supported", "supported_under_sensitivity_family", "detail"]].to_string(index=False))

# %% Honest notes: closeness to Dummy, every setup
for setup in SETUPS:
    cv_hb = {m: fold_scores(setup, "hb_model", "xgboost", primary["hb_model"])[m].mean() for m in ("mae", "auc_roc", "pr_auc")}
    cv_dummy = {"mae": fold_scores(setup, "hb_model", "dummy_mean", primary["hb_model"])["mae"].mean()}
    for text in dummy_closeness_notes("regression", cv_hb, cv_dummy, None, ev["dummy_close"]):
        notes.append({"topic": f"Hb model, CV ({setup})", "note": text})
    test_hb = {m: test_value(hb_test_table, setup, "xgboost", m) for m in ("mae", "auc_roc", "pr_auc")}
    test_dummy = {"mae": test_value(hb_test_table, setup, "dummy_mean", "mae")}
    hb_y = test_data("hb_model", setup).y
    for text in dummy_closeness_notes("regression", test_hb, test_dummy, float(np.mean(hb_y < thr)), ev["dummy_close"]):
        notes.append({"topic": f"Hb model, test ({setup})", "note": text})
    test_ced = {m: test_value(ced_test_table, setup, "xgboost", m) for m in ("auc_roc", "pr_auc")}
    ced_y = test_data("ced_model", setup).y
    for text in dummy_closeness_notes("classification", test_ced, {}, float(np.mean(ced_y)), ev["dummy_close"]):
        notes.append({"topic": f"CED model, test ({setup})", "note": text})

hb_decision = test_data("hb_model", DECISION)
if not hb_decision.true_class.eq("Severe").any():
    notes.append({"topic": "WHO classes", "note": "The test set has no Severe case, so Severe performance cannot be "
                  "assessed. Under hb_zero_policy 'missing' the dataset has no measured Hb below 7 g/dL."})
if np.allclose(np.asarray(hb_decision.y) % 1, 0):
    notes.append({"topic": "Data", "note": "Hb is recorded as whole numbers in this data version. The Mild class "
                  "then means Hb exactly 10 and regression error has a floor from rounding."})
notes_table = pd.DataFrame(notes)
save_table(notes_table, "c15_honesty_notes", config)
print("\nNotes")
for n in notes:
    print(f"- [{n['topic']}] {n['note']}")
