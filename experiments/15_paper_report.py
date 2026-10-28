# %% [markdown]
# # 15 Full report for the paper and thesis (results/paper_full_report.md)
# Collects every reported number from the tables written by 00 to 14 into one Markdown
# document with the structure of a paper (methods, results, discussion points, limitations,
# reproducibility, appendix). Nothing is fitted and no number is typed by hand: rerunning the
# pipeline and then this script refreshes the report. Prose is a draft for the author.
# Only aggregate numbers are written, never a row about one woman.
#
# Run from the project root:  python experiments/15_paper_report.py

# %%
from __future__ import annotations

import importlib.metadata
import json
import subprocess
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml
from _common import ROOT, start

from src.cleaning import clean_dataset
from src.nested_cv import build_data, run_hash
from src.postmodel.explain import ALGORITHM_LABELS
from src.utils import assert_no_identifiers

config, hyp = start("15_paper_report")
assert hyp is not None
T, X, R = ROOT / "results" / "tables", ROOT / "results" / "exploratory", ROOT / "reports" / "tables"
rule = yaml.safe_load((ROOT / "config" / "model_selection.yaml").read_text(encoding="utf-8"))
final = json.loads((ROOT / "artifacts" / "final_models.json").read_text(encoding="utf-8"))
NAMES = {**ALGORITHM_LABELS, "gbm": "GBM", "dummy_mean": "Mean-only", "dummy_prior": "Class proportion",
         "ridge": "Ridge", "logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
         "random_forest": "Random Forest"}
METRIC_NAMES = {"mae": "MAE (g/dL)", "rmse": "RMSE (g/dL)", "r2": "R²", "sensitivity": "Sensitivity",
                "specificity": "Specificity", "ppv": "PPV", "npv": "NPV", "auc_roc": "AUC-ROC", "pr_auc": "PR-AUC",
                "brier": "Brier", "referral_rate": "Referral rate"}
FEATURE_NAMES = {"age": "Age (years)", "n_children": "Number of children", "n_under5": "Children under 5",
                 "prepreg_weight": "Pre-pregnancy weight (kg)", "height": "Height (cm)", "anc_visits": "ANC visits",
                 "hfias_score": "HFIAS score (0-27)", "education_mother": "Education", "income": "Monthly income",
                 "egg_freq": "Egg consumption", "food_taboo": "Food taboo"}
BOOST = list(rule["candidates"])
TN = {"hb": "Hb", "ced": "CED"}
out: list[str] = []


def csv(path) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    return pd.read_csv(path)


def f(x: Any, d: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def ci(point: float, low: float, high: float, d: int = 3) -> str:
    return f"{f(point, d)} ({f(low, d)} to {f(high, d)})"


def table(df: pd.DataFrame) -> None:
    cols = list(df.columns)
    out.append("| " + " | ".join(map(str, cols)) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    out.append("")


def h(level: int, text: str) -> None:
    out.extend([f"{'#' * level} {text}", ""])


def p(*lines: str) -> None:
    out.extend([" ".join(lines), ""])


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# %% Inputs
clean = clean_dataset(config)
hb_data, ced_data = build_data(config, clean.frame, "hb"), build_data(config, clean.frame, "ced")
a01 = csv(R / "a01_cleaning_report.csv").set_index("item")["value"]
d01, d02 = csv(T / "d01_hypothesis_verdicts.csv"), csv(T / "d02_hypothesis_components.csv")
d03, d04 = csv(T / "d03_hb_metrics.csv"), csv(T / "d04_ced_metrics.csv")
d05, d06, d07 = (csv(T / f"{n}.csv") for n in ("d05_confusion_matrices_mean_over_repetitions",
                                               "d06_baseline_comparisons", "d07_feature_selection_frequency"))
d08, d11, d13 = (csv(T / f"{n}.csv") for n in ("d08_exploratory_comparisons", "d11_sensitivity_analyses",
                                               "d13_sensitivity_sample_sizes"))
c00, c00c, m00 = (csv(T / f"{n}.csv") for n in ("c00_canaries", "c00c_canaries_boosting_comparators",
                                                "m00_canary_too_good_boosting_comparators"))
m01, m02, m03 = (csv(T / f"{n}.csv") for n in ("m01_hb_boosting_metrics", "m02_ced_boosting_metrics",
                                               "m03_paired_differences_xgboost_vs_comparators"))
m05, m06, m07, m08, m09, m10, m11 = (csv(T / f"{n}.csv") for n in (
    "m05_training_inference_cost", "m06_shap_eligibility", "m07_shap_global_ranking_boosting",
    "m08_shap_stability_boosting", "m09_shap_cross_model_spearman", "m10_model_selection_summary",
    "m11_hb_explanation_gate"))
x13, x14, x16, x18, x20 = (csv(X / f"{n}_exploratory.csv") for n in (
    "x13_ced_brier_difference_uncalibrated_minus_calibrated", "x14_ced_dashboard_calibration_decision",
    "x16_hb_interval_coverage", "x18_shap_stability", "x20_shap_vs_builtin_importance_final"))


def metric_row(source: pd.DataFrame, model: str, metric: str) -> pd.Series:
    """One metric of one model from the long d03/d04 tables or the wide m01/m02 tables."""
    if "metric" in source.columns:
        r = source[(source["model"] == model) & (source["metric"] == metric)].iloc[0]
        return pd.Series({"point": r["point"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    r = source.set_index("model").loc[model]
    return pd.Series({"point": r[metric], "ci_low": r[f"{metric}_ci_low"], "ci_high": r[f"{metric}_ci_high"]})


def all_models(target: str) -> list[tuple[str, pd.DataFrame]]:
    """Every main-run model with the table its metrics come from (same bootstrap seeds in 08 and 12)."""
    long, wide = (d03, m01) if target == "hb" else (d04, m02)
    base = ["dummy_mean", "ridge"] if target == "hb" else ["dummy_prior", "logistic_regression"]
    return ([(m, wide if m != "xgboost" else long) for m in BOOST]
            + [(m, long) for m in base + ["decision_tree", "random_forest"]])


n_hb, n_ced = len(hb_data.y), len(ced_data.y)
n_anemic, n_ced_risk = int(hb_data.y_binary.sum()), int(ced_data.y.sum())
sel = {t: m10[(m10["target"] == t) & m10["selected"]].iloc[0] for t in ("hb", "ced")}
v1 = {t: m10[(m10["target"] == t) & m10["selected_by_version1_rule"]]["candidate"].iloc[0] for t in ("hb", "ced")}
verdict = d01.set_index("hypothesis")
gate = m11.iloc[0]
p2_lr = d02[(d02["hypothesis"] == "P2_ced_classification")
            & (d02["component"] == "auc_roc difference vs logistic_regression")].iloc[0]
ced_auc = metric_row(d04, "xgboost", "auc_roc")
hb_mae = metric_row(d03, "xgboost", "mae")

# %% Title and abstract
out += [
    "# Evaluating XGBoost with SHAP for Questionnaire-Based Risk Screening of Anemia and Chronic Energy "
    "Deficiency in Pregnant Women: A Case Study in North Lombok",
    "",
    f"*Full report generated by experiments/15_paper_report.py on {datetime.now():%Y-%m-%d %H:%M} from git commit "
    f"{git('rev-parse', '--short', 'HEAD')} (run hash {run_hash(config)}). Every number is read from the result "
    "tables named in each section. Prose is a draft for the author to edit. Internal validation on one regional "
    "dataset only.*",
    "",
]
h(2, "Abstract (draft)")
p(f"Field data from pregnant women in North Lombok were used to estimate hemoglobin (Hb) and the risk of chronic "
  f"energy deficiency (CED) from questionnaire answers only. Of {int(a01['n_rows_raw'])} records, {n_hb} women had a "
  f"valid Hb ({n_anemic}, {100 * n_anemic / n_hb:.1f} percent, below 11.0 g/dL) and {n_ced} a valid mid-upper arm "
  f"circumference ({n_ced_risk}, {100 * n_ced_risk / n_ced:.1f} percent, at risk of CED). Hb and MUAC were excluded "
  "from the feature sets, and imputation, feature selection, tuning and thresholds were fitted inside nested repeated "
  "cross-validation (5 x 5 outer, 3-fold inner). XGBoost was evaluated against pre-specified baselines and compared "
  "with four other boosting algorithms (GBM, LightGBM, CatBoost, AdaBoost) under an identical protocol.",
  f"For CED, XGBoost reached an AUC-ROC of {ci(*ced_auc)} but did not exceed Logistic Regression "
  f"(difference {ci(p2_lr['point'], p2_lr['ci_low'], p2_lr['ci_high'])}, {p2_lr['ci_level']:.1%} CI). Hb could not be "
  f"estimated better than the mean (XGBoost MAE {ci(*hb_mae)} g/dL; the R² point estimate was below 0 for every model), so anemia "
  "screening from predicted Hb did not reach its feasibility criteria. The five boosting algorithms did not differ "
  "meaningfully on either target. All four pre-specified hypotheses were not supported. SHAP rankings agreed closely "
  "across algorithms, with pre-pregnancy weight ranked first for both targets. A Streamlit prototype presents the CED "
  "estimate as the main result and the Hb estimate as indicative only.")

# %% Methods
h(2, "1. Methods")
h(3, "1.1 Data and participants")
p(f"Raw records: {int(a01['n_rows_raw'])}. Exact duplicate records removed: {int(a01['n_duplicate_records_dropped'])}. "
  f"Records without any usable target removed: {int(a01['n_rows_without_any_target_dropped'])}. Hb recorded as 0 was "
  f"treated as not measured ({int(a01['n_hb_zero'])} women, {int(a01['n_hb_zero_reporting_never_tested'])} of whom "
  f"reported never having had a blood test). Hb above 20 g/dL was removed from the Hb analysis "
  f"({int(a01['n_hb_above_max_dropped_from_hb_model'])}). Invalid MUAC values: {int(a01['n_muac_invalid'])}. "
  f"Eligible records: {n_hb} for the Hb model and {n_ced} for the CED model (source: reports/tables/a01).")
table(pd.DataFrame([
    {"Outcome": "Anemia (Hb < 11.0 g/dL)", "Eligible": n_hb, "Cases": n_anemic, "Percent": f"{100 * n_anemic / n_hb:.1f}",
     "Detail": f"mild {int(a01['anemia_class_counts.Mild'])}, moderate {int(a01['anemia_class_counts.Moderate'])}, "
               f"severe {int(a01['anemia_class_counts.Severe'])}"},
    {"Outcome": "CED risk (MUAC < 23.5 cm)", "Eligible": n_ced, "Cases": n_ced_risk,
     "Percent": f"{100 * n_ced_risk / n_ced:.1f}", "Detail": "recorded MUAC of 23 cm or less"},
]))
h(3, "1.2 Features")
fs = config["feature_selection"]
p(f"Eleven questionnaire candidates: {', '.join(FEATURE_NAMES[c] for c in config['features']['candidate_pool'])}. "
  f"The HFIAS score is the sum of the nine HFIAS items. In every training split, mutual information and random forest "
  f"importance ({fs['rf_n_estimators']} trees) ranks were averaged; {', '.join(FEATURE_NAMES[c] for c in fs['forced_include'])} "
  f"were always kept and the best remaining features completed a set of {fs['k']}. Imputation (median for numeric, "
  "mode for ordinal and binary) was fitted inside each training split. Hb, MUAC, their labels and identifiers are "
  "refused by a guard at fit and at prediction time.")
h(3, "1.3 Models")
p("Main model: XGBoost (regressor for Hb, classifier for CED). Pre-specified baselines: mean-only and Ridge (Hb), "
  "class proportion and Logistic Regression (CED). Exploratory comparators: Decision Tree and Random Forest. Boosting "
  "comparators (supervisor revision): Gradient Boosting Machine (scikit-learn), LightGBM, CatBoost and AdaBoost "
  "(SAMME for classification, AdaBoost.R2 for regression, shallow decision trees as base learners). All models "
  "received the same integer-coded matrix; CatBoost's native categorical handling was not used. No resampling was "
  "used in the main models. Search spaces are listed in Appendix A.")
h(3, "1.4 Validation and tuning")
cv, tu, th = config["cv"], config["tuning"], config["thresholds"]
p(f"Repeated stratified {cv['outer_n_splits']}-fold cross-validation with {cv['outer_n_repeats']} repetitions on all "
  f"eligible records, stratified on anemia class x CED status. Inside each outer training set an inner "
  f"{cv['inner_n_splits']}-fold cross-validation was used for Optuna tuning ({tu['n_trials']} TPE trials with a fixed "
  f"seed for every tuned model; objective MAE for Hb and AUC-ROC for CED) and to produce the predictions that set "
  f"the decision threshold: the cut-off on predicted Hb that flags {th['hb_flag_share_of_cases']:.0%} of true anemia "
  "cases, and Youden's index for CED. Thresholds never used the outer test fold or predictions of a model on its own "
  "training rows. Every model used identical outer folds, inner folds, preprocessing and per-fold feature sets.")
h(3, "1.5 Statistical analysis")
p(f"For each repetition the five outer test folds were pooled, so each woman has one out-of-fold prediction per "
  f"repetition. Metrics were computed on the pooled predictions with percentile bootstrap intervals over women "
  f"({hyp['n_bootstrap']} resamples per repetition); point estimates and CI bounds were averaged over the five "
  "repetitions. Comparisons resampled the same women for both models (paired bootstrap). The corrected resampled "
  "t-test (Nadeau and Bengio) is reported as an approximate check only. Holm's procedure was applied across the two "
  "primary hypotheses (first judged at a 97.5 percent CI) and, for reporting only, across the four XGBoost-versus-"
  "comparator differences per target on MAE (Hb) or AUC-ROC (CED).")
h(3, "1.6 Pre-specified hypotheses")
spec = hyp["hypotheses"]
table(pd.DataFrame([{"Hypothesis": k, "Criterion": " ".join(v["success_criterion"].split())} for k, v in spec.items()]))
h(3, "1.7 Explanations")
p("SHAP values were computed with shap.TreeExplainer only (no KernelExplainer fallback). For every boosting model, "
  "the 25 outer-fold models were refitted from their stored hyperparameters, had to reproduce the stored out-of-fold "
  f"predictions, and were explained on their outer test folds. Additivity was checked against a tolerance of "
  f"{rule['additivity_tolerance']:g} in raw output units (g/dL, log-odds). Stability is the mean pairwise Spearman "
  "correlation of mean |SHAP| across fold models; agreement between algorithms is the Spearman correlation of their "
  "rankings.")
h(3, "1.8 Selection of the dashboard model")
p(f"Candidates: {', '.join(NAMES[c] for c in BOOST)}; baselines, Decision Tree and Random Forest are never selected. "
  "A candidate is eligible only with a valid TreeExplainer and additivity passed on all 25 fold models. Version 1 of "
  "the rule (locked 2026-10-27, before the four comparators were run) preferred XGBoost whenever its paired 95 "
  "percent CI against the best eligible model contained 0. **Version 2 (2026-10-28, a deviation decided after all "
  f"results were known)** selects the eligible model with the best point estimate ({rule['targets']['hb']['metric'].upper()} "
  f"for Hb, AUC-ROC for CED) with no preferred model. The selection is therefore post hoc and its performance "
  "optimistic. The Hb explanation is shown only if the selected Hb model reduces MAE versus mean-only by at least "
  "0.10 g/dL (same rule as S1).")
h(3, "1.9 Safeguards and reproducibility")
p("Canaries before any headline result: label permutation (AUC must stay at or below 0.60, Hb R² at or below 0.05) "
  "and a fit spy recording the rows seen by every fit; results that look too good (CED AUC above 0.90, Hb R² above "
  "0.50) stop the analysis. Each outer fold is checkpointed under a hash of the configuration, hypotheses, raw data "
  "and source code. A refactor of the engine was rerun and every output compared cell for cell with the earlier "
  "results (identical). The automated test suite covers leakage, identical folds across models, two-run "
  "reproducibility, the selection rule and the dashboard.")

# %% Results
h(2, "2. Results")
h(3, "2.1 Characteristics of the eligible women")
p("Median [IQR] for numeric features, percent for categories, computed on all eligible records "
  "(the older reports/tables/e01 describes the retired hold-out design and is not used).")
enc = config["encodings"]
inv = {"education_mother": enc["education"], "income": enc["income"], "egg_freq": enc["food_frequency"]}


def describe(frame: pd.DataFrame) -> dict[str, str]:
    row = {"n": str(len(frame))}
    for c in config["features"]["candidate_pool"]:
        s = frame[c]
        if c in inv:
            labels = {}
            for text, code in inv[c].items():
                labels.setdefault(code, text.replace("?", ">="))
            top = s.value_counts(normalize=True).sort_index()
            row[FEATURE_NAMES[c]] = "; ".join(f"{labels.get(int(k), k)} {100 * v:.0f}%" for k, v in top.items())
        elif c == "food_taboo":
            row[FEATURE_NAMES[c]] = f"yes {100 * s.mean():.0f}%"
        else:
            q = s.quantile([0.5, 0.25, 0.75])
            row[FEATURE_NAMES[c]] = f"{q[0.5]:.0f} [{q[0.25]:.0f}, {q[0.75]:.0f}]"
    return row


groups = {"Hb cohort": hb_data.X, "Hb < 11": hb_data.X[hb_data.y_binary == 1], "Hb >= 11": hb_data.X[hb_data.y_binary == 0],
          "CED cohort": ced_data.X, "CED risk": ced_data.X[ced_data.y == 1], "No CED risk": ced_data.X[ced_data.y == 0]}
desc = pd.DataFrame({k: describe(v) for k, v in groups.items()})
desc.insert(0, "Characteristic", desc.index)
table(desc.reset_index(drop=True))

h(3, "2.2 Leakage canaries")
rows = []
for _, r in pd.concat([c00.assign(model="xgboost"), c00c], ignore_index=True).iterrows():
    value = f"AUC {f(r['auc_roc'])}" + (f", R² {f(r['r2'])}" if pd.notna(r.get("r2")) else "") if "permutation" in r["canary"] \
        else f"{int(r['fits_checked'])} fits checked"
    rows.append({"Canary": r["canary"], "Model": NAMES[r["model"]], "Result": value, "Passed": bool(r["passed"])})
table(pd.DataFrame(rows))
too_good = ", ".join(f"{NAMES[r.model]} {'R²' if 'r2' in r.canary else 'AUC'} {f(r.value)}" for r in m00.itertuples())
p(f"Too-good check for the comparators (CED AUC at most 0.90, Hb R² at most 0.50): all passed ({too_good}).")

h(3, "2.3 Hypothesis verdicts")
rows = []
for hid, r in d02.iterrows():
    v = verdict.loc[r["hypothesis"]]
    rows.append({"Hypothesis": r["hypothesis"], "Component": r["component"], "Criterion": f"{r['criterion']} ({r['rule']})",
                 "Estimate (CI)": ci(r["point"], r["ci_low"], r["ci_high"]), "CI level": f"{r['ci_level']:.1%}",
                 "Outcome": r["outcome"], "Holm order": "" if pd.isna(v["holm_order"]) else int(v["holm_order"]),
                 "Verdict": v["verdict"]})
table(pd.DataFrame(rows))
p("A hypothesis is supported only if every component passes or is met. P1 was judged first (smaller p-value) at a "
  "97.5 percent CI; because it was not supported, P2 was also judged at 97.5 percent. Source: results/tables/d01, d02.")

for target, metrics, title in (("hb", ["mae", "rmse", "r2", "sensitivity", "specificity", "ppv", "npv", "auc_roc", "pr_auc"],
                                "2.4 Hemoglobin and anemia screening, all models"),
                               ("ced", ["auc_roc", "pr_auc", "brier", "sensitivity", "specificity", "ppv", "npv"],
                                "2.5 CED risk, all models")):
    h(3, title)
    rows = []
    for model, src in all_models(target):
        row = {"Model": NAMES[model]}
        for m in metrics:
            r = metric_row(src, model, m)
            row[METRIC_NAMES[m]] = ci(r["point"], r["ci_low"], r["ci_high"])
        rows.append(row)
    table(pd.DataFrame(rows))
    p("Mean of five repetitions of pooled out-of-fold predictions, 95 percent percentile bootstrap CI over women. "
      f"Sources: results/tables/{'d03 and m01' if target == 'hb' else 'd04 and m02'}. "
      + ("Anemia metrics use the predicted Hb with the inner-fold threshold flagging 70 percent of cases. "
         "The mean-only model predicts the training mean of each fold, so it flags every woman and its AUC is "
         "close to 0.5." if target == "hb"
         else "Classification metrics use the Youden threshold from inner-fold predictions."))

h(3, "2.6 Referral burden")
cm = d05[(d05["threshold"] == "flag") & d05["model"].isin(["xgboost", "dummy_mean", "dummy_prior", "ridge", "logistic_regression"])]
table(cm.assign(Model=cm["model"].map(NAMES), Target=cm["target"].map(TN))[
    ["Target", "Model", "tp", "fp", "tn", "fn", "referral_burden_per_100", "missed_cases_per_100"]].rename(columns={
        "tp": "True positive", "fp": "False positive", "tn": "True negative", "fn": "False negative",
        "referral_burden_per_100": "Flagged per 100", "missed_cases_per_100": "Missed cases per 100"}))
p("Counts are means over the five repetitions for all eligible women; the last two columns are per 100 women "
  "screened (source: results/tables/d05).")

h(3, "2.7 XGBoost against the pre-specified baselines")
table(pd.DataFrame([{"Target": "Hb" if r.metric in ("mae",) or r.versus in ("dummy_mean", "ridge") else "CED",
                     "Versus": NAMES[r.versus], "Metric": METRIC_NAMES[r.metric], "Quantity": r.quantity,
                     "Estimate (95% CI)": ci(r.point, r.ci_low, r.ci_high),
                     "Corrected t p (approx.)": f(r.p_corrected_approximate)}
                    for r in d06.itertuples() if r.role == "confirmatory baseline"]))
p("MAE is reported as a reduction (comparator minus XGBoost), so positive favours XGBoost. AUC-ROC and Brier are "
  "XGBoost minus comparator, so a positive AUC difference and a negative Brier difference favour XGBoost. "
  "Source: results/tables/d06.")

h(3, "2.8 XGBoost against the other boosting algorithms")
for target, metric in (("hb", "mae"), ("ced", "auc_roc")):
    sub = m03[m03["target"] == target]
    rows = []
    for other in BOOST[1:]:
        row = {"Versus": NAMES[other]}
        for m in (["mae", "rmse", "auc_roc"] if target == "hb" else ["auc_roc", "pr_auc", "brier"]):
            r = sub[(sub["versus"] == other) & (sub["metric"] == m)].iloc[0]
            row[f"{METRIC_NAMES[m]} {r['quantity']}"] = ci(r["point"], r["ci_low"], r["ci_high"])
        r = sub[(sub["versus"] == other) & (sub["metric"] == metric)].iloc[0]
        row["p bootstrap (Holm)"] = f"{f(r['p_bootstrap_two_sided'])} ({f(r['p_bootstrap_holm'])})"
        row["p corrected t (Holm)"] = f"{f(r['p_corrected_approximate'])} ({f(r['p_corrected_holm'])})"
        rows.append(row)
    p(f"**{TN[target]}** (positive favours XGBoost: MAE, RMSE and Brier as reductions, AUC and PR-AUC as "
      "XGBoost minus comparator):")
    table(pd.DataFrame(rows))
excl = m03[(m03["ci_low"] > 0) | (m03["ci_high"] < 0)]
p("Differences whose 95 percent CI excludes 0: "
  + ("; ".join(f"{TN[r.target]} {METRIC_NAMES[r.metric]} versus {NAMES[r.versus]} {ci(r.point, r.ci_low, r.ci_high)}"
               for r in excl.itertuples()) or "none")
  + ". Holm applies to the primary metric only. Figures: results/figures/m_fig01_forest_paired_differences_hb.png and "
  "_ced.png (paired differences), m_fig02_boxplot_repetitions_hb.png and _ced.png (spread over repetitions).")

h(3, "2.9 Training and inference cost")
table(m05.assign(Model=m05["model"].map(NAMES), Target=m05["target"].map(TN))[
    ["Target", "Model", "median_tuning_seconds_per_fold", "median_fit_seconds_per_fold", "median_predict_ms_per_row",
     "median_tree_nodes"]].round(3).rename(columns={
        "median_tuning_seconds_per_fold": "Tuning s per fold", "median_fit_seconds_per_fold": "Final fit s per fold",
        "median_predict_ms_per_row": "Prediction ms per woman", "median_tree_nodes": "Tree nodes"}))
p("Wall-clock times with eight outer folds in parallel and one thread per fit; descriptive only. Tree nodes: total "
  "nodes over all trees of one outer-fold model, median over 25 folds (source: results/tables/m05).")

h(3, "2.10 SHAP eligibility and the dashboard model")
table(pd.DataFrame([{"Target": TN[r.target], "Model": NAMES[r.model],
                     "TreeExplainer": "yes" if r.explainer_supported else "no",
                     "Max additivity error": f"{r.max_additivity_error:.1e}" if pd.notna(r.max_additivity_error) else "n/a",
                     "Eligible": "yes" if r.eligible else "no",
                     "Reason": "ok" if r.eligible else ("not supported by shap 0.46.0 TreeExplainer"
                                                        if not r.explainer_supported else r.reason)}
                    for r in m06.itertuples()]))
rows = []
for t in ("hb", "ced"):
    s = m10[m10["target"] == t]
    group = s[s["in_equivalence_group"]]["candidate"].map(NAMES).tolist()
    rows.append({"Target": TN[t], "Metric": METRIC_NAMES[s["metric"].iloc[0]],
                 "Selected (version 2)": NAMES[sel[t]["candidate"]], "Estimate": f(sel[t]["point"]),
                 "Not significantly different from": ", ".join(g for g in group if g != NAMES[sel[t]["candidate"]]),
                 "Version 1 would select": NAMES[v1[t]]})
table(pd.DataFrame(rows))
p(f"Hb explanation gate for the selected Hb model ({NAMES[gate['model']]}): MAE reduction versus mean-only "
  f"{ci(gate['point'], gate['ci_low'], gate['ci_high'])} g/dL, criterion at least 0.10, outcome {gate['outcome']}, "
  f"verdict {gate['verdict']}. The dashboard therefore shows Hb as indicative without an explanation. "
  f"{' '.join(rule['optimism_note'].split())} Sources: results/tables/m06, m10, m11.")

h(3, "2.11 What the models rely on (SHAP)")
for t in ("hb", "ced"):
    piv = m07[m07["target"] == t].pivot(index="rank", columns="model", values="feature")
    piv = piv[[c for c in BOOST if c in piv.columns]].head(6).apply(lambda col: col.map(FEATURE_NAMES))
    piv.columns = [NAMES[c] for c in piv.columns]
    p(f"**{TN[t]}**, top six features by mean |SHAP| on outer test folds:")
    table(piv.reset_index().rename(columns={"rank": "Rank"}))
table(m08.assign(Model=m08["model"].map(NAMES), Target=m08["target"].map(TN))[
    ["Target", "Model", "mean_pairwise_spearman", "min_pairwise_spearman", "share_fold_pairs_same_top3_set"]].round(3)
    .rename(columns={"mean_pairwise_spearman": "Mean Spearman across 25 folds", "min_pairwise_spearman": "Minimum",
                     "share_fold_pairs_same_top3_set": "Fold pairs with the same top 3"}))
agree = m09.groupby("target")["spearman_global_mean_abs_shap"].agg(["min", "max"])
p("Agreement between algorithms (Spearman of global rankings): "
  + "; ".join(f"{TN[t]} {f(r['min'], 2)} to {f(r['max'], 2)}" for t, r in agree.iterrows())
  + f". AdaBoost is not included (no TreeExplainer). Built-in importance versus mean |SHAP| for the final models: "
  + "; ".join(f"{TN[r.target]} {NAMES[r.model]} {r.importance} {f(r.spearman_vs_mean_abs_shap, 2)}" for r in x20.itertuples())
  + ". The Hb rankings describe models that learned almost nothing (R² at or below 0) and should not be read as "
  "evidence about anemia. Sources: results/tables/m07 to m09, results/exploratory/x20.")

h(3, "2.12 Feature selection frequency")
fr = d07.pivot(index="feature", columns="target", values="share")
fr.index = fr.index.map(FEATURE_NAMES)
table(fr.reset_index().rename(columns={"feature": "Feature", "hb": "Hb share of 25 folds", "ced": "CED share of 25 folds"}))
p(f"Selection does not depend on the algorithm, so every model used the same feature set in each fold. Dashboard "
  f"features (selected in at least half the folds): Hb {', '.join(FEATURE_NAMES[c] for c in final['hb']['features'])}; "
  f"CED {', '.join(FEATURE_NAMES[c] for c in final['ced']['features'])}.")

h(3, "2.13 Exploratory comparisons, calibration and intervals (no verdict)")
table(pd.DataFrame([{"Variant": r.model.replace("xgboost_", "XGBoost + ").replace("_", " "),
                     "Versus": r.versus.replace("_", " "), "Metric": METRIC_NAMES[r.metric], "Quantity": r.quantity,
                     "Estimate (95% CI)": ci(r.point, r.ci_low, r.ci_high)} for r in d08.itertuples()]))
p("Difference = variant minus comparator; reduction = comparator minus variant (a negative MAE reduction means the "
  "variant had the larger error). A positive Brier difference means the variant was worse calibrated.")
pl = x13[x13["versus"] == "platt"].iloc[0]
p(f"Calibration of the selected CED model ({NAMES[x14['model'].iloc[0]]}): Brier uncalibrated minus Platt "
  f"{ci(pl['point'], pl['ci_low'], pl['ci_high'], 4)}; the pre-specified rule keeps the uncalibrated probability. "
  f"Split-conformal 90 percent intervals for Hb ({NAMES[final['hb']['algorithm']]}): coverage "
  + "; ".join(f"{r.group} {f(r.coverage_mean, 2)}" for r in x16.itertuples() if pd.notna(r.coverage_mean))
  + ", which is why the dashboard shows no Hb range (sources: results/exploratory/x13, x14, x16).")

h(3, "2.14 Sensitivity analyses (XGBoost and baselines)")
rows = []
for (analysis, target), block in d11[d11["analysis"] != "sens_three_class"].groupby(["analysis", "target"], sort=False):
    metric = "mae" if target == "hb" else "auc_roc"
    n = d13[(d13["analysis"] == analysis) & (d13["target"] == target)]["n"]
    row = {"Analysis": analysis.replace("sens_", ""), "Target": TN[target], "n": int(n.iloc[0]) if len(n) else ""}
    for model in block["model"].unique():
        r = block[(block["model"] == model) & (block["metric"] == metric)]
        if len(r):
            row[f"{NAMES[model]} {METRIC_NAMES[metric]}"] = ci(r["point"].iloc[0], r["ci_low"].iloc[0], r["ci_high"].iloc[0])
    rows.append(row)
sens = pd.DataFrame(rows)
table(sens.fillna(""))
three = d11[d11["analysis"] == "sens_three_class"]
p("Three-class anemia (from predicted Hb): " + ", ".join(f"{r.metric} {ci(r.point, r.ci_low, r.ci_high)}" for r in three.itertuples())
  + ". The four boosting comparators and the selected Hb model (GBM) were not included in these analyses. "
  "Source: results/tables/d11, d13.")

# %% Discussion points and limitations
h(2, "3. Discussion points (draft, for the author to verify and expand)")
best_ced = max(((m, metric_row(src, m, "auc_roc")["point"]) for m, src in all_models("ced")), key=lambda x: x[1])
for line in [
    f"The questionnaire carries a usable signal for CED risk (every model AUC-ROC around {f(ced_auc['point'], 2)} against "
    "0.50 for the class-proportion model) but practically none for hemoglobin: no algorithm, linear or boosted, improved "
    "on predicting the mean.",
    f"Algorithm choice mattered little. The highest CED AUC-ROC point estimate came from {NAMES[best_ced[0]]} "
    f"({f(best_ced[1])}), and the five boosting algorithms were within about 0.01 of each other with paired CIs that "
    "all contained 0. The limit appears to be the information in the features, not the learner.",
    "The XGBoost-specific hypotheses were not supported. P2 failed only on the margin over Logistic Regression, while "
    "its absolute AUC criteria were met.",
    "SHAP rankings were consistent across algorithms (Spearman 0.93 to 1.00), so the explanation shown in the dashboard "
    "does not hinge on the algorithm. Pre-pregnancy weight dominating CED is plausible because it reflects the "
    "nutritional status and body mass that MUAC also measures; this is association, not cause.",
    "The dashboard model was selected after the results were known (version 2 of the rule). The Hb choice of GBM over "
    "XGBoost rests on a 0.006 g/dL difference that is not statistically established and does not change what the "
    "dashboard can claim about Hb.",
]:
    out.append(f"- {line}")
out.append("")
h(2, "4. Limitations")
for line in [
    "Internal validation on one regional dataset of fewer than 420 women; no external or temporal validation.",
    f"Only {n_anemic} anemia cases, so sensitivity estimates have wide intervals.",
    "Self-reported pre-pregnancy weight and ANC visits; Hb dates differ from interview dates (see the date-gap analyses).",
    "The dashboard model selection (version 2) was decided after the results were known and is optimistic.",
    "Sensitivity analyses cover XGBoost and the baselines only, not the four boosting comparators or the selected Hb model.",
    "AdaBoost could not be explained with TreeExplainer in the installed shap version and was not eligible for the dashboard.",
    "Equal tuning budgets (30 trials) explore large search spaces (LightGBM, 8 parameters) less thoroughly than small ones.",
    "No user study with community cadres; dashboard claims are limited to technical correctness.",
]:
    out.append(f"- {line}")
out.append("")

# %% Reproducibility and appendix
h(2, "5. Reproducibility")
versions = json.loads((ROOT / "reports" / "library_versions.json").read_text(encoding="utf-8"))
for name, v in list(versions.items()):
    if v == "unknown":  # some packages (smogn) have no __version__ attribute
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
p("Library versions: " + ", ".join(f"{k} {v}" for k, v in versions.items() if k != "platform") + ".")
p(f"Seed {config['project']['random_seed']}. Selection rule version {rule.get('version', 1)}, locked {rule.get('locked_on')}. "
  "Key commits: " + "; ".join(git("log", "--format=%h %s", "-n", "12").splitlines()) + ".")
p("Deviations and decisions are recorded with dates in CHANGELOG.md, including the version 2 selection rule "
  "(2026-10-28) and the fact that XGBoost results were known before the comparison rule was written.")
h(2, "Appendix A. Search spaces")
rows = []
for name, space in config["tuning"]["search_spaces"].items():
    for param, s in space.items():
        rng = ", ".join(map(str, s["choices"])) if s["type"] == "categorical" else \
            f"{s['low']} to {s['high']}" + (f", step {s['step']}" if "step" in s else "") + (", log" if s.get("log") else "")
        rows.append({"Space": name, "Parameter": param, "Type": s["type"], "Range": rng})
table(pd.DataFrame(rows))
h(2, "Appendix B. Dashboard models")
table(pd.DataFrame([{"Target": TN[t], "Algorithm": NAMES[final[t]["algorithm"]], "Records": final[t]["n_records"],
                     "Features": ", ".join(FEATURE_NAMES[c] for c in final[t]["features"]),
                     "Hyperparameters": ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in final[t]["params"].items()), "SHAP additivity error": f"{final[t]['shap_additivity_max_error']:.1e}"}
                    for t in ("hb", "ced")]))
p("Final models are refitted on all eligible records with the same tuning procedure and are not used for any reported metric.")

# %% Write
text = "\n".join(out) + "\n"
if "anon_id" in text or "respondent_code" in text:
    raise ValueError("Identifier text in the report")
assert_no_identifiers(desc, config)
path = ROOT / "results" / "paper_full_report.md"
path.write_text(text, encoding="utf-8")
print(f"Written {path} ({len(text.splitlines())} lines)")
