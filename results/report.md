# Results report (2026-10-27 21:47)

**WARNING: the Hb XGBoost model is not better than the mean-only Dummy (MAE reduction 0.002 g/dL, 95% CI -0.022 to 0.026).**

All results are internal validation on one regional dataset of fewer than 420 women. They show whether the approach is feasible in this dataset. They do not show that it generalizes to other regions or that it is clinically useful.

Reported metrics come only from the outer cross-validation with per-fold feature selection. The final models (script 09) use a fixed feature list (selected in at least half of the 25 outer folds) and are not used for any reported metric.

## Hypotheses

| Hypothesis | Role | Verdict | CI level | Holm order | p-value |
|---|---|---|---|---|---|
| P1_anemia_screening | primary | not_supported | 0.975 | 1 | 0.9484 |
| P2_ced_classification | primary | not_supported | 0.975 | 2 | 0.9995 |
| S1_hb_vs_mean | secondary | not_supported | 0.95 |  | 1.0000 |
| S2_hb_vs_ridge | secondary | not_supported | 0.95 |  | 0.9982 |

| Hypothesis | Component | Criterion | Point | CI | Outcome |
|---|---|---|---|---|---|
| P1_anemia_screening | sensitivity | >= 0.7 (point_and_ci) | 0.722 | 0.610 to 0.828 (97.5%) | inconclusive |
| P1_anemia_screening | specificity | >= 0.4 (point_and_ci) | 0.334 | 0.276 to 0.394 (97.5%) | fail |
| P1_anemia_screening | auc_roc | > 0.5 (ci_lower_bound) | 0.535 | 0.456 to 0.611 (97.5%) | not_met |
| P2_ced_classification | auc_roc | >= 0.7 (point_and_ci) | 0.773 | 0.710 to 0.830 (97.5%) | pass |
| P2_ced_classification | auc_roc | >= 0.5 (ci_lower_bound) | 0.773 | 0.710 to 0.830 (97.5%) | met |
| P2_ced_classification | auc_roc difference vs logistic_regression | >= 0.03 (point_and_ci) | -0.021 | -0.049 to 0.007 (97.5%) | fail |
| S1_hb_vs_mean | mae reduction vs dummy_mean | >= 0.1 (point_and_ci) | 0.002 | -0.022 to 0.026 (95.0%) | fail |
| S2_hb_vs_ridge | mae reduction vs ridge | >= 0.05 (point_and_ci) | 0.005 | -0.017 to 0.027 (95.0%) | fail |

## Dashboard feature lists (selected in at least half of the 25 outer folds)

* hb: age, anc_visits, hfias_score, prepreg_weight, height, education_mother, n_children
* ced: anc_visits, education_mother, hfias_score, prepreg_weight, age, egg_freq, height, n_children

## Files

* results\tables\c00b_canary_too_good.csv
* results\tables\d01_hypothesis_verdicts.csv
* results\tables\d02_hypothesis_components.csv
* results\tables\d03_hb_metrics.csv
* results\tables\d04_ced_metrics.csv
* results\tables\d05_confusion_matrices_mean_over_repetitions.csv
* results\tables\d06_baseline_comparisons.csv
* results\tables\d07_feature_selection_frequency.csv
* results\tables\d08_exploratory_comparisons.csv
* results\tables\d09_conformal_intervals.csv
* results\tables\d10_ced_reliability_curve.csv
* results\tables\d12_grouped_cv_per_subdistrict_hb.csv
* results\tables\d12_grouped_cv_per_subdistrict_ced.csv
* results\tables\d11_sensitivity_analyses.csv
* results\tables\d13_sensitivity_sample_sizes.csv
