# Changelog

Every change to criteria, threshold rules, feature protocol, model setup or data handling is recorded here with its date and reason. No model had been trained on the real data when any entry below was made, and the earlier 334/84 hold-out split was never opened.

## 2026-10-17 (approved) and 2026-10-18 (implemented)

Reason for all entries: align the code with the thesis proposal (docs/proposal.pdf, dated 2026-05-25), which is the protocol of record. Approved by the researcher on 2026-10-17.

### Hypotheses
- `config/hypotheses.yaml` now holds only the four hypotheses of proposal Table 1.1 (P1, P2, S1, S2) as structured components, plus the exploratory list. The eight-hypothesis file locked on 2026-10-14 is archived in `archive/holdout_design_2026-10-14/`.
- Verdict rule: a component judged on the point estimate and the CI passes if both are on the pass side, fails if both are on the fail side, and is inconclusive otherwise. A criterion on a CI bound (the AUC lower bounds) is met or not met. The AUC of at least 0.70 in P2 uses the point-and-CI rule. A hypothesis is supported if every component passes or is met, not supported if any fails or is not met, and inconclusive otherwise. S1 and S2 use 95 percent CIs.
- Holm across P1 and P2: component p-value = (resamples on the fail side + 1) / (2000 + 1), averaged over the five repetitions. Hypothesis p-value = largest component p-value. The primary hypothesis with the smaller p-value is judged with a 97.5 percent CI. The other is judged with a 95 percent CI if the first is supported, and with a 97.5 percent CI otherwise. Criterion values never change.
- Averaging: each repetition has its own bootstrap with its own seed. Point estimates and CI bounds are averaged over the five repetitions, and the range of the point estimate is reported.
- Locked on 2026-10-17 with the reason "Align with proposal Table 1.1 dated 2026-05-25".

### Evaluation design
- The 80/20 hold-out split and the one-time hold-out claim are retired. Evaluation is repeated stratified 5 x 5 outer cross-validation on all eligible records (396 Hb, 410 CED), with an inner 3-fold CV for tuning and for the decision thresholds (proposal 3.5). The retired split file is archived.
- Metrics are computed on pooled out-of-fold predictions per repetition with percentile bootstrap intervals over women (2,000 resamples). Comparisons resample the same women for both models. The corrected resampled t-test is reported only as an approximate adjustment.

### Models and tuning
- Main models use no resampling. SMOTE-NC, class weighting and SMOGN are exploratory comparisons, each retuned with the same budget.
- Baselines: mean-only and Ridge for Hb, class proportion and Logistic Regression for CED. Decision Tree and Random Forest are exploratory.
- **Protocol choice not specified in the proposal:** 30 Optuna trials (TPE, seeded) for every tuned model (XGBoost, Ridge, Logistic Regression, Decision Tree, Random Forest). Dummy models are not tuned.
- **Protocol choice not specified in the proposal:** tuning objective is MAE for Hb models and ROC AUC for CED models and the direct anemia classifier, because AUC is the primary CED criterion.

### Features
- Per-fold selection (proposal 3.3): mutual information rank and random forest (300 trees) rank are averaged. Pre-pregnancy weight, ANC frequency and HFIAS are always kept, then the 4 best remaining features, exactly 7 per fold. Ties are broken by the lower MI rank, then alphabetically. Selection is refit inside every inner split and cached per split, so it is computed once and reused across trials. The 2026-10-14 inner-5-fold "at least 50 percent" rule is retired.
- Dashboard list: features selected in at least 50 percent of the 25 outer folds. Final models are fit on all eligible records with that list and tuned by the same procedure. They are not used for any reported metric.

### Thresholds
- Hb: cut-off on predicted Hb that flags 70 percent of true anemia cases in the inner-fold predictions of each outer training set, applied unchanged to the outer test fold.
- CED: Youden's index on inner-fold predictions, plus cut-offs reaching 0.80 and 0.90 inner-fold sensitivity.

### Data handling
- Plausibility limits set to age 15 to 50 years, pre-pregnancy weight 30 to 120 kg, height 130 to 190 cm (proposal 3.2). The same four values are flagged as before.
- Sample sizes asserted in code: 420 raw, 419 deduplicated, 396 Hb model, 410 CED model.
- Stratification: anemia class x CED status. A missing component is its own category (NoHb in the CED model, NoMUAC in the Hb model). Any group under 5 records merges into the same-anemia-class group with the opposite CED status. Groups with a missing component merge into the largest group sharing the known component. Applied before any split. Final cell counts are logged.
- Dates for the date-gap sensitivity analysis: format d/m/yyyy, read as m/d/yyyy when the second number exceeds 12. A date is usable only if it is a real date between 2024-01-01 and 2026-01-31. Only the absolute gap in days is kept. Counts differ slightly from the proposal (377 usable of 396 versus 381, median 31 days in both, 192 versus 196 above 30 days, 84 versus 88 above 90 days). They were not forced to match.

### Sensitivity analyses (proposal 3.8 item 3)
- CED cut-offs: at risk when recorded MUAC is 22 or less, and 24 or less, with labels and stratification rebuilt.
- Trimester-specific Hb cut-offs on women with a usable gestational age (bare numbers read as weeks, 38 bare values of 9 or less counted as ambiguous).
- Date-gap exclusions above 90 and above 30 days.
- GroupKFold by subdistrict (five groups, leave one out, one pass).
- All 11 candidate features, and the fixed preliminary set (Hb 6, CED 7 features from Table 3.2), labelled optimistic.
- Three-class anemia analysis from the main Hb predictions.

### Infrastructure
- Checkpoint per outer fold and per repetition, keyed by a hash of the config, the hypotheses file, the raw data and the source code. Timestamped logs and all outputs in results/. Progress shows elapsed time and the estimated time left.

## 2026-10-17

- **Sample sizes asserted.** Reason: pre-flight gate item 2.
- **pypdf installed in the local venv** to read docs/proposal.pdf. It is a reading tool and not a project dependency.

## 2026-10-21 (Phase D, after the locked results)

No hypothesis, criterion, verdict or file in results/tables changed. The main run checkpoint hash (df62f171da4a28ac) is unchanged because the new modules live in src/postmodel/, which the hash does not cover.

- **Exploratory post hoc analyses** (researcher's audit request): descriptive diagnosis (results/exploratory x01 to x07) and a near-duplicate sensitivity analysis removing 2 (strict) or 5 (loose) probable repeated entries (x08 to x10). No conclusion changed.
- **De-identified data copy**: data/raw/dataset_deidentified.csv without name, phone or WhatsApp number, email, enumerator name and GPS columns. Verified to give an identical cleaned analysis frame. The original file is not modified.
- **Phase D settings** in config/explain.yaml, written before any Phase D result. Dashboard calibration rule fixed in advance: use Platt recalibration only if the cross-fitted Brier improvement has a 95 percent CI above 0. Result: CI -0.0036 to 0.0004, so the dashboard keeps the uncalibrated probability.
- **Hb explanation hidden** in the dashboard (proposal 3.6), because S1 was not supported. Hb SHAP is still reported for the thesis and labelled accordingly.
- **SHAP**: the 25 fold models were refitted from their stored parameters and reproduced the locked out-of-fold predictions (largest difference below 1e-6) with identical selected features. Explainers saved to artifacts/explainer_ced.joblib and artifacts/explainer_hb.joblib.

## 2026-10-21 (Phase E, dashboard)

- **Hb display, researcher decision (option C)**: the dashboard shows the Hb estimate only as an indicative point value, without the conformal range, because the 90 percent interval covered only 69 percent of anemic women (results/exploratory/x16). The SQLite hb_lower and hb_upper columns stay empty. Recorded as hb_show_range: false in config/explain.yaml.
- **Dashboard** (app/streamlit_app.py, app/pages/1_Riwayat_sesi.py, app/components): presentation layer only. Logic in src/postmodel/inference.py and recommendations.py, data in src/postmodel/db.py. CED risk is the main result. The Hb explanation is hidden (S1 not supported).
- **Preprocessing replication**: dashboard answers pass through src/cleaning.clean_features and the saved final pipelines. The HFIAS score uses the same item scoring (items_combined) as training. Only the dataset-level "looks binary" format check is skipped, because one food-secure woman answering "Tidak" to every item would wrongly trigger it. A test confirms identical scores and predictions to the offline path.
- **Advice messages** in config/recommendations.yaml are a DRAFT until reviewed by a health worker (proposal 3.6). The dashboard shows a banner until review is recorded.
- **Privacy**: .streamlit/config.toml serves the app on localhost only and disables usage statistics. The database stores no names, phone numbers, coordinates or respondent codes.

## 2026-10-21 (dashboard fix and layout, researcher request)

- **Fix**: ModuleNotFoundError "src" under `streamlit run`. Both pages now put the project root and app/ on sys.path before any project import. The automated tests had missed it because pytest already had the root on sys.path. A new test runs each page in a fresh interpreter without the project root, and it reproduces the original error when the fix is removed.
- **Main page renamed** from app/streamlit_app.py to app/Skrining.py. Run with `streamlit run app/Skrining.py` from the project root.
- **Layout**: two-column form with the nine HFIAS items in an expander, results in tabs (Hasil, Penjelasan, Saran), metrics in bordered containers. **Theme**: teal (primary #0f766e) and toolbarMode minimal in .streamlit/config.toml.
- No change to inference logic, model artifacts, hypotheses or results/tables.

## 2026-10-21 (readable session codes, researcher request)

- **Researcher asked to store the mother's name so the history is not random.** Not done, because proposal 3.7 states that the prototype never stores names. Researcher chose readable automatic codes instead.
- **Session codes**: every saved session gets a code of the local date plus a daily number, for example 20261021-003, shown after saving and in the history page. Cadres write it in the paper register. The link from code to name stays on paper. A counter table guarantees a code is never reused after a deletion. Older databases are migrated by adding the column.
- **Test pollution found and removed**: the database path used to be fixed when the module was first imported, so one test run wrote one test session to the real database data/app/sessions.sqlite. That record (default form profile, created during the test) and its code counter were deleted, leaving the database empty as before. The path is now read at every call, and the test asserts that the real database stays untouched.

## 2026-10-27 (supervisor revision: boosting comparison and dashboard model selection)

Reason: the title is now "Evaluating XGBoost with SHAP ...". The supervisor asked to compare XGBoost with four other boosting algorithms and to use the best performing model in the dashboard.

### Lock of the selection rule (before any new model was run)
- **config/model_selection.yaml locked on 2026-10-27 in commit 0cc4098a5eef38be59dbbf43a647e2dd5b878711** (2026-10-27 23:10:36 +0700). No GBM, LightGBM, CatBoost or AdaBoost model had been trained on the real data at that commit.
- **Results already seen when the rule was written:** every XGBoost, baseline, Decision Tree and Random Forest result had been available since 2026-10-18 (results/report.md, d01 to d13, and the Phase D/E outputs of 2026-10-21). The rule, including the preference for XGBoost inside the equivalence group, was therefore written with knowledge of the XGBoost results, but before any result of the four new models existed.
- **Checks seen before the lock (eligibility only, no performance metric):** on synthetic data, shap.TreeExplainer was built for GBM, LightGBM and CatBoost and the SHAP additivity error was about 1e-15; refits with the same seed were identical. shap 0.46.0 TreeExplainer rejected AdaBoost (InvalidModelError). The XGBoost additivity error of the locked run (up to 1.4e-5, results/exploratory/x19, float32 arithmetic) was used to set the additivity tolerance to 1e-4. None of these checks involved Hb, MUAC or any performance metric.
- Rule content (researcher approved, full text in the file): eligibility first (valid TreeExplainer, no KernelExplainer fallback, additivity passed on all 25 outer-fold models); provisional winner by MAE (Hb) or AUC-ROC (CED) on pooled out-of-fold predictions averaged over the 5 repetitions; equivalence group = candidates whose 95 percent paired bootstrap CI of metric(candidate) minus metric(winner) contains 0, relative to the provisional winner and not transitive; XGBoost if in the group, otherwise the smallest median tree-node count, then candidate order. Fixed bootstrap seeds Hb 42 and CED 142 (the seeds of 08). Decision Tree, Random Forest and baselines are never candidates. If no candidate is eligible, selection stops and no model is chosen.
- Hb explanation gate (proposal 3.6): judged exactly like S1 for the selected Hb model (MAE reduction versus mean-only of at least 0.10 g/dL, point_and_ci, 95 percent, seed 42). Only "supported" enables the Hb SHAP explanation. "inconclusive" and "not_supported" keep the Hb output indicative.
- The selected model is chosen from five candidates on the same data, so its cross-validated performance is optimistic. All five are reported side by side.

### Models and protocol
- New models for both targets: GradientBoostingRegressor/Classifier, LGBMRegressor/Classifier, CatBoostRegressor/Classifier, AdaBoostRegressor/Classifier. Same outer folds (5 x 5), inner 3-fold, preprocessing, per-fold feature set, 30 Optuna TPE trials with fixed seed, threshold rules (70 percent of anemia cases, Youden), no resampling. Search spaces in config/config.yaml tuning.search_spaces.
- **AdaBoostClassifier uses algorithm="SAMME" explicitly**, because SAMME.R (the default in scikit-learn 1.5.2) is deprecated and raises a FutureWarning.
- CatBoost receives the same integer-coded matrix as every model. Its native categorical handling is not used, and that sensitivity analysis is skipped (no nominal features exist). LightGBM: subsample_freq=1 so that subsample takes effect, deterministic=True, one thread. CatBoost: thread_count=1, verbose=0, no files written.
- The four comparators are not added to the sensitivity analyses (07). If a selected model is not XGBoost, the researcher decides whether 07 is run for it.
- Hypotheses P1, P2, S1 and S2 stay about XGBoost. The comparison is descriptive (paired bootstrap CIs, corrected resampled t-test, Holm across the four comparisons per target on MAE or AUC-ROC for reporting only).

### Refactor proven not to change results (commit 16db993)
- run_fold records wall-clock times (descriptive only). lightgbm and catboost added to the tracked library versions. requirements.txt pins lightgbm 4.5.0 and catboost 1.2.7 (plus graphviz 0.21 and plotly 7.1.0 pulled in by catboost), installed with a constraints file: no existing package changed version.
- Scripts 03 to 11 were rerun under the new checkpoint hash. All 146 compared files (out-of-fold predictions, fold tables without the timing columns, results/tables, results/exploratory, artifacts/final_models.json) were identical cell for cell to the pre-refactor backup. The only text difference, x22, came from a backup copy that predated the 2026-10-21 option C narrative change.

### Infrastructure
- The project is now a git repository (first commit e73abad is the state before this revision). Per-woman outputs (results/oof, results/checkpoints, results/logs, the archived hold-out ID list) are git-ignored. core.autocrlf is off so git never changes the bytes that the checkpoint hash reads.

## 2026-10-27/28 (boosting comparison run, after the lock in 0cc4098)

- **Runs**: 04 and 05 trained all nine models per target under the identical protocol (new checkpoint hash). XGBoost, baseline, Decision Tree and Random Forest outputs were again identical cell for cell to the pre-refactor backup (145 tables plus artifacts/final_models.json). Canaries for the four comparators passed (c00c: permuted-label AUC 0.46 to 0.55, R2 at most 0; fit spy clean, 80 fits per model; m00: no result above the too-good limits).
- **Not run**: 07 (sensitivity analyses) was not rerun. 08 uses the 07 out-of-fold files of the 2026-10-27 refactor run, which were proven identical to the backup. The comparators are not part of 07 (researcher decision), recorded as a limitation.
- **Reporting fix in 12 before any result was reported** (not part of the locked rule, no effect on selection): the two-sided bootstrap p-value was first computed as the mean of per-repetition p-values. When repetitions disagreed in sign this gave small p-values (for example 0.02) while the repetition-averaged 95 percent CI contained 0. It now reads the p-value from the quantile-averaged bootstrap distribution, the same averaging as the CI, so p < 0.05 exactly when the CI excludes 0. Table m04 (the selection input) was byte-identical before and after the fix.
- **Eligibility (m06)**: XGBoost, GBM, LightGBM and CatBoost passed (max additivity error 1.4e-5 for XGBoost Hb, below 4e-14 for the others). AdaBoost is ineligible for both targets: shap 0.46.0 TreeExplainer does not support it, and no other explainer was used.
- **Selection (m10, artifacts/model_selection.json)**: Hb: provisional winner GBM (MAE 0.993), equivalence group XGBoost, GBM, LightGBM, CatBoost, so XGBoost is selected by step 4. CED: XGBoost has the best eligible AUC-ROC (0.773) and is selected. The dashboard keeps XGBoost for both targets, so 07 needs no rerun for a winner.
- **Hb explanation gate (m11)**: XGBoost MAE reduction versus mean-only 0.002 g/dL (95 percent CI -0.022 to 0.026), outcome fail, verdict not_supported, identical to S1. The Hb explanation stays hidden and Hb stays indicative.
- **Calibration (10)**: recomputed for the selected CED model (XGBoost) with the explain.yaml rule. Brier difference uncalibrated minus Platt -0.0016 (CI -0.0036 to 0.0004), so the dashboard keeps the uncalibrated probability, unchanged.

## 2026-10-28 (DEVIATION: selection rule version 2, set after the results were known)

- **Researcher instruction**: XGBoost must not be the reference. If another boosting model performs better it goes to the dashboard. Baselines (and Decision Tree, Random Forest) never do.
- **Change**: config/model_selection.yaml version 2, locked 2026-10-28 in commit 4ea49bd. Only step 4 changed: choice best_point_estimate, preferred null. The eligible boosting model with the best point estimate (MAE for Hb, AUC-ROC for CED) is selected. Eligibility, candidates, metrics, bootstrap seeds, additivity tolerance and the Hb gate are unchanged. Version 1 (commit 0cc4098) is still applied side by side in table m10 (column selected_by_version1_rule).
- **This is post hoc**: every comparison result (m01 to m11) was known when version 2 was written, and it was known that version 2 would select GBM for Hb. The selection is therefore data-driven and its performance optimistic. It is reported as such.
- **Outcome**: Hb: GBM (MAE 0.993 versus XGBoost 0.998, difference 0.006 g/dL, 95 percent paired CI containing 0, so not statistically established; version 1 selected XGBoost). CED: XGBoost (AUC-ROC 0.773, best eligible; same as version 1).
- **Hb gate for GBM (m11)**: MAE reduction versus mean-only 0.008 g/dL (95 percent CI -0.020 to 0.035), fail, not_supported. The Hb output stays indicative without explanation. GBM interval coverage among anemic women 0.70 (x16), so hb_show_range stays false.
- **Final models (09)**: Hb GBM tuned on all 396 records with the same procedure (inner 3-fold, 30 trials), same seven dashboard features, SHAP additivity 1e-14. CED unchanged. 10 and 11 rerun for the selected models; the CED calibration decision is unchanged (uncalibrated).
- **07** is not run for GBM (researcher decision of 2026-10-27), recorded as a limitation: the sensitivity analyses cover XGBoost and the baselines only.
- Hypotheses P1, P2, S1 and S2 and their verdicts (d01, d02) are unchanged: they concern XGBoost.
- 297 tests pass. The dashboard loads GBM for Hb and XGBoost for CED from artifacts/final_models.json.

## 2026-10-28 (dashboard follows the version 2 selection)

- **Tentang model panel** (tab Penjelasan): names the algorithm behind each estimate (CED XGBoost, Hb Gradient Boosting Machine), its cross-validated metric, the models it did not differ from significantly ("keunggulannya kecil dan belum terbukti"), and that selection from five candidates on the same data makes the result slightly optimistic. Content comes from artifacts/final_models.json, never hard-coded.
- **Riwayat sesi**: the history table shows the model of each saved session (model_KEK, model_Hb; "tidak tercatat" for sessions saved before algorithm names were stored). A session computed with a different model than the current one is flagged with a prompt to recompute. Recomputing stores the current algorithm names.
- **Bug fix found by the new history test**: the history page crashed as soon as one session existed, because pyarrow 26.0.0 (pulled in by streamlit, not pinned, installed 2026-10-14) refuses NumPy 1.26. It had gone unnoticed because the database was always empty. pyarrow pinned to 18.1.0 in requirements.txt, installed with a constraints file: no other package changed. The modelling pipeline does not use pyarrow.
- Verified: 299 tests pass, including both pages rendered with the real artifacts and a temporary database; `streamlit run app/Skrining.py` serves both pages. The real database stays empty.
