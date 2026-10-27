# %% [markdown]
# # 13 SHAP for the boosting candidates and dashboard eligibility (supervisor revision 2026-10-27)
# For every candidate and target, each of the 25 outer-fold pipelines of the main run is
# refitted with its stored hyperparameters and must reproduce the stored out-of-fold
# predictions. SHAP (shap.TreeExplainer only, never KernelExplainer) is computed on each
# outer TEST fold, so rankings are out-of-sample. Features not selected in a fold count as 0.
#
# * m06: eligibility for the dashboard (config/model_selection.yaml step 1): a valid
#   TreeExplainer and the additivity check passed on all 25 fold models. AdaBoost has no
#   TreeExplainer in the installed shap and is reported as ineligible, with no fallback.
# * m07: global ranking of mean |SHAP| per model.
# * m08: stability of the ranking across the 25 fold models (mean pairwise Spearman).
# * m09: Spearman correlation of the rankings between models, on the global ranking and
#   per fold (same outer test women), excluding models without a TreeExplainer.
# SHAP describes how each model reached its output, not what causes anemia or CED.
#
# Run from the project root:  python experiments/13_shap_boosting.py

# %%
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from _common import ROOT, start
from scipy.stats import spearmanr

from src.cleaning import clean_dataset
from src.model_selection import load_rule
from src.models import CED_MODELS, HB_MODELS
from src.nested_cv import build_data, make_pipeline_factory, outer_splitter
from src.postmodel.explain import ExplainerNotSupportedError, additivity_error, shap_table, tree_explainer
from src.runs import load_folds, load_oof, main_run, save_result_table

config, _ = start("13_shap_boosting")
rule = load_rule(require_locked=not config["project"].get("smoke_test"))
TOL = float(rule["additivity_tolerance"])
CANDS = list(rule["candidates"])
POOL = list(config["features"]["candidate_pool"])
REPRO_TOL = 1e-5  # stored predictions went through CSV, so allow float rounding only
df = clean_dataset(config).frame
written: list[str] = []


def save(table: pd.DataFrame, name: str) -> None:
    written.append(str(save_result_table(table, name, config).relative_to(ROOT)))


# %% Fold models per candidate
elig_rows, global_rows, stab_rows = [], [], []
fold_vectors: dict[tuple[str, str], list[np.ndarray]] = {}
for target, models in (("hb", HB_MODELS), ("ced", CED_MODELS)):
    data = build_data(config, df, target)
    col = "pred" if target == "hb" else "proba"
    for name in CANDS:
        spec = models[name]
        run = main_run(spec, config)
        splits = list(outer_splitter(run, data, config).split(data.X))
        folds = load_folds(config, f"main__{target}__{name}__main")
        oof = load_oof(config, f"main__{target}__{name}__main").set_index(["repeat", "anon_id"])
        vectors: list[np.ndarray] = []
        max_pred_diff = max_add = 0.0
        supported, reason = True, ""
        for i, (tr, te) in enumerate(splits):
            row = folds.iloc[i]
            assert int(row["repeat"]) == i // 5 and int(row["fold"]) == i % 5
            pipe = make_pipeline_factory(run, config)(json.loads(row["params"])).fit(data.X.iloc[tr], data.y[tr])
            X_te = data.X.iloc[te]
            pred = pipe.predict(X_te) if target == "hb" else pipe.predict_proba(X_te)[:, 1]
            stored = oof.loc[[(int(row["repeat"]), a) for a in X_te.index], col].to_numpy()
            max_pred_diff = max(max_pred_diff, float(np.max(np.abs(pred - stored))))
            try:
                explainer = tree_explainer(pipe)
            except ExplainerNotSupportedError as exc:
                supported, reason = False, str(exc)
                break
            values, _, _ = shap_table(pipe, X_te, explainer)
            max_add = max(max_add, additivity_error(pipe, X_te, explainer))
            vectors.append(values.reindex(columns=POOL, fill_value=0.0).abs().mean().to_numpy())
        if max_pred_diff > REPRO_TOL:
            raise SystemExit(f"{target} {name}: refitted fold models do not reproduce the stored predictions "
                             f"(max difference {max_pred_diff:.2e}). Stop and investigate.")
        passed = supported and max_add <= TOL
        elig_rows.append({"target": target, "model": name, "explainer": "shap.TreeExplainer",
                          "explainer_supported": supported,
                          "fold_models_checked": len(vectors) if supported else 0,
                          "max_additivity_error": max_add if supported else np.nan, "tolerance": TOL,
                          "additivity_passed": passed, "max_abs_prediction_difference_vs_stored": max_pred_diff,
                          "eligible": passed,
                          "reason": "ok" if passed else (reason or f"additivity error {max_add:.2e} above {TOL:g}")})
        print(elig_rows[-1])
        if not supported:
            continue
        fold_vectors[(target, name)] = vectors
        mat = np.vstack(vectors)
        g = pd.DataFrame({"target": target, "model": name, "feature": POOL, "mean_abs_shap": mat.mean(axis=0),
                          "share_of_folds_nonzero": (mat > 0).mean(axis=0)})
        g = g.sort_values(["mean_abs_shap", "feature"], ascending=[False, True], ignore_index=True)
        g.insert(2, "rank", range(1, len(g) + 1))
        global_rows.append(g)
        rhos = [spearmanr(a, b).statistic for a, b in itertools.combinations(vectors, 2)]
        tops = [tuple(sorted(np.array(POOL)[np.argsort(-v, kind="stable")[:3]])) for v in vectors]
        stab_rows.append({"target": target, "model": name, "fold_models": len(vectors),
                          "mean_pairwise_spearman": float(np.nanmean(rhos)), "min_pairwise_spearman": float(np.nanmin(rhos)),
                          "share_fold_pairs_same_top3_set": float(np.mean([a == b for a, b in itertools.combinations(tops, 2)]))})

elig = pd.DataFrame(elig_rows)
save(elig, "m06_shap_eligibility")
global_table = pd.concat(global_rows, ignore_index=True)
global_table["note"] = "out-of-sample (outer test folds), describes the model, not causes"
save(global_table, "m07_shap_global_ranking_boosting")
save(pd.DataFrame(stab_rows), "m08_shap_stability_boosting")

# %% Agreement between models
cross = []
for target in ("hb", "ced"):
    names = [n for n in CANDS if (target, n) in fold_vectors]
    for a, b in itertools.combinations(names, 2):
        va, vb = fold_vectors[(target, a)], fold_vectors[(target, b)]
        per_fold = [spearmanr(x, y).statistic for x, y in zip(va, vb, strict=True)]
        cross.append({"target": target, "model_a": a, "model_b": b,
                      "spearman_global_mean_abs_shap": float(spearmanr(np.mean(va, axis=0), np.mean(vb, axis=0)).statistic),
                      "mean_spearman_same_fold": float(np.nanmean(per_fold)),
                      "min_spearman_same_fold": float(np.nanmin(per_fold))})
cross_table = pd.DataFrame(cross)
cross_table["note"] = "AdaBoost excluded: no TreeExplainer in the installed shap (no KernelExplainer fallback)."
save(cross_table, "m09_shap_cross_model_spearman")

print(elig.to_string(index=False))
print(global_table[global_table["rank"] <= 5][["target", "model", "rank", "feature", "mean_abs_shap"]].round(4).to_string(index=False))
print(pd.DataFrame(stab_rows).round(3).to_string(index=False))
print(cross_table.drop(columns="note").round(3).to_string(index=False))
print("Written:", *written, sep="\n  ")
