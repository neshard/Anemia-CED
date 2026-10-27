# %% [markdown]
# # 11 SHAP analysis (Phase D, proposal 3.6)
# Exploratory, pre-specified. No verdict. results/tables is never modified.
#
# 1. Fold models: each of the 25 outer-fold XGBoost pipelines of the main run is refitted
#    with its stored hyperparameters and checked to reproduce the locked out-of-fold
#    predictions exactly. SHAP is computed on each outer TEST fold, so the global ranking is
#    out-of-sample. Features not selected in a fold contribute 0 in that fold.
# 2. Stability: mean pairwise Spearman correlation of mean |SHAP| across the 25 fold models,
#    and how often a woman keeps the same top-3 features across the 5 repetitions.
# 3. Final models (fit on all eligible records, fixed dashboard list): TreeExplainer,
#    beeswarm, built-in XGBoost importance (gain, cover, weight) against mean |SHAP|
#    (Spearman), and the SHAP additivity check.
# 4. Example explanation for the dashboard on a SYNTHETIC profile (median answers), never a
#    real woman.
# Hb SHAP is reported but marked: the Hb model did not beat the mean-only baseline, so its
# explanation is not shown in the dashboard (proposal 3.6, config/explain.yaml).
#
# Run from the project root:  python experiments/11_shap_analysis.py

# %%
from __future__ import annotations

import itertools
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import yaml  # noqa: E402
from _common import ROOT, start  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.cleaning import clean_dataset  # noqa: E402
from src.models import CED_MODELS, HB_MODELS  # noqa: E402
from src.nested_cv import build_data, make_pipeline_factory, outer_splitter  # noqa: E402
from src.postmodel.explain import (  # noqa: E402
    additivity_error,
    builtin_importance,
    explain_ced_row,
    explain_hb_row,
    shap_table,
    tree_explainer,
)
from src.runs import main_run  # noqa: E402

config, _ = start("11_shap_analysis")
explain_cfg = yaml.safe_load((ROOT / "config" / "explain.yaml").read_text(encoding="utf-8"))
LABEL = explain_cfg["label"]
LABELS = explain_cfg["feature_labels_id"]
OUT = ROOT / "results" / "exploratory"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
POOL = list(config["features"]["candidate_pool"])
TOPK = int(explain_cfg["shap"]["top_k_stability"])
df = clean_dataset(config).frame
final = json.loads((ROOT / "artifacts" / "final_models.json").read_text(encoding="utf-8"))
written: list[str] = []


def save(table: pd.DataFrame, name: str) -> None:
    if "anon_id" in table.columns:
        raise ValueError("Identifier column in an output table")
    path = OUT / f"{name}_exploratory.csv"
    table.insert(0, "label", LABEL)
    table.to_csv(path, index=False)
    written.append(str(path.relative_to(ROOT)))


def savefig(fig, name: str) -> None:  # type: ignore[no-untyped-def]
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(f"results/exploratory/figures/{name}.png")


# %% 1 and 2. Fold models: reproduce, explain out-of-sample, measure stability
global_rows, stability_rows, repro_rows = [], [], []
for target, spec in (("hb", HB_MODELS["xgboost"]), ("ced", CED_MODELS["xgboost"])):
    data = build_data(config, df, target)
    run = main_run(spec, config)
    splits = list(outer_splitter(run, data, config).split(data.X))
    folds = pd.read_csv(ROOT / "results" / "folds" / f"main__{target}__xgboost__main.csv")
    oof = pd.read_csv(ROOT / "results" / "oof" / f"main__{target}__xgboost__main.csv").set_index(["repeat", "anon_id"])
    col = "pred" if target == "hb" else "proba"
    per_woman: list[pd.DataFrame] = []
    fold_vectors: list[np.ndarray] = []
    max_pred_diff = max_add_err = 0.0
    selection_match = True
    for i, (tr, te) in enumerate(splits):
        row = folds.iloc[i]
        assert int(row["repeat"]) == i // 5 and int(row["fold"]) == i % 5
        pipe = make_pipeline_factory(run, config)(json.loads(row["params"])).fit(data.X.iloc[tr], data.y[tr])
        X_te = data.X.iloc[te]
        pred = pipe.predict(X_te) if target == "hb" else pipe.predict_proba(X_te)[:, 1]
        stored = oof.loc[[(int(row["repeat"]), a) for a in X_te.index], col].to_numpy()
        max_pred_diff = max(max_pred_diff, float(np.max(np.abs(pred - stored))))
        selection_match &= list(pipe.named_steps["select"].selected_features_) == row["selected_features"].split("|")
        values, _, _ = shap_table(pipe, X_te)
        max_add_err = max(max_add_err, additivity_error(pipe, X_te))
        full_vals = values.reindex(columns=POOL, fill_value=0.0)
        fold_vectors.append(full_vals.abs().mean().to_numpy())
        per_woman.append(full_vals.assign(repeat=int(row["repeat"])))
    repro_rows.append({"target": target, "fold_models": len(splits), "max_abs_prediction_difference": max_pred_diff,
                       "selected_features_identical": bool(selection_match),
                       "max_shap_additivity_error": max_add_err})

    allw = pd.concat(per_woman)
    by_rep = allw.groupby("repeat")[POOL].apply(lambda b: b.abs().mean())
    g = pd.DataFrame({"feature": POOL, "label_id": [LABELS[f] for f in POOL],
                      "mean_abs_shap": by_rep.mean().to_numpy(), "min_over_repetitions": by_rep.min().to_numpy(),
                      "max_over_repetitions": by_rep.max().to_numpy()})
    g = g.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    g.insert(0, "rank", range(1, len(g) + 1))
    g.insert(0, "target", target)
    global_rows.append(g)

    rhos = [spearmanr(a, b).statistic for a, b in itertools.combinations(fold_vectors, 2)]
    same_topk, same_top1 = [], []
    for _, w in allw.groupby(level=0):
        tops = [tuple(sorted(r[POOL].abs().sort_values(ascending=False).index[:TOPK])) for _, r in w.iterrows()]
        firsts = [r[POOL].abs().idxmax() for _, r in w.iterrows()]
        pairs = list(itertools.combinations(range(len(tops)), 2))
        same_topk.append(np.mean([tops[a] == tops[b] for a, b in pairs]))
        same_top1.append(np.mean([firsts[a] == firsts[b] for a, b in pairs]))
    stability_rows.append({"target": target, "fold_models": len(fold_vectors),
                           "mean_pairwise_spearman_mean_abs_shap": float(np.mean(rhos)),
                           "min_pairwise_spearman": float(np.min(rhos)),
                           f"share_same_top{TOPK}_set_across_repetitions": float(np.mean(same_topk)),
                           "share_same_top1_across_repetitions": float(np.mean(same_top1)),
                           "women": int(len(same_topk))})

global_table = pd.concat(global_rows, ignore_index=True)
global_table["note"] = np.where(global_table["target"] == "hb",
                                "Hb model did not beat the mean-only baseline. Not shown in the dashboard.",
                                "out-of-sample (outer test folds)")
save(global_table.round(4), "x17_shap_global_ranking_out_of_sample")
save(pd.DataFrame(stability_rows).round(4), "x18_shap_stability")
save(pd.DataFrame(repro_rows), "x19_fold_model_reproduction_and_additivity")

for target in ("ced", "hb"):
    g = global_table[global_table["target"] == target].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(g["label_id"], g["mean_abs_shap"], color="#3987e5", edgecolor="#fcfcfb", linewidth=1.5, height=0.75)
    ax.errorbar(g["mean_abs_shap"], g["label_id"], xerr=[g["mean_abs_shap"] - g["min_over_repetitions"],
                g["max_over_repetitions"] - g["mean_abs_shap"]], fmt="none", ecolor="#52514e", lw=1, capsize=2)
    unit = "log-odds" if target == "ced" else "g/dL"
    title = ("CED model: mean |SHAP| on outer test folds" if target == "ced"
             else "Hb model: mean |SHAP| (model did not beat the mean baseline)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel(f"Mean |SHAP| ({unit}). Bars: range over 5 repetitions")
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    savefig(fig, f"x_fig02_shap_global_{target}")

# %% 3. Final models: explainer, beeswarm, built-in importance, additivity
imp_rows, final_rows = [], []
for target in ("ced", "hb"):
    pipe = joblib.load(ROOT / "artifacts" / f"final_{target}_xgboost.joblib")
    data = build_data(config, df, target, analysis="final")
    X = data.X
    values, base, Xt = shap_table(pipe, X)
    mean_abs = values.abs().mean()
    imp = builtin_importance(pipe).join(mean_abs.rename("mean_abs_shap"))
    for kind in ("gain", "cover", "weight"):
        imp_rows.append({"target": target, "importance": kind, "n_features": len(imp),
                         "spearman_vs_mean_abs_shap": float(spearmanr(imp[kind], imp["mean_abs_shap"]).statistic)})
    final_rows.append({"target": target, "n_records": len(X), "features": "|".join(Xt.columns),
                       "base_value": base, "max_additivity_error": additivity_error(pipe, X),
                       "note": "in-sample SHAP of the final model, for the dashboard explainer only"})
    explainer = tree_explainer(pipe)
    joblib.dump(explainer, ROOT / "artifacts" / f"explainer_{target}.joblib")
    written.append(f"artifacts/explainer_{target}.joblib")
    plt.close("all")
    shap.summary_plot(values.to_numpy(), Xt, feature_names=[LABELS.get(c, c) for c in Xt.columns], show=False,
                      plot_size=(7.5, 4.2))
    fig = plt.gcf()
    fig.suptitle(("CED final model" if target == "ced" else "Hb final model (not shown in dashboard)")
                 + ", SHAP per woman, in-sample", x=0.01, ha="left", fontsize=10)
    savefig(fig, f"x_fig03_shap_beeswarm_final_{target}")
save(pd.DataFrame(imp_rows).round(4), "x20_shap_vs_builtin_importance_final")
save(pd.DataFrame(final_rows).round(8), "x21_final_model_explainer_check")

# %% 4. Example explanation on a synthetic median profile (not a real woman)
profile = df[POOL].median(numeric_only=True).round().to_frame().T
profile.index = ["synthetic_median_profile"]
ced_pipe = joblib.load(ROOT / "artifacts" / "final_ced_xgboost.joblib")
hb_pipe = joblib.load(ROOT / "artifacts" / "final_hb_xgboost.joblib")
fig, ced_text, prob = explain_ced_row(ced_pipe, profile, float(final["ced"]["threshold_youden"]), explain_cfg, config)
savefig(fig, "x_fig04_example_waterfall_ced_synthetic_profile")
_, hb_text, hb_pred = explain_hb_row(hb_pipe, profile, float(final["hb"]["interval_half_width"]), explain_cfg)
example = OUT / "x22_example_narrative_synthetic_profile_exploratory.txt"
example.write_text(f"[{LABEL}] Synthetic median profile, not a real woman.\n\nCED:\n{ced_text}\n\nHb:\n{hb_text}\n",
                   encoding="utf-8")
written.append(str(example.relative_to(ROOT)))

print(pd.DataFrame(repro_rows).to_string(index=False))
print(global_table[["target", "rank", "feature", "mean_abs_shap", "min_over_repetitions", "max_over_repetitions"]].round(4).to_string(index=False))
print(pd.DataFrame(stability_rows).round(3).to_string(index=False))
print(pd.DataFrame(imp_rows).round(3).to_string(index=False))
print(pd.DataFrame(final_rows).drop(columns="note").to_string(index=False))
print(example.read_text(encoding="utf-8"))
print("Written:", *written, sep="\n  ")
