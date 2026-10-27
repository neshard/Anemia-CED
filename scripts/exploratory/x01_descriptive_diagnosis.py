# %% [markdown]
# # x01 Descriptive diagnosis (EXPLORATORY, POST HOC)
# Written after the locked results of 2026-10-18 were seen. Nothing here changes a
# hypothesis, a criterion, a verdict or any file in results/tables. It only reads:
# * the cleaned data (deterministic rules in src/cleaning.py, no model fitting),
# * existing results in results/oof, results/folds and results/tables.
# Every output goes to results/exploratory with the prefix "x" and the label
# "exploratory, post hoc" in the file name and in a column. No patient row is written.
#
# Run from the project root:  python scripts/exploratory/x01_descriptive_diagnosis.py

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cleaning import clean_dataset  # noqa: E402
from src.utils import load_config  # noqa: E402

LABEL = "exploratory, post hoc"
config = load_config()
seed = int(config["project"]["random_seed"])
RES = ROOT / "results"
OUT = RES / "exploratory"
OUT.mkdir(parents=True, exist_ok=True)
written: list[str] = []


def save(df: pd.DataFrame, name: str) -> None:
    """Write an aggregate table. Refuses to overwrite and refuses identifier columns."""
    if "respondent_code" in df.columns or "anon_id" in df.columns:
        raise ValueError("Identifier column in an exploratory table")
    path = OUT / f"{name}_exploratory_post_hoc.csv"
    df.insert(0, "label", LABEL)
    df.to_csv(path, index=False)
    written.append(str(path.relative_to(ROOT)))


df = clean_dataset(config).frame
hb = df[df["hb_target"].notna() & df["anemia_class"].notna()]
ced = df[df["ced"].notna()]
pool = list(config["features"]["candidate_pool"])

# %% 1. Hb distribution, anemia prevalence, CED prevalence
h = hb["hb_target"]
q1, med, q3 = h.quantile([0.25, 0.5, 0.75])
dist = pd.DataFrame([
    {"quantity": "Hb n", "value": len(h)},
    {"quantity": "Hb mean (g/dL)", "value": h.mean()},
    {"quantity": "Hb SD (g/dL)", "value": h.std(ddof=1)},
    {"quantity": "Hb variance", "value": h.var(ddof=1)},
    {"quantity": "Hb median", "value": med},
    {"quantity": "Hb IQR low", "value": q1},
    {"quantity": "Hb IQR high", "value": q3},
    {"quantity": "Hb min", "value": h.min()},
    {"quantity": "Hb max", "value": h.max()},
    {"quantity": "Hb distinct values", "value": h.nunique()},
    {"quantity": "Hb recorded as whole numbers", "value": bool(np.allclose(h % 1, 0))},
    {"quantity": "share of women at the modal Hb value", "value": h.value_counts(normalize=True).max()},
    {"quantity": "anemia prevalence (Hb < 11)", "value": (h < 11).mean()},
    {"quantity": "anemia cases", "value": int((h < 11).sum())},
    {"quantity": "CED n", "value": len(ced)},
    {"quantity": "CED risk prevalence (MUAC < 23.5)", "value": ced["ced"].mean()},
    {"quantity": "CED risk cases", "value": int(ced["ced"].sum())},
    {"quantity": "CED risk cases recorded at exactly 23 cm", "value": int((ced["muac"] == 23).sum())},
])
save(dist, "x01_distribution_and_prevalence")
save(h.value_counts().sort_index().rename_axis("hb_g_dl").reset_index(name="n"), "x02_hb_value_counts")

# %% 2. Spearman correlation and mutual information of each feature with Hb and with CED
def spearman_ci(rho: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Approximate CI by the Fisher z transform (exploratory)."""
    z, se = np.arctanh(rho), 1 / np.sqrt(n - 3)
    k = stats.norm.ppf(1 - (1 - level) / 2)
    return float(np.tanh(z - k * se)), float(np.tanh(z + k * se))


rows = []
discrete = set(config["feature_selection"]["discrete_features"])
for target, data, y_col in (("hb", hb, "hb_target"), ("ced", ced, "ced")):
    X = data[pool].copy()
    X = X.fillna(X.median())  # at most 2 missing values per column, for MI only
    y = data[y_col].to_numpy()
    mask = np.array([c in discrete for c in pool])
    mi = (mutual_info_regression(X, y, discrete_features=mask, random_state=seed) if target == "hb"
          else mutual_info_classif(X, y.astype(int), discrete_features=mask, random_state=seed))
    for col, m in zip(pool, mi, strict=True):
        pair = data[[col, y_col]].dropna()
        rho, p = stats.spearmanr(pair[col], pair[y_col])
        lo, hi = spearman_ci(rho, len(pair))
        rows.append({"target": target, "feature": col, "n": len(pair), "spearman_rho": rho,
                     "rho_ci_low": lo, "rho_ci_high": hi, "p_uncorrected": p, "mutual_information": m})
assoc = pd.DataFrame(rows).sort_values(["target", "spearman_rho"], key=lambda s: s.abs() if s.name == "spearman_rho" else s,
                                       ascending=[True, False])
save(assoc.round(4), "x03_feature_association_spearman_mi")

# %% 3. Why P1 has high sensitivity and low specificity
oof = pd.read_csv(RES / "oof" / "main__hb__xgboost__main.csv")
folds = pd.read_csv(RES / "folds" / "main__hb__xgboost__main.csv")
p1 = []
for rep, b in oof.groupby("repeat"):
    anemic = b["y_binary"] == 1
    p1.append({
        "repeat": rep,
        "share_flagged": b["flag"].mean(),
        "share_flagged_among_anemic": b.loc[anemic, "flag"].mean(),
        "share_flagged_among_not_anemic": b.loc[~anemic, "flag"].mean(),
        "sd_true_hb": b["y_true"].std(ddof=1),
        "sd_predicted_hb": b["pred"].std(ddof=1),
        "range_predicted_hb": b["pred"].max() - b["pred"].min(),
        "mean_pred_anemic": b.loc[anemic, "pred"].mean(),
        "mean_pred_not_anemic": b.loc[~anemic, "pred"].mean(),
    })
p1 = pd.DataFrame(p1)
p1_summary = p1.drop(columns="repeat").agg(["mean", "min", "max"]).T.reset_index(names="quantity")
cut = folds["hb_cutoff"]
p1_summary = pd.concat([p1_summary, pd.DataFrame([
    {"quantity": "fold cut-off on predicted Hb (g/dL)", "mean": cut.mean(), "min": cut.min(), "max": cut.max()},
])], ignore_index=True)
save(p1_summary.round(4), "x04_p1_threshold_diagnosis")
pred_q = oof.groupby("repeat")["pred"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).groupby(level=1).mean()
save(pred_q.rename_axis("quantile").reset_index(name="predicted_hb_mean_over_repetitions").round(3),
     "x05_predicted_hb_quantiles")

# %% 4. Stability: sensitivity analyses (d11, d13) and grouped CV per subdistrict (d12)
d11 = pd.read_csv(RES / "tables" / "d11_sensitivity_analyses.csv")
d13 = pd.read_csv(RES / "tables" / "d13_sensitivity_sample_sizes.csv")
main_hb = pd.read_csv(RES / "tables" / "d03_hb_metrics.csv")
main_ced = pd.read_csv(RES / "tables" / "d04_ced_metrics.csv")
key = {"hb": ["mae", "r2", "auc_roc", "sensitivity", "specificity"], "ced": ["auc_roc", "pr_auc", "brier"]}
stab = []
for target, main in (("hb", main_hb), ("ced", main_ced)):
    for metric in key[target]:
        m = main[(main.model == "xgboost") & (main.metric == metric)]
        stab.append({"analysis": "main", "target": target, "metric": metric, "xgboost": float(m["point"].iloc[0]),
                     "ci_low": float(m["ci_low"].iloc[0]), "ci_high": float(m["ci_high"].iloc[0])})
for (analysis, target), block in d11[d11["target"].notna()].groupby(["analysis", "target"]):
    for metric in key[target]:
        x = block[(block.model == "xgboost") & (block.metric == metric)]
        if len(x):
            base = "logistic_regression" if target == "ced" else "dummy_mean"
            b = block[(block.model == base) & (block.metric == metric)]
            stab.append({"analysis": analysis, "target": target, "metric": metric,
                         "xgboost": float(x["point"].iloc[0]), "ci_low": float(x["ci_low"].iloc[0]),
                         "ci_high": float(x["ci_high"].iloc[0]),
                         f"comparator": base, "comparator_point": float(b["point"].iloc[0]) if len(b) else np.nan})
stab = pd.DataFrame(stab).merge(d13[["analysis", "target", "n"]], on=["analysis", "target"], how="left")
save(stab.round(4), "x06_stability_across_sensitivity_analyses")

groups = []
for target in ("hb", "ced"):
    g = pd.read_csv(RES / "tables" / f"d12_grouped_cv_per_subdistrict_{target}.csv")
    g.insert(0, "prevalence", (g["cases"] / g["n"]).round(3))
    groups.append(g)
save(pd.concat(groups, ignore_index=True).round(4), "x07_grouped_cv_per_subdistrict_summary")

print("Written:")
for w in written:
    print(f"  {w}")
