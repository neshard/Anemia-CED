# %% [markdown]
# # 10 Calibration of CED probabilities and Hb interval coverage (Phase D)
# Exploratory, pre-specified in proposal 2.8 and 3.5. No verdict. No model is refitted.
# Inputs: the locked out-of-fold predictions in results/oof. Outputs go to
# results/exploratory with the prefix x. results/tables is never modified.
#
# CED: Platt and isotonic recalibration, cross-fitted over the outer folds of each
# repetition (a calibrator never sees the rows it is scored on). Brier score and AUC with
# paired bootstrap CIs, calibration slope and intercept, reliability bins.
# The dashboard rule in config/explain.yaml was fixed before these results existed.
#
# Hb: empirical coverage and width of the split-conformal intervals, overall and by
# anemia status.
#
# Run from the project root:  python experiments/10_calibration_and_intervals.py

# %%
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from _common import ROOT, start  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from src.postmodel.calibration import calibration_slope_intercept, cross_fitted, reliability_bins  # noqa: E402
from src.protocol_stats import paired_bootstrap, paired_table, summary_table  # noqa: E402

config, hyp = start("10_calibration_and_intervals")
explain_cfg = yaml.safe_load((ROOT / "config" / "explain.yaml").read_text(encoding="utf-8"))
LABEL = explain_cfg["label"]
OUT = ROOT / "results" / "exploratory"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
B, SEED = int(hyp["n_bootstrap"]), int(config["project"]["random_seed"])
written: list[str] = []
# The selected algorithm per target (14_select_final_model.py), XGBoost until a selection exists.
_sel_path = ROOT / "artifacts" / "model_selection.json"
_sel = json.loads(_sel_path.read_text(encoding="utf-8")) if _sel_path.exists() else {}
ALG = {t: _sel.get(t, {}).get("algorithm", "xgboost") for t in ("hb", "ced")}
print(f"Calibration and intervals for the selected models: {ALG}")


def save(df: pd.DataFrame, name: str) -> None:
    if "anon_id" in df.columns:
        raise ValueError("Identifier column in an output table")
    path = OUT / f"{name}_exploratory.csv"
    df.insert(0, "label", LABEL)
    df.to_csv(path, index=False)
    written.append(str(path.relative_to(ROOT)))


# %% CED: cross-fitted recalibration of the selected model's main out-of-fold probabilities
oof = pd.read_csv(ROOT / "results" / "oof" / f"main__ced__{ALG['ced']}__main.csv")
lr = pd.read_csv(ROOT / "results" / "oof" / "main__ced__logistic_regression__main.csv")
frames = {"uncalibrated": oof[["anon_id", "repeat", "fold", "y_true", "proba"]].copy(),
          "logistic_regression_reference": lr[["anon_id", "repeat", "fold", "y_true", "proba"]].copy()}
for method in explain_cfg["calibration"]["methods"]:
    parts = []
    for _, block in oof.groupby("repeat"):
        block = block.copy()
        block["proba"] = cross_fitted(block, method)
        parts.append(block[["anon_id", "repeat", "fold", "y_true", "proba"]])
    frames[method] = pd.concat(parts, ignore_index=True)


def cal_metrics(d: pd.DataFrame) -> dict[str, float]:
    y, p = d["y_true"].to_numpy(int), d["proba"].to_numpy(float)
    return {"brier": float(brier_score_loss(y, p)), "auc_roc": float(roc_auc_score(y, p))}


rb = paired_bootstrap(frames, cal_metrics, B, SEED + 700)
summary = summary_table(rb, list(frames), ["brier", "auc_roc"], 0.95)
slopes = []
for name, d in frames.items():
    vals = [calibration_slope_intercept(b["y_true"].to_numpy(int), b["proba"].to_numpy(float)) for _, b in d.groupby("repeat")]
    slopes.append({"model": name, "calibration_slope": np.mean([v[0] for v in vals]),
                   "slope_min": np.min([v[0] for v in vals]), "slope_max": np.max([v[0] for v in vals]),
                   "calibration_intercept": np.mean([v[1] for v in vals]),
                   "mean_predicted": float(d["proba"].mean()), "observed_rate": float(d["y_true"].mean())})
save(summary.round(4), "x11_ced_calibration_metrics")
save(pd.DataFrame(slopes).round(4), "x12_ced_calibration_slope_intercept")

# Positive difference means the calibrated version has the LOWER (better) Brier score.
diff = paired_table(rb, "uncalibrated", [m for m in explain_cfg["calibration"]["methods"]], "brier", "difference", 0.95)
save(diff.round(5), "x13_ced_brier_difference_uncalibrated_minus_calibrated")
platt = diff[diff["versus"] == "platt"].iloc[0]
use_platt = bool(platt["ci_low"] > 0)
decision = pd.DataFrame([{
    "rule": explain_cfg["calibration"]["dashboard_rule"],
    "brier_uncalibrated_minus_platt": platt["point"], "ci_low": platt["ci_low"], "ci_high": platt["ci_high"],
    "dashboard_uses": "platt-recalibrated probability" if use_platt else "uncalibrated probability",
    "model": ALG["ced"],
}])
if use_platt:
    print("NOTE: the rule selects Platt recalibration, which the dashboard does not apply yet. Ask the researcher.")
save(decision.round(5), "x14_ced_dashboard_calibration_decision")

bins = []
for name, d in frames.items():
    per_rep = [reliability_bins(b["y_true"], b["proba"], int(explain_cfg["calibration"]["n_bins"]),
                                explain_cfg["calibration"]["bin_strategy"]).assign(repeat=r) for r, b in d.groupby("repeat")]
    avg = pd.concat(per_rep).groupby("bin")[["mean_predicted", "observed_rate", "n"]].mean().reset_index()
    bins.append(avg.assign(model=name))
bins = pd.concat(bins, ignore_index=True)
save(bins.round(4), "x15_ced_reliability_bins")

fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.plot([0, 1], [0, 1], color="#898781", lw=1, ls="--", label="perfect calibration")
styles = {"uncalibrated": ("#3987e5", "o"), "platt": ("#e34948", "s"), "isotonic": ("#52514e", "^")}
for name, (color, marker) in styles.items():
    b = bins[bins["model"] == name]
    ax.plot(b["mean_predicted"], b["observed_rate"], color=color, marker=marker, ms=5, lw=2, label=name)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted probability (decile bins)")
ax.set_ylabel("Observed CED risk rate")
ax.set_title("CED reliability, out-of-fold, mean of 5 repetitions", loc="left", fontsize=10)
ax.grid(color="#e1e0d9", lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "x_fig01_ced_reliability.png", dpi=300, facecolor="white")
plt.close(fig)
written.append("results/exploratory/figures/x_fig01_ced_reliability.png")

# %% Hb: split-conformal interval coverage and width
hb = pd.read_csv(ROOT / "results" / "oof" / f"main__hb__{ALG['hb']}__main.csv")
rows = []
for group, mask in (("all", hb["y_true"].notna()), ("anemic (Hb < 11)", hb["y_binary"] == 1),
                    ("not anemic", hb["y_binary"] == 0)):
    per_rep = []
    for _, b in hb[mask].groupby("repeat"):
        inside = (b["y_true"] >= b["interval_lo"]) & (b["y_true"] <= b["interval_hi"])
        per_rep.append((inside.mean(), (b["interval_hi"] - b["interval_lo"]).mean(), len(b)))
    cov = [r[0] for r in per_rep]
    rows.append({"group": group, "n": per_rep[0][2], "coverage_mean": np.mean(cov), "coverage_min": np.min(cov),
                 "coverage_max": np.max(cov), "mean_width_g_dl": np.mean([r[1] for r in per_rep]),
                 "nominal": config["intervals"]["nominal"]})
final = json.loads((ROOT / "artifacts" / "final_models.json").read_text(encoding="utf-8"))
rows.append({"group": "final model (dashboard)", "n": final["hb"]["n_records"], "coverage_mean": np.nan,
             "coverage_min": np.nan, "coverage_max": np.nan, "mean_width_g_dl": 2 * final["hb"]["interval_half_width"],
             "nominal": config["intervals"]["nominal"]})
save(pd.DataFrame(rows).round(4), "x16_hb_interval_coverage")

print(summary.round(4).to_string(index=False))
print(pd.DataFrame(slopes).round(3).to_string(index=False))
print(diff[["versus", "point", "ci_low", "ci_high"]].round(5).to_string(index=False))
print(decision.to_string(index=False))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("Written:", *written, sep="\n  ")
