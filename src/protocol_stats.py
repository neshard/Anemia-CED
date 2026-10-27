"""Metrics on pooled out-of-fold predictions and the paired bootstrap (proposal Section 3.8).

For each repetition the five outer test folds are pooled, so every woman has
exactly one out-of-fold prediction. Metrics are computed on that pooled set.
Each repetition gets its own percentile bootstrap over women (2,000
resamples, own seed). A comparison between two models resamples the same
women for both, which gives a paired interval. Point estimates and CI bounds
are then averaged over the five repetitions, and the range of the point
estimate is reported as well.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.labels import who_anemia_class

MetricFn = Callable[[pd.DataFrame], dict[str, float]]


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def _safe_auc(truth: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(truth, score)) if len(np.unique(truth)) == 2 else float("nan")


def _safe_ap(truth: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(truth, score)) if len(np.unique(truth)) == 2 else float("nan")


def confusion(truth: np.ndarray, flag: np.ndarray) -> dict[str, int]:
    t = np.asarray(truth).astype(bool)
    f = np.asarray(flag).astype(bool)
    return {"tp": int((t & f).sum()), "fp": int((~t & f).sum()), "tn": int((~t & ~f).sum()), "fn": int((t & ~f).sum())}


def screening(truth: np.ndarray, flag: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Sensitivity, specificity, PPV, NPV, AUC-ROC, PR-AUC and the referral burden (share flagged)."""
    c = confusion(truth, flag)
    return {
        "sensitivity": _ratio(c["tp"], c["tp"] + c["fn"]),
        "specificity": _ratio(c["tn"], c["tn"] + c["fp"]),
        "ppv": _ratio(c["tp"], c["tp"] + c["fp"]),
        "npv": _ratio(c["tn"], c["tn"] + c["fn"]),
        "auc_roc": _safe_auc(np.asarray(truth).astype(int), score),
        "pr_auc": _safe_ap(np.asarray(truth).astype(int), score),
        "referral_rate": float(np.mean(np.asarray(flag).astype(bool))),
    }


def hb_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Regression error and the anemia screen. The screening score is the negative predicted Hb."""
    y, pred = df["y_true"].to_numpy(float), df["pred"].to_numpy(float)
    out = {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
    }
    out.update(screening(df["y_binary"].to_numpy(), df["flag"].to_numpy(), -pred))
    if "interval_lo" in df.columns:
        inside = (y >= df["interval_lo"].to_numpy()) & (y <= df["interval_hi"].to_numpy())
        out["interval_coverage"] = float(inside.mean())
        out["interval_mean_width"] = float((df["interval_hi"] - df["interval_lo"]).mean())
    return out


def ced_metrics(df: pd.DataFrame, flag_col: str = "flag") -> dict[str, float]:
    """Classification metrics at the given threshold column plus Brier score."""
    y, proba, flag = df["y_true"].to_numpy(int), df["proba"].to_numpy(float), df[flag_col].to_numpy(int)
    out = screening(y, flag, proba)
    out.update({
        "brier": float(brier_score_loss(y, proba)),
        "accuracy": float((flag == y).mean()),
        "f1": float(f1_score(y, flag, zero_division=0)),
    })
    return out


def three_class_metrics(df: pd.DataFrame) -> dict[str, float]:
    """WHO classes from predicted Hb (Normal, Mild, Moderate. No Severe case exists)."""
    labels = ["Normal", "Mild", "Moderate"]
    truth = pd.Series(who_anemia_class(df["y_true"])).replace("Severe", "Moderate")
    pred = pd.Series(who_anemia_class(df["pred"])).replace("Severe", "Moderate")
    return {
        "macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, pred, labels=labels, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Paired bootstrap per repetition
# ---------------------------------------------------------------------------

@dataclass
class RepBootstrap:
    """Point estimates and bootstrap draws per model and repetition."""

    points: dict[tuple[str, int], dict[str, float]]
    draws: dict[tuple[str, int], pd.DataFrame]
    repeats: list[int]
    n_boot: int


def paired_bootstrap(oof_by_model: Mapping[str, pd.DataFrame], metric_fn: MetricFn, n_boot: int, seed: int) -> RepBootstrap:
    """Bootstrap every repetition, resampling the same women for every model.

    Each repetition has its own seed (``seed + repeat``). The women are
    aligned by anonymous ID, so model A and model B always see the same
    resampled women.
    """
    models = list(oof_by_model)
    repeats = sorted(set.intersection(*[set(df["repeat"].unique()) for df in oof_by_model.values()]))
    points: dict[tuple[str, int], dict[str, float]] = {}
    draws: dict[tuple[str, int], pd.DataFrame] = {}
    for rep in repeats:
        frames = {m: oof_by_model[m][oof_by_model[m]["repeat"] == rep].sort_values("anon_id").reset_index(drop=True)
                  for m in models}
        ids = frames[models[0]]["anon_id"].to_numpy()
        for m in models[1:]:
            if not np.array_equal(frames[m]["anon_id"].to_numpy(), ids):
                raise ValueError(f"Models {models[0]} and {m} do not cover the same women in repetition {rep}")
        rng = np.random.default_rng(seed + int(rep))
        idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
        for m in models:
            points[(m, rep)] = metric_fn(frames[m])
            draws[(m, rep)] = pd.DataFrame([metric_fn(frames[m].iloc[row]) for row in idx])
    return RepBootstrap(points, draws, repeats, n_boot)


def quantity_samples(rb: RepBootstrap, model: str, metric: str, versus: str | None = None,
                     quantity: str = "value") -> tuple[list[float], list[np.ndarray]]:
    """Per-repetition point and bootstrap draws of a metric, a difference or a reduction.

    * value: the metric of ``model``.
    * difference: metric(model) minus metric(versus), same resampled women.
    * reduction: metric(versus) minus metric(model), for errors where lower is better.
    """
    points, boots = [], []
    for rep in rb.repeats:
        a_pt, a_bt = rb.points[(model, rep)][metric], rb.draws[(model, rep)][metric].to_numpy()
        if quantity == "value":
            points.append(a_pt)
            boots.append(a_bt)
            continue
        b_pt, b_bt = rb.points[(versus, rep)][metric], rb.draws[(versus, rep)][metric].to_numpy()
        if quantity == "difference":
            points.append(a_pt - b_pt)
            boots.append(a_bt - b_bt)
        elif quantity == "reduction":
            points.append(b_pt - a_pt)
            boots.append(b_bt - a_bt)
        else:
            raise ValueError(f"Unknown quantity '{quantity}'")
    return points, boots


def summarize(points: Sequence[float], boots: Sequence[np.ndarray], level: float) -> dict[str, float]:
    """Average point and CI bounds over repetitions, plus the range of the point estimate."""
    alpha = (1 - level) / 2
    lows = [float(np.nanquantile(b, alpha)) for b in boots]
    highs = [float(np.nanquantile(b, 1 - alpha)) for b in boots]
    return {
        "point": float(np.mean(points)),
        "ci_low": float(np.mean(lows)),
        "ci_high": float(np.mean(highs)),
        "ci_level": level,
        "point_min": float(np.min(points)),
        "point_max": float(np.max(points)),
        "n_repetitions": len(points),
    }


def one_sided_p(boots: Sequence[np.ndarray], criterion: float, op: str) -> float:
    """(resamples on the fail side + 1) / (B + 1), averaged over repetitions.

    For a criterion "value >= c" the fail side is value < c. For "value > c"
    it is value <= c. Undefined draws (NaN) count as failures, which can only
    make the p-value larger.
    """
    ps = []
    for b in boots:
        b = np.asarray(b, dtype=float)
        fail = np.isnan(b) | ((b < criterion) if op == ">=" else (b <= criterion))
        ps.append((int(fail.sum()) + 1) / (len(b) + 1))
    return float(np.mean(ps))


def summary_table(rb: RepBootstrap, models: Sequence[str], metrics: Sequence[str], level: float) -> pd.DataFrame:
    """Every metric of every model with averaged point, CI and range."""
    rows = []
    for m in models:
        for metric in metrics:
            if metric not in rb.points[(m, rb.repeats[0])]:
                continue
            pts, bts = quantity_samples(rb, m, metric)
            rows.append({"model": m, "metric": metric, **summarize(pts, bts, level)})
    return pd.DataFrame(rows)


def paired_table(rb: RepBootstrap, model: str, comparators: Sequence[str], metric: str, quantity: str,
                 level: float) -> pd.DataFrame:
    """Paired differences of ``model`` against each comparator."""
    rows = []
    for c in comparators:
        pts, bts = quantity_samples(rb, model, metric, c, quantity)
        rows.append({"model": model, "versus": c, "metric": metric, "quantity": quantity, **summarize(pts, bts, level)})
    return pd.DataFrame(rows)
