"""Metrics, cross-validated evaluation, bootstrap intervals, and paired tests.

``cv_evaluate`` is the single CV loop used by tuning, baselines and ablations.
It builds a fresh pipeline per fold, fits it on the training indices only,
asserts that training and validation indices are disjoint, and scores the
validation fold. Because the same splitter object (same seed) is used for
every model, fold k means the same rows for every model, which is what the
paired tests require.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.labels import WHO_CLASSES, who_anemia_class
from src.splits import assert_disjoint

# Direction of each metric: +1 means higher is better, -1 means lower is better.
METRIC_DIRECTION: dict[str, int] = {
    "mae": -1, "rmse": -1, "r2": 1, "mae_hb_below_11": -1,
    "sensitivity": 1, "specificity": 1, "ppv": 1, "npv": 1, "auc_roc": 1, "pr_auc": 1,
    "accuracy": 1, "precision": 1, "recall": 1, "f1": 1, "brier": -1, "average_precision": 1,
    "macro_f1": 1, "macro_precision": 1, "macro_recall": 1, "ovr_auc": 1,
}


def _safe(fn: Callable[[], float]) -> float:
    """Return NaN when a metric is undefined for this fold (for example one class only)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = float(fn())
        return value
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


# ---------------------------------------------------------------------------
# Metric sets
# ---------------------------------------------------------------------------

def screening_metrics(y_true_pos: np.ndarray, pred_pos: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Sensitivity, specificity, PPV, NPV, AUC-ROC and PR-AUC for a binary screen."""
    t = np.asarray(y_true_pos).astype(int)
    p = np.asarray(pred_pos).astype(int)
    tp = int(((t == 1) & (p == 1)).sum())
    tn = int(((t == 0) & (p == 0)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    return {
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "auc_roc": _safe(lambda: roc_auc_score(t, score)),
        "pr_auc": _safe(lambda: average_precision_score(t, score)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, anemia_threshold: float = 11.0) -> dict[str, float]:
    """MAE, RMSE, R2, MAE on anemic cases, and the derived Hb < threshold screen.

    The screening score is the negative predicted Hb, so a lower predicted Hb
    ranks as higher anemia risk when computing AUC.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    low = y_true < anemia_threshold
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": _safe(lambda: r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "mae_hb_below_11": float(mean_absolute_error(y_true[low], y_pred[low])) if low.any() else float("nan"),
    }
    out.update(screening_metrics(low, y_pred < anemia_threshold, -y_pred))
    return out


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, precision, recall, specificity, NPV, F1, AUC-ROC, PR-AUC and Brier score."""
    t = np.asarray(y_true).astype(int)
    pr = np.asarray(proba, dtype=float)
    pred = (pr >= threshold).astype(int)
    screen = screening_metrics(t, pred, pr)
    return {
        "accuracy": float((pred == t).mean()),
        "precision": screen["ppv"],
        "recall": screen["sensitivity"],
        "sensitivity": screen["sensitivity"],
        "specificity": screen["specificity"],
        "ppv": screen["ppv"],
        "npv": screen["npv"],
        "f1": _safe(lambda: f1_score(t, pred, zero_division=0)),
        "auc_roc": screen["auc_roc"],
        "pr_auc": screen["pr_auc"],
        "average_precision": screen["pr_auc"],
        "brier": float(brier_score_loss(t, pr)),
    }


def multiclass_metrics(y_true: np.ndarray, proba: np.ndarray, class_names: Sequence[str]) -> dict[str, float]:
    """Macro metrics and one-vs-rest AUC for the direct WHO classifier.

    It also reports the binary anemia screen implied by the class
    probabilities, where the anemia score is 1 minus P(Normal).
    """
    t = np.asarray(y_true).astype(int)
    pr = np.asarray(proba, dtype=float)
    pred = pr.argmax(axis=1)
    labels = list(range(len(class_names)))
    out = {
        "macro_f1": _safe(lambda: f1_score(t, pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": _safe(lambda: precision_score(t, pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": _safe(lambda: recall_score(t, pred, labels=labels, average="macro", zero_division=0)),
        "ovr_auc": _safe(lambda: roc_auc_score(t, pr, multi_class="ovr", average="macro", labels=labels)),
    }
    normal = list(class_names).index("Normal")
    anemic_true = t != normal
    out.update(screening_metrics(anemic_true, pred != normal, 1.0 - pr[:, normal]))
    return out


def who_metrics_from_hb(true_class: Sequence[str], pred_hb: np.ndarray, merge_severe: bool = False) -> dict[str, Any]:
    """WHO class metrics when classes are derived from predicted Hb.

    A point prediction has no per-class probability, so one-vs-rest AUC is
    not defined for the regressor. Instead the cumulative AUC at each WHO
    cut-off is reported (Hb < 11, < 10, < 7), using the negative predicted Hb
    as the score. That is the AUC a screen at that cut-off would achieve.
    """
    labels = list(WHO_CLASSES)
    truth = pd.Series(list(true_class), dtype=object)
    pred = pd.Series(who_anemia_class(pred_hb), dtype=object)
    if merge_severe:
        truth = truth.replace("Severe", "Moderate+Severe").replace("Moderate", "Moderate+Severe")
        pred = pred.replace("Severe", "Moderate+Severe").replace("Moderate", "Moderate+Severe")
        labels = ["Normal", "Mild", "Moderate+Severe"]
    cm = confusion_matrix(truth, pred, labels=labels)
    out: dict[str, Any] = {
        "labels": labels,
        "confusion_matrix": cm,
        "support": {lab: int((truth == lab).sum()) for lab in labels},
        "macro_f1": _safe(lambda: f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": _safe(lambda: precision_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": _safe(lambda: recall_score(truth, pred, labels=labels, average="macro", zero_division=0)),
    }
    order = {"Severe": 0, "Moderate": 1, "Moderate+Severe": 1, "Mild": 2, "Normal": 3}
    rank = truth.map(order).to_numpy(dtype=float)
    for cut_name, cut_rank in (("below_11", 3), ("below_10", 2), ("below_7", 1)):
        positive = rank < cut_rank
        out[f"cumulative_auc_{cut_name}"] = _safe(lambda positive=positive: roc_auc_score(positive, -np.asarray(pred_hb)))
    return out


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """Per-fold metrics and out-of-fold predictions (positional row index)."""

    folds: pd.DataFrame
    oof: pd.DataFrame

    def mean(self, metric: str) -> float:
        return float(self.folds[metric].mean())


RESAMPLER_KEYS = ("n_synthetic", "n_interpolation", "n_gaussian_noise", "n_removed_by_undersampling",
                  "share_interpolation", "share_gaussian_noise")


def cv_evaluate(
    make_pipeline: Callable[[], Any],
    X: pd.DataFrame,
    y: np.ndarray,
    cv: Any,
    task: str,
    *,
    class_names: Sequence[str] | None = None,
    threshold: float = 0.5,
    anemia_threshold: float = 11.0,
) -> CVResult:
    """Fit a fresh pipeline per training fold and score the matching validation fold."""
    y = np.asarray(y)
    n_splits = int(getattr(cv, "n_splits", cv.get_n_splits()))
    rows: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        assert_disjoint(train_idx, val_idx, f"cv fold {fold}")
        pipe = make_pipeline()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        X_val = X.iloc[val_idx]
        if task == "regression":
            pred = pipe.predict(X_val)
            metrics = regression_metrics(y[val_idx], pred, anemia_threshold)
            oof = pd.DataFrame({"pred": pred})
        elif task == "classification":
            proba = pipe.predict_proba(X_val)[:, 1]
            metrics = binary_metrics(y[val_idx], proba, threshold)
            oof = pd.DataFrame({"proba": proba})
        elif task == "multiclass":
            if class_names is None:
                raise ValueError("multiclass evaluation needs class_names")
            proba = pipe.predict_proba(X_val)
            metrics = multiclass_metrics(y[val_idx], proba, class_names)
            oof = pd.DataFrame(proba, columns=[f"proba_{c}" for c in class_names])
        else:
            raise ValueError(f"Unknown task '{task}'")

        record: dict[str, Any] = {
            "fold": fold, "repeat": fold // n_splits, "n_train": len(train_idx), "n_val": len(val_idx), **metrics,
        }
        steps = getattr(pipe, "named_steps", {})
        if "resample" in steps and hasattr(steps["resample"], "report_"):
            report = steps["resample"].report_
            record.update({k: report.get(k) for k in RESAMPLER_KEYS if k in report})
        rows.append(record)
        oof.insert(0, "y_true", y[val_idx])
        oof.insert(0, "row", val_idx)
        oof.insert(0, "repeat", fold // n_splits)
        oof.insert(0, "fold", fold)
        oof_parts.append(oof)
    return CVResult(folds=pd.DataFrame(rows), oof=pd.concat(oof_parts, ignore_index=True))


def summarize_folds(folds: pd.DataFrame, metrics: Sequence[str]) -> dict[str, float]:
    """Mean and SD across folds for each metric."""
    out: dict[str, float] = {}
    for m in metrics:
        if m in folds.columns:
            out[f"{m}_mean"] = float(folds[m].mean())
            out[f"{m}_sd"] = float(folds[m].std(ddof=1))
    return out


# ---------------------------------------------------------------------------
# Bootstrap intervals on the hold-out test set
# ---------------------------------------------------------------------------

def bootstrap_metrics(
    metric_fn: Callable[[np.ndarray, np.ndarray], Mapping[str, float]],
    y_true: np.ndarray,
    y_score: np.ndarray,
    metrics: Sequence[str],
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Point estimate and percentile bootstrap interval for each metric.

    Resamples test rows with replacement. Replicates where a metric is
    undefined (for example a resample with no positive case) are skipped
    for that metric, and the number of valid replicates is reported.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point = metric_fn(y_true, y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        values = metric_fn(y_true[idx], y_score[idx])
        for m in metrics:
            v = values.get(m, float("nan"))
            if np.isfinite(v):
                draws[m].append(v)
    alpha = (1 - level) / 2
    rows = []
    for m in metrics:
        sample = np.asarray(draws[m])
        rows.append(
            {
                "metric": m,
                "value": point.get(m, float("nan")),
                "ci_low": float(np.quantile(sample, alpha)) if len(sample) else float("nan"),
                "ci_high": float(np.quantile(sample, 1 - alpha)) if len(sample) else float("nan"),
                "n_valid_bootstrap": len(sample),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired statistical comparison on repeated-CV scores
# ---------------------------------------------------------------------------

def corrected_resampled_ttest(diff: np.ndarray, n_train: float, n_test: float, level: float = 0.95) -> dict[str, float]:
    """Nadeau and Bengio (2003) corrected resampled t-test on paired fold differences.

    Folds of repeated CV share training rows, so their scores are correlated
    and the ordinary paired t-test is too liberal. The variance is inflated
    by ``n_test / n_train``.
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    j = len(d)
    if j < 2:
        return {"t": float("nan"), "p": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_pairs": j}
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    se = float(np.sqrt((1 / j + n_test / n_train) * var))
    if se == 0:
        p = 0.0 if mean != 0 else 1.0
        return {"t": float("inf") if mean else 0.0, "p": p, "ci_low": mean, "ci_high": mean, "n_pairs": j}
    t = mean / se
    p = float(2 * stats.t.sf(abs(t), j - 1))
    half = float(stats.t.ppf(1 - (1 - level) / 2, j - 1) * se)
    return {"t": t, "p": p, "ci_low": mean - half, "ci_high": mean + half, "n_pairs": j}


def wilcoxon_test(diff: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p value. NaN when all differences are zero."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return float("nan")
    return float(stats.wilcoxon(d).pvalue)


def cohens_dz(diff: np.ndarray) -> float:
    """Paired effect size: mean difference over SD of differences."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    return float(d.mean() / sd) if sd and np.isfinite(sd) else float("nan")


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p values. NaN entries stay NaN."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out


def paired_comparison(
    reference: pd.DataFrame,
    comparator: pd.DataFrame,
    metric: str,
    level: float = 0.95,
) -> dict[str, float]:
    """Compare two models evaluated on the same folds.

    The difference is oriented so that a positive value means the reference
    model is better, whatever the direction of the metric.
    """
    merged = reference[["fold", metric, "n_train", "n_val"]].merge(
        comparator[["fold", metric]], on="fold", suffixes=("_ref", "_cmp")
    )
    sign = METRIC_DIRECTION[metric]
    diff = sign * (merged[f"{metric}_ref"] - merged[f"{metric}_cmp"]).to_numpy()
    test = corrected_resampled_ttest(diff, merged["n_train"].mean(), merged["n_val"].mean(), level)
    return {
        "metric": metric,
        "reference_mean": float(merged[f"{metric}_ref"].mean()),
        "comparator_mean": float(merged[f"{metric}_cmp"].mean()),
        "mean_improvement": float(np.nanmean(diff)),
        "improvement_ci_low": test["ci_low"],
        "improvement_ci_high": test["ci_high"],
        "t_corrected": test["t"],
        "p_corrected_t": test["p"],
        "p_wilcoxon": wilcoxon_test(diff),
        "cohens_dz": cohens_dz(diff),
        "n_pairs": test["n_pairs"],
    }


# ---------------------------------------------------------------------------
# Honest reporting helpers
# ---------------------------------------------------------------------------

def dummy_closeness_notes(
    task: str, model_scores: Mapping[str, float], dummy_scores: Mapping[str, float], prevalence: float | None,
    cfg: Mapping[str, float],
) -> list[str]:
    """Plain statements flagging when a model is close to the Dummy baseline."""
    notes: list[str] = []
    if task == "regression":
        rel = (dummy_scores["mae"] - model_scores["mae"]) / dummy_scores["mae"]
        pair = f"({model_scores['mae']:.3f} vs {dummy_scores['mae']:.3f} g/dL)"
        if rel < 0:
            notes.append(f"MAE is {-rel:.1%} WORSE than the Dummy mean predictor {pair}. The Hb regressor does "
                         "not beat predicting the average.")
        elif rel < cfg["mae_relative_improvement"]:
            notes.append(f"MAE is only {rel:.1%} better than the Dummy mean predictor {pair}. The Hb regressor "
                         "adds little over predicting the average.")
    auc = model_scores.get("auc_roc", float("nan"))
    if np.isfinite(auc) and auc < 0.5 + cfg["auc_margin"]:
        notes.append(f"AUC-ROC {auc:.3f} is close to chance (0.5).")
    pr = model_scores.get("pr_auc", float("nan"))
    if prevalence is not None and np.isfinite(pr) and pr < prevalence + cfg["pr_auc_margin"]:
        notes.append(f"PR-AUC {pr:.3f} is close to the prevalence {prevalence:.3f}, which is what a random ranking gets.")
    return notes


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Single-hue sequential ramp (light to dark blue). Magnitude only, so one hue.
SEQUENTIAL_BLUE: tuple[str, ...] = ("#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
INK_DARK = "#1f1f1e"
INK_LIGHT = "#ffffff"
INK_MUTED = "#5f5e5a"


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], title: str, normalize: bool = True) -> Any:
    """Confusion matrix heatmap with counts and row percentages in every cell.

    Colour encodes the row share (recall per true class), so classes of very
    different size are comparable. The count is printed so small classes
    such as Severe are never hidden behind a percentage.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0) if normalize else cm
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    size = 1.1 * len(labels) + 1.6
    fig, ax = plt.subplots(figsize=(size + 0.6, size))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1 if normalize else max(cm.max(), 1))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ink = INK_LIGHT if share[i, j] > 0.55 else INK_DARK
            if row_sums[i, 0]:
                text = f"{int(cm[i, j])}\n{share[i, j]:.0%}"
            else:
                text = "no cases" if j == 0 else ""
            ax.text(j, i, text, ha="center", va="center", color=ink, fontsize=9)
    ax.set_xticks(range(len(labels)), labels, color=INK_DARK, fontsize=9)
    ax.set_yticks(range(len(labels)), [f"{lab} (n={int(row_sums[k, 0])})" for k, lab in enumerate(labels)],
                  color=INK_DARK, fontsize=9)
    ax.set_xlabel("Predicted", color=INK_MUTED)
    ax.set_ylabel("True", color=INK_MUTED)
    ax.set_title(title, color=INK_DARK, fontsize=10, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def save_figure(fig: Any, name: str, figures_dir: Any) -> Any:
    """Save a figure as PNG at 300 dpi and close it."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    out = Path(figures_dir) / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Operating point for the Hb screen (hypothesis H3)
# ---------------------------------------------------------------------------

def choose_hb_cutoff(
    y_true: np.ndarray, pred: np.ndarray, anemia_threshold: float, min_sensitivity: float
) -> dict[str, float]:
    """Choose the cut-off on predicted Hb from out-of-fold training predictions.

    A woman screens positive when predicted Hb is below the cut-off. Raising
    the cut-off raises sensitivity and lowers specificity, so the rule picks
    the LOWEST cut-off whose sensitivity reaches ``min_sensitivity``. That is
    the most specific cut-off that still meets the sensitivity target.

    Candidates are the midpoints between consecutive distinct predictions,
    plus one value above the largest prediction (everyone positive), so the
    target is always reachable. Only training out-of-fold predictions are
    used. The hold-out set later receives this fixed cut-off once.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    positive = y < anemia_threshold
    if not positive.any() or positive.all():
        raise ValueError("Out-of-fold data must contain anemic and non-anemic rows")
    values = np.unique(p)
    candidates = np.concatenate([(values[:-1] + values[1:]) / 2, [values[-1] + 1e-6]])
    for cut in candidates:
        flagged = p < cut
        sens = float(flagged[positive].mean())
        if sens >= min_sensitivity:
            spec = float((~flagged[~positive]).mean())
            return {"cutoff": float(cut), "oof_sensitivity": sens, "oof_specificity": spec,
                    "n_oof_predictions": int(len(p)), "min_sensitivity": float(min_sensitivity)}
    raise RuntimeError("No cut-off reached the sensitivity target, which should be impossible")
