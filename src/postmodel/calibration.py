"""Calibration of CED probabilities on existing out-of-fold predictions (proposal 2.8).

Recalibration is evaluated by cross-fitting over the outer folds of each
repetition: the calibrator for fold k is fitted on the out-of-fold
predictions of the other four folds and applied to fold k. A calibrator is
therefore never fitted on the rows it is scored on. No model is refitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_calibrator(proba: np.ndarray, y: np.ndarray, method: str):  # type: ignore[no-untyped-def]
    """Platt scaling (logistic regression on the logit) or isotonic regression."""
    if method == "platt":
        model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(proba).reshape(-1, 1), y)
        return lambda p: model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(proba, y)
        return lambda p: iso.predict(np.asarray(p, dtype=float))
    raise ValueError(f"Unknown calibration method '{method}'")


def cross_fitted(oof_rep: pd.DataFrame, method: str) -> np.ndarray:
    """Calibrated probabilities for one repetition, cross-fitted over its outer folds."""
    out = np.full(len(oof_rep), np.nan)
    folds = oof_rep["fold"].to_numpy()
    p, y = oof_rep["proba"].to_numpy(float), oof_rep["y_true"].to_numpy(int)
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        out[test] = fit_calibrator(p[train], y[train], method)(p[test])
    return out


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration slope (logistic regression of y on logit p) and calibration-in-the-large.

    A slope of 1 and an intercept of 0 mean perfect calibration. The intercept
    is estimated with the slope fixed at 1 through an offset.
    """
    lp = _logit(p)
    slope = float(LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0, 0])
    # Calibration-in-the-large: solve for a so that mean(sigmoid(a + lp)) equals the event rate.
    a, target = 0.0, float(np.mean(y))
    for _ in range(100):
        prob = 1 / (1 + np.exp(-(a + lp)))
        grad = float(np.mean(prob * (1 - prob)))
        step = (float(np.mean(prob)) - target) / max(grad, 1e-9)
        a -= step
        if abs(step) < 1e-10:
            break
    return slope, float(a)


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Mean predicted probability and observed rate per bin."""
    p = np.asarray(p, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    df = pd.DataFrame({"bin": bins, "p": p, "y": np.asarray(y, dtype=float)})
    return df.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index()
