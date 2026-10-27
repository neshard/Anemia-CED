"""Prediction intervals for Hb (hypothesis H8).

Method ``oof_conformal``: the half-width is the conformal quantile of the
absolute out-of-fold residuals from ONE CV repeat, so every training row
contributes exactly one residual from a model that did not see it. The
interval is the final model's prediction plus or minus that half-width.

Why out-of-fold residuals: residuals of the final model on its own training
rows are too small (the model has seen them), which would make the interval
too narrow. Out-of-fold residuals behave like residuals on new women. The
finite-sample guarantee of split conformal prediction holds exactly only when
the calibration rows are scored by the same model as the new row. Here the
fold models stand in for the final model, so coverage is close to nominal
but not guaranteed. H8 measures the achieved coverage on the hold-out set
for exactly that reason.

The residuals come from the training set only. The hold-out set is used once,
to measure coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class ConformalInterval:
    """Symmetric interval of fixed half-width around a point prediction."""

    half_width: float
    nominal: float
    n_calibration: int
    method: str = "oof_conformal"

    def predict(self, point: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds for each point prediction."""
        p = np.asarray(point, dtype=float)
        return p - self.half_width, p + self.half_width

    def to_dict(self) -> dict[str, float | int | str]:
        return {"half_width": self.half_width, "nominal": self.nominal,
                "n_calibration": self.n_calibration, "method": self.method}


def conformal_quantile(abs_residuals: npt.ArrayLike, nominal: float) -> float:
    """Finite-sample conformal quantile: the ceil((n + 1) * nominal)-th smallest residual.

    When that rank exceeds n the interval would have to be infinite to keep
    the guarantee, so the largest residual is returned and the caller should
    treat the result as approximate.
    """
    r = np.sort(np.abs(np.asarray(abs_residuals, dtype=float)))
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        raise ValueError("No residuals to calibrate on")
    rank = math.ceil((n + 1) * nominal)
    return float(r[min(rank, n) - 1])


def fit_oof_conformal(oof: pd.DataFrame, nominal: float) -> ConformalInterval:
    """Calibrate from out-of-fold predictions of the first CV repeat.

    ``oof`` is the table written by ``cv_evaluate`` (columns repeat, y_true, pred).
    """
    first = oof[oof["repeat"] == oof["repeat"].min()]
    if first["row"].duplicated().any():
        raise ValueError("A row appears twice in one CV repeat. The OOF table is malformed.")
    residuals = (first["y_true"] - first["pred"]).abs().to_numpy()
    return ConformalInterval(conformal_quantile(residuals, nominal), nominal, len(residuals))


def empirical_coverage(y_true: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike) -> float:
    """Share of true values inside the closed interval."""
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))
