"""Resamplers used as imblearn pipeline steps (training folds only).

imblearn calls ``fit_resample`` during ``Pipeline.fit`` and skips the step
during ``predict``, ``predict_proba`` and ``score``. Validation folds, the
hold-out test set and dashboard inputs are therefore never resampled. Any
statistic a resampler needs (scaler means, feature ranges, relevance fences)
is computed inside ``fit_resample`` from the training fold it receives.

``InFoldSMOTENC``
    Wraps imblearn SMOTENC for the CED classifier. Distances are computed on
    standardised continuous columns (SMOTE is distance based, and an
    unscaled height in cm would dominate a 0 to 5 frequency code). The output
    is returned on the original scale, integer columns are rounded and
    categorical columns keep their original categories.

``SMOGNResampler``
    SMOGN for the Hb regressor (Branco, Torgo and Ribeiro 2017). The
    ``native`` backend is a seeded implementation that records whether each
    synthetic row came from SMOTER interpolation or from Gaussian noise. The
    ``smogn`` package does not expose that, and it draws from the global
    random state, which makes seeding per fold fragile. The ``smogn`` backend
    is kept for comparison.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from scipy.interpolate import PchipInterpolator
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


def _round_and_clip(
    X: pd.DataFrame, reference: pd.DataFrame, integer_columns: Sequence[str], skip: Sequence[str] = ()
) -> pd.DataFrame:
    """Clip synthetic values to the training range and round integer columns."""
    out = X.copy()
    for col in out.columns:
        if col in skip:
            continue
        out[col] = out[col].clip(reference[col].min(), reference[col].max())
        if col in integer_columns:
            out[col] = out[col].round()
    return out


# ---------------------------------------------------------------------------
# SMOTENC for CED
# ---------------------------------------------------------------------------

class InFoldSMOTENC(BaseEstimator):
    """SMOTENC with in-fold scaling, rounding of integer columns, and a report.

    The categorical column positions are resolved from the column names of
    the frame that reaches this step, because in-fold feature selection can
    change which columns are present. If none of the categorical columns
    survived selection, plain SMOTE is the correct algorithm and is used. If
    only categorical columns remain, interpolation is impossible and random
    over-sampling is used instead. The report records which path ran.
    """

    def __init__(
        self,
        categorical_columns: Sequence[str] = (),
        integer_columns: Sequence[str] = (),
        k_neighbors: int = 5,
        sampling_strategy: Any = "auto",
        random_state: int | None = None,
    ) -> None:
        self.categorical_columns = categorical_columns
        self.integer_columns = integer_columns
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X: pd.DataFrame, y: Any) -> tuple[pd.DataFrame, Any]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("InFoldSMOTENC expects a DataFrame")
        y_series = pd.Series(np.asarray(y), name="y")
        cats = [c for c in X.columns if c in set(self.categorical_columns)]
        cont = [c for c in X.columns if c not in cats]
        minority = int(y_series.value_counts().min())
        k = max(1, min(self.k_neighbors, minority - 1))

        scaler = StandardScaler().fit(X[cont]) if cont else None
        Xs = X.reset_index(drop=True).copy()
        if scaler is not None:
            Xs[cont] = scaler.transform(Xs[cont])

        if cats and cont:
            sampler: Any = SMOTENC(
                categorical_features=[Xs.columns.get_loc(c) for c in cats],
                sampling_strategy=self.sampling_strategy,
                k_neighbors=k,
                random_state=self.random_state,
            )
            method = "SMOTENC"
        elif cont:
            sampler = SMOTE(sampling_strategy=self.sampling_strategy, k_neighbors=k, random_state=self.random_state)
            method = "SMOTE (no categorical column present)"
        else:
            sampler = RandomOverSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
            method = "RandomOverSampler (only categorical columns present)"

        Xr, yr = sampler.fit_resample(Xs, y_series)
        Xr = pd.DataFrame(Xr, columns=X.columns)
        if scaler is not None:
            Xr[cont] = scaler.inverse_transform(Xr[cont])
        Xr = _round_and_clip(Xr, X, self.integer_columns, skip=cats)

        counts_before = y_series.value_counts().to_dict()
        counts_after = pd.Series(yr).value_counts().to_dict()
        self.report_ = {
            "method": method,
            "k_neighbors_used": k,
            "categorical_columns_used": cats,
            "n_in": int(len(X)),
            "n_out": int(len(Xr)),
            "n_synthetic": int(len(Xr) - len(X)),
            "class_counts_before": {str(k_): int(v) for k_, v in counts_before.items()},
            "class_counts_after": {str(k_): int(v) for k_, v in counts_after.items()},
        }
        return Xr, np.asarray(yr)


# ---------------------------------------------------------------------------
# SMOGN for Hb
# ---------------------------------------------------------------------------

def relevance_manual(y: np.ndarray, control_points: Sequence[Sequence[float]]) -> np.ndarray:
    """Relevance from user control points joined by a monotone PCHIP curve.

    Outside the control points the relevance is held at the nearest end
    value. For Hb with points (10, 1) and (11, 0), every value at or below
    10 g/dL is fully relevant and every value at or above 11 has relevance 0.
    """
    pts = np.asarray(control_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("control_points must be at least two (value, relevance) pairs")
    order = np.argsort(pts[:, 0])
    xs, phis = pts[order, 0], pts[order, 1]
    if np.any(np.diff(xs) <= 0):
        raise ValueError("control point values must be distinct")
    curve = PchipInterpolator(xs, phis, extrapolate=False)
    out = curve(np.clip(y, xs[0], xs[-1]))
    return np.clip(out, 0.0, 1.0)


def relevance_auto(y: np.ndarray, extreme_type: str = "both", coef: float = 1.5) -> tuple[np.ndarray, list[list[float]]]:
    """Relevance from boxplot fences of ``y`` (Ribeiro 2011 style).

    Called inside ``fit_resample`` so the fences come from the training fold.
    Returns the relevance and the control points used, for the report.
    """
    q1, median, q3 = np.percentile(y, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - coef * iqr, q3 + coef * iqr
    points: list[list[float]] = []
    if extreme_type in ("low", "both"):
        points.append([lo, 1.0])
    points.append([median, 0.0])
    if extreme_type in ("high", "both"):
        points.append([hi, 1.0])
    if extreme_type not in ("low", "high", "both"):
        raise ValueError("auto_extreme_type must be low, high or both")
    if len(points) < 2 or len({p[0] for p in points}) < len(points):
        return np.zeros_like(y, dtype=float), points
    return relevance_manual(y, points), points


def _bumps(sorted_rare: np.ndarray) -> list[tuple[int, int, bool]]:
    """Split a sorted relevance mask into consecutive runs (start, end, is_rare)."""
    runs: list[tuple[int, int, bool]] = []
    start = 0
    for i in range(1, len(sorted_rare) + 1):
        if i == len(sorted_rare) or sorted_rare[i] != sorted_rare[start]:
            runs.append((start, i, bool(sorted_rare[start])))
            start = i
    return runs


def _heom(a: np.ndarray, B: np.ndarray, num_mask: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Heterogeneous Euclidean-overlap distance from row ``a`` to each row of ``B``.

    Numeric columns use absolute difference over the training range. Nominal
    columns use 0 when equal and 1 otherwise. This is the metric SMOGN uses.
    """
    diff = np.empty_like(B, dtype=float)
    diff[:, num_mask] = np.abs(B[:, num_mask] - a[num_mask]) / ranges[num_mask]
    diff[:, ~num_mask] = (B[:, ~num_mask] != a[~num_mask]).astype(float)
    return np.sqrt((diff**2).sum(axis=1))


class SMOGNResampler(BaseEstimator):
    """SMOGN over- and under-sampling for a continuous target.

    Algorithm (Branco, Torgo and Ribeiro 2017), ``balance`` strategy:

    1. Relevance ``phi(y)`` from control points or boxplot fences.
       Rows with ``phi >= relevance_threshold`` are rare.
    2. Sort by ``y`` and split into consecutive rare and normal bumps.
       Each bump's target size is ``n / number_of_bumps``.
    3. Normal bumps above the target size are randomly under-sampled
       (when ``undersample_normal`` is true).
    4. Rare bumps below the target size get synthetic rows. For each new row
       a seed is taken from the bump and one of its ``k`` nearest neighbours
       (HEOM distance, within the bump) is drawn. If that neighbour is
       closer than half the median distance from the seed to all rows of the bump, the
       row is a SMOTER interpolation. Otherwise it is the seed plus Gaussian
       noise scaled by ``perturbation`` times the bump's standard deviation.

    After ``fit_resample`` the attribute ``report_`` holds the counts,
    including ``n_interpolation`` and ``n_gaussian_noise``.
    """

    def __init__(
        self,
        categorical_columns: Sequence[str] = (),
        integer_columns: Sequence[str] = (),
        k_neighbors: int = 5,
        relevance_method: str = "manual",
        control_points: Sequence[Sequence[float]] = ((10.0, 1.0), (11.0, 0.0)),
        auto_extreme_type: str = "low",
        relevance_threshold: float = 0.5,
        perturbation: float = 0.02,
        undersample_normal: bool = True,
        backend: str = "native",
        random_state: int | None = None,
    ) -> None:
        self.categorical_columns = categorical_columns
        self.integer_columns = integer_columns
        self.k_neighbors = k_neighbors
        self.relevance_method = relevance_method
        self.control_points = control_points
        self.auto_extreme_type = auto_extreme_type
        self.relevance_threshold = relevance_threshold
        self.perturbation = perturbation
        self.undersample_normal = undersample_normal
        self.backend = backend
        self.random_state = random_state

    # -- relevance ---------------------------------------------------------
    def _relevance(self, y: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
        if self.relevance_method == "manual":
            return relevance_manual(y, self.control_points), [list(map(float, p)) for p in self.control_points]
        if self.relevance_method == "auto":
            return relevance_auto(y, self.auto_extreme_type)
        raise ValueError(f"Unknown relevance_method '{self.relevance_method}'")

    # -- entry point ---------------------------------------------------------
    def fit_resample(self, X: pd.DataFrame, y: Any) -> tuple[pd.DataFrame, np.ndarray]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SMOGNResampler expects a DataFrame")
        X = X.reset_index(drop=True)
        y_arr = np.asarray(y, dtype=float)
        if self.backend == "native":
            return self._native(X, y_arr)
        if self.backend == "smogn":
            return self._smogn_package(X, y_arr)
        raise ValueError(f"Unknown SMOGN backend '{self.backend}'")

    # -- native implementation ----------------------------------------------
    def _native(self, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        phi, points = self._relevance(y)
        rare = phi >= self.relevance_threshold
        n = len(y)
        report: dict[str, Any] = {
            "backend": "native",
            "control_points": points,
            "n_in": n,
            "n_rare_in": int(rare.sum()),
        }
        if rare.all() or not rare.any():
            report.update(n_out=n, n_synthetic=0, n_interpolation=0, n_gaussian_noise=0,
                          n_removed_by_undersampling=0, note="no rare and normal split, data unchanged")
            report.update(share_interpolation=np.nan, share_gaussian_noise=np.nan)
            self.report_ = report
            return X.copy(), y.copy()

        cats = [c for c in X.columns if c in set(self.categorical_columns)]
        num_mask = np.array([c not in cats for c in X.columns])
        values = X.to_numpy(dtype=float)
        ranges = np.ptp(values, axis=0)
        ranges[ranges == 0] = 1.0

        order = np.argsort(y, kind="mergesort")
        bumps = _bumps(rare[order])
        target = int(round(n / len(bumps)))

        keep_idx: list[np.ndarray] = []
        synth_X: list[np.ndarray] = []
        synth_y: list[float] = []
        n_interp = n_noise = n_removed = 0

        for start, end, is_rare in bumps:
            idx = order[start:end]
            if not is_rare:
                if self.undersample_normal and len(idx) > target:
                    chosen = rng.choice(idx, size=target, replace=False)
                    n_removed += len(idx) - target
                    keep_idx.append(np.sort(chosen))
                else:
                    keep_idx.append(idx)
                continue
            keep_idx.append(idx)
            n_new = target - len(idx)
            if n_new <= 0:
                continue
            new_X, new_y, k_interp, k_noise = self._generate(values[idx], y[idx], n_new, num_mask, ranges, rng)
            synth_X.append(new_X)
            synth_y.extend(new_y)
            n_interp += k_interp
            n_noise += k_noise

        kept = np.concatenate(keep_idx)
        X_out = X.iloc[kept].reset_index(drop=True)
        y_out = y[kept]
        if synth_X:
            synthetic = pd.DataFrame(np.vstack(synth_X), columns=X.columns)
            synthetic = _round_and_clip(synthetic, X, self.integer_columns)
            X_out = pd.concat([X_out, synthetic], ignore_index=True)
            y_out = np.concatenate([y_out, np.clip(synth_y, y.min(), y.max())])
            X_out = X_out.astype(X.dtypes.to_dict())

        n_synth = n_interp + n_noise
        report.update(
            n_out=int(len(y_out)),
            n_synthetic=int(n_synth),
            n_interpolation=int(n_interp),
            n_gaussian_noise=int(n_noise),
            n_removed_by_undersampling=int(n_removed),
            n_bumps=len(bumps),
            share_interpolation=round(n_interp / n_synth, 4) if n_synth else np.nan,
            share_gaussian_noise=round(n_noise / n_synth, 4) if n_synth else np.nan,
        )
        self.report_ = report
        return X_out, y_out

    def _generate(
        self,
        Xb: np.ndarray,
        yb: np.ndarray,
        n_new: int,
        num_mask: np.ndarray,
        ranges: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[float], int, int]:
        """Create ``n_new`` synthetic rows from one rare bump."""
        size = len(yb)
        k = min(self.k_neighbors, size - 1)
        sd_x = Xb.std(axis=0, ddof=1) if size > 1 else np.zeros(Xb.shape[1])
        sd_x = np.nan_to_num(sd_x)
        sd_y = float(np.nan_to_num(yb.std(ddof=1))) if size > 1 else 0.0

        neighbours: list[np.ndarray] = []
        safe: list[float] = []
        dists: list[np.ndarray] = []
        for i in range(size):
            d = _heom(Xb[i], Xb, num_mask, ranges)
            # Safe distance as in the reference smogn code: half the median
            # distance from the seed to every row of the bump (self included).
            safe.append(float(np.median(d)) / 2)
            d[i] = np.inf
            nn = np.argsort(d, kind="mergesort")[:k] if k > 0 else np.array([], dtype=int)
            neighbours.append(nn)
            dists.append(d)

        seeds = np.resize(rng.permutation(size), n_new)
        rows: list[np.ndarray] = []
        targets: list[float] = []
        n_interp = n_noise = 0
        for s in seeds:
            seed_x, seed_y = Xb[s], yb[s]
            if k > 0:
                j = int(rng.choice(neighbours[s]))
                use_interp = dists[s][j] < safe[s]
            else:
                use_interp = False

            if use_interp:
                nb_x, nb_y = Xb[j], yb[j]
                new = seed_x.copy()
                u = rng.random(num_mask.sum())
                new[num_mask] = seed_x[num_mask] + u * (nb_x[num_mask] - seed_x[num_mask])
                pick = rng.random((~num_mask).sum()) < 0.5
                new[~num_mask] = np.where(pick, seed_x[~num_mask], nb_x[~num_mask])
                d1 = float(_heom(new, seed_x[None, :], num_mask, ranges)[0])
                d2 = float(_heom(new, nb_x[None, :], num_mask, ranges)[0])
                new_y = (seed_y + nb_y) / 2 if d1 + d2 == 0 else (d2 * seed_y + d1 * nb_y) / (d1 + d2)
                n_interp += 1
            else:
                new = seed_x.copy()
                new[num_mask] = seed_x[num_mask] + rng.normal(0.0, self.perturbation * sd_x[num_mask] + 0.0)
                flip = rng.random((~num_mask).sum()) < self.perturbation
                if flip.any():
                    donors = Xb[rng.integers(0, size, flip.sum()), :][:, ~num_mask]
                    cat_vals = new[~num_mask]
                    cat_vals[flip] = donors[np.arange(flip.sum()), np.flatnonzero(flip)]
                    new[~num_mask] = cat_vals
                new_y = seed_y + rng.normal(0.0, self.perturbation * sd_y) if sd_y > 0 else seed_y
                n_noise += 1
            rows.append(new)
            targets.append(float(new_y))
        return np.vstack(rows), targets, n_interp, n_noise

    # -- smogn package backend ------------------------------------------------
    def _smogn_package(self, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        """Run the ``smogn`` package for comparison. It cannot report sample origin."""
        import smogn

        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        data = X.copy()
        data["__target__"] = y
        kwargs: dict[str, Any] = dict(
            data=data, y="__target__", k=self.k_neighbors, pert=self.perturbation,
            samp_method="balance", under_samp=self.undersample_normal, rel_thres=self.relevance_threshold,
        )
        if self.relevance_method == "manual":
            kwargs.update(rel_method="manual", rel_ctrl_pts_rg=[[float(a), float(b), 0.0] for a, b in self.control_points])
        else:
            kwargs.update(rel_method="auto", rel_xtrm_type=self.auto_extreme_type)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = smogn.smoter(**kwargs)
        y_out = out.pop("__target__").to_numpy(dtype=float)
        X_out = _round_and_clip(out[X.columns].reset_index(drop=True), X, self.integer_columns)
        self.report_ = {
            "backend": "smogn",
            "n_in": int(len(X)),
            "n_out": int(len(X_out)),
            "n_synthetic": None,
            "n_interpolation": None,
            "n_gaussian_noise": None,
            "share_interpolation": np.nan,
            "share_gaussian_noise": np.nan,
            "note": "smogn package does not expose the origin of synthetic rows",
        }
        return X_out, y_out


# ---------------------------------------------------------------------------
# Factories from config
# ---------------------------------------------------------------------------

def make_smotenc(config: dict[str, Any]) -> InFoldSMOTENC:
    """Build the CED resampler from config."""
    cfg = config["resampling"]["smotenc"]
    return InFoldSMOTENC(
        categorical_columns=tuple(cfg["categorical_columns"]),
        integer_columns=tuple(config["preprocessing"]["integer_columns"]),
        k_neighbors=int(cfg["k_neighbors"]),
        sampling_strategy=cfg["sampling_strategy"],
        random_state=int(config["project"]["random_seed"]),
    )


def make_smogn(config: dict[str, Any]) -> SMOGNResampler:
    """Build the Hb resampler from config."""
    cfg = config["resampling"]["smogn"]
    return SMOGNResampler(
        categorical_columns=tuple(cfg["categorical_columns"]),
        integer_columns=tuple(config["preprocessing"]["integer_columns"]),
        k_neighbors=int(cfg["k_neighbors"]),
        relevance_method=str(cfg["relevance_method"]),
        control_points=tuple(tuple(p) for p in cfg["control_points"]),
        auto_extreme_type=str(cfg["auto_extreme_type"]),
        relevance_threshold=float(cfg["relevance_threshold"]),
        perturbation=float(cfg["perturbation"]),
        undersample_normal=bool(cfg["undersample_normal"]),
        backend=str(cfg["backend"]),
        random_state=int(config["project"]["random_seed"]),
    )
