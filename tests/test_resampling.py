"""SMOTENC wrapper and SMOGN implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.resampling import InFoldSMOTENC, SMOGNResampler, relevance_auto, relevance_manual


def _clf_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "height": rng.normal(152, 6, n).round(),
            "n_children": rng.integers(0, 4, n).astype(float),
            "egg_freq": rng.integers(0, 6, n).astype(float),
            "food_taboo": rng.choice([0.0, 1.0], n, p=[0.85, 0.15]),
        }
    )
    y = (rng.random(n) < 0.25).astype(int)
    return X, y


def _reg_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    X, _ = _clf_data(n, seed)
    rng = np.random.default_rng(seed + 1)
    y = np.round(rng.normal(11.5, 1.2, n), 1)
    return X, y


# --- SMOTENC ---------------------------------------------------------------

def test_smotenc_balances_and_respects_types() -> None:
    X, y = _clf_data()
    s = InFoldSMOTENC(categorical_columns=["food_taboo"], integer_columns=["n_children", "egg_freq"], random_state=0)
    Xr, yr = s.fit_resample(X, y)
    assert np.bincount(yr)[0] == np.bincount(yr)[1]
    assert set(Xr["food_taboo"].unique()) <= {0.0, 1.0}
    assert (Xr["n_children"] % 1 == 0).all() and (Xr["egg_freq"] % 1 == 0).all()
    assert Xr["height"].between(X["height"].min(), X["height"].max()).all()
    pd.testing.assert_frame_equal(Xr.iloc[: len(X)], X.reset_index(drop=True))  # originals unchanged
    assert s.report_["method"] == "SMOTENC" and s.report_["n_synthetic"] == len(Xr) - len(X)


def test_smotenc_falls_back_when_categorical_not_present() -> None:
    X, y = _clf_data()
    s = InFoldSMOTENC(categorical_columns=["food_taboo"], random_state=0)
    s.fit_resample(X.drop(columns="food_taboo"), y)
    assert s.report_["method"].startswith("SMOTE ")


def test_smotenc_is_reproducible() -> None:
    X, y = _clf_data()
    a, _ = InFoldSMOTENC(["food_taboo"], random_state=3).fit_resample(X, y)
    b, _ = InFoldSMOTENC(["food_taboo"], random_state=3).fit_resample(X, y)
    pd.testing.assert_frame_equal(a, b)


# --- relevance ---------------------------------------------------------------

def test_manual_relevance_shape() -> None:
    phi = relevance_manual(np.array([7.0, 10.0, 10.5, 11.0, 14.0]), [[10, 1], [11, 0]])
    assert phi[0] == 1 and phi[1] == 1 and phi[3] == 0 and phi[4] == 0
    assert 0 < phi[2] < 1


def test_manual_relevance_validates_points() -> None:
    with pytest.raises(ValueError):
        relevance_manual(np.array([1.0]), [[10, 1]])


def test_auto_relevance_low_tail() -> None:
    y = np.concatenate([np.full(50, 12.0), np.linspace(11, 13, 50), [7.0, 7.5]])
    phi, points = relevance_auto(y, "low")
    assert phi[-1] == 1.0 and phi[0] == 0.0 and len(points) == 2


# --- SMOGN ---------------------------------------------------------------------

def test_smogn_native_counts_and_origins() -> None:
    X, y = _reg_data()
    s = SMOGNResampler(categorical_columns=["food_taboo"], integer_columns=["n_children", "egg_freq"], random_state=0)
    Xr, yr = s.fit_resample(X, y)
    rep = s.report_
    assert rep["n_interpolation"] + rep["n_gaussian_noise"] == rep["n_synthetic"] > 0
    assert rep["share_interpolation"] + rep["share_gaussian_noise"] == pytest.approx(1.0)
    assert len(Xr) == len(yr) == rep["n_out"] == rep["n_in"] + rep["n_synthetic"] - rep["n_removed_by_undersampling"]
    rare_before = np.mean(y <= 10.5)
    rare_after = np.mean(yr <= 10.5)
    assert rare_after > rare_before
    assert yr.min() >= y.min() and yr.max() <= y.max()
    assert set(Xr["food_taboo"].unique()) <= {0.0, 1.0}
    assert (Xr["n_children"] % 1 == 0).all()


def test_smogn_native_reproducible() -> None:
    X, y = _reg_data()
    a = SMOGNResampler(random_state=5).fit_resample(X, y)
    b = SMOGNResampler(random_state=5).fit_resample(X, y)
    pd.testing.assert_frame_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_smogn_without_rare_values_returns_input() -> None:
    X, _ = _reg_data()
    y = np.full(len(X), 13.0)
    s = SMOGNResampler(random_state=0)
    Xr, yr = s.fit_resample(X, y)
    assert len(Xr) == len(X) and s.report_["n_synthetic"] == 0


def test_smogn_no_undersampling_keeps_all_originals() -> None:
    X, y = _reg_data()
    s = SMOGNResampler(undersample_normal=False, random_state=0)
    Xr, _ = s.fit_resample(X, y)
    assert s.report_["n_removed_by_undersampling"] == 0 and len(Xr) >= len(X)


def test_smogn_package_backend_runs() -> None:
    pytest.importorskip("smogn")
    X, y = _reg_data(n=120)
    s = SMOGNResampler(backend="smogn", random_state=0)
    Xr, yr = s.fit_resample(X, y)
    assert len(Xr) == len(yr) and not Xr.isna().any().any()
    assert np.isnan(s.report_["share_interpolation"])


def test_smogn_uses_both_generation_branches() -> None:
    # With the reference safe distance (half the median distance to all bump rows),
    # a typical bump produces a real mix of SMOTER interpolation and Gaussian noise.
    X, y = _reg_data(n=400)
    s = SMOGNResampler(random_state=0)
    s.fit_resample(X, y)
    assert 0.05 < s.report_["share_interpolation"] < 0.95
