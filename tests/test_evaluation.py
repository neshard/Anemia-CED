"""Metrics, CV loop, bootstrap intervals, and paired statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.evaluation import (
    binary_metrics,
    bootstrap_metrics,
    cohens_dz,
    corrected_resampled_ttest,
    cv_evaluate,
    dummy_closeness_notes,
    holm_adjust,
    multiclass_metrics,
    paired_comparison,
    regression_metrics,
    who_metrics_from_hb,
)
from src.splits import KeyStratifiedRepeatedKFold


def test_regression_screening_uses_low_hb_as_positive() -> None:
    y = np.array([9.0, 10.0, 12.0, 13.0])
    m = regression_metrics(y, y, 11.0)
    assert m["mae"] == 0 and m["sensitivity"] == 1 and m["specificity"] == 1 and m["auc_roc"] == 1
    m = regression_metrics(y, y[::-1], 11.0)
    assert m["auc_roc"] == 0 and m["sensitivity"] == 0


def test_binary_metrics_known_values() -> None:
    y = np.array([1, 1, 0, 0])
    m = binary_metrics(y, np.array([0.9, 0.4, 0.6, 0.1]), 0.5)
    assert m["recall"] == 0.5 and m["specificity"] == 0.5 and m["precision"] == 0.5
    assert m["auc_roc"] == pytest.approx(0.75)


def test_metrics_undefined_return_nan() -> None:
    m = binary_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert np.isnan(m["auc_roc"]) and np.isnan(m["recall"])


def test_multiclass_anemia_score() -> None:
    names = ["Normal", "Mild", "Moderate"]
    proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.7, 0.2, 0.1]])
    m = multiclass_metrics(np.array([0, 1, 2, 0]), proba, names)
    assert m["macro_f1"] == 1.0 and m["auc_roc"] == 1.0 and m["ovr_auc"] == 1.0


def test_who_from_hb_and_three_class_merge() -> None:
    truth = ["Normal", "Mild", "Moderate", "Severe"]
    pred = np.array([12.0, 10.5, 8.0, 6.0])
    four = who_metrics_from_hb(truth, pred)
    assert four["macro_f1"] == 1.0 and four["cumulative_auc_below_7"] == 1.0
    three = who_metrics_from_hb(truth, pred, merge_severe=True)
    assert three["labels"] == ["Normal", "Mild", "Moderate+Severe"] and three["support"]["Moderate+Severe"] == 2


def test_corrected_ttest_is_more_conservative_than_naive() -> None:
    rng = np.random.default_rng(0)
    diff = rng.normal(0.05, 0.1, 25)
    corrected = corrected_resampled_ttest(diff, n_train=240, n_test=60)
    from scipy import stats

    naive = stats.ttest_1samp(diff, 0).pvalue
    assert corrected["p"] > naive
    assert corrected["ci_low"] < diff.mean() < corrected["ci_high"]


def test_corrected_ttest_zero_variance() -> None:
    assert corrected_resampled_ttest(np.zeros(10), 80, 20)["p"] == 1.0


def test_holm_adjustment() -> None:
    adj = holm_adjust([0.01, 0.04, 0.03, np.nan])
    np.testing.assert_allclose(adj[:3], [0.03, 0.06, 0.06])
    assert np.isnan(adj[3])


def test_paired_comparison_orientation() -> None:
    ref = pd.DataFrame({"fold": range(10), "mae": np.full(10, 1.0), "n_train": 80, "n_val": 20})
    worse = pd.DataFrame({"fold": range(10), "mae": np.linspace(1.2, 1.4, 10)})
    res = paired_comparison(ref, worse, "mae")
    assert res["mean_improvement"] > 0  # lower MAE for the reference counts as an improvement
    ref_auc = pd.DataFrame({"fold": range(10), "auc_roc": np.full(10, 0.8), "n_train": 80, "n_val": 20})
    res = paired_comparison(ref_auc, pd.DataFrame({"fold": range(10), "auc_roc": np.full(10, 0.7)}), "auc_roc")
    assert res["mean_improvement"] == pytest.approx(0.1)


def test_cohens_dz() -> None:
    assert cohens_dz(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)


def test_bootstrap_contains_point_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    a = bootstrap_metrics(lambda t, s: binary_metrics(t, s), y, p, ["auc_roc", "recall"], n_boot=300, seed=3)
    b = bootstrap_metrics(lambda t, s: binary_metrics(t, s), y, p, ["auc_roc", "recall"], n_boot=300, seed=3)
    pd.testing.assert_frame_equal(a, b)
    row = a.set_index("metric").loc["auc_roc"]
    assert row["ci_low"] <= row["value"] <= row["ci_high"]


def test_cv_evaluate_records_oof_and_disjoint_folds() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=100)})
    y = 2 * X["a"].to_numpy() + 11
    cv = KeyStratifiedRepeatedKFold(np.where(y > 11, "hi", "lo"), n_splits=5, n_repeats=2, random_state=0)
    res = cv_evaluate(LinearRegression, X, y, cv, "regression")
    assert len(res.folds) == 10 and res.folds["repeat"].max() == 1
    assert res.oof.groupby("repeat")["row"].nunique().eq(100).all()  # every row predicted once per repeat
    assert res.mean("mae") < 1e-8


def test_dummy_notes_flag_weak_model() -> None:
    cfg = {"mae_relative_improvement": 0.05, "auc_margin": 0.05, "pr_auc_margin": 0.05}
    notes = dummy_closeness_notes("regression", {"mae": 0.99, "auc_roc": 0.52, "pr_auc": 0.22}, {"mae": 1.0}, 0.2, cfg)
    assert len(notes) == 3
    assert dummy_closeness_notes("regression", {"mae": 0.5, "auc_roc": 0.8}, {"mae": 1.0}, None, cfg) == []
