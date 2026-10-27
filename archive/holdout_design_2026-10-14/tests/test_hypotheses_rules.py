"""hypotheses.yaml engine, final-feature rule, H3 cut-off and H8 interval."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.evaluation import choose_hb_cutoff
from src.feature_selection import RankAggregationSelector, stability_select
from src.hypotheses import (
    HypothesisSpecError,
    adjust_pvalues,
    comparison_plan,
    decide,
    parse_ci_level,
    pooled_oof_metric,
    validate,
)
from src.intervals import conformal_quantile, empirical_coverage, fit_oof_conformal
from src.utils import load_hypotheses


@pytest.fixture()
def hyp(config: dict) -> dict:
    # The engine in src/hypotheses.py implements the superseded eight-hypothesis schema.
    # It is tested against that archived file until the proposal protocol replaces it.
    config["paths"]["hypotheses"] = "config/hypotheses_superseded_2026-10-14.yaml"
    return copy.deepcopy(load_hypotheses(config))


def test_project_hypotheses_validate(hyp: dict, config: dict) -> None:
    validate(hyp, config)


def test_unknown_key_and_value_rejected(hyp: dict, config: dict) -> None:
    bad = copy.deepcopy(hyp)
    bad["hypotheses"]["H5_ced_classification"]["min_aucc"] = 0.7
    with pytest.raises(HypothesisSpecError, match="min_aucc"):
        validate(bad, config)
    bad = copy.deepcopy(hyp)
    bad["holm_family"] = "everything"
    with pytest.raises(HypothesisSpecError, match="holm_family"):
        validate(bad, config)


def test_decision_setup_must_be_trained_and_ablated(hyp: dict, config: dict) -> None:
    config["ablation_setups"] = ["proposal_fixed"]
    with pytest.raises(HypothesisSpecError, match="ablation_setups"):
        validate(hyp, config)


def test_h8_nominal_must_match_config(hyp: dict, config: dict) -> None:
    config["intervals"]["nominal"] = 0.8
    with pytest.raises(HypothesisSpecError, match="nominal"):
        validate(hyp, config)


def test_ci_level_parsing() -> None:
    assert parse_ci_level("two_sided_90") == 0.90 and parse_ci_level(0.95) == 0.95
    with pytest.raises(HypothesisSpecError):
        parse_ci_level("one_sided_95")


def test_plan_resolves_names(hyp: dict, config: dict) -> None:
    plan = {(c.hypothesis, c.comparator[1], c.comparator[2], c.reference[2]) for c in comparison_plan(hyp, config)}
    assert ("H6_resampling_ced", "xgboost", "ablation_none", "ablation_smotenc") in plan
    assert ("H6_resampling_ced", "xgboost", "ablation_none", "ablation_scale_pos_weight") in plan
    assert ("H7_resampling_hb", "xgboost", "ablation_none", "ablation_smogn") in plan
    h4 = [c for c in comparison_plan(hyp, config) if c.hypothesis == "H4_regression_vs_direct"][0]
    assert h4.ci_level == pytest.approx(0.90) and h4.comparator[0] == "anemia_binary"


def _table(hypothesis: list[str], p: list[float], improvement: list[float], kind: str = "superiority") -> pd.DataFrame:
    n = len(p)
    return pd.DataFrame({
        "setup": ["primary"] * n, "hypothesis": hypothesis, "kind": [kind] * n, "p_corrected_t": p,
        "decision_improvement": improvement, "mean_improvement": improvement, "paired_dz": [1.0] * n,
        "reference": ["ced_model/xgboost/x"] * n, "comparator": [f"ced_model/c{i}/y" for i in range(n)],
    })


def test_holm_per_hypothesis_versus_all(hyp: dict) -> None:
    table = _table(["H2", "H2", "H5"], [0.02, 0.03, 0.04], [0.1, 0.1, 0.1])
    out = adjust_pvalues(table, hyp, "p_corrected_t")
    np.testing.assert_allclose(out["p_per_hypothesis"], [0.04, 0.04, 0.04])
    np.testing.assert_allclose(out["p_all_superiority"], [0.06, 0.06, 0.06])
    assert (out["p_adjusted"] == out["p_per_hypothesis"]).all()  # holm_family: per_hypothesis


def _decide_one(hyp: dict, hid: str, comps: pd.DataFrame, holdout: dict) -> pd.Series:
    hyp = copy.deepcopy(hyp)
    hyp["hypotheses"] = {hid: hyp["hypotheses"][hid]}
    return decide(hyp, comps, holdout).iloc[0]


def test_pass_rule_any_for_h6(hyp: dict) -> None:
    comps = _table(["H6_resampling_ced"] * 2, [0.01, 0.9], [0.05, -0.01], "ablation")
    comps["p_adjusted"] = comps["p_corrected_t"]
    assert _decide_one(hyp, "H6_resampling_ced", comps, {})["supported"]
    hyp["hypotheses"]["H6_resampling_ced"]["pass_rule"] = "all"
    assert not _decide_one(hyp, "H6_resampling_ced", comps, {})["supported"]


def test_min_meaningful_difference_applies(hyp: dict) -> None:
    comps = _table(["H6_resampling_ced"], [0.001], [0.01], "ablation")  # significant but below 0.02
    comps["p_adjusted"] = comps["p_corrected_t"]
    assert not _decide_one(hyp, "H6_resampling_ced", comps, {})["supported"]


def test_h5_needs_cv_and_holdout_criteria(hyp: dict) -> None:
    comps = _table(["H5_ced_classification"] * 2, [0.001, 0.001], [0.3, 0.05])
    comps["p_adjusted"] = comps["p_corrected_t"]
    good = {"H5_ced_classification": {"auc": 0.75, "auc_ci_low": 0.6}}
    assert _decide_one(hyp, "H5_ced_classification", comps, good)["supported"]
    low_ci = {"H5_ced_classification": {"auc": 0.75, "auc_ci_low": 0.45}}
    assert not _decide_one(hyp, "H5_ced_classification", comps, low_ci)["supported"]


def test_h3_uses_only_structured_keys(hyp: dict) -> None:
    holdout = {"H3_anemia_screening": {"cutoff": 11.2, "sensitivity": 0.75, "specificity": 0.45, "auc_ci_low": 0.40}}
    verdict = _decide_one(hyp, "H3_anemia_screening", pd.DataFrame(columns=["hypothesis"]), holdout)
    has_auc_key = "auc_ci_lower_min" in hyp["hypotheses"]["H3_anemia_screening"]
    assert bool(verdict["supported"]) == (not has_auc_key)
    holdout["H3_anemia_screening"]["specificity"] = 0.30
    assert not _decide_one(hyp, "H3_anemia_screening", pd.DataFrame(columns=["hypothesis"]), holdout)["supported"]


def test_h4_non_inferiority(hyp: dict) -> None:
    comps = pd.DataFrame({"hypothesis": ["H4_regression_vs_direct"], "mean_improvement": [-0.02],
                          "improvement_ci_low": [-0.08], "ci_level": [0.9]})
    assert _decide_one(hyp, "H4_regression_vs_direct", comps, {})["supported"]
    comps["improvement_ci_low"] = -0.12
    assert not _decide_one(hyp, "H4_regression_vs_direct", comps, {})["supported"]


def test_h8_coverage_window(hyp: dict) -> None:
    empty = pd.DataFrame(columns=["hypothesis"])
    ok = _decide_one(hyp, "H8_hb_interval_coverage", empty, {"H8_hb_interval_coverage": {"coverage": 0.9, "half_width": 2}})
    low = _decide_one(hyp, "H8_hb_interval_coverage", empty, {"H8_hb_interval_coverage": {"coverage": 0.8, "half_width": 2}})
    assert ok["supported"] and not low["supported"]


def test_pooled_oof_metric() -> None:
    oof = pd.DataFrame({"repeat": [0, 0, 0, 1, 1, 1], "y_true": [9, 10, 12, 9, 10, 12], "pred": [10, 10, 12, 11, 10, 12]})
    np.testing.assert_allclose(pooled_oof_metric(oof, "mae_hb_below_11").to_numpy(), [0.5, 1.0])


# --- final-feature rule ------------------------------------------------------

def _xy(n: int = 240, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 9)), columns=[f"f{i}" for i in range(9)])
    y = 3 * X["f0"] + 2 * X["f1"] + X["f2"] + rng.normal(scale=0.3, size=n)
    return X, y.to_numpy()


def test_stability_rule_keeps_forced_and_caps() -> None:
    X, y = _xy()
    chosen, rates = stability_select(X, y, "regression", k=4, inner_folds=5, min_selection_rate=0.5,
                                     max_features=3, forced_include=["f8"], n_estimators=30,
                                     discrete_features=(), random_state=0)
    assert chosen[0] == "f8" and len(chosen) <= 3
    assert all(rates[f] >= 0.5 for f in chosen) and rates["f8"] == 1.0
    assert set(chosen) <= {"f8", "f0", "f1", "f2"}


def test_stability_rule_drops_unstable_features() -> None:
    X, y = _xy()
    chosen, rates = stability_select(X, y, "regression", k=7, inner_folds=5, min_selection_rate=1.01,
                                     max_features=7, forced_include=["f5"], n_estimators=30,
                                     discrete_features=(), random_state=0)
    assert chosen == ["f5"]  # nothing else can reach a rate above 1


def test_selector_cache_returns_identical_selection() -> None:
    X, y = _xy()
    a = RankAggregationSelector(k=4, n_estimators=30, inner_folds=3, max_features=4, random_state=1).fit(X, y)
    b = RankAggregationSelector(k=4, n_estimators=30, inner_folds=3, max_features=4, random_state=1).fit(X, y)
    assert a.selected_features_ == b.selected_features_
    c = RankAggregationSelector(k=4, n_estimators=30, inner_folds=3, max_features=2, random_state=1).fit(X, y)
    assert len(c.selected_features_) <= 2  # different settings are not served from the cache


# --- H3 cut-off and H8 interval ------------------------------------------------

def test_cutoff_is_most_specific_meeting_sensitivity() -> None:
    y = np.array([9, 10, 10, 12, 12, 13, 13, 14], dtype=float)
    pred = np.array([10.0, 10.5, 11.5, 11.0, 11.8, 12.5, 13.0, 13.5])
    out = choose_hb_cutoff(y, pred, 11.0, min_sensitivity=2 / 3)
    assert out["oof_sensitivity"] >= 2 / 3
    assert out["cutoff"] == pytest.approx(10.75)  # flags 10.0 and 10.5, not yet 11.0
    full = choose_hb_cutoff(y, pred, 11.0, min_sensitivity=1.0)
    assert full["oof_sensitivity"] == 1.0 and full["cutoff"] > 11.5


def test_conformal_quantile_and_coverage() -> None:
    r = np.arange(1, 20, dtype=float)  # n = 19, ceil(20 * 0.9) = 18th smallest
    assert conformal_quantile(r, 0.9) == 18.0
    assert empirical_coverage([1, 2, 3, 10], [0, 0, 0, 0], [5, 5, 5, 5]) == 0.75


def test_interval_uses_one_repeat_of_oof() -> None:
    oof = pd.DataFrame({"repeat": [0, 0, 0, 1, 1, 1], "row": [0, 1, 2, 0, 1, 2],
                        "y_true": [10, 11, 12, 10, 11, 12], "pred": [10.5, 11, 13, 20, 20, 20]})
    interval = fit_oof_conformal(oof, 0.5)
    assert interval.n_calibration == 3 and interval.half_width == 0.5
