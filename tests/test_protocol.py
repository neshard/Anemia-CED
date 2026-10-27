"""Protocol tests requested on 2026-10-17, plus the nested CV engine and the verdict rules."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import src.nested_cv as nested
from src.cleaning import clean_dataset, parse_survey_date
from src.hypotheses import EXPECTED_IDS, HypothesisSpecError, decide, judge_component, validate
from src.labels import ced_label_at_most, trimester_anemia
from src.models import ALL_SPECS, CED_MODELS, HB_MODELS
from src.nested_cv import (
    build_data,
    cutoff_flagging_share,
    make_pipeline_factory,
    run_fold,
    run_hash,
    run_nested,
    sensitivity_threshold,
    tune_inner,
    youden_threshold,
)
from src.protocol_stats import RepBootstrap, one_sided_p, paired_bootstrap, summarize
from src.runs import main_run
from src.utils import load_hypotheses


# --- requested tests -----------------------------------------------------------

def test_hypotheses_file_has_only_the_four_proposal_hypotheses(config: dict) -> None:
    hyp = load_hypotheses(config)
    assert tuple(sorted(hyp["hypotheses"])) == tuple(sorted(EXPECTED_IDS))
    assert hyp["primary_hypotheses"] == ["P1_anemia_screening", "P2_ced_classification"]
    assert hyp["secondary_hypotheses"] == ["S1_hb_vs_mean", "S2_hb_vs_ridge"]
    assert len(hyp["exploratory"]) >= 5
    validate(hyp)
    p1 = {(c["metric"], c["op"], c["value"], c["rule"]) for c in hyp["hypotheses"]["P1_anemia_screening"]["components"]}
    assert p1 == {("sensitivity", ">=", 0.70, "point_and_ci"), ("specificity", ">=", 0.40, "point_and_ci"),
                  ("auc_roc", ">", 0.50, "ci_lower_bound")}
    p2 = hyp["hypotheses"]["P2_ced_classification"]["components"]
    assert {(c["value"], c["rule"]) for c in p2} == {(0.70, "point_and_ci"), (0.50, "ci_lower_bound"), (0.03, "point_and_ci")}
    assert hyp["hypotheses"]["S1_hb_vs_mean"]["components"][0]["value"] == 0.10
    assert hyp["hypotheses"]["S2_hb_vs_ridge"]["components"][0]["value"] == 0.05


def test_extra_hypothesis_is_rejected(config: dict) -> None:
    hyp = copy.deepcopy(load_hypotheses(config))
    hyp["hypotheses"]["H9_extra"] = copy.deepcopy(hyp["hypotheses"]["S1_hb_vs_mean"])
    with pytest.raises(HypothesisSpecError):
        validate(hyp)


def test_equal_trial_counts_for_every_tuned_model(config: dict) -> None:
    assert isinstance(config["tuning"]["n_trials"], int) and config["tuning"]["n_trials"] == 30
    tuned = [s for s in ALL_SPECS if s.space is not None]
    counts = {main_run(s, config).n_trials for s in tuned}
    assert counts == {30}
    names = {s.name for s in tuned}
    assert {"xgboost", "ridge", "logistic_regression", "decision_tree", "random_forest"} <= names
    assert all(s.space is None for s in ALL_SPECS if s.name.startswith("dummy"))


def test_no_resampler_in_any_main_pipeline(config: dict) -> None:
    for spec in [*HB_MODELS.values(), *CED_MODELS.values()]:
        run = main_run(spec, config)
        assert run.resampler == "none"
        pipe = make_pipeline_factory(run, config)({})
        assert "resample" not in pipe.named_steps
    explo = make_pipeline_factory(main_run(HB_MODELS["xgboost"], config, resampler="smogn"), config)({})
    assert "resample" in explo.named_steps  # the check above would catch a resampler


@pytest.fixture()
def small(isolated_config: dict) -> dict:
    cfg = copy.deepcopy(isolated_config)
    cfg["feature_selection"]["rf_n_estimators"] = 20
    cfg["tuning"]["n_trials"] = 2
    cfg["runtime"]["n_jobs"] = 1
    cfg["cv"]["outer_n_repeats"] = 1
    return cfg


def test_thresholds_never_come_from_outer_test_or_training_predictions(small: dict, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    df = clean_dataset(small).frame
    data = build_data(small, df, "hb")
    run = main_run(HB_MODELS["xgboost"], small)
    splitter = nested.outer_splitter(run, data, small)
    train_idx, test_idx = next(splitter.split(data.X))

    captured: dict = {}
    original = nested.cutoff_flagging_share

    def spy(pred, truth, share):  # type: ignore[no-untyped-def]
        captured["pred"] = np.asarray(pred).copy()
        return original(pred, truth, share)

    monkeypatch.setattr(nested, "cutoff_flagging_share", spy)
    record = run_fold(run, data, small, 0, 0, train_idx, test_idx, spy=True)

    assert len(captured["pred"]) == len(train_idx)  # no outer test row in the threshold input
    seed = int(small["project"]["random_seed"])
    params, inner_pred, _ = tune_inner(run, data, small, train_idx, seed)
    np.testing.assert_allclose(captured["pred"], inner_pred)  # inner out-of-fold predictions only
    fitted = make_pipeline_factory(run, small)(params).fit(data.X.iloc[train_idx], data.y[train_idx])
    own = fitted.predict(data.X.iloc[train_idx])
    assert not np.allclose(captured["pred"], own)  # not the model's predictions on its own training rows
    assert record["threshold_source"] == "inner_out_of_fold"
    assert record["spy"]["ok"] and record["spy"]["n_fits_checked"] > 0


def test_ced_thresholds_use_inner_predictions_only(small: dict, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    df = clean_dataset(small).frame
    data = build_data(small, df, "ced")
    run = main_run(CED_MODELS["logistic_regression"], small)
    train_idx, test_idx = next(nested.outer_splitter(run, data, small).split(data.X))
    seen: list[int] = []
    original = nested.youden_threshold
    monkeypatch.setattr(nested, "youden_threshold", lambda p, t: seen.append(len(p)) or original(p, t))
    run_fold(run, data, small, 0, 0, train_idx, test_idx)
    assert seen == [len(train_idx)]


# --- engine ----------------------------------------------------------------------

def test_nested_run_predicts_every_woman_once_and_resumes(small: dict) -> None:
    df = clean_dataset(small).frame
    data = build_data(small, df, "ced")
    run = main_run(CED_MODELS["dummy_prior"], small)
    first = run_nested(run, data, small)
    assert first.oof.groupby("repeat")["anon_id"].nunique().eq(len(data.y)).all()
    assert all(len(r["selected_features"]) == 7 for r in first.folds)
    again = run_nested(run, data, small)  # every fold loaded from its checkpoint
    pd.testing.assert_frame_equal(first.oof, again.oof)


def test_checkpoint_key_changes_with_config(small: dict) -> None:
    other = copy.deepcopy(small)
    other["tuning"]["n_trials"] = 3
    assert run_hash(small) != run_hash(other)
    same = copy.deepcopy(small)
    same["runtime"]["n_jobs"] = 4  # parallelism does not change results
    assert run_hash(small) == run_hash(same)


def test_fit_spy_detects_a_leak(small: dict) -> None:
    log = [("outer_final", frozenset({"P0001", "P0002"}))]
    with pytest.raises(nested.ThresholdLeakageError):
        nested._check_spy(log, {"P0001"}, {"P0002"}, [], pd.Index([]))


def test_threshold_rules() -> None:
    pred = np.array([9.0, 9.5, 10.2, 10.8, 11.5, 12.0, 12.5, 13.0])
    truth = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
    cut = cutoff_flagging_share(pred, truth, 0.70)
    assert (pred[truth] < cut).mean() >= 0.70
    assert cut == pytest.approx(10.5)  # lowest cut-off reaching 70 percent (3 of 4 cases)
    proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    y = np.array([0, 0, 1, 0, 1, 1], dtype=bool)
    assert youden_threshold(proba, y) == pytest.approx(0.7)
    assert sensitivity_threshold(proba, y, 1.0) == pytest.approx(0.3)


# --- verdict rules -----------------------------------------------------------------

def _rb(values: dict[str, list[np.ndarray]], points: dict[str, list[float]]) -> RepBootstrap:
    reps = list(range(len(next(iter(points.values())))))
    pts, draws = {}, {}
    for model in points:
        for r in reps:
            pts[(model, r)] = {"m": points[model][r]}
            draws[(model, r)] = pd.DataFrame({"m": values[model][r]})
    return RepBootstrap(pts, draws, reps, len(values[next(iter(values))][0]))


def test_point_and_ci_rule_outcomes() -> None:
    comp = {"metric": "m", "quantity": "value", "op": ">=", "value": 0.70, "rule": "point_and_ci"}
    passing = _rb({"x": [np.linspace(0.72, 0.80, 2000)]}, {"x": [0.76]})
    failing = _rb({"x": [np.linspace(0.50, 0.65, 2000)]}, {"x": [0.58]})
    unsure = _rb({"x": [np.linspace(0.65, 0.80, 2000)]}, {"x": [0.72]})
    assert judge_component(comp, passing, "x", 0.95)["outcome"] == "pass"
    assert judge_component(comp, failing, "x", 0.95)["outcome"] == "fail"
    assert judge_component(comp, unsure, "x", 0.95)["outcome"] == "inconclusive"
    lower = {"metric": "m", "quantity": "value", "op": ">", "value": 0.50, "rule": "ci_lower_bound"}
    assert judge_component(lower, unsure, "x", 0.95)["outcome"] == "met"
    assert judge_component(lower, _rb({"x": [np.linspace(0.45, 0.6, 2000)]}, {"x": [0.55]}), "x", 0.95)["outcome"] == "not_met"


def test_p_value_formula_and_averaging() -> None:
    boots = [np.array([0.6, 0.8, 0.9, 0.95]), np.array([0.8, 0.8, 0.8, 0.8])]
    assert one_sided_p(boots, 0.70, ">=") == pytest.approx(((1 + 1) / 5 + (0 + 1) / 5) / 2)
    s = summarize([0.7, 0.8], [np.linspace(0.6, 0.8, 101), np.linspace(0.7, 0.9, 101)], 0.95)
    assert s["point"] == pytest.approx(0.75) and s["point_min"] == 0.7 and s["point_max"] == 0.8


def _full_rb(hb_good: bool, ced_good: bool) -> dict[str, RepBootstrap]:
    rng = np.random.default_rng(0)
    n = 2000

    def around(center: float, sd: float = 0.01) -> np.ndarray:
        return rng.normal(center, sd, n)

    hb_metrics = ["sensitivity", "specificity", "auc_roc", "mae"]
    hb = {"xgboost": {"sensitivity": 0.8 if hb_good else 0.5, "specificity": 0.5, "auc_roc": 0.7, "mae": 0.6},
          "dummy_mean": {"mae": 0.9, "sensitivity": 0, "specificity": 0, "auc_roc": 0.5},
          "ridge": {"mae": 0.8, "sensitivity": 0, "specificity": 0, "auc_roc": 0.5}}
    ced = {"xgboost": {"auc_roc": 0.85 if ced_good else 0.6}, "logistic_regression": {"auc_roc": 0.7}}
    out = {}
    for target, table in (("hb", hb), ("ced", ced)):
        pts, draws = {}, {}
        for model, metrics in table.items():
            for r in range(5):
                pts[(model, r)] = dict(metrics)
                draws[(model, r)] = pd.DataFrame({k: around(v) for k, v in metrics.items()})
        out[target] = RepBootstrap(pts, draws, list(range(5)), n)
    del hb_metrics
    return out


def test_holm_order_and_ci_levels(config: dict) -> None:
    hyp = load_hypotheses(config)
    verdicts, comps = decide(hyp, _full_rb(hb_good=True, ced_good=True))
    v = verdicts.set_index("hypothesis")
    assert set(v.loc[["P1_anemia_screening", "P2_ced_classification"], "ci_level_used"]) <= {0.975, 0.95}
    first = v.loc[v["holm_order"] == 1].iloc[0]
    second = v.loc[v["holm_order"] == 2].iloc[0]
    assert first["ci_level_used"] == 0.975
    assert second["ci_level_used"] == (0.95 if first["verdict"] == "supported" else 0.975)
    assert v.loc["S1_hb_vs_mean", "ci_level_used"] == 0.95
    assert v.loc["S1_hb_vs_mean", "verdict"] == "supported" and v.loc["S2_hb_vs_ridge", "verdict"] == "supported"


def test_failed_first_primary_keeps_second_at_strict_level(config: dict) -> None:
    hyp = load_hypotheses(config)
    verdicts, _ = decide(hyp, _full_rb(hb_good=False, ced_good=False))
    v = verdicts.set_index("hypothesis")
    assert (v.loc[["P1_anemia_screening", "P2_ced_classification"], "verdict"] == "not_supported").all()
    assert (v.loc[["P1_anemia_screening", "P2_ced_classification"], "ci_level_used"] == 0.975).all()


def test_paired_bootstrap_uses_same_women() -> None:
    a = pd.DataFrame({"anon_id": ["P1", "P2", "P3"] * 1, "repeat": 0, "v": [1.0, 2.0, 3.0]})
    b = a.assign(v=[10.0, 20.0, 30.0])
    rb = paired_bootstrap({"a": a, "b": b}, lambda d: {"s": float(d["v"].sum())}, 50, 0)
    ratio = rb.draws[("b", 0)]["s"] / rb.draws[("a", 0)]["s"]
    assert np.allclose(ratio, 10.0)  # identical resampled women for both models


# --- data helpers --------------------------------------------------------------------

def test_survey_date_rule() -> None:
    window = ("2024-01-01", "2026-01-31")
    assert parse_survey_date("16/12/2025", window) == pd.Timestamp("2025-12-16")
    assert parse_survey_date("12/16/2025", window) == pd.Timestamp("2025-12-16")
    assert parse_survey_date("12/16/0025", window) is pd.NaT
    assert parse_survey_date("05/06/2015", window) is pd.NaT
    assert parse_survey_date("31/02/2025", window) is pd.NaT
    assert parse_survey_date("kemarin", window) is pd.NaT


def test_trimester_and_ced_cutoff_labels() -> None:
    out = trimester_anemia([10.6, 10.6, 10.6, np.nan], [10, 20, 30, 20], 13, 27, {"first": 11.0, "second": 10.5, "third": 11.0})
    assert out[:3].tolist() == [1.0, 0.0, 1.0] and np.isnan(out[3])
    assert ced_label_at_most([22, 23, 24, np.nan], 22)[:3].tolist() == [1.0, 0.0, 0.0]
    assert ced_label_at_most([22, 23, 24, 25], 24).tolist() == [1.0, 1.0, 1.0, 0.0]


def test_build_data_rebuilds_labels_and_key_for_ced_cutoffs(small: dict) -> None:
    df = clean_dataset(small).frame
    main = build_data(small, df, "ced")
    le24 = build_data(small, df, "ced", analysis="le24", ced_cutoff_cm=24)
    assert le24.y.sum() >= main.y.sum()
    assert min(le24.meta["cell_counts"].values()) >= 5
