"""Boosting comparators (supervisor revision 2026-10-27): fairness, leakage, reproducibility, selection, dashboard gate."""

from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.cleaning import clean_dataset
from src.model_selection import SelectionRuleError, hb_explanation_gate, load_rule, select_model
from src.models import BOOSTING_CANDIDATES, CED_MODELS, HB_MODELS, tree_node_count
from src.nested_cv import build_data, make_pipeline_factory, outer_splitter, run_fold, run_nested
from src.postmodel.db import SessionStore
from src.postmodel.explain import ExplainerNotSupportedError, additivity_error, explainer_supported, tree_explainer
from src.postmodel.inference import ModelNotEligibleError, ScreeningService
from src.protocol_stats import RepBootstrap
from src.runs import main_run
from src.utils import PROJECT_ROOT, LeakageError

NEW = [m for m in BOOSTING_CANDIDATES if m != "xgboost"]
SMALL_PARAMS = {"gbm": {"n_estimators": 30}, "lightgbm": {"n_estimators": 30, "num_leaves": 7},
                "catboost": {"iterations": 30, "depth": 3}, "adaboost": {"n_estimators": 20, "max_depth": 2},
                "xgboost": {"n_estimators": 30, "max_depth": 3}}


@pytest.fixture()
def small(isolated_config: dict) -> dict:
    cfg = copy.deepcopy(isolated_config)
    cfg["feature_selection"]["rf_n_estimators"] = 20
    cfg["tuning"]["n_trials"] = 2
    cfg["runtime"]["n_jobs"] = 1
    cfg["cv"]["outer_n_repeats"] = 1
    return cfg


# --- registry and protocol --------------------------------------------------------

def test_every_candidate_is_registered_for_both_targets(config: dict) -> None:
    spaces = config["tuning"]["search_spaces"]
    for models, task in ((HB_MODELS, "regression"), (CED_MODELS, "classification")):
        for name in BOOSTING_CANDIDATES:
            spec = models[name]
            assert spec.task == task and spec.space in spaces and not spec.scale
            assert spec.role == ("main" if name == "xgboost" else "boosting_comparator")
            assert hasattr(spec.factory({}, 0, config), "fit")
    assert list(config["comparison"]["candidates"]) == list(BOOSTING_CANDIDATES)


def test_comparator_settings_follow_the_protocol(config: dict) -> None:
    ada = CED_MODELS["adaboost"].factory({"max_depth": 2}, 0, config)
    assert ada.algorithm == "SAMME" and ada.estimator.max_depth == 2
    cat = CED_MODELS["catboost"].factory({}, 7, config).get_params()
    assert cat["random_seed"] == 7 and cat["thread_count"] == 1 and cat.get("cat_features") is None
    lgbm = HB_MODELS["lightgbm"].factory({}, 7, config).get_params()
    assert lgbm["verbosity"] == -1 and lgbm["subsample_freq"] == 1 and lgbm["deterministic"]


@pytest.mark.parametrize("name", NEW)
@pytest.mark.parametrize("target,bad", [("hb", "hb"), ("hb", "anemia_binary"), ("ced", "muac"), ("ced", "ced")])
def test_leakage_guard_rejects_targets_for_new_models(small: dict, name: str, target: str, bad: str) -> None:
    df = clean_dataset(small).frame
    data = build_data(small, df, target)
    spec = (HB_MODELS if target == "hb" else CED_MODELS)[name]
    pipe = make_pipeline_factory(main_run(spec, small), small)(SMALL_PARAMS[name])
    X = data.X.assign(**{bad: 1.0})
    with pytest.raises(LeakageError):
        pipe.fit(X, data.y)


@pytest.mark.parametrize("target", ["hb", "ced"])
def test_all_candidates_share_folds_features_and_pass_the_fit_spy(small: dict, target: str) -> None:
    """Identical outer folds, inner seeds, threshold rows and per-fold features. No fit sees evaluation rows."""
    df = clean_dataset(small).frame
    data = build_data(small, df, target)
    models = HB_MODELS if target == "hb" else CED_MODELS
    ref = None
    for name in BOOSTING_CANDIDATES:
        result = run_nested(main_run(models[name], small), data, small, spy=True)
        assert all(r["spy"]["ok"] for r in result.folds)  # raises inside run_fold on any leak
        keys = [(r["repeat"], r["fold"], r["inner_seed"], r["threshold_source_ids_hash"], tuple(r["selected_features"]),
                 tuple(sorted(r["oof"]["anon_id"]))) for r in result.folds]
        ref = ref or keys
        assert keys == ref, f"{name} did not use the folds of xgboost"
        assert all(r["n_tree_nodes"] > 0 for r in result.folds)


@pytest.mark.parametrize("name", BOOSTING_CANDIDATES)
@pytest.mark.parametrize("target", ["hb", "ced"])
def test_two_runs_give_identical_numbers(small: dict, name: str, target: str) -> None:
    df = clean_dataset(small).frame
    data = build_data(small, df, target)
    run = main_run((HB_MODELS if target == "hb" else CED_MODELS)[name], small)
    tr, te = next(outer_splitter(run, data, small).split(data.X))
    a = run_fold(run, data, small, 0, 0, tr, te)
    b = run_fold(run, data, small, 0, 0, tr, te)
    pd.testing.assert_frame_equal(a["oof"], b["oof"])
    assert a["params"] == b["params"] and a["n_tree_nodes"] == b["n_tree_nodes"]
    assert a.get("hb_cutoff") == b.get("hb_cutoff") and a.get("threshold_youden") == b.get("threshold_youden")


def test_tree_node_count(small: dict) -> None:
    df = clean_dataset(small).frame
    data = build_data(small, df, "ced")
    for name in BOOSTING_CANDIDATES:
        pipe = make_pipeline_factory(main_run(CED_MODELS[name], small), small)(SMALL_PARAMS[name]).fit(data.X, data.y)
        n = tree_node_count(pipe.named_steps["model"])
        assert isinstance(n, int) and n > 0
    ridge = make_pipeline_factory(main_run(HB_MODELS["ridge"], small), small)({}).fit(*_hb(small))
    assert tree_node_count(ridge.named_steps["model"]) is None


def _hb(cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    data = build_data(cfg, clean_dataset(cfg).frame, "hb")
    return data.X, data.y


# --- SHAP ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["xgboost", "gbm", "lightgbm", "catboost"])
@pytest.mark.parametrize("target", ["hb", "ced"])
def test_tree_explainer_additivity_for_supported_models(small: dict, name: str, target: str) -> None:
    data = build_data(small, clean_dataset(small).frame, target)
    spec = (HB_MODELS if target == "hb" else CED_MODELS)[name]
    pipe = make_pipeline_factory(main_run(spec, small), small)(SMALL_PARAMS[name]).fit(data.X, data.y)
    assert explainer_supported(pipe)
    assert additivity_error(pipe, data.X) <= float(load_rule(require_locked=False)["additivity_tolerance"])


@pytest.mark.parametrize("target", ["hb", "ced"])
def test_adaboost_has_no_explainer_and_no_fallback(small: dict, target: str) -> None:
    data = build_data(small, clean_dataset(small).frame, target)
    spec = (HB_MODELS if target == "hb" else CED_MODELS)["adaboost"]
    pipe = make_pipeline_factory(main_run(spec, small), small)(SMALL_PARAMS["adaboost"]).fit(data.X, data.y)
    assert not explainer_supported(pipe)
    with pytest.raises(ExplainerNotSupportedError):
        tree_explainer(pipe)


# --- selection rule -------------------------------------------------------------------

def test_selection_rule_file_is_locked_and_consistent() -> None:
    rule = load_rule()
    assert rule["status"] == "locked" and rule["locked_on"] and rule["version"] == 2
    # version 2 (2026-10-28): no preferred model, the best eligible point estimate is selected
    assert rule["candidates"] == list(BOOSTING_CANDIDATES) and rule["preferred"] is None
    assert rule["choice"] == "best_point_estimate"
    assert rule["targets"]["hb"] == {"metric": "mae", "direction": "lower_is_better"}
    assert rule["targets"]["ced"] == {"metric": "auc_roc", "direction": "higher_is_better"}
    assert rule["hb_explanation_gate"]["enable_on"] == ["supported"]
    seed = int(yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))["project"]["random_seed"])
    # selection seeds are written out and equal the evaluation seeds of 08 (S1 uses the Hb one)
    assert rule["paired_bootstrap_seed"] == {"hb": seed, "ced": seed + 100}
    assert rule["hb_explanation_gate"]["bootstrap_seed"] == seed
    dash = yaml.safe_load((PROJECT_ROOT / "config" / "dashboard.yaml").read_text(encoding="utf-8"))
    assert float(dash["additivity_tolerance"]) == float(rule["additivity_tolerance"])


def _rule() -> dict:
    """Version 1 rule (XGBoost preferred inside the equivalence group), still applied side by side in 14."""
    return {"candidates": list(BOOSTING_CANDIDATES), "preferred": "xgboost", "status": "locked",
            "targets": {"ced": {"metric": "auc_roc", "direction": "higher_is_better"},
                        "hb": {"metric": "mae", "direction": "lower_is_better"}}}


def _rule_v2() -> dict:
    return {**_rule(), "preferred": None, "choice": "best_point_estimate"}


def _ci(pairs: dict[tuple[str, str], tuple[float, float]]) -> dict[tuple[str, str], tuple[float, float]]:
    """Fill every pair not given with a CI that excludes 0. A given pair also sets its reverse (-high, -low), like m04."""
    out = {(a, b): (0.01, 0.02) for a in BOOSTING_CANDIDATES for b in BOOSTING_CANDIDATES if a != b}
    for (a, b), (low, high) in pairs.items():
        out[(a, b)] = (low, high)
        out[(b, a)] = (-high, -low)
    return out


ALL_OK = dict.fromkeys(BOOSTING_CANDIDATES, True)
COST = {"xgboost": 900.0, "gbm": 700.0, "lightgbm": 300.0, "catboost": 500.0, "adaboost": 100.0}


def test_selection_is_deterministic() -> None:
    points = {"xgboost": 0.76, "gbm": 0.78, "lightgbm": 0.77, "catboost": 0.75, "adaboost": 0.74}
    args = (_rule(), "ced", points, _ci({("gbm", "lightgbm"): (-0.01, 0.03)}), ALL_OK, COST)
    first, second = select_model(*args), select_model(*args)
    assert first == second and first.selected == "lightgbm"  # gbm best, lightgbm equivalent and cheaper
    pd.testing.assert_frame_equal(first.table, second.table)


def test_xgboost_preferred_inside_the_group() -> None:
    points = {"xgboost": 0.76, "gbm": 0.78, "lightgbm": 0.70, "catboost": 0.70, "adaboost": 0.70}
    res = select_model(_rule(), "ced", points, _ci({("gbm", "xgboost"): (-0.01, 0.05)}), ALL_OK, COST)
    assert res.provisional_winner == "gbm" and res.selected == "xgboost"


def test_lower_is_better_for_hb() -> None:
    points = {"xgboost": 1.10, "gbm": 1.05, "lightgbm": 1.20, "catboost": 1.30, "adaboost": 1.40}
    res = select_model(_rule(), "hb", points, _ci({}), ALL_OK, COST)
    assert res.provisional_winner == "gbm" and res.selected == "gbm"


def test_group_is_relative_to_the_provisional_winner_and_not_transitive() -> None:
    points = {"xgboost": 0.70, "gbm": 0.80, "lightgbm": 0.78, "catboost": 0.70, "adaboost": 0.70}
    # lightgbm ~ gbm, xgboost ~ lightgbm but not ~ gbm: xgboost stays out of the group
    ci = _ci({("gbm", "lightgbm"): (-0.01, 0.05), ("lightgbm", "xgboost"): (-0.01, 0.10)})
    res = select_model(_rule(), "ced", points, ci, ALL_OK, COST)
    assert res.equivalence_group == ("gbm", "lightgbm") and res.selected == "lightgbm"


def test_ineligible_models_are_skipped_before_the_provisional_winner() -> None:
    points = {"xgboost": 0.70, "gbm": 0.71, "lightgbm": 0.70, "catboost": 0.70, "adaboost": 0.90}
    eligible = {**ALL_OK, "adaboost": False}
    # adaboost is best and "equivalent" to gbm, but it is removed first and can never be selected
    ci = _ci({("adaboost", "gbm"): (-0.01, 0.30), ("gbm", "adaboost"): (-0.30, 0.01)})
    res = select_model(_rule(), "ced", points, ci, eligible, {**COST, "adaboost": 1.0})
    assert res.provisional_winner == "gbm" and "adaboost" not in res.equivalence_group
    assert res.selected != "adaboost" and not res.table.set_index("candidate").loc["adaboost", "selected"]


def test_all_candidates_ineligible_is_reported_clearly() -> None:
    points = {"xgboost": 0.70, "gbm": 0.71, "lightgbm": 0.70, "catboost": 0.70, "adaboost": 0.90}
    with pytest.raises(SelectionRuleError, match="no eligible candidate.*never falls back"):
        select_model(_rule(), "ced", points, _ci({}), dict.fromkeys(BOOSTING_CANDIDATES, False), COST)


@pytest.mark.parametrize("target,points", [
    ("ced", {"xgboost": 0.70, "gbm": 0.71, "lightgbm": 0.80, "catboost": 0.69, "adaboost": 0.60}),
    ("hb", {"xgboost": 1.30, "gbm": 1.25, "lightgbm": 1.10, "catboost": 1.40, "adaboost": 1.50}),
])
def test_single_member_group_selects_the_provisional_winner(target: str, points: dict) -> None:
    """Every CI against the provisional winner excludes 0: the group is the winner alone, even if it is expensive."""
    res = select_model(_rule(), target, points, _ci({}), ALL_OK, {**COST, "lightgbm": 10_000.0})
    assert res.provisional_winner == "lightgbm"
    assert res.equivalence_group == ("lightgbm",) and res.selected == "lightgbm"


def test_difference_is_candidate_minus_provisional_winner() -> None:
    """Only one orientation is given: the function must read it, or negate the reverse, consistently."""
    points = {"xgboost": 0.76, "gbm": 0.80, "lightgbm": 0.70, "catboost": 0.70, "adaboost": 0.70}
    only_reverse = {k: v for k, v in _ci({}).items() if k != ("xgboost", "gbm")}
    only_reverse[("gbm", "xgboost")] = (0.01, 0.07)  # gbm minus xgboost excludes 0
    res = select_model(_rule(), "ced", points, only_reverse, ALL_OK, COST)
    row = res.table.set_index("candidate").loc["xgboost"]
    assert (row["diff_candidate_minus_winner_ci_low"], row["diff_candidate_minus_winner_ci_high"]) == (-0.07, -0.01)
    assert res.selected == "gbm"


def test_rule_refuses_tree_forest_and_baselines_as_candidates(tmp_path: Path) -> None:
    text = (PROJECT_ROOT / "config" / "model_selection.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "rule.yaml"
    bad.write_text(text.replace("candidates: [xgboost,", "candidates: [random_forest, xgboost,", 1), encoding="utf-8")
    with pytest.raises(SelectionRuleError, match="cannot be candidates"):
        load_rule(bad, require_locked=False)


def test_cost_ties_go_to_the_candidate_order() -> None:
    points = {"xgboost": 0.60, "gbm": 0.80, "lightgbm": 0.79, "catboost": 0.79, "adaboost": 0.6}
    ci = _ci({("gbm", "lightgbm"): (-0.01, 0.03), ("gbm", "catboost"): (-0.01, 0.03)})
    res = select_model(_rule(), "ced", points, ci, ALL_OK, {**COST, "gbm": 300.0, "lightgbm": 300.0, "catboost": 300.0})
    assert res.selected == "gbm"


def _gate_rb(reduction: float, spread: float) -> RepBootstrap:
    rng = np.random.default_rng(0)
    points, draws = {}, {}
    for rep in range(5):
        points[("m", rep)] = {"mae": 1.0}
        points[("dummy_mean", rep)] = {"mae": 1.0 + reduction}
        draws[("m", rep)] = pd.DataFrame({"mae": np.full(2000, 1.0)})
        draws[("dummy_mean", rep)] = pd.DataFrame({"mae": 1.0 + reduction + rng.normal(0, spread, 2000)})
    return RepBootstrap(points, draws, list(range(5)), 2000)


@pytest.mark.parametrize("reduction,spread,verdict,enabled", [(0.30, 0.02, "supported", True),
                                                              (0.12, 0.10, "inconclusive", False),
                                                              (0.00, 0.02, "not_supported", False)])
def test_hb_gate_enables_shap_only_when_supported(reduction: float, spread: float, verdict: str, enabled: bool) -> None:
    gate = hb_explanation_gate(load_rule(require_locked=False), _gate_rb(reduction, spread), "m")
    assert gate["verdict"] == verdict and gate["hb_explanation_enabled"] is enabled


# --- dashboard ----------------------------------------------------------------------

HB_FEATS = ["age", "anc_visits", "hfias_score", "prepreg_weight", "height", "education_mother", "n_children"]
CED_FEATS = [*HB_FEATS, "egg_freq"]


def _service_root(tmp_path: Path, cfg: dict, ced_algorithm: str, meta_extra: dict | None = None) -> Path:
    cfg = copy.deepcopy(cfg)
    cfg["feature_selection"]["rf_n_estimators"] = 20
    df = clean_dataset(cfg).frame
    root = tmp_path / "root"
    (root / "artifacts").mkdir(parents=True)
    meta = {}
    for target, models, alg, feats in (("hb", HB_MODELS, "xgboost", HB_FEATS), ("ced", CED_MODELS, ced_algorithm, CED_FEATS)):
        data = build_data(cfg, df, target)
        run = main_run(models[alg], cfg, feature_mode="fixed", fixed_features=tuple(feats))
        pipe = make_pipeline_factory(run, cfg)(SMALL_PARAMS[alg]).fit(data.X, data.y)
        joblib.dump(pipe, root / "artifacts" / f"final_{target}_{alg}.joblib")
        meta[target] = {"features": feats, "threshold_youden": 0.3, "interval_half_width": 2.0, "algorithm": alg,
                        "artifact": f"artifacts/final_{target}_{alg}.joblib", "explainer": "artifacts/none.joblib",
                        **(meta_extra or {}).get(target, {})}
    (root / "artifacts" / "final_models.json").write_text(json.dumps(meta), encoding="utf-8")
    shutil.copytree(PROJECT_ROOT / "config", root / "config")
    return root


def test_dashboard_loads_the_configured_algorithm(tmp_path: Path, isolated_config: dict) -> None:
    root = _service_root(tmp_path, isolated_config, "lightgbm")
    service = ScreeningService(root, isolated_config)
    assert service.algorithms == {"hb": "xgboost", "ced": "lightgbm"}
    from src.postmodel.inference import median_answers

    result = service.predict(median_answers(service))
    assert "LightGBM" in result.ced_text and result.extra["ced_algorithm"] == "lightgbm"
    inputs, prediction = service.storage_records(median_answers(service), result)
    store = SessionStore(tmp_path / "s.sqlite")
    sid = store.create(inputs, prediction, "u1")
    assert store.read(sid)["prediction"]["ced_algorithm"] == "lightgbm"


def test_dashboard_refuses_a_model_that_failed_additivity(tmp_path: Path, isolated_config: dict) -> None:
    root = _service_root(tmp_path, isolated_config, "catboost", {"ced": {"shap_additivity_passed": False}})
    with pytest.raises(ModelNotEligibleError):
        ScreeningService(root, isolated_config)


def test_dashboard_refuses_a_model_without_tree_explainer(tmp_path: Path, isolated_config: dict) -> None:
    root = _service_root(tmp_path, isolated_config, "adaboost")
    with pytest.raises(ModelNotEligibleError):
        ScreeningService(root, isolated_config)


def test_dashboard_refuses_a_mismatched_explainer(tmp_path: Path, isolated_config: dict) -> None:
    """An explainer saved for another model fails the load-time additivity check."""
    root = _service_root(tmp_path, isolated_config, "gbm")
    other = joblib.load(root / "artifacts" / "final_hb_xgboost.joblib")
    joblib.dump(tree_explainer(other), root / "artifacts" / "none.joblib")
    with pytest.raises((ModelNotEligibleError, ValueError, IndexError)):
        ScreeningService(root, isolated_config)


def test_old_database_gets_algorithm_columns(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE predictions (session_id TEXT PRIMARY KEY, predicted_hb REAL, ced_probability REAL "
                      "NOT NULL, ced_label INTEGER NOT NULL, model_note TEXT);"
                      "CREATE TABLE inputs (session_id TEXT PRIMARY KEY, age REAL, self_reported_fields TEXT NOT NULL);")
    con.close()
    cols = SessionStore(path).column_names()
    assert {"hb_algorithm", "ced_algorithm"} <= set(cols["predictions"])
    assert {"n_under5", "income", "food_taboo"} <= set(cols["inputs"])


# --- version 2 rule (2026-10-28) ---------------------------------------------------------

def test_v2_selects_the_best_point_estimate_even_inside_an_equivalence_group() -> None:
    points = {"xgboost": 0.76, "gbm": 0.78, "lightgbm": 0.70, "catboost": 0.70, "adaboost": 0.70}
    res = select_model(_rule_v2(), "ced", points, _ci({("gbm", "xgboost"): (-0.01, 0.05)}), ALL_OK, COST)
    assert res.selected == "gbm" and "xgboost" in res.equivalence_group
    assert "not statistically established" in res.reason
    v1 = select_model(_rule(), "ced", points, _ci({("gbm", "xgboost"): (-0.01, 0.05)}), ALL_OK, COST)
    assert v1.selected == "xgboost"  # the superseded rule would have kept XGBoost


def test_v2_ignores_cost_and_never_selects_an_ineligible_model() -> None:
    points = {"xgboost": 1.10, "gbm": 1.05, "lightgbm": 1.20, "catboost": 1.30, "adaboost": 0.90}
    eligible = {**ALL_OK, "adaboost": False}
    res = select_model(_rule_v2(), "hb", points, _ci({}), eligible, {**COST, "gbm": 1e9})
    assert res.selected == "gbm"


def test_v2_rule_must_not_name_a_preferred_model(tmp_path: Path) -> None:
    text = (PROJECT_ROOT / "config" / "model_selection.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "rule.yaml"
    bad.write_text(re.sub(r"(?m)^preferred: null$", "preferred: xgboost", text, count=1), encoding="utf-8")
    with pytest.raises(SelectionRuleError, match="prefers no model"):
        load_rule(bad)
