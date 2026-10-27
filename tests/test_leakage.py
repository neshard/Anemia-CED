"""The target leakage guard and the configured feature lists."""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils import (
    LeakageError,
    PrivacyError,
    assert_feature_lists_safe,
    assert_no_identifiers,
    assert_no_target_leakage,
    forbidden_columns_for,
)


def test_configured_feature_lists_are_safe(config: dict) -> None:
    assert_feature_lists_safe(config)


def test_candidate_pool_and_forced_features(config: dict) -> None:
    pool = config["features"]["candidate_pool"]
    assert len(pool) == 11 and len(set(pool)) == 11
    assert config["feature_selection"]["forced_include"] == ["prepreg_weight", "anc_visits", "hfias_score"]
    assert len(config["features"]["fixed_preliminary"]["hb_model"]) == 6
    assert len(config["features"]["fixed_preliminary"]["ced_model"]) == 7


@pytest.mark.parametrize("bad", ["hb", "hb_target", "anemia_class", "num__hb", "anemia_class_Mild", "strat_key"])
def test_hb_model_rejects_hb_derivatives(config: dict, bad: str) -> None:
    cols = [*config["features"]["candidate_pool"], bad]
    with pytest.raises(LeakageError):
        assert_no_target_leakage(cols, forbidden_columns_for(config, "hb_model"))


@pytest.mark.parametrize("bad", ["muac", "ced", "remainder__muac", "ced_1"])
def test_ced_model_rejects_muac_derivatives(config: dict, bad: str) -> None:
    cols = [*config["features"]["candidate_pool"], bad]
    with pytest.raises(LeakageError):
        assert_no_target_leakage(cols, forbidden_columns_for(config, "ced_model"))


def test_guard_accepts_dataframe(config: dict) -> None:
    X = pd.DataFrame(columns=["age", "muac"])
    with pytest.raises(LeakageError):
        assert_no_target_leakage(X, forbidden_columns_for(config, "ced_model"))


def test_misconfigured_feature_list_fails_at_startup(config: dict) -> None:
    config["features"]["fixed_preliminary"]["ced_model"] = [*config["features"]["fixed_preliminary"]["ced_model"], "muac"]
    with pytest.raises(LeakageError):
        assert_feature_lists_safe(config)


def test_privacy_guard(config: dict) -> None:
    with pytest.raises(PrivacyError):
        assert_no_identifiers(pd.DataFrame(columns=["respondent_code", "age"]), config)
