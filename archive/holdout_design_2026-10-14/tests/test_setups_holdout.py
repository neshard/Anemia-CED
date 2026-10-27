"""Feature setups, the expected row count, and the one-time hold-out claim."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.cleaning import DataSchemaError, load_raw
from src.preprocessing import model_input_columns
from src.splits import load_or_create_holdout, make_strat_key
from src.training import (
    HoldoutAlreadyUsedError,
    HoldoutToken,
    claim_holdout,
    load_split_frame,
    load_test_data,
    load_training_data,
)
from src.utils import LeakageError, assert_feature_lists_safe


def test_expected_row_count_enforced(config: dict, raw_csv: Path) -> None:
    config["data"]["expected_row_count"] = 999
    with pytest.raises(DataSchemaError, match="999"):
        load_raw(config, raw_csv)
    config["data"]["expected_row_count"] = 200
    frame, _ = load_raw(config, raw_csv)
    assert len(frame) == 200


def test_setup_columns(config: dict) -> None:
    assert model_input_columns(config, "hb_model", "primary") == config["features"]["candidate_pool"]
    assert model_input_columns(config, "ced_model", "proposal_fixed") == config["features"]["ced_model"]
    plus = model_input_columns(config, "hb_model", "proposal_plus_education")
    assert plus == [*config["features"]["hb_model"], "education_mother"]
    with pytest.raises(KeyError):
        model_input_columns(config, "hb_model", "unknown_setup")


def test_primary_setup_forces_required_features(config: dict) -> None:
    forced = config["feature_setups"][config["primary_setup"]]["forced_include"]
    assert set(forced) == {"hfias_score", "anc_visits", "prepreg_weight"}
    assert "education_mother" in config["features"]["candidate_pool"]


def test_setup_validation_catches_leakage_and_bad_forced(config: dict) -> None:
    bad = copy.deepcopy(config)
    bad["feature_setups"]["proposal_fixed"]["ced_model"].append("muac")
    with pytest.raises(LeakageError):
        assert_feature_lists_safe(bad)
    bad = copy.deepcopy(config)
    bad["feature_setups"]["primary"]["forced_include"] = ["not_in_pool"]
    with pytest.raises(ValueError, match="outside candidate_pool"):
        assert_feature_lists_safe(bad)


def _prepare(isolated_config: dict) -> None:
    df, _, _ = load_split_frame(isolated_config)
    key, _ = make_strat_key(df["anemia_class"], df["ced"], {}, min_cell_count=8)
    load_or_create_holdout(df, key, isolated_config)


def test_holdout_can_be_claimed_once(isolated_config: dict) -> None:
    _prepare(isolated_config)
    token = claim_holdout(isolated_config, "first evaluation")
    test = load_test_data(isolated_config, "hb_model", token, "proposal_fixed")
    assert len(test.y) > 0
    load_test_data(isolated_config, "ced_model", token, "primary")  # same claim covers every setup
    with pytest.raises(HoldoutAlreadyUsedError, match="first evaluation"):
        claim_holdout(isolated_config, "second look")


def test_test_data_needs_a_token(isolated_config: dict, tmp_path: Path) -> None:
    _prepare(isolated_config)
    with pytest.raises(HoldoutAlreadyUsedError):
        load_test_data(isolated_config, "hb_model", HoldoutToken(str(tmp_path / "missing.json"), "now"))


def test_training_and_test_rows_are_disjoint(isolated_config: dict) -> None:
    _prepare(isolated_config)
    train = load_training_data(isolated_config, "hb_model", "primary")
    token = claim_holdout(isolated_config, "disjointness test")
    test = load_test_data(isolated_config, "hb_model", token, "primary")
    assert not set(train.anon_ids) & set(test.anon_ids)
