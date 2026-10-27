"""Checks on the real dataset. Skipped when data/raw/dataset.csv is absent.

Assertions are on aggregate properties only. No test prints a patient row.
"""

from __future__ import annotations

import pytest

from src.cleaning import clean_dataset
from src.utils import assert_no_target_leakage, forbidden_columns_for, resolve_path


@pytest.fixture(scope="module")
def real(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    from src.utils import load_config

    cfg = load_config()
    if not resolve_path(cfg, "raw_data").exists():
        pytest.skip("real dataset not present")
    return cfg, clean_dataset(cfg)


def test_real_schema_and_targets(real) -> None:  # type: ignore[no-untyped-def]
    cfg, result = real
    df, rep = result.frame, result.report
    assert rep["n_rows_analysis"] <= rep["n_rows_raw"]
    assert df["anon_id"].is_unique
    assert "respondent_code" not in df.columns
    assert df["hfias_score"].dropna().between(0, 27).all()
    if cfg["cleaning"]["hb_zero_policy"] == "missing":
        assert rep["n_severe_from_hb_zero"] == 0


def test_real_feature_matrices_have_no_targets(real) -> None:  # type: ignore[no-untyped-def]
    cfg, result = real
    for model in ("hb_model", "ced_model"):
        X = result.frame[cfg["features"]["candidate_pool"]]
        assert_no_target_leakage(X, forbidden_columns_for(cfg, model), context=model)


def test_real_sample_sizes_match_protocol(real) -> None:  # type: ignore[no-untyped-def]
    cfg, result = real
    rep, df = result.report, result.frame
    assert rep["n_rows_raw"] == 420
    assert rep["n_rows_raw"] - rep["n_duplicate_records_dropped"] == 419
    assert int((df["hb_target"].notna() & df["anemia_class"].notna()).sum()) == 396
    assert int(df["ced"].notna().sum()) == 410


def test_expected_counts_mismatch_stops(real) -> None:  # type: ignore[no-untyped-def]
    import copy

    from src.cleaning import DataSchemaError

    cfg, _ = real
    bad = copy.deepcopy(cfg)
    bad["data"]["expected_counts"]["hb_model"] = 395
    with pytest.raises(DataSchemaError, match="hb_model"):
        clean_dataset(bad)
