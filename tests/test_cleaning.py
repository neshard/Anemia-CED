"""Parsers, plausibility ranges, schema checks, and privacy of the cleaned frame."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    DataSchemaError,
    UnmappedCategoryError,
    apply_plausible_ranges,
    clean_dataset,
    load_raw,
    map_categories,
    parse_gestational_age,
    parse_gpa,
    to_numeric_flexible,
)


def _same(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all((math.isnan(x) and math.isnan(y)) or x == y for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("G2P1A0", (2, 1, 0)),
        ("G2P1A0 ", (2, 1, 0)),
        ("G_3 P_2 A_0", (3, 2, 0)),
        ("G(2) P(1) ", (2, 1, np.nan)),
        ("G(1) ", (1, np.nan, np.nan)),
        ("2/1/0", (2, 1, 0)),
        ("2,1,0", (2, 1, 0)),
        ("2_1_0", (2, 1, 0)),
        ("3-1-1", (3, 1, 1)),
        ("G-2 P-1 A-0", (2, 1, 0)),
        ("G: 4 P:3 A:0", (4, 3, 0)),
        ("G9p8a0", (9, 8, 0)),
        ("G/2/P/1/A/0", (2, 1, 0)),
        ("Kehamilan ke 3, kelahiran 2, Abortus 0", (3, 2, 0)),
        ("G(2) P(1, sudah meninggal) ", (2, 1, np.nan)),
        ("-", (np.nan, np.nan, np.nan)),
        ("", (np.nan, np.nan, np.nan)),
        ("G1P3A0", (np.nan, np.nan, np.nan)),   # para above gravida is inconsistent
    ],
)
def test_parse_gpa(text: str, expected: tuple[float, float, float]) -> None:
    assert _same(parse_gpa(text), tuple(float(v) for v in expected))


def test_parse_gpa_bare_number_policy() -> None:
    assert _same(parse_gpa("3", "nan"), (np.nan, np.nan, np.nan))
    assert _same(parse_gpa("3", "gravida"), (3.0, np.nan, np.nan))
    with pytest.raises(ValueError):
        parse_gpa("3", "guess")


@pytest.mark.parametrize(
    ("text", "unit", "expected"),
    [
        ("20", "weeks", 20.0),
        ("5", "months", 5 * 4.345),
        ("20 minggu", "months", 20.0),
        ("5 bulan", "weeks", 5 * 4.345),
        ("6 bln", "weeks", 6 * 4.345),
        ("24 mgg", "weeks", 24.0),
        ("7,5 bulan", "weeks", 7.5 * 4.345),
        ("tidak tahu", "weeks", np.nan),
    ],
)
def test_parse_gestational_age(text: str, unit: str, expected: float) -> None:
    out = parse_gestational_age(text, unit, 4.345)
    assert (np.isnan(out) and np.isnan(expected)) or out == pytest.approx(expected)


def test_numeric_flexible_accepts_comma_and_dot() -> None:
    out = to_numeric_flexible(pd.Series(["10,5", "11.2", "12", "abc", None]))
    assert out.iloc[:3].tolist() == [10.5, 11.2, 12.0] and out.iloc[3:].isna().all()


def test_plausible_ranges_set_nan_and_count() -> None:
    df = pd.DataFrame({"height": [0.0, 150.0, 256.0], "age": [3.0, 25.0, np.nan]})
    out, counts = apply_plausible_ranges(df, {"height": [120, 200], "age": [12, 55]})
    assert out["height"].isna().tolist() == [True, False, True]
    assert counts == {"height": 2, "age": 1}
    assert df["height"].iloc[0] == 0.0  # input is not modified in place


def test_map_categories_rejects_unknown() -> None:
    with pytest.raises(UnmappedCategoryError):
        map_categories(pd.Series(["SD", "S3"]), {"SD": 1}, "education_mother")


def test_load_raw_missing_column(config: dict, raw_csv: Path) -> None:
    config["columns"]["age"] = "Kolom yang tidak ada"
    with pytest.raises(DataSchemaError, match="age"):
        load_raw(config, raw_csv)


def test_load_raw_drops_identifiers_and_unmapped(config: dict, raw_csv: Path) -> None:
    frame, info = load_raw(config, raw_csv)
    assert "respondent_code" not in frame.columns
    assert not any("GPS" in c or "Latitude" in c for c in frame.columns)
    assert frame["anon_id"].is_unique and frame["anon_id"].str.fullmatch(r"P\d{4}").all()
    assert info["n_rows_raw"] == len(frame)


def test_clean_dataset_end_to_end_synthetic(isolated_config: dict) -> None:
    result = clean_dataset(isolated_config)
    df = result.frame
    for col in ("hb_target", "anemia_class", "ced", "hfias_score", "gravida", "gestational_age_weeks"):
        assert col in df.columns
    assert "hb" not in df.columns
    assert df["hfias_score"].between(0, 27).all()
    assert df["egg_freq"].dropna().isin([0, 1, 2, 3, 4, 5]).all()
    assert not any(c.startswith("hfias_q") for c in df.columns)
    assert result.report["hfias_is_valid_score"] is True


def test_duplicate_records_dropped_by_default(config: dict, raw_csv: Path) -> None:
    df = pd.read_csv(raw_csv, sep=";", encoding="cp1252", dtype=str)
    dup = df.iloc[[0]].copy()
    dup[config["columns"]["respondent_code"]] = "OTHER-CODE"  # same woman, different code
    pd.concat([df, dup]).to_csv(raw_csv, sep=";", encoding="cp1252", index=False)
    frame, info = load_raw(config, raw_csv)
    assert info["n_duplicate_records_found"] == 1 and len(frame) == len(df)
    assert frame["anon_id"].is_unique
    config["cleaning"]["drop_duplicate_records"] = False
    frame, info = load_raw(config, raw_csv)
    assert len(frame) == len(df) + 1 and info["n_duplicate_records_dropped"] == 0
