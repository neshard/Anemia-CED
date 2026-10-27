"""HFIAS scoring per Coates et al. 2007."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.hfias import HFIASBinaryWarning, HFIASFormatError, hfias_category, score_hfias

ANSWERS = {0: "Tidak", 1: "Jarang (1-2x)", 2: "Kadang-kadang (3-10x)", 3: "Sering (>10x)"}
ITEMS = [f"hfias_q{i}" for i in range(1, 10)]


def _frame(rows: list[list[int]]) -> pd.DataFrame:
    """Rows of nine codes. A frame always includes all four answers so it never looks binary."""
    return pd.DataFrame([[ANSWERS[c] for c in row] for row in rows], columns=ITEMS)


ALL_LEVELS = [0, 1, 2, 3, 0, 0, 0, 0, 0]


def test_score_range_extremes(config: dict) -> None:
    res = score_hfias(_frame([[0] * 9, [3] * 9, ALL_LEVELS]), config["hfias"])
    assert res.score.tolist() == [0.0, 27.0, 6.0]
    assert res.is_valid_hfias


def test_unknown_answer_raises(config: dict) -> None:
    df = _frame([ALL_LEVELS, [0] * 9])
    df.iloc[1, 0] = "Mungkin"
    with pytest.raises(HFIASFormatError, match="Mungkin"):
        score_hfias(df, config["hfias"])


def test_missing_policy_nan_and_zero(config: dict) -> None:
    df = _frame([ALL_LEVELS, [1] * 9])
    df.iloc[1, 4] = np.nan
    assert np.isnan(score_hfias(df, config["hfias"]).score.iloc[1])
    cfg = copy.deepcopy(config["hfias"])
    cfg["missing_policy"] = "zero"
    assert score_hfias(df, cfg).score.iloc[1] == 8.0


def test_binary_data_flagged_in_combined_mode(config: dict) -> None:
    df = pd.DataFrame([["Ya"] * 9, ["Tidak"] * 9], columns=ITEMS)
    with pytest.raises(HFIASFormatError, match="binary"):
        score_hfias(df, config["hfias"])


def test_binary_mode_requires_opt_in_and_warns(config: dict) -> None:
    df = pd.DataFrame([["Ya"] * 9, ["Tidak"] * 9], columns=ITEMS)
    cfg = copy.deepcopy(config["hfias"])
    cfg["format"] = "binary"
    with pytest.raises(HFIASFormatError):
        score_hfias(df, cfg)
    cfg["allow_binary"] = True
    with pytest.warns(HFIASBinaryWarning):
        res = score_hfias(df, cfg)
    assert res.score.tolist() == [9.0, 0.0] and not res.is_valid_hfias


def test_split_format_ignores_frequency_when_no(config: dict) -> None:
    occ = [f"occ{i}" for i in range(9)]
    freq = [f"freq{i}" for i in range(9)]
    df = pd.DataFrame(
        [
            ["Tidak"] * 9 + ["Sering"] * 9,          # frequency filled by mistake, must be ignored
            ["Ya"] * 9 + ["Kadang-kadang"] * 9,
            ["Ya"] + ["Tidak"] * 8 + ["Jarang"] + [None] * 8,
        ],
        columns=occ + freq,
    )
    cfg = copy.deepcopy(config["hfias"])
    cfg.update(format="split", occurrence_columns=occ, frequency_columns=freq)
    res = score_hfias(df, cfg)
    assert res.score.tolist() == [0.0, 18.0, 1.0]


def test_split_yes_without_frequency_is_missing(config: dict) -> None:
    occ = [f"occ{i}" for i in range(9)]
    freq = [f"freq{i}" for i in range(9)]
    df = pd.DataFrame([["Ya"] * 9 + [None] * 9], columns=occ + freq)
    cfg = copy.deepcopy(config["hfias"])
    cfg.update(format="split", occurrence_columns=occ, frequency_columns=freq)
    assert np.isnan(score_hfias(df, cfg).score.iloc[0])


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        ([0] * 9, 0),
        ([1, 0, 0, 0, 0, 0, 0, 0, 0], 0),   # rarely worried only is still food secure
        ([2, 0, 0, 0, 0, 0, 0, 0, 0], 1),
        ([0, 1, 1, 1, 0, 0, 0, 0, 0], 1),
        ([0, 0, 2, 0, 0, 0, 0, 0, 0], 2),
        ([0, 0, 0, 0, 1, 2, 0, 0, 0], 2),
        ([0, 0, 0, 0, 3, 0, 0, 0, 0], 3),
        ([0, 0, 0, 0, 0, 0, 1, 0, 0], 3),
        ([3, 3, 3, 3, 2, 2, 0, 0, 0], 2),
    ],
)
def test_coates_categories(codes: list[int], expected: int) -> None:
    items = pd.DataFrame([codes], columns=ITEMS, dtype="Float64")
    assert int(hfias_category(items).iloc[0]) == expected
