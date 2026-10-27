"""WHO thresholds, CED cut-off, and the Hb == 0 and Hb > 20 rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels import anemia_binary, build_labels, ced_label, who_anemia_class


@pytest.mark.parametrize(
    ("hb", "expected"),
    [
        (11.0, "Normal"),
        (10.99, "Mild"),
        (10.0, "Mild"),
        (9.99, "Moderate"),
        (7.0, "Moderate"),
        (6.99, "Severe"),
        (14.2, "Normal"),
        (0.5, "Severe"),
    ],
)
def test_who_boundaries(hb: float, expected: str) -> None:
    assert who_anemia_class([hb])[0] == expected


def test_who_missing_is_none() -> None:
    assert who_anemia_class([np.nan])[0] is None


def test_anemia_binary_boundary() -> None:
    out = anemia_binary([10.99, 11.0, np.nan])
    assert out[0] == 1.0 and out[1] == 0.0 and np.isnan(out[2])


def test_ced_boundary() -> None:
    out = ced_label([23.49, 23.5, 30.0, np.nan])
    assert list(out[:3]) == [1.0, 0.0, 0.0] and np.isnan(out[3])


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hb": [0.0, 0.0, 25.0, 10.5, 12.0, np.nan],
            "muac": [22.0, 25.0, 24.0, 10.0, 26.0, 23.0],
            "hb_ever_tested": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_policy_missing(config: dict) -> None:
    config["cleaning"]["hb_zero_policy"] = "missing"
    out, report = build_labels(_frame(), config)
    assert out["hb_target"].isna().tolist() == [True, True, True, False, False, True]
    assert out["anemia_class"].tolist()[:2] == [None, None]
    assert report["n_hb_zero"] == 2 and report["n_hb_zero_reporting_never_tested"] == 1
    assert report["n_severe_total"] == 0
    assert "hb" not in out.columns


def test_policy_severe_drop(config: dict) -> None:
    config["cleaning"]["hb_zero_policy"] = "severe_drop"
    out, report = build_labels(_frame(), config)
    assert out["anemia_class"].tolist()[:2] == ["Severe", "Severe"]
    assert out["hb_target"].iloc[:2].isna().all()
    assert report["n_severe_from_hb_zero"] == 2


def test_policy_severe_floor(config: dict) -> None:
    config["cleaning"]["hb_zero_policy"] = "severe_floor"
    config["cleaning"]["hb_zero_floor_value"] = 5.0
    out, _ = build_labels(_frame(), config)
    assert out["hb_target"].iloc[:2].tolist() == [5.0, 5.0]
    assert out["anemia_binary"].iloc[:2].tolist() == [1.0, 1.0]


def test_hb_above_max_has_no_label_but_keeps_ced(config: dict) -> None:
    out, report = build_labels(_frame(), config)
    row = out.iloc[2]
    assert np.isnan(row["hb_target"]) and row["anemia_class"] is None
    assert row["ced"] == 0.0
    assert report["n_hb_above_max_dropped_from_hb_model"] == 1


def test_invalid_muac_has_no_ced_label(config: dict) -> None:
    out, report = build_labels(_frame(), config)
    assert np.isnan(out["ced"].iloc[3]) and report["n_muac_invalid"] == 1


def test_unknown_policy_raises(config: dict) -> None:
    config["cleaning"]["hb_zero_policy"] = "impute"
    with pytest.raises(ValueError):
        build_labels(_frame(), config)
