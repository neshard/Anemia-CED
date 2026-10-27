"""Shared fixtures. All fixtures build synthetic data so tests never touch patient records."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402


@pytest.fixture()
def config() -> dict[str, Any]:
    """A fresh copy of the project config for each test.

    Synthetic test data has its own size, so the real-data row-count check is disabled here.
    """
    cfg = load_config()
    cfg["data"]["expected_row_count"] = None
    cfg["data"]["expected_counts"] = None
    return cfg


HFIAS_ANSWERS = ["Tidak", "Jarang (1-2x)", "Kadang-kadang (3-10x)", "Sering (>10x)"]


def synthetic_raw_frame(config: dict[str, Any], n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a raw-format frame using the configured headers and plausible synthetic values."""
    rng = np.random.default_rng(seed)
    cols = config["columns"]
    hb = np.round(rng.normal(11.3, 1.3, n), 1)
    muac = np.round(rng.normal(25.0, 2.8, n), 1)
    data: dict[str, Any] = {
        cols["respondent_code"]: [f"X-{i}" for i in range(n)],
        cols["health_center"]: rng.choice([f"PKM{i}" for i in range(6)], n),
        cols["district"]: rng.choice(["D1", "D2", "D3"], n),
        cols["age"]: rng.integers(17, 42, n).astype(str),
        cols["n_children"]: rng.integers(0, 4, n).astype(str),
        cols["n_under5"]: rng.integers(0, 2, n).astype(str),
        cols["education_mother"]: rng.choice(list(config["encodings"]["education"])[:5], n),
        cols["education_husband"]: rng.choice(list(config["encodings"]["education"])[:5], n),
        cols["income"]: rng.choice(list(config["encodings"]["income"]), n),
        cols["gpa_raw"]: [f"G{g}P{g - 1}A0" for g in rng.integers(1, 5, n)],
        cols["gestational_age_raw"]: rng.integers(6, 40, n).astype(str),
        cols["muac"]: [str(v).replace(".", ",") for v in muac],
        cols["prepreg_weight"]: rng.integers(38, 80, n).astype(str),
        cols["height"]: rng.integers(140, 168, n).astype(str),
        cols["hb"]: [str(v).replace(".", ",") for v in hb],
        cols["hb_ever_tested"]: ["Ya, pernah"] * n,
        cols["anc_visits"]: rng.integers(1, 9, n).astype(str),
        cols["food_taboo"]: rng.choice(["Ya", "Tidak"], n, p=[0.1, 0.9]),
        cols["egg_freq"]: rng.choice(["Tidak pernah", "1x/minggu", "2-3x/minggu", "?1x/hari"], n),
        cols["hb_date_raw"]: [f"{d}/11/2025" for d in rng.integers(1, 29, n)],
        cols["interview_date_raw"]: [f"{d}/12/2025" for d in rng.integers(1, 29, n)],
    }
    for i in range(1, 10):
        data[cols[f"hfias_q{i}"]] = rng.choice(HFIAS_ANSWERS, n, p=[0.8, 0.1, 0.06, 0.04])
    data["Titik Koordinat GPS (Latitude)"] = ["-8.3"] * n
    return pd.DataFrame(data)


@pytest.fixture()
def raw_csv(tmp_path: Path, config: dict[str, Any]) -> Path:
    """Write a synthetic raw CSV with the project's separator and encoding."""
    path = tmp_path / "dataset.csv"
    synthetic_raw_frame(config).to_csv(
        path, sep=config["data"]["csv_sep"], encoding=config["data"]["csv_encoding"], index=False
    )
    return path


@pytest.fixture()
def isolated_config(tmp_path: Path, config: dict[str, Any], raw_csv: Path) -> dict[str, Any]:
    """Config whose data and split paths point into a temporary directory."""
    cfg = dict(config)
    cfg["paths"] = {**config["paths"], "raw_data": str(raw_csv), "reports_tables": str(tmp_path / "tables"),
                    "artifacts_dir": str(tmp_path / "artifacts"), "results_dir": str(tmp_path / "results")}
    return cfg
