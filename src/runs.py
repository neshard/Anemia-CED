"""Building run specifications and saving run outputs to results/.

Every main run is built by ``main_run`` so that XGBoost and every baseline
get identical folds, preprocessing, inner splits and tuning budget, and no
resampler. Exploratory and sensitivity runs change exactly one thing.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.models import ModelSpec
from src.nested_cv import AnalysisData, NestedResult, RunSpec, fold_table, run_nested
from src.utils import assert_no_identifiers, resolve_path


def main_run(spec: ModelSpec, config: Mapping[str, Any], **overrides: Any) -> RunSpec:
    """The protocol run: per-fold selection, no resampling, the shared trial budget, 5 x 5 outer CV."""
    base = {
        "spec": spec,
        "variant": "main",
        "resampler": "none",
        "feature_mode": "per_fold",
        "n_trials": int(config["tuning"]["n_trials"]),
        "outer": "stratified",
        "n_repeats": int(config["cv"]["outer_n_repeats"]),
    }
    base.update(overrides)
    return RunSpec(**base)


def results_dir(config: Mapping[str, Any], sub: str) -> Path:
    path = resolve_path(config, "results_dir") / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_tag(data: AnalysisData, run: RunSpec) -> str:
    return f"{data.analysis}__{data.target}__{run.spec.name}__{run.variant}"


def run_and_save(run: RunSpec, data: AnalysisData, config: Mapping[str, Any], *, spy: bool = False,
                 n_jobs: int | None = None) -> NestedResult:
    """Run (or resume) a nested analysis and write its out-of-fold predictions and fold table."""
    tag = run_tag(data, run)
    result = run_nested(run, data, config, spy=spy, n_jobs=n_jobs, label=tag)
    assert_no_identifiers(result.oof, config)
    result.oof.to_csv(results_dir(config, "oof") / f"{tag}.csv", index=False)
    fold_table(result).to_csv(results_dir(config, "folds") / f"{tag}.csv", index=False)
    return result


def load_oof(config: Mapping[str, Any], tag: str) -> pd.DataFrame:
    path = resolve_path(config, "results_dir") / "oof" / f"{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the script that produces {tag} first.")
    return pd.read_csv(path)


def load_folds(config: Mapping[str, Any], tag: str) -> pd.DataFrame:
    return pd.read_csv(resolve_path(config, "results_dir") / "folds" / f"{tag}.csv")


def save_result_table(df: pd.DataFrame, name: str, config: Mapping[str, Any]) -> Path:
    """Aggregate table in results/tables after a privacy check."""
    assert_no_identifiers(df, config)
    path = results_dir(config, "tables") / f"{name}.csv"
    df.to_csv(path, index=False)
    return path
