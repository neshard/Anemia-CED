"""Shared utilities: configuration, seeding, version logging, and guards.

The two guards in this module (``assert_no_target_leakage`` and
``assert_no_identifiers``) are called by every module that builds a feature
matrix or writes a table. They exist so that a leakage or privacy mistake
stops the run instead of silently producing an optimistic or unsafe result.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import platform
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# joblib warns on some Windows machines when it cannot count physical cores.
# Setting the count explicitly silences that and keeps parallel runs deterministic in size.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"

TRACKED_LIBRARIES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "imblearn",
    "xgboost",
    "shap",
    "smogn",
    "optuna",
    "statsmodels",
    "matplotlib",
    "yaml",
    "joblib",
    "streamlit",
)


class LeakageError(AssertionError):
    """Raised when a target or a target-derived column reaches a feature matrix."""


class PrivacyError(AssertionError):
    """Raised when an identifier column is about to be printed or exported."""


class HypothesesNotLockedError(RuntimeError):
    """Raised when an experiment starts before config/hypotheses.yaml is locked."""


def get_logger(name: str = "anemia_ced") -> logging.Logger:
    """Return a module logger with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration.

    Relative paths inside ``paths`` are resolved against the project root so
    scripts behave the same whatever the current working directory is. The
    environment variable ``ANEMIA_CED_CONFIG`` selects another config file,
    which is how the synthetic smoke run points every script at fake data.
    """
    if path is not None:
        config_path = Path(path)
    elif os.environ.get("ANEMIA_CED_CONFIG"):
        config_path = Path(os.environ["ANEMIA_CED_CONFIG"])
    else:
        config_path = DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    return config


def resolve_path(config: Mapping[str, Any], key: str) -> Path:
    """Return the absolute path for ``config['paths'][key]``."""
    raw = Path(config["paths"][key])
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def load_hypotheses(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read the pre-specified hypotheses file."""
    with resolve_path(config, "hypotheses").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_locked_hypotheses(config: Mapping[str, Any]) -> dict[str, Any]:
    """Stop unless hypotheses are locked, so success criteria are fixed before any result exists.

    The synthetic smoke run (``project.smoke_test: true``) is exempt because
    it produces no real result.
    """
    hypotheses = load_hypotheses(config)
    if config["project"].get("smoke_test"):
        return hypotheses
    if hypotheses.get("status") != "locked":
        raise HypothesesNotLockedError(
            "config/hypotheses.yaml has status '" + str(hypotheses.get("status")) + "'. Review the "
            "hypotheses and success criteria, set status: locked and locked_on: <date>, then rerun."
        )
    return hypotheses


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and the hash seed.

    Estimators still receive ``random_state`` explicitly. This function only
    covers code paths that draw from the global generators.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def library_versions(libraries: Iterable[str] = TRACKED_LIBRARIES) -> dict[str, str]:
    """Return installed versions of the tracked libraries plus Python itself."""
    versions: dict[str, str] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in libraries:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not installed"
    return versions


def log_library_versions(config: Mapping[str, Any]) -> dict[str, str]:
    """Write library versions to the configured JSON file and return them."""
    versions = library_versions()
    out_path = resolve_path(config, "versions_file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(versions, indent=2), encoding="utf-8")
    get_logger().info("Library versions written to %s", out_path)
    return versions


def file_fingerprint(path: str | Path) -> str:
    """SHA-256 of a file. Used to detect that the raw data changed under a saved split."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_name(column: str) -> str:
    """Strip ColumnTransformer prefixes such as ``num__`` from a column name."""
    return str(column).split("__")[-1]


def _matches_forbidden(column: str, forbidden: str) -> bool:
    """True if ``column`` is ``forbidden`` or a one-hot style expansion of it.

    The prefix rule is deliberately broad. A future feature whose name starts
    with ``hb_`` or ``muac_`` will be rejected and must be renamed, which is
    preferable to letting a target derivative through.
    """
    base = _base_name(column)
    return base == forbidden or base.startswith(f"{forbidden}_")


def forbidden_columns_for(config: Mapping[str, Any], model: str) -> list[str]:
    """Return the full forbidden list for ``model`` ("hb_model" or "ced_model")."""
    spec = config["forbidden_feature_columns"]
    combined = list(spec.get("all_models", [])) + list(spec.get(model, []))
    combined += list(config.get("identifier_columns", []))
    return sorted(set(combined))


def assert_no_target_leakage(
    columns: Iterable[str] | pd.DataFrame,
    forbidden: Iterable[str],
    context: str = "feature matrix",
) -> None:
    """Fail the run if any forbidden column is present.

    Called on every feature matrix right before ``fit`` or ``predict``.
    The check covers raw names, one-hot expansions (``anemia_class_Mild``) and
    ColumnTransformer prefixes (``num__hb``). Hb must never feed the anemia
    model and MUAC must never feed the CED model, because either would turn a
    screening model into a lookup of the measured outcome.
    """
    names = list(columns.columns) if isinstance(columns, pd.DataFrame) else list(columns)
    hits = sorted({name for name in names for bad in forbidden if _matches_forbidden(name, bad)})
    if hits:
        raise LeakageError(f"Forbidden columns found in {context}: {hits}")


def assert_feature_lists_safe(config: Mapping[str, Any]) -> None:
    """Validate every configured feature list against the forbidden lists at startup.

    Covers the candidate pool, the forced includes and the fixed preliminary
    sets, so a target cannot enter the model through any configured list.
    """
    both = sorted(set(forbidden_columns_for(config, "hb_model")) | set(forbidden_columns_for(config, "ced_model")))
    pool = list(config["features"]["candidate_pool"])
    assert_no_target_leakage(pool, both, context="config features.candidate_pool")
    forced = list(config["feature_selection"]["forced_include"])
    assert_no_target_leakage(forced, both, context="config feature_selection.forced_include")
    outside = [f for f in forced if f not in pool]
    if outside:
        raise ValueError(f"feature_selection.forced_include has columns outside candidate_pool: {outside}")
    for model, cols in config["features"].get("fixed_preliminary", {}).items():
        assert_no_target_leakage(cols, forbidden_columns_for(config, model), f"features.fixed_preliminary.{model}")
        outside = [f for f in cols if f not in pool]
        if outside:
            raise ValueError(f"features.fixed_preliminary.{model} has columns outside candidate_pool: {outside}")


def assert_no_identifiers(df: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """Fail if a frame about to be exported still carries identifier columns."""
    present = [c for c in config.get("identifier_columns", []) if c in df.columns]
    if present:
        raise PrivacyError(f"Identifier columns must not be exported: {present}")


def save_json(obj: Any, path: Path) -> Path:
    """Write JSON with NaN turned into null."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    path.write_text(text.replace("NaN", "null"), encoding="utf-8")
    return path


def save_table(df: pd.DataFrame, name: str, config: Mapping[str, Any], index: bool = False) -> Path:
    """Save an aggregate table as CSV in reports/tables after a privacy check."""
    assert_no_identifiers(df, config)
    out_dir = resolve_path(config, "reports_tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=index)
    return out_path
