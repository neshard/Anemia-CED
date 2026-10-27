"""Optuna search spaces.

The tuning loop itself lives in ``src.nested_cv.tune_inner``: it runs only on
the inner 3-fold splits of an outer training set, so the outer test fold
never influences a hyperparameter. Every tuned model gets the same number of
trials (config tuning.n_trials).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial: optuna.Trial, space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Draw one parameter set from a config search space."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                               log=bool(spec.get("log", False)))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unknown parameter type '{kind}' for {name}")
    return params


def make_sampler(seed: int) -> optuna.samplers.BaseSampler:
    """Seeded TPE sampler so every tuning run is reproducible."""
    return optuna.samplers.TPESampler(seed=seed)
