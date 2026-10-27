"""Hyperparameter tuning with Optuna or GridSearchCV (config ``tuning.backend``).

Both backends run on the hold-out TRAINING set only and score every
candidate with the full pipeline (imputation, resampling, model) refitted in
each tuning fold, so resampling never touches a tuning validation fold.

The tuning folds use a different seed from the 5 x 5 evaluation folds.
Reusing the evaluation folds would select hyperparameters on exactly the
splits later used to compare models, which inflates the reported CV score of
whichever model had the larger search.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.evaluation import METRIC_DIRECTION, cv_evaluate

optuna.logging.set_verbosity(optuna.logging.WARNING)

SKLEARN_SCORING: dict[str, str] = {
    "mae": "neg_mean_absolute_error",
    "average_precision": "average_precision",
    "auc_roc": "roc_auc",
    "macro_f1": "f1_macro",
}


@dataclass
class TuningResult:
    """Best parameters (without the ``model__`` prefix), best CV score, and the trial log."""

    best_params: dict[str, Any]
    best_score: float
    trials: pd.DataFrame
    backend: str


def suggest_params(trial: optuna.Trial, space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Draw one parameter set from a config search space."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unknown parameter type '{kind}' for {name}")
    return params


def grid_from_space(space: Mapping[str, Mapping[str, Any]], points: int = 3) -> dict[str, list[Any]]:
    """Turn a search space into an explicit grid with ``points`` values per numeric parameter."""
    grid: dict[str, list[Any]] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "categorical":
            grid[name] = list(spec["choices"])
        elif kind == "int":
            values = np.linspace(int(spec["low"]), int(spec["high"]), points)
            grid[name] = sorted({int(round(v)) for v in values})
        elif kind == "float":
            low, high = float(spec["low"]), float(spec["high"])
            values = np.geomspace(low, high, points) if spec.get("log") else np.linspace(low, high, points)
            grid[name] = [float(v) for v in values]
        else:
            raise ValueError(f"Unknown parameter type '{kind}' for {name}")
    return grid


def tune(
    make_pipeline: Callable[[Mapping[str, Any]], Any],
    space: Mapping[str, Mapping[str, Any]] | None,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: Any,
    task: str,
    metric: str,
    n_trials: int,
    seed: int,
    backend: str = "optuna",
    grid_points: int = 3,
    class_names: list[str] | None = None,
) -> TuningResult:
    """Search hyperparameters and return the best set.

    ``make_pipeline(params)`` must return an unfitted pipeline whose model
    receives ``params``. A model without a search space (the Dummy baselines)
    is scored once with empty parameters so its CV score is still logged.
    """
    direction = METRIC_DIRECTION[metric]

    def score(params: Mapping[str, Any]) -> float:
        result = cv_evaluate(lambda: make_pipeline(params), X, y, cv, task, class_names=class_names)
        return result.mean(metric)

    if not space:
        value = score({})
        return TuningResult({}, value, pd.DataFrame([{"trial": 0, "value": value}]), "none")

    if backend == "optuna":
        study = optuna.create_study(
            direction="maximize" if direction > 0 else "minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(lambda trial: score(suggest_params(trial, space)), n_trials=n_trials, show_progress_bar=False)
        trials = study.trials_dataframe(attrs=("number", "value", "params"))
        return TuningResult(dict(study.best_params), float(study.best_value), trials, "optuna")

    if backend == "grid":
        if metric not in SKLEARN_SCORING:
            raise ValueError(f"Grid backend has no sklearn scorer for metric '{metric}'")
        grid = {f"model__{k}": v for k, v in grid_from_space(space, grid_points).items()}
        n_combos = int(np.prod([len(v) for v in grid.values()]))
        search = GridSearchCV(make_pipeline({}), grid, scoring=SKLEARN_SCORING[metric], cv=cv, n_jobs=1, refit=False)
        search.fit(X, y)
        best = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
        best_score = float(search.best_score_) * (-1 if metric == "mae" else 1)
        trials = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
        trials["n_combinations"] = n_combos
        return TuningResult(best, best_score, trials, "grid")

    raise ValueError(f"Unknown tuning backend '{backend}'")

