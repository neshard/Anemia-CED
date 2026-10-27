"""Per-fold feature selection by rank aggregation (proposal Section 3.3).

Rule, fixed before the final experiments:

* Mutual information (MI) and random forest importance (RF, 300 trees) each
  rank all candidates from 1 (best). The two ranks are averaged.
* Pre-pregnancy weight, ANC frequency and the HFIAS score are always kept.
* The 4 remaining features with the lowest average rank complete the set,
  so every fold uses exactly 7 features.
* Ties are broken by the lower MI rank, then alphabetically.

The selector is a pipeline step, so it is refitted on every outer training
fold and inside every inner split used for tuning. It never sees the rows it
is evaluated on. Selection does not depend on model hyperparameters, so the
result is cached by a hash of its training data and reused across Optuna
trials. A cache hit returns exactly what a fresh fit would.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

EXPLORATORY_LABEL = "EXPLORATORY: computed once on all eligible records. Not used to build any model."


def rank_features(
    X: pd.DataFrame,
    y: Any,
    task: str,
    n_estimators: int = 300,
    discrete_features: Sequence[str] = (),
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank every column by MI and RF importance and order by the approved rule."""
    if X.isna().any().any():
        raise ValueError("rank_features needs imputed input. Run it after the imputer step.")
    y_arr = np.asarray(y)
    discrete_mask = np.array([c in set(discrete_features) for c in X.columns])
    if task == "regression":
        mi = mutual_info_regression(X, y_arr, discrete_features=discrete_mask, random_state=random_state)
        forest: Any = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=1)
    elif task == "classification":
        mi = mutual_info_classif(X, y_arr, discrete_features=discrete_mask, random_state=random_state)
        forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=1)
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")
    forest.fit(X, y_arr)
    table = pd.DataFrame({"feature": X.columns, "mi": mi, "rf_importance": forest.feature_importances_})
    table["mi_rank"] = table["mi"].rank(ascending=False, method="average")
    table["rf_rank"] = table["rf_importance"].rank(ascending=False, method="average")
    table["mean_rank"] = (table["mi_rank"] + table["rf_rank"]) / 2
    table = table.sort_values(["mean_rank", "mi_rank", "feature"], ignore_index=True)
    table["position"] = np.arange(1, len(table) + 1)
    return table


def choose_top_k(ranking: pd.DataFrame, k: int, forced_include: Iterable[str] = ()) -> list[str]:
    """Forced features first, then the best remaining by the ranking order, exactly ``k`` in total."""
    available = list(ranking["feature"])
    forced = [f for f in forced_include if f in available]
    if len(forced) > k:
        raise ValueError(f"forced_include has {len(forced)} features but k is {k}")
    rest = [f for f in available if f not in forced]
    chosen = forced + rest[: k - len(forced)]
    if len(chosen) != min(k, len(available)):
        raise ValueError(f"Selected {len(chosen)} features, expected {k}")
    return chosen


_SELECTION_CACHE: dict[str, tuple[pd.DataFrame, list[str]]] = {}


def _cache_key(X: pd.DataFrame, y: Any, settings: tuple[Any, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    digest.update(np.ascontiguousarray(np.asarray(y, dtype=float)).tobytes())
    digest.update(repr((list(X.columns), settings)).encode())
    return digest.hexdigest()


def clear_selection_cache() -> None:
    """Empty the cache (used by tests)."""
    _SELECTION_CACHE.clear()


class RankAggregationSelector(BaseEstimator, TransformerMixin):
    """Pipeline step: keep forced features plus the best ranked others, ``k`` in total."""

    def __init__(
        self,
        k: int = 7,
        task: str = "regression",
        n_estimators: int = 300,
        forced_include: Sequence[str] = (),
        discrete_features: Sequence[str] = (),
        random_state: int = 42,
    ) -> None:
        self.k = k
        self.task = task
        self.n_estimators = n_estimators
        self.forced_include = forced_include
        self.discrete_features = discrete_features
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any) -> RankAggregationSelector:
        """Select on the training data passed in only."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RankAggregationSelector expects a DataFrame with feature names")
        settings = (self.k, self.task, self.n_estimators, tuple(self.forced_include),
                    tuple(self.discrete_features), self.random_state)
        key = _cache_key(X, y, settings)
        if key not in _SELECTION_CACHE:
            ranking = rank_features(X, y, self.task, self.n_estimators, self.discrete_features, self.random_state)
            _SELECTION_CACHE[key] = (ranking, choose_top_k(ranking, self.k, self.forced_include))
        ranking, selected = _SELECTION_CACHE[key]
        self.ranking_ = ranking.copy()
        self.selected_features_ = list(selected)
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.selected_features_)


def selection_frequency(selected_per_fold: Sequence[Sequence[str]], candidates: Sequence[str],
                        min_share: float) -> pd.DataFrame:
    """How often each candidate was selected across outer folds, and the dashboard list flag."""
    n = len(selected_per_fold)
    counts = {c: sum(c in fold for fold in selected_per_fold) for c in candidates}
    table = pd.DataFrame({"feature": list(counts), "times_selected": list(counts.values())})
    table["n_folds"] = n
    table["share"] = (table["times_selected"] / n).round(4)
    table["in_dashboard_list"] = table["share"] >= min_share
    return table.sort_values(["share", "feature"], ascending=[False, True], ignore_index=True)


def exploratory_ranking(X: pd.DataFrame, y: Any, imputer: Any, task: str, n_estimators: int = 300,
                        discrete_features: Sequence[str] = (), random_state: int = 42) -> pd.DataFrame:
    """One ranking on all eligible records, clearly labelled as exploratory and never used for a model."""
    imputed = imputer.fit_transform(X)
    table = rank_features(imputed, y, task, n_estimators, discrete_features, random_state)
    table.insert(0, "note", EXPLORATORY_LABEL)
    return table
