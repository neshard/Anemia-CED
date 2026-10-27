"""In-fold feature selection by rank aggregation of MI and RF importance.

The selector is a pipeline step, so it is refitted on every training fold
and never sees validation or test rows. Selecting features on the full
dataset and then cross-validating would leak, because the choice of features
would already reflect the validation rows.

Two reporting helpers sit next to it:

* ``selection_stability`` refits the selector in every CV training fold and
  counts how often each feature was chosen. This is the honest picture of how
  stable the selection is.
* ``exploratory_ranking`` ranks features once on the hold-out training set.
  It is labelled EXPLORATORY and is never used to build a model. It
  deliberately excludes the hold-out test set so even exploration cannot
  peek at the final evaluation data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import KFold, StratifiedKFold

EXPLORATORY_LABEL = "EXPLORATORY: computed once on the hold-out training set. Not used to build any model."


def rank_features(
    X: pd.DataFrame,
    y: Any,
    task: str,
    n_estimators: int = 300,
    discrete_features: Sequence[str] = (),
    random_state: int = 42,
) -> pd.DataFrame:
    """Score every column with mutual information and random forest impurity importance.

    Ranks use 1 for the best feature. The aggregate is the mean of the two
    ranks. Ties are broken by MI, then by name, so the result is
    deterministic. The HFIAS composite enters as one column, because the
    nine items were already summed during cleaning.
    """
    if X.isna().any().any():
        raise ValueError("rank_features needs imputed input. Run it after the imputer step.")
    y_arr = np.asarray(y)
    discrete_mask = np.array([c in set(discrete_features) for c in X.columns])
    if task == "regression":
        mi = mutual_info_regression(X, y_arr, discrete_features=discrete_mask, random_state=random_state)
        forest: BaseEstimator = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    elif task == "classification":
        mi = mutual_info_classif(X, y_arr, discrete_features=discrete_mask, random_state=random_state)
        forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")
    forest.fit(X, y_arr)

    table = pd.DataFrame(
        {"feature": X.columns, "mi": mi, "rf_importance": forest.feature_importances_}  # type: ignore[attr-defined]
    )
    table["mi_rank"] = table["mi"].rank(ascending=False, method="average")
    table["rf_rank"] = table["rf_importance"].rank(ascending=False, method="average")
    table["mean_rank"] = (table["mi_rank"] + table["rf_rank"]) / 2
    table = table.sort_values(["mean_rank", "mi", "feature"], ascending=[True, False, True], ignore_index=True)
    table["position"] = np.arange(1, len(table) + 1)
    return table


def choose_top_k(ranking: pd.DataFrame, k: int, forced_include: Iterable[str] = ()) -> list[str]:
    """Forced features first (when present), then the best remaining by aggregate rank, up to ``k``."""
    available = list(ranking["feature"])
    forced = [f for f in forced_include if f in available]
    if len(forced) > k:
        raise ValueError(f"forced_include has {len(forced)} features but k is {k}")
    rest = [f for f in available if f not in forced]
    return forced + rest[: max(k - len(forced), 0)]


# Fitted selections keyed by a hash of (training data, settings). Tuning trials
# reuse the same folds, and the selection does not depend on model
# hyperparameters, so recomputing it per trial would only waste time. The key
# covers every input, so a cache hit returns exactly what a fresh fit would.
_SELECTION_CACHE: dict[str, tuple[pd.DataFrame, list[str], pd.Series | None]] = {}


def _cache_key(X: pd.DataFrame, y: Any, settings: tuple[Any, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes())
    digest.update(np.ascontiguousarray(np.asarray(y, dtype=float)).tobytes())
    digest.update(repr((list(X.columns), settings)).encode())
    return digest.hexdigest()


def stability_select(
    X: pd.DataFrame,
    y: Any,
    task: str,
    k: int,
    inner_folds: int,
    min_selection_rate: float,
    max_features: int,
    forced_include: Sequence[str],
    n_estimators: int,
    discrete_features: Sequence[str],
    random_state: int,
) -> tuple[list[str], pd.Series]:
    """Keep features chosen in at least ``min_selection_rate`` of inner folds.

    Inside the data it receives (a CV training fold, or the whole training
    set for the final model) the selector runs ``inner_folds`` folds, ranks
    features on each inner training part and takes the top ``k`` (forced
    features first). Features reaching the rate are kept, ordered by rate and
    then by mean rank, forced features always first, at most
    ``max_features`` in total. Because the same rule runs inside every outer
    CV fold, the CV estimate describes exactly the procedure used for the
    final model.
    """
    y_arr = np.asarray(y)
    if task == "classification":
        splitter: Any = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        splits = splitter.split(X, y_arr)
    else:
        splitter = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        splits = splitter.split(X)
    counts = pd.Series(0, index=X.columns, dtype=float)
    rank_sum = pd.Series(0.0, index=X.columns)
    for train_idx, _ in splits:
        ranking = rank_features(X.iloc[train_idx], y_arr[train_idx], task, n_estimators, discrete_features, random_state)
        chosen = choose_top_k(ranking, k, forced_include)
        counts[chosen] += 1
        rank_sum += ranking.set_index("feature")["mean_rank"].reindex(X.columns)
    rates = counts / inner_folds
    mean_rank = rank_sum / inner_folds
    forced = [f for f in forced_include if f in X.columns]
    if len(forced) > max_features:
        raise ValueError(f"{len(forced)} forced features exceed max_features={max_features}")
    eligible = [f for f in X.columns if f not in forced and rates[f] >= min_selection_rate]
    eligible.sort(key=lambda f: (-rates[f], mean_rank[f], f))
    return forced + eligible[: max_features - len(forced)], rates.rename("inner_selection_rate")


class RankAggregationSelector(BaseEstimator, TransformerMixin):
    """Pipeline step that selects features by mean MI and RF rank.

    With ``inner_folds`` of 0 it keeps the top ``k`` ranked on the data it is
    fitted on. With ``inner_folds`` of 2 or more it applies the stability
    rule of ``stability_select``.
    """

    def __init__(
        self,
        k: int = 7,
        task: str = "regression",
        n_estimators: int = 300,
        forced_include: Sequence[str] = (),
        discrete_features: Sequence[str] = (),
        random_state: int = 42,
        inner_folds: int = 0,
        min_selection_rate: float = 0.5,
        max_features: int = 7,
    ) -> None:
        self.k = k
        self.task = task
        self.n_estimators = n_estimators
        self.forced_include = forced_include
        self.discrete_features = discrete_features
        self.random_state = random_state
        self.inner_folds = inner_folds
        self.min_selection_rate = min_selection_rate
        self.max_features = max_features

    def fit(self, X: pd.DataFrame, y: Any) -> RankAggregationSelector:
        """Select on the training data passed in only."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RankAggregationSelector expects a DataFrame with feature names")
        settings = (self.k, self.task, self.n_estimators, tuple(self.forced_include), tuple(self.discrete_features),
                    self.random_state, self.inner_folds, self.min_selection_rate, self.max_features)
        key = _cache_key(X, y, settings)
        if key not in _SELECTION_CACHE:
            ranking = rank_features(X, y, self.task, self.n_estimators, self.discrete_features, self.random_state)
            if self.inner_folds and self.inner_folds >= 2:
                selected, rates = stability_select(
                    X, y, self.task, self.k, self.inner_folds, self.min_selection_rate, self.max_features,
                    self.forced_include, self.n_estimators, self.discrete_features, self.random_state,
                )
            else:
                selected, rates = choose_top_k(ranking, self.k, self.forced_include), None
            _SELECTION_CACHE[key] = (ranking, selected, rates)
        ranking, selected, rates = _SELECTION_CACHE[key]
        self.ranking_ = ranking.copy()
        self.selected_features_ = list(selected)
        self.inner_selection_rates_ = rates.copy() if rates is not None else None
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.selected_features_)


def selection_stability(
    X: pd.DataFrame,
    y: Any,
    splitter: Any,
    make_selector_pipeline: Callable[[], Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit imputation plus selection in every training fold and count selections.

    ``make_selector_pipeline`` returns an unfitted pipeline whose last step is
    a ``RankAggregationSelector``. Only training indices are passed to fit.

    Returns a summary (one row per feature) and the long per-fold table.
    """
    y_arr = np.asarray(y)
    per_fold: list[pd.DataFrame] = []
    for fold, (train_idx, _val_idx) in enumerate(splitter.split(X)):
        pipe = make_selector_pipeline()
        pipe.fit(X.iloc[train_idx], y_arr[train_idx])
        selector: RankAggregationSelector = pipe.steps[-1][1]
        ranking = selector.ranking_.copy()
        ranking["fold"] = fold
        ranking["selected"] = ranking["feature"].isin(selector.selected_features_)
        per_fold.append(ranking)

    long = pd.concat(per_fold, ignore_index=True)
    n_folds = long["fold"].nunique()
    summary = (
        long.groupby("feature")
        .agg(
            times_selected=("selected", "sum"),
            mean_rank=("mean_rank", "mean"),
            sd_rank=("mean_rank", "std"),
            mean_mi=("mi", "mean"),
            mean_rf_importance=("rf_importance", "mean"),
        )
        .reset_index()
    )
    summary["n_folds"] = n_folds
    summary["selection_rate"] = (summary["times_selected"] / n_folds).round(3)
    summary = summary.sort_values(["selection_rate", "mean_rank"], ascending=[False, True], ignore_index=True)
    return summary, long


def exploratory_ranking(
    X_train: pd.DataFrame,
    y_train: Any,
    make_imputer: Callable[[], Any],
    task: str,
    n_estimators: int = 300,
    discrete_features: Sequence[str] = (),
    random_state: int = 42,
) -> pd.DataFrame:
    """One ranking on the hold-out training set, clearly labelled as exploratory."""
    imputed = make_imputer().fit_transform(X_train)
    table = rank_features(imputed, y_train, task, n_estimators, discrete_features, random_state)
    table.insert(0, "note", EXPLORATORY_LABEL)
    return table
