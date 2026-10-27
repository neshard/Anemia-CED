"""Per-fold selection rule (proposal 3.3 as approved on 2026-10-17)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_selection import (
    EXPLORATORY_LABEL,
    RankAggregationSelector,
    choose_top_k,
    clear_selection_cache,
    exploratory_ranking,
    rank_features,
    selection_frequency,
)
from src.preprocessing import build_imputer

POOL = ["age", "n_children", "n_under5", "prepreg_weight", "height", "anc_visits", "hfias_score",
        "education_mother", "income", "egg_freq", "food_taboo"]
FORCED = ("prepreg_weight", "anc_visits", "hfias_score")


def _data(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(POOL))), columns=POOL)
    y = 3 * X["age"] - 2 * X["height"] + X["income"] + 0.2 * rng.normal(size=n)
    return X, y.to_numpy()


def test_exactly_seven_with_forced_first() -> None:
    X, y = _data()
    sel = RankAggregationSelector(k=7, n_estimators=30, forced_include=FORCED, random_state=0).fit(X, y)
    assert len(sel.selected_features_) == 7
    assert tuple(sel.selected_features_[:3]) == FORCED
    assert {"age", "height", "income"} <= set(sel.selected_features_)


def test_ties_broken_by_mi_rank_then_alphabetically() -> None:
    ranking = pd.DataFrame({"feature": ["b", "a", "c", "d"], "mean_rank": [2.0, 2.0, 2.0, 1.0],
                            "mi_rank": [1.0, 3.0, 1.0, 4.0]})
    ordered = ranking.sort_values(["mean_rank", "mi_rank", "feature"])["feature"].tolist()
    assert ordered == ["d", "b", "c", "a"]
    X, y = _data()
    table = rank_features(X, y, "regression", n_estimators=30, random_state=0)
    keys = list(zip(table["mean_rank"], table["mi_rank"], table["feature"], strict=True))
    assert keys == sorted(keys)


def test_choose_top_k_errors() -> None:
    with pytest.raises(ValueError):
        choose_top_k(pd.DataFrame({"feature": ["a", "b"]}), 1, ["a", "b"])


def test_cache_returns_identical_selection_and_is_keyed_by_data() -> None:
    clear_selection_cache()
    X, y = _data()
    a = RankAggregationSelector(k=7, n_estimators=30, forced_include=FORCED).fit(X, y).selected_features_
    b = RankAggregationSelector(k=7, n_estimators=30, forced_include=FORCED).fit(X, y).selected_features_
    assert a == b
    c = RankAggregationSelector(k=7, n_estimators=30, forced_include=FORCED).fit(X.iloc[:150], y[:150])
    assert len(c.selected_features_) == 7


def test_rejects_nan_input() -> None:
    X, y = _data()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="imputed"):
        rank_features(X, y, "regression")


def test_selection_frequency_and_dashboard_list() -> None:
    folds = [["a", "b"], ["a", "c"], ["a", "b"], ["b", "d"]]
    table = selection_frequency(folds, ["a", "b", "c", "d", "e"], 0.5).set_index("feature")
    assert table.loc["a", "share"] == 0.75 and table.loc["b", "share"] == 0.75
    assert table.loc["c", "share"] == 0.25 and table.loc["e", "times_selected"] == 0
    assert set(table.index[table["in_dashboard_list"]]) == {"a", "b"}


def test_exploratory_table_is_labelled(config: dict) -> None:
    X, y = _data()
    table = exploratory_ranking(X, y, build_imputer(config), "regression", n_estimators=30)
    assert (table["note"] == EXPLORATORY_LABEL).all()
