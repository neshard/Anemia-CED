"""Stratification key (approved rule of 2026-10-17) and CV splitters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.splits import (
    BoundGroupKFold,
    KeyStratifiedRepeatedKFold,
    SplitIntegrityError,
    assert_disjoint,
    build_strat_key,
    inner_splits,
    merge_small_cells,
    raw_strat_key,
)


def test_missing_components_are_own_category() -> None:
    key = raw_strat_key(pd.Series(["Normal", None, "Mild"]), pd.Series([0.0, 1.0, np.nan]))
    assert key.tolist() == ["Normal|NoCED", "NoHb|CED", "Mild|NoMUAC"]


def test_small_known_cell_goes_to_opposite_ced_same_class() -> None:
    classes = pd.Series(["Normal"] * 40 + ["Moderate"] * 10 + ["Moderate"] * 3)
    ced = pd.Series([0.0] * 30 + [1.0] * 10 + [0.0] * 10 + [1.0] * 3)
    key, log, cells = build_strat_key(classes, ced, 5)
    assert log == [{"from": "Moderate|CED", "to": "Moderate|NoCED", "n_rows": 3}]
    assert cells.min() >= 5


def test_small_missing_component_cell_goes_to_largest_sharing_known_part() -> None:
    classes = pd.Series([None] * 4 + ["Normal"] * 30 + ["Mild"] * 12)
    ced = pd.Series([1.0] * 4 + [1.0] * 30 + [1.0] * 12)
    key, log, _ = build_strat_key(classes, ced, 5)
    assert log[0] == {"from": "NoHb|CED", "to": "Normal|CED", "n_rows": 4}
    classes = pd.Series(["Mild"] * 3 + ["Mild"] * 20 + ["Normal"] * 30)
    ced = pd.Series([np.nan] * 3 + [0.0] * 20 + [0.0] * 30)
    _, log, _ = build_strat_key(classes, ced, 5)
    assert log[0] == {"from": "Mild|NoMUAC", "to": "Mild|NoCED", "n_rows": 3}


@pytest.mark.parametrize("seed", range(40))
def test_no_group_below_five_remains_after_merging(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 420))
    classes = pd.Series(rng.choice(["Normal", "Mild", "Moderate", "Severe", None], n, p=[0.7, 0.14, 0.08, 0.02, 0.06]))
    ced = pd.Series(rng.choice([0.0, 1.0, np.nan], n, p=[0.7, 0.27, 0.03]))
    key, _, cells = build_strat_key(classes, ced, 5)
    assert cells.min() >= 5 or key.nunique() == 1
    assert len(key) == n


def test_merge_is_deterministic() -> None:
    classes = pd.Series(["Normal"] * 20 + ["Mild"] * 4 + ["Moderate"] * 4)
    ced = pd.Series([0.0] * 20 + [1.0] * 4 + [1.0] * 4)
    assert build_strat_key(classes, ced)[1] == build_strat_key(classes, ced)[1]


def test_key_splitter_only_uses_key_and_covers_every_row_once_per_repeat() -> None:
    key = np.array(["a"] * 30 + ["b"] * 20)
    cv = KeyStratifiedRepeatedKFold(key, n_splits=5, n_repeats=3, random_state=0)
    folds = list(cv.split(np.zeros((50, 1))))
    assert len(folds) == 15
    for r in range(3):
        tests = np.concatenate([folds[r * 5 + k][1] for k in range(5)])
        assert sorted(tests) == list(range(50))
    for tr, te in folds:
        assert_disjoint(tr, te)


def test_key_splitter_rejects_tiny_cells_and_misaligned_x() -> None:
    with pytest.raises(ValueError, match="smaller than n_splits"):
        KeyStratifiedRepeatedKFold(np.array(["a"] * 10 + ["b"] * 3), n_splits=5)
    cv = KeyStratifiedRepeatedKFold(np.array(["a"] * 10 + ["b"] * 10), n_splits=5, n_repeats=1)
    with pytest.raises(ValueError):
        next(cv.split(np.zeros((19, 1))))


def test_inner_splits_merge_cells_too_small_for_three_folds() -> None:
    key = np.array(["a"] * 30 + ["b"] * 2)  # "b" cannot be split three ways
    splits = inner_splits(key, 3, seed=0)
    assert len(splits) == 3
    for tr, te in splits:
        assert_disjoint(tr, te)


def test_group_kfold_is_leave_one_group_out_with_five_groups() -> None:
    groups = np.repeat(["Gangga", "Pemenang", "Tanjung", "Bayan", "Kayangan"], [115, 103, 87, 75, 40])
    cv = BoundGroupKFold(groups, n_splits=5)
    held = []
    for tr, te in cv.split(np.zeros(len(groups))):
        assert not set(groups[tr]) & set(groups[te])
        assert len(set(groups[te])) == 1
        held.append(groups[te][0])
    assert sorted(held) == sorted(set(groups))


def test_assert_disjoint_raises() -> None:
    with pytest.raises(SplitIntegrityError):
        assert_disjoint([1, 2, 3], [3, 4])


def test_merge_small_cells_on_existing_keys() -> None:
    key = pd.Series(["Normal|NoCED"] * 10 + ["Mild|CED"] * 2 + ["Mild|NoCED"] * 6)
    merged, log = merge_small_cells(key, 5)
    assert log == [{"from": "Mild|CED", "to": "Mild|NoCED", "n_rows": 2}]
    assert merged.value_counts().min() >= 5
