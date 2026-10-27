"""Stratification key and cross-validation splitters.

Protocol (proposal Section 3.5 and the approved rules of 2026-10-17): there
is no separate hold-out set. Every model is evaluated with repeated
stratified 5 x 5 outer cross-validation on all eligible records. The
stratification key is built once, before any split, so the folds are
reproducible.

The splitters take the key separately from ``y``. The Hb target is
continuous, so sklearn could not stratify on it directly.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, StratifiedKFold

NO_HB = "NoHb"
NO_MUAC = "NoMUAC"
CED_YES = "CED"
CED_NO = "NoCED"


class SplitIntegrityError(RuntimeError):
    """Raised when training and evaluation indices overlap."""


def assert_disjoint(train: Sequence[Any], test: Sequence[Any], context: str = "split") -> None:
    """Fail if any element is in both sets."""
    overlap = set(np.asarray(train).tolist()) & set(np.asarray(test).tolist())
    if overlap:
        raise SplitIntegrityError(f"{context}: {len(overlap)} items appear in both train and test")


# ---------------------------------------------------------------------------
# Stratification key
# ---------------------------------------------------------------------------

def raw_strat_key(anemia_class: pd.Series, ced: pd.Series) -> pd.Series:
    """``"<anemia class>|<CED status>"`` with missing components as their own category."""
    left = anemia_class.astype(object).where(anemia_class.notna(), NO_HB).astype(str)
    right = ced.map(lambda v: NO_MUAC if pd.isna(v) else (CED_YES if int(v) == 1 else CED_NO))
    return (left + "|" + right).rename("strat_key")


def _merge_target(cell: str, counts: pd.Series) -> str:
    """Where a small cell goes, following the approved rule.

    * Both components known: the group with the same anemia class and the
      opposite CED status. If that group does not exist, the largest group
      with the same anemia class.
    * One component missing (NoHb or NoMUAC): the largest group that shares
      the known component.
    * Otherwise: the largest remaining group.
    """
    others = counts.drop(index=cell)
    if "|" not in cell:
        return str(max(others.index, key=lambda c: (others[c], c)))
    anemia, ced = cell.split("|", 1)
    parts = {c: (c.split("|", 1) if "|" in c else (c, "")) for c in others.index}
    if anemia != NO_HB and ced != NO_MUAC:
        opposite = f"{anemia}|{CED_NO if ced == CED_YES else CED_YES}"
        if opposite in others.index:
            return opposite
        same_class = [c for c, (a, _) in parts.items() if a == anemia]
        if same_class:
            return max(same_class, key=lambda c: (others[c], c))
    elif anemia == NO_HB:
        same_ced = [c for c, (_, b) in parts.items() if b == ced]
        if same_ced:
            return max(same_ced, key=lambda c: (others[c], c))
    else:
        same_class = [c for c, (a, _) in parts.items() if a == anemia]
        if same_class:
            return max(same_class, key=lambda c: (others[c], c))
    return str(max(others.index, key=lambda c: (others[c], c)))


def merge_small_cells(key: pd.Series, min_cell: int) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Merge every group under ``min_cell`` records, smallest first, until none remains.

    Returns the merged key and a merge log (counts only).
    """
    key = key.copy()
    log: list[dict[str, Any]] = []
    while key.nunique() > 1:
        counts = key.value_counts()
        small = counts[counts < min_cell]
        if small.empty:
            break
        cell = str(min(small.index, key=lambda c: (small[c], c)))
        target = _merge_target(cell, counts)
        log.append({"from": cell, "to": target, "n_rows": int(counts[cell])})
        key = key.replace(cell, target)
    return key, log


def build_strat_key(
    anemia_class: pd.Series, ced: pd.Series, min_cell: int = 5
) -> tuple[pd.Series, list[dict[str, Any]], pd.Series]:
    """Stratification key for one model's eligible records.

    Built before any splitting. Returns the key, the merge log and the final
    cell counts, all of which are logged by the caller.
    """
    key, log = merge_small_cells(raw_strat_key(anemia_class, ced), min_cell)
    return key, log, key.value_counts().sort_index()


# ---------------------------------------------------------------------------
# Cross-validation splitters
# ---------------------------------------------------------------------------

class KeyStratifiedRepeatedKFold:
    """Repeated stratified K-fold on a stored key, not on ``y``.

    The key must be aligned row by row with the ``X`` passed to ``split``.
    ``y`` and ``groups`` are accepted and ignored.
    """

    def __init__(self, strat_key: npt.ArrayLike, n_splits: int = 5, n_repeats: int = 5, random_state: int = 42) -> None:
        self.strat_key = np.asarray(strat_key)
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        counts = pd.Series(self.strat_key).value_counts()
        too_small = counts[counts < n_splits]
        if not too_small.empty:
            raise ValueError(
                f"Stratification cells smaller than n_splits={n_splits}: {too_small.to_dict()}. "
                "Merge them with merge_small_cells first."
            )

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        if n != len(self.strat_key):
            raise ValueError(f"X has {n} rows but the stratification key has {len(self.strat_key)}")
        inner = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        yield from inner.split(np.zeros((n, 1)), self.strat_key)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits * self.n_repeats


def inner_splits(strat_key: npt.ArrayLike, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified inner folds of an outer training set.

    Cells too small for ``n_splits`` inside this training set are merged with
    the same rule as the outer key, so every inner fold can be stratified.
    """
    key, _ = merge_small_cells(pd.Series(np.asarray(strat_key)), n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros((len(key), 1)), key.to_numpy()))


class BoundGroupKFold:
    """GroupKFold with the groups bound at construction (one deterministic pass).

    With five subdistricts and five folds this is leave one subdistrict out.
    """

    def __init__(self, groups: npt.ArrayLike, n_splits: int = 5) -> None:
        self.groups = np.asarray(groups)
        n_groups = len(pd.unique(self.groups))
        if n_groups < 2:
            raise ValueError("GroupKFold needs at least two groups")
        self.n_splits = min(n_splits, n_groups)
        self.n_repeats = 1

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        if n != len(self.groups):
            raise ValueError(f"X has {n} rows but groups has {len(self.groups)}")
        yield from GroupKFold(n_splits=self.n_splits).split(np.zeros((n, 1)), groups=self.groups)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits
