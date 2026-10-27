"""Stratification key, the persisted hold-out split, and CV splitters.

Ordering matters for leakage. The hold-out split is made right after the
deterministic cleaning, before anything is fitted. It is saved to disk by
anonymous ID together with a fingerprint of the raw file, so later scripts
reload the exact same test set instead of re-drawing it. Hyperparameter
tuning, feature selection and resampling only ever see the training part.

The CV splitters take the stratification key separately from ``y``. That is
needed because the Hb target is continuous (sklearn cannot stratify on it)
and because SMOGN changes the training rows inside each fold, so ``y`` at fit
time is not the ``y`` that defined the folds.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, train_test_split

from src.labels import WHO_CLASSES
from src.utils import file_fingerprint, get_logger, resolve_path

LOGGER = get_logger(__name__)
NA_TOKEN = "NA"
SEVERITY_ORDER: tuple[str, ...] = (*WHO_CLASSES, NA_TOKEN)


class SplitIntegrityError(RuntimeError):
    """Raised when a saved split no longer matches the data it was made from."""


# ---------------------------------------------------------------------------
# Stratification key
# ---------------------------------------------------------------------------

def raw_strat_key(anemia_class: pd.Series, ced: pd.Series) -> pd.Series:
    """Concatenate WHO class and CED label as ``"<class>|<0 or 1>"``.

    A missing component becomes ``NA`` so rows that are usable for only one
    model still take part in the split and stay balanced.
    """
    left = anemia_class.astype(object).where(anemia_class.notna(), NA_TOKEN).astype(str)
    right = ced.map(lambda v: NA_TOKEN if pd.isna(v) else str(int(v)))
    return (left + "|" + right).rename("strat_key")


def _closest_cell(cell: str, counts: pd.Series) -> str:
    """Pick the merge target for a rare cell.

    Prefer another cell with the same CED part and the nearest anemia
    severity, which keeps CED balance exact and changes anemia balance as
    little as possible. Ties go to the larger cell. A cell whose anemia
    class is unknown has no meaningful severity neighbour, so it goes to the
    largest cell with the same CED part. Without any same-CED cell, use the
    largest remaining cell.
    """
    klass, ced = cell.split("|")
    others = counts.drop(index=cell)
    same_ced = [c for c in others.index if c.split("|")[1] == ced]
    if same_ced and klass == NA_TOKEN:
        return max(same_ced, key=lambda c: others[c])
    if same_ced:
        rank = SEVERITY_ORDER.index(klass)
        return min(same_ced, key=lambda c: (abs(SEVERITY_ORDER.index(c.split("|")[0]) - rank), -others[c]))
    return str(others.idxmax())


def make_strat_key(
    anemia_class: pd.Series,
    ced: pd.Series,
    merge_map: Mapping[str, str] | None = None,
    min_cell_count: int = 8,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Build the stratification key and merge rare cells.

    Explicit merges from config run first (for example Severe with CED risk
    into Moderate with CED risk). Remaining cells smaller than
    ``min_cell_count`` are merged one at a time, smallest first, so every
    cell is large enough to appear in every training fold.

    Returns the key and a log of merges with counts only.
    """
    key = raw_strat_key(anemia_class, ced)
    merges: list[dict[str, Any]] = []
    for source, target in (merge_map or {}).items():
        n = int((key == source).sum())
        if n:
            key = key.replace(source, target)
            merges.append({"from": source, "to": target, "n_rows": n, "reason": "config merge map"})

    while key.nunique() > 1:
        counts = key.value_counts()
        smallest = str(counts.idxmin())
        if counts[smallest] >= min_cell_count:
            break
        target = _closest_cell(smallest, counts)
        merges.append({"from": smallest, "to": target, "n_rows": int(counts[smallest]), "reason": "below min_cell_count"})
        key = key.replace(smallest, target)
    return key, merges


# ---------------------------------------------------------------------------
# Hold-out split
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HoldoutSplit:
    """Train and test membership by anonymous ID plus provenance metadata."""

    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    raw_fingerprint: str
    seed: int
    test_size: float
    created_with_hb_zero_policy: str

    def masks(self, anon_ids: pd.Series) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Boolean train and test masks aligned with ``anon_ids``."""
        ids = anon_ids.to_numpy()
        return np.isin(ids, self.train_ids), np.isin(ids, self.test_ids)


def assert_disjoint(train: Sequence[Any], test: Sequence[Any], context: str = "split") -> None:
    """Fail if any element is in both sets."""
    overlap = set(train) & set(test)
    if overlap:
        raise SplitIntegrityError(f"{context}: {len(overlap)} items appear in both train and test")


def create_holdout(
    anon_ids: pd.Series,
    strat_key: pd.Series,
    test_size: float,
    seed: int,
    raw_fingerprint: str,
    hb_zero_policy: str,
) -> HoldoutSplit:
    """Draw the stratified hold-out split once."""
    train_ids, test_ids = train_test_split(
        anon_ids.to_numpy(), test_size=test_size, stratify=strat_key.to_numpy(), random_state=seed
    )
    assert_disjoint(train_ids, test_ids, "holdout")
    return HoldoutSplit(
        train_ids=tuple(sorted(map(str, train_ids))),
        test_ids=tuple(sorted(map(str, test_ids))),
        raw_fingerprint=raw_fingerprint,
        seed=seed,
        test_size=test_size,
        created_with_hb_zero_policy=hb_zero_policy,
    )


def save_holdout(split: HoldoutSplit, path: Path) -> None:
    """Write the split as JSON. It holds anonymous IDs only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(split), indent=1), encoding="utf-8")


def load_holdout(path: Path) -> HoldoutSplit:
    """Read a split written by ``save_holdout``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data["train_ids"] = tuple(data["train_ids"])
    data["test_ids"] = tuple(data["test_ids"])
    return HoldoutSplit(**data)


def load_or_create_holdout(
    frame: pd.DataFrame,
    strat_key: pd.Series,
    config: Mapping[str, Any],
    raw_path: Path | None = None,
    force_recreate: bool = False,
) -> HoldoutSplit:
    """Return the saved split, or create and save it on first use.

    The split is reused across every experiment and every Hb == 0 policy so
    that all reported test metrics refer to the same women. If the raw file
    changed since the split was made, this raises instead of silently
    re-drawing, because a new draw would let earlier tuning decisions leak
    into a different test set. Pass ``force_recreate=True`` only when a new
    data version is deliberately adopted, and before any tuning on it.
    """
    split_path = resolve_path(config, "split_file")
    data_path = raw_path if raw_path is not None else resolve_path(config, "raw_data")
    fingerprint = file_fingerprint(data_path)

    if split_path.exists() and not force_recreate:
        split = load_holdout(split_path)
        if split.raw_fingerprint != fingerprint:
            raise SplitIntegrityError(
                "Raw data changed since the hold-out split was created. Re-create it explicitly "
                "with force_recreate=True before any tuning on the new data."
            )
        known = set(split.train_ids) | set(split.test_ids)
        current = set(frame["anon_id"].astype(str))
        if known != current:
            raise SplitIntegrityError(
                f"Saved split covers {len(known)} IDs but the analysis frame has {len(current)}. "
                "A cleaning rule changed which rows are usable. Re-create the split deliberately."
            )
        assert_disjoint(split.train_ids, split.test_ids, "holdout (loaded)")
        LOGGER.info("Loaded hold-out split: %d train, %d test", len(split.train_ids), len(split.test_ids))
        return split

    split = create_holdout(
        frame["anon_id"].astype(str),
        strat_key,
        float(config["splits"]["test_size"]),
        int(config["project"]["random_seed"]),
        fingerprint,
        str(config["cleaning"]["hb_zero_policy"]),
    )
    save_holdout(split, split_path)
    LOGGER.info("Created hold-out split: %d train, %d test", len(split.train_ids), len(split.test_ids))
    return split


def describe_split(strat_key: pd.Series, train_mask: npt.ArrayLike, test_mask: npt.ArrayLike) -> pd.DataFrame:
    """Counts and shares per stratification cell in train and test (aggregates only)."""
    train = strat_key[np.asarray(train_mask)].value_counts()
    test = strat_key[np.asarray(test_mask)].value_counts()
    table = pd.DataFrame({"n_train": train, "n_test": test}).fillna(0).astype(int)
    table["share_train"] = (table["n_train"] / table["n_train"].sum()).round(4)
    table["share_test"] = (table["n_test"] / table["n_test"].sum()).round(4)
    return table.rename_axis("strat_key").reset_index().sort_values("strat_key", ignore_index=True)


# ---------------------------------------------------------------------------
# Cross-validation splitters
# ---------------------------------------------------------------------------

class KeyStratifiedRepeatedKFold:
    """Repeated stratified K-fold that stratifies on a stored key, not on ``y``.

    The key is bound at construction and must be aligned row by row with the
    ``X`` later passed to ``split``. ``y`` and ``groups`` are accepted and
    ignored so the object works as ``cv=`` in ``GridSearchCV`` and
    ``cross_validate`` with a continuous target.
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
                "Rebuild the key on this subset with make_strat_key(min_cell_count=n_splits)."
            )

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
        """Yield positional train and validation indices."""
        n = len(X)
        if n != len(self.strat_key):
            raise ValueError(f"X has {n} rows but the stratification key has {len(self.strat_key)}")
        inner = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        for train_idx, val_idx in inner.split(np.zeros((n, 1)), self.strat_key):
            yield train_idx, val_idx

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Total number of folds across repeats."""
        return self.n_splits * self.n_repeats


class BoundGroupKFold:
    """GroupKFold with the groups bound at construction.

    Holding out whole health centres estimates how the model transfers to a
    centre it has not seen. This is reported next to the stratified CV, not
    instead of it. Stratification is not possible here, so a fold can have
    a different class balance from the others.
    """

    def __init__(self, groups: npt.ArrayLike, n_splits: int = 5) -> None:
        self.groups = np.asarray(groups)
        n_groups = len(pd.unique(self.groups))
        if n_groups < 2:
            raise ValueError("GroupKFold needs at least two groups")
        self.n_splits = min(n_splits, n_groups)

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
        """Yield positional train and validation indices with no group in both."""
        n = len(X)
        if n != len(self.groups):
            raise ValueError(f"X has {n} rows but groups has {len(self.groups)}")
        for train_idx, val_idx in GroupKFold(n_splits=self.n_splits).split(np.zeros((n, 1)), groups=self.groups):
            yield train_idx, val_idx

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of folds."""
        return self.n_splits


def make_cv_splitters(
    train_frame: pd.DataFrame,
    target_mask: npt.ArrayLike,
    config: Mapping[str, Any],
) -> dict[str, KeyStratifiedRepeatedKFold | BoundGroupKFold]:
    """Build CV splitters for the training rows usable by one model.

    ``train_frame`` must already be restricted to the hold-out training set.
    ``target_mask`` selects the rows with a valid label for the model in
    question (for example ``hb_target`` not NaN). The stratification key is
    rebuilt on that subset so rare cells are merged for the rows that
    actually enter CV.
    """
    cfg = config["splits"]
    subset = train_frame.loc[np.asarray(target_mask)]
    key, _ = make_strat_key(
        subset["anemia_class"], subset["ced"], cfg["strat_key_merge_map"], min_cell_count=int(cfg["cv_n_splits"])
    )
    splitters: dict[str, KeyStratifiedRepeatedKFold | BoundGroupKFold] = {
        "stratified": KeyStratifiedRepeatedKFold(
            key, int(cfg["cv_n_splits"]), int(cfg["cv_n_repeats"]), int(config["project"]["random_seed"])
        )
    }
    group_col = cfg.get("group_column")
    if group_col and group_col in subset.columns:
        splitters["group"] = BoundGroupKFold(subset[group_col].to_numpy(), int(cfg["cv_n_splits"]))
    return splitters
