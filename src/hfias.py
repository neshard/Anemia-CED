"""Household Food Insecurity Access Scale (HFIAS) scoring.

Reference: Coates J, Swindale A, Bilinsky P. Household Food Insecurity Access
Scale (HFIAS) for Measurement of Food Access: Indicator Guide, Version 3.
FANTA, 2007.

Each of the nine items has an occurrence question (yes or no) and a
frequency-of-occurrence follow-up (rarely = 1, sometimes = 2, often = 3).
An item scores 0 when the event did not occur, so the composite ranges from
0 to 27. The models receive only the composite. The nine items are never
features themselves, which matches the requirement that feature selection
works on the composite score.

Scoring is a fixed arithmetic rule with no parameters learned from data, so
running it on the full dataset before the train and test split cannot leak
information between them.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

HFIAS_N_ITEMS: int = 9
HFIAS_MAX_SCORE: int = 27
HFIAS_CATEGORY_LABELS: dict[int, str] = {
    0: "Food secure",
    1: "Mildly food insecure",
    2: "Moderately food insecure",
    3: "Severely food insecure",
}


class HFIASFormatError(ValueError):
    """Raised when HFIAS responses do not match the configured format."""


class HFIASBinaryWarning(UserWarning):
    """Emitted when only yes or no items are available. The result is not a valid HFIAS score."""


@dataclass(frozen=True)
class HFIASResult:
    """Item codes (0 to 3), the composite score, the Coates category, and the format used."""

    items: pd.DataFrame
    score: pd.Series
    category: pd.Series
    response_format: str
    is_valid_hfias: bool


def _map_strict(series: pd.Series, mapping: Mapping[str, int], item: str) -> pd.Series:
    """Map response text to codes and refuse unknown answers.

    Unknown text raises instead of becoming NaN because a silent NaN would be
    imputed later and hide a mapping mistake in the config.
    """
    cleaned = series.astype("string").str.strip()
    normalized = {str(k).strip(): v for k, v in mapping.items()}
    unknown = sorted(set(cleaned.dropna().unique()) - set(normalized))
    if unknown:
        raise HFIASFormatError(
            f"Item {item} has responses not in the configured mapping: {unknown}. "
            "Add them to config hfias mappings."
        )
    return cleaned.map(normalized).astype("Float64")


def looks_binary(df: pd.DataFrame, item_columns: Sequence[str]) -> bool:
    """True when every item has at most two distinct answers across all rows.

    A genuine HFIAS export normally shows three or four distinct answers on at
    least the first item. If all nine items have only two, the frequency
    follow-up was probably not collected.
    """
    distinct = set()
    for col in item_columns:
        distinct |= set(df[col].dropna().astype(str).str.strip().unique())
    return len(distinct) <= 2


def items_combined(df: pd.DataFrame, item_columns: Sequence[str], mapping: Mapping[str, int]) -> pd.DataFrame:
    """Score items where a single answer already combines occurrence and frequency.

    This is the layout of the North Lombok questionnaire, for example
    "Tidak" = 0, "Jarang (1-2x)" = 1, "Kadang-kadang (3-10x)" = 2, "Sering (>10x)" = 3.
    """
    codes = set(mapping.values())
    if not codes <= {0, 1, 2, 3}:
        raise HFIASFormatError(f"combined_mapping codes must lie in 0..3, got {sorted(codes)}")
    return pd.DataFrame({col: _map_strict(df[col], mapping, col) for col in item_columns}, index=df.index)


def items_split(
    df: pd.DataFrame,
    occurrence_columns: Sequence[str],
    frequency_columns: Sequence[str],
    occurrence_mapping: Mapping[str, int],
    frequency_mapping: Mapping[str, int],
    item_names: Sequence[str],
) -> pd.DataFrame:
    """Score items stored as separate occurrence and frequency columns.

    Following Coates 2007, when the occurrence answer is "no" the item is 0
    and the frequency column is ignored, because enumerators skip it. When
    occurrence is "yes" but frequency is missing the item is NaN.
    """
    if not (len(occurrence_columns) == len(frequency_columns) == len(item_names) == HFIAS_N_ITEMS):
        raise HFIASFormatError("split format needs nine occurrence and nine frequency columns")
    out: dict[str, pd.Series] = {}
    for name, occ_col, freq_col in zip(item_names, occurrence_columns, frequency_columns, strict=True):
        occ = _map_strict(df[occ_col], occurrence_mapping, occ_col)
        if not set(occ.dropna().unique()) <= {0, 1}:
            raise HFIASFormatError(f"occurrence codes for {occ_col} must be 0 or 1")
        said_yes = (occ == 1).fillna(False).astype(bool)
        said_no = (occ == 0).fillna(False).astype(bool)
        freq = _map_strict(df[freq_col].where(said_yes), frequency_mapping, freq_col)
        if not set(freq.dropna().unique()) <= {1, 2, 3}:
            raise HFIASFormatError(f"frequency codes for {freq_col} must be 1, 2 or 3")
        item = pd.Series(pd.NA, index=df.index, dtype="Float64")
        item[said_no] = 0
        item[said_yes] = freq[said_yes]
        out[name] = item
    return pd.DataFrame(out, index=df.index)


def hfias_category(items: pd.DataFrame) -> pd.Series:
    """Assign the four HFIA categories from Coates 2007, Indicator 2.

    The categories are checked from most to least severe so a household lands
    in the highest category whose condition it meets. That ordering is the
    same as the guide's rule that each category requires the more severe
    items to be zero.

    Returns integer codes 0 to 3 (see ``HFIAS_CATEGORY_LABELS``). Rows with
    any missing item return NA.
    """
    if items.shape[1] != HFIAS_N_ITEMS:
        raise HFIASFormatError("hfias_category needs exactly nine item columns in order")
    q = [items.iloc[:, i] for i in range(HFIAS_N_ITEMS)]
    severe = (q[4] == 3) | (q[5] == 3) | (q[6] > 0) | (q[7] > 0) | (q[8] > 0)
    moderate = (q[2] >= 2) | (q[3] >= 2) | q[4].isin([1, 2]) | q[5].isin([1, 2])
    mild = (q[0] >= 2) | (q[1] >= 1) | (q[2] == 1) | (q[3] == 1)
    category = pd.Series(0, index=items.index, dtype="Int64")
    category[mild.fillna(False).astype(bool)] = 1
    category[moderate.fillna(False).astype(bool)] = 2
    category[severe.fillna(False).astype(bool)] = 3
    category[items.isna().any(axis=1)] = pd.NA
    return category


def score_hfias(df: pd.DataFrame, hfias_config: Mapping[str, Any]) -> HFIASResult:
    """Compute item codes, the 0 to 27 composite, and the category.

    The response format comes from config. If the data look binary while the
    config says ``combined``, scoring stops with an explanation, because a
    count of "yes" answers is not comparable to a published HFIAS score.
    """
    response_format = str(hfias_config["format"])
    item_names = list(hfias_config["items"])
    is_valid = True

    if response_format == "combined":
        if looks_binary(df, item_names):
            raise HFIASFormatError(
                "HFIAS items look binary (two distinct answers). The frequency of occurrence "
                "is missing, so a 0 to 27 score cannot be computed. Set hfias.format to "
                "'binary' and hfias.allow_binary to true to proceed with a 0 to 9 count."
            )
        items = items_combined(df, item_names, hfias_config["combined_mapping"])
    elif response_format == "split":
        items = items_split(
            df,
            hfias_config["occurrence_columns"],
            hfias_config["frequency_columns"],
            hfias_config["occurrence_mapping"],
            hfias_config["frequency_mapping"],
            item_names,
        )
    elif response_format == "binary":
        if not hfias_config.get("allow_binary", False):
            raise HFIASFormatError(
                "HFIAS format is 'binary'. This yields a 0 to 9 count of affirmative answers, "
                "not a Coates 2007 score. Set hfias.allow_binary to true to accept this."
            )
        warnings.warn(
            "HFIAS items are binary only. The composite is a 0 to 9 count and is NOT a valid "
            "HFIAS score. Report it as a proxy in the thesis.",
            HFIASBinaryWarning,
            stacklevel=2,
        )
        items = pd.DataFrame(
            {c: _map_strict(df[c], hfias_config["occurrence_mapping"], c) for c in item_names}, index=df.index
        )
        is_valid = False
    else:
        raise HFIASFormatError(f"Unknown hfias.format '{response_format}'")

    policy = str(hfias_config.get("missing_policy", "nan"))
    if policy == "zero":
        score = items.fillna(0).sum(axis=1)
    elif policy == "nan":
        score = items.sum(axis=1, min_count=HFIAS_N_ITEMS)
    else:
        raise HFIASFormatError(f"Unknown hfias.missing_policy '{policy}'")

    score = score.astype("float64").rename("hfias_score")
    upper = HFIAS_MAX_SCORE if is_valid else HFIAS_N_ITEMS
    finite = score.dropna()
    if ((finite < 0) | (finite > upper)).any():
        raise HFIASFormatError(f"HFIAS score outside 0..{upper}. Check the mapping.")

    category = hfias_category(items) if is_valid else pd.Series(pd.NA, index=df.index, dtype="Int64")
    items_float = pd.DataFrame(
        items.to_numpy(dtype="float64", na_value=np.nan), index=items.index, columns=items.columns
    )
    return HFIASResult(
        items=items_float,
        score=score,
        category=category.rename("hfias_category"),
        response_format=response_format,
        is_valid_hfias=is_valid,
    )
