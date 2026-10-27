"""Loading and deterministic cleaning of the questionnaire data.

Everything in this module is a fixed, row-wise rule: parsing text,
converting units, mapping questionnaire answers to codes, and setting
physically implausible values to NaN. None of it estimates a parameter from
the data (no medians, no frequencies, no scalers). That is why it is safe to
run on the full dataset before the train and test split. Every step that
does learn from data (imputation, scaling, encoding with learned categories,
feature selection, resampling) is deferred to in-fold pipelines in later
phases.

Privacy: ``load_raw`` reads only the columns mapped in config. GPS
coordinates, phone status, dates and free text never enter memory as part of
the analysis frame. The respondent code is used once to count duplicates and
then dropped in favour of a sequential anonymous ID.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hfias import score_hfias
from src.labels import build_labels
from src.utils import get_logger, resolve_path

LOGGER = get_logger(__name__)

NUMERIC_FEATURES: tuple[str, ...] = ("age", "n_children", "n_under5", "prepreg_weight", "height", "anc_visits")
NUMERIC_TARGETS: tuple[str, ...] = ("hb", "muac")


class DataSchemaError(ValueError):
    """Raised when the raw file does not contain the configured columns."""


class UnmappedCategoryError(ValueError):
    """Raised when a categorical answer has no code in config ``encodings``."""


@dataclass
class CleanResult:
    """Cleaned analysis frame plus an aggregate-only cleaning report."""

    frame: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Low level parsers
# ---------------------------------------------------------------------------

def to_numeric_flexible(series: pd.Series) -> pd.Series:
    """Parse numbers written with either a dot or a comma as the decimal mark.

    Indonesian spreadsheets often export "10,5". A plain ``pd.to_numeric``
    would turn that into NaN and silently lose the label.
    """
    text = series.astype("string").str.strip()
    text = text.str.replace(r"^(-?\d+),(\d+)$", r"\1.\2", regex=True)
    return pd.to_numeric(text, errors="coerce").astype("float64")


_GPA_LETTER_RE = {
    # "[^A-Za-z0-9]" rather than "\W" because "\W" excludes the underscore in "G_2".
    "G": re.compile(r"G[^A-Za-z0-9]{0,3}(\d+)", re.IGNORECASE),
    "P": re.compile(r"P[^A-Za-z0-9]{0,3}(\d+)", re.IGNORECASE),
    "A": re.compile(r"A[^A-Za-z0-9]{0,3}(\d+)", re.IGNORECASE),
}
_GPA_WORDS_RE = {
    "G": re.compile(r"kehamilan\D{0,6}(\d+)", re.IGNORECASE),
    "P": re.compile(r"kelahiran\D{0,6}(\d+)", re.IGNORECASE),
    "A": re.compile(r"abortus\D{0,6}(\d+)", re.IGNORECASE),
}


def parse_gpa(value: object, bare_number_policy: str = "nan") -> tuple[float, float, float]:
    """Parse obstetric shorthand into (gravida, para, abortus).

    Handles the formats seen in the export, for example ``G2P1A0``,
    ``G_2 P_1 A_0``, ``G(2) P(1)``, ``2/1/0``, ``2,1,0``, ``2-1-0`` and
    "Kehamilan ke 2, kelahiran 1, Abortus 0". A component that is absent is
    NaN. A lone number is ambiguous and follows ``bare_number_policy``
    ("nan" or "gravida"). Results where para plus abortus exceeds gravida are
    internally inconsistent and returned as all NaN.
    """
    nan3 = (np.nan, np.nan, np.nan)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return nan3
    text = str(value).strip()
    if not text or not re.search(r"\d", text):
        return nan3

    if re.search(r"kehamilan|kelahiran|abortus", text, re.IGNORECASE):
        patterns = _GPA_WORDS_RE
    elif re.search(r"[GgPpAa]", text):
        patterns = _GPA_LETTER_RE
    else:
        numbers = [float(n) for n in re.findall(r"\d+", text)]
        if len(numbers) == 3:
            g, p, a = numbers
        elif len(numbers) == 1:
            if bare_number_policy == "gravida":
                return (numbers[0], np.nan, np.nan)
            if bare_number_policy == "nan":
                return nan3
            raise ValueError(f"Unknown gpa bare_number_policy '{bare_number_policy}'")
        else:
            return nan3
        return _gpa_consistent(g, p, a)

    found = []
    for key in ("G", "P", "A"):
        match = patterns[key].search(text)
        found.append(float(match.group(1)) if match else np.nan)
    return _gpa_consistent(*found)


def _gpa_consistent(g: float, p: float, a: float) -> tuple[float, float, float]:
    """Return NaNs when P plus A exceeds G, which cannot happen in a valid record."""
    if not np.isnan(g) and np.nansum([p, a]) > g:
        return (np.nan, np.nan, np.nan)
    return (g, p, a)


_WEEK_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(minggu|mgg|mg|week|wk|w)\b", re.IGNORECASE)
_MONTH_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(bulan|bln|bl|month|mo)\b", re.IGNORECASE)
_BARE_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*$")


def parse_gestational_age(value: object, bare_number_unit: str = "weeks", weeks_per_month: float = 4.345) -> float:
    """Convert a gestational age answer to weeks.

    Text with "minggu" or "week" is read as weeks. Text with "bulan" or
    "month" is converted with ``weeks_per_month``. A bare number uses
    ``bare_number_unit``. Anything else is NaN.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    for regex, factor in ((_WEEK_RE, 1.0), (_MONTH_RE, weeks_per_month)):
        match = regex.search(text)
        if match:
            return float(match.group(1).replace(",", ".")) * factor
    match = _BARE_RE.match(text)
    if match:
        number = float(match.group(1).replace(",", "."))
        if bare_number_unit == "weeks":
            return number
        if bare_number_unit == "months":
            return number * weeks_per_month
        raise ValueError(f"Unknown bare_number_unit '{bare_number_unit}'")
    return np.nan


_DATE_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{1,4})\s*$")


def parse_survey_date(value: object, window: tuple[str, str]) -> pd.Timestamp:
    """Parse a questionnaire date, returning NaT when it is not usable.

    Rule (logged in config cleaning.dates): the export writes d/m/yyyy. When
    the second number exceeds 12 it can only be a day, so the value is read
    as m/d/yyyy. A date is usable only if it is a real calendar date inside
    ``window``. Years such as 25, 225, 1015 or 2015 fall outside and are not
    usable. Nothing is guessed or repaired.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    match = _DATE_RE.match(str(value))
    if not match:
        return pd.NaT
    first, second, year = (int(g) for g in match.groups())
    day, month = (second, first) if second > 12 else (first, second)
    try:
        stamp = pd.Timestamp(year=year, month=month, day=day)
    except (ValueError, OverflowError):
        return pd.NaT
    low, high = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    return stamp if low <= stamp <= high else pd.NaT


def map_categories(series: pd.Series, mapping: Mapping[str, int], name: str) -> pd.Series:
    """Map questionnaire answers to integer codes and refuse unmapped answers.

    The maps are fixed by the questionnaire, not estimated, so applying them
    before the split is leakage free. An unmapped answer raises so that a
    typo in config cannot be hidden by later imputation.
    """
    text = series.astype("string").str.strip()
    normalized = {str(k).strip(): v for k, v in mapping.items()}
    unknown = sorted(set(text.dropna().unique()) - set(normalized))
    if unknown:
        raise UnmappedCategoryError(f"Column {name} has answers without a code in config: {unknown}")
    return text.map(normalized).astype("float64")


def apply_plausible_ranges(df: pd.DataFrame, ranges: Mapping[str, list[float]]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Set values outside the configured closed interval to NaN.

    Implausible feature values are treated as missing rather than deleting
    the row, so the label is not lost. The NaNs are imputed later with a
    median fitted inside each training fold.
    """
    out = df.copy()
    counts: dict[str, int] = {}
    for col, (low, high) in ranges.items():
        if col not in out.columns:
            continue
        bad = out[col].notna() & ((out[col] < low) | (out[col] > high))
        counts[col] = int(bad.sum())
        out.loc[bad, col] = np.nan
    return out, counts


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_raw(config: Mapping[str, Any], path: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the CSV, keep only mapped columns, rename them, and anonymise.

    All cells are read as text so that parsing is explicit and auditable.
    """
    csv_path = Path(path) if path is not None else resolve_path(config, "raw_data")
    raw = pd.read_csv(
        csv_path,
        sep=config["data"]["csv_sep"],
        encoding=config["data"]["csv_encoding"],
        dtype=str,
        keep_default_na=True,
    )
    expected = config["data"].get("expected_row_count")
    if expected is not None and len(raw) != int(expected):
        raise DataSchemaError(
            f"{csv_path.name} has {len(raw)} data rows but config expects {expected} "
            f"(source: {config['data'].get('source_sheet', 'unknown')}). Check that the right export is in data/raw."
        )
    mapping: dict[str, str] = dict(config["columns"])
    missing = [f"{canon} -> '{header}'" for canon, header in mapping.items() if header not in raw.columns]
    if missing:
        raise DataSchemaError("Configured columns not found in the raw file:\n  " + "\n  ".join(missing))

    frame = raw[list(mapping.values())].copy()
    frame.columns = list(mapping.keys())
    info: dict[str, Any] = {"n_rows_raw": int(len(frame)), "n_columns_raw": int(raw.shape[1])}
    del raw

    if "respondent_code" in frame.columns:
        info["n_duplicate_respondent_codes"] = int(frame["respondent_code"].duplicated().sum())
    frame = frame.drop(columns=[c for c in config.get("identifier_columns", []) if c in frame.columns])

    # A row identical to another on every analysed column is almost certainly the same
    # woman entered twice. Keeping both could place her in training and in test at once.
    duplicated = frame.duplicated(keep="first")
    info["n_duplicate_records_found"] = int(duplicated.sum())
    if config["cleaning"].get("drop_duplicate_records", True):
        frame = frame.loc[~duplicated]
        info["n_duplicate_records_dropped"] = int(duplicated.sum())
    else:
        info["n_duplicate_records_dropped"] = 0
    frame.insert(0, "anon_id", [f"P{i:04d}" for i in range(1, len(frame) + 1)])
    return frame.reset_index(drop=True), info


def clean_features(frame: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply all deterministic feature cleaning to a canonical-named frame.

    This function is shared with the dashboard, which builds a one-row frame
    with the same canonical names, so offline and online inputs pass through
    identical rules.
    """
    cfg_clean = config["cleaning"]
    enc = config["encodings"]
    df = frame.copy()
    report: dict[str, Any] = {}

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = to_numeric_flexible(df[col])

    if "gestational_age_raw" in df.columns:
        ga_cfg = cfg_clean["gestational_age"]
        df["gestational_age_weeks"] = [
            parse_gestational_age(v, ga_cfg["bare_number_unit"], ga_cfg["weeks_per_month"])
            for v in df["gestational_age_raw"]
        ]
        bare = df["gestational_age_raw"].astype("string").str.fullmatch(r"\s*\d+([.,]\d+)?\s*").fillna(False)
        report["ga_bare_numbers"] = int(bare.sum())
        report["ga_bare_numbers_le_9"] = int((bare & (df["gestational_age_weeks"] <= 9)).sum())
        report["ga_unparsed"] = int(df["gestational_age_weeks"].isna().sum())
        df = df.drop(columns=["gestational_age_raw"])

    if "gpa_raw" in df.columns:
        policy = cfg_clean["gpa"]["bare_number_policy"]
        parsed = [parse_gpa(v, policy) for v in df["gpa_raw"]]
        df[["gravida", "para", "abortus"]] = pd.DataFrame(parsed, index=df.index, columns=["gravida", "para", "abortus"])
        report["gpa_parsed_gravida"] = int(df["gravida"].notna().sum())
        report["gpa_unparsed"] = int(df["gravida"].isna().sum())
        df = df.drop(columns=["gpa_raw"])

    if "hb_date_raw" in df.columns and "interview_date_raw" in df.columns:
        window = tuple(cfg_clean["dates"]["plausible_window"])
        hb_date = df["hb_date_raw"].map(lambda v: parse_survey_date(v, window))
        interview = df["interview_date_raw"].map(lambda v: parse_survey_date(v, window))
        both = hb_date.notna() & interview.notna()
        gap = (hb_date - interview).abs().dt.days
        df["hb_date_gap_days"] = gap.where(both).astype("float64")
        report["hb_date_usable"] = int(hb_date.notna().sum())
        report["interview_date_usable"] = int(interview.notna().sum())
        report["date_ambiguous_day_month"] = int(
            df["hb_date_raw"].astype("string").str.extract(r"^\s*(\d{1,2})/(\d{1,2})/").astype(float)
            .pipe(lambda d: ((d[0] <= 12) & (d[1] <= 12) & (d[0] != d[1])).sum())
        )
        # Only the gap in days is kept. Raw dates are quasi-identifiers and are dropped here.
        df = df.drop(columns=["hb_date_raw", "interview_date_raw"])

    df, range_counts = apply_plausible_ranges(df, cfg_clean["plausible_ranges"])
    report["set_to_nan_out_of_range"] = range_counts

    categorical_maps = {
        "education_mother": enc["education"],
        "education_husband": enc["education"],
        "income": enc["income"],
        "egg_freq": enc["food_frequency"],
        "food_taboo": enc["yes_no"],
        "hb_ever_tested": enc["hb_ever_tested"],
    }
    for col, mapping in categorical_maps.items():
        if col in df.columns:
            df[col] = map_categories(df[col], mapping, col)

    hfias_items = list(config["hfias"]["items"])
    if all(c in df.columns for c in hfias_items):
        result = score_hfias(df, config["hfias"])
        df["hfias_score"] = result.score
        df["hfias_category"] = result.category.astype("float64")
        report["hfias_format"] = result.response_format
        report["hfias_is_valid_score"] = result.is_valid_hfias
        report["hfias_score_missing"] = int(result.score.isna().sum())
        df = df.drop(columns=hfias_items)

    return df, report


def clean_dataset(config: Mapping[str, Any], path: str | Path | None = None) -> CleanResult:
    """Full deterministic preparation: load, clean features, parse targets, build labels.

    Rows without any usable target (neither a valid Hb nor a valid MUAC) are
    removed because they cannot contribute to training or evaluation of
    either model.
    """
    frame, load_info = load_raw(config, path)
    features, feature_report = clean_features(frame, config)
    for col in NUMERIC_TARGETS:
        features[col] = to_numeric_flexible(features[col])

    labelled, label_report = build_labels(features, config)
    no_target = labelled["hb_target"].isna() & labelled["ced"].isna() & labelled["anemia_class"].isna()
    labelled = labelled.loc[~no_target].reset_index(drop=True)

    report: dict[str, Any] = {
        **load_info,
        **feature_report,
        **label_report,
        "n_rows_without_any_target_dropped": int(no_target.sum()),
        "n_rows_analysis": int(len(labelled)),
    }
    expected = config["data"].get("expected_counts")
    if expected:
        actual = {
            "deduplicated": report["n_rows_raw"] - report["n_duplicate_records_dropped"],
            "hb_model": int((labelled["hb_target"].notna() & labelled["anemia_class"].notna()).sum()),
            "ced_model": int(labelled["ced"].notna().sum()),
        }
        wrong = {k: (actual[k], int(v)) for k, v in expected.items() if actual[k] != int(v)}
        if wrong:
            raise DataSchemaError(f"Sample sizes differ from config data.expected_counts (actual, expected): {wrong}")
    LOGGER.info("Cleaning done: %d of %d rows usable", report["n_rows_analysis"], report["n_rows_raw"])
    return CleanResult(frame=labelled, report=report)


def flatten_report(report: Mapping[str, Any]) -> pd.DataFrame:
    """Turn the nested cleaning report into a two-column table for reports/tables."""
    rows: list[tuple[str, Any]] = []
    for key, value in report.items():
        if isinstance(value, Mapping):
            rows.extend((f"{key}.{k}", v) for k, v in value.items())
        else:
            rows.append((key, value))
    return pd.DataFrame(rows, columns=["item", "value"])
