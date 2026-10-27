"""Target construction: Hb regression target, WHO anemia classes, and CED label.

Labels are derived from measured Hb and MUAC with fixed clinical cut-offs.
No parameter is estimated from data, so building labels before the split
does not leak. The Hb regression target and the WHO class are kept in
separate columns because the Hb == 0 policy can make them disagree (a row can
carry the class "Severe" while having no regression target).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

WHO_CLASSES: tuple[str, ...] = ("Normal", "Mild", "Moderate", "Severe")
HB_ZERO_POLICIES: tuple[str, ...] = ("missing", "severe_drop", "severe_floor")


def who_anemia_class(
    hb: npt.ArrayLike,
    normal: float = 11.0,
    mild: float = 10.0,
    moderate: float = 7.0,
) -> npt.NDArray[np.object_]:
    """Map Hb in g/dL to WHO anemia severity for pregnant women.

    Normal >= 11.0, Mild 10.0 to below 11.0, Moderate 7.0 to below 10.0,
    Severe below 7.0. A value equal to a cut-off belongs to the higher class,
    so 11.0 is Normal and 7.0 is Moderate. Using "below the next cut-off"
    instead of the printed 10.9 or 9.9 keeps values such as 10.95 (from a
    regressor) in Mild rather than in a gap. Missing values return None.
    """
    values = np.asarray(hb, dtype="float64")
    classes = np.select(
        [values >= normal, values >= mild, values >= moderate, values < moderate],
        list(WHO_CLASSES),
        default="",
    ).astype(object)
    classes[np.isnan(values)] = None
    return classes


def anemia_binary(hb: npt.ArrayLike, threshold: float = 11.0) -> npt.NDArray[np.float64]:
    """1.0 when Hb is below ``threshold``, 0.0 otherwise, NaN when unknown."""
    values = np.asarray(hb, dtype="float64")
    out = (values < threshold).astype("float64")
    out[np.isnan(values)] = np.nan
    return out


def ced_label(muac: npt.ArrayLike, threshold: float = 23.5) -> npt.NDArray[np.float64]:
    """1.0 when MUAC is below ``threshold`` cm (CED risk), 0.0 otherwise, NaN when unknown."""
    values = np.asarray(muac, dtype="float64")
    out = (values < threshold).astype("float64")
    out[np.isnan(values)] = np.nan
    return out


def build_labels(df: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add ``hb_target``, ``anemia_class``, ``anemia_binary`` and ``ced`` to the frame.

    Rules from config ``cleaning``:

    * Hb above ``hb_max_valid`` is a data entry error. The row gets no Hb
      target and no anemia class, so it is excluded from Hb training and
      evaluation. The CED label is unaffected.
    * Hb == 0 follows ``hb_zero_policy``. ``missing`` treats it as not
      measured. ``severe_drop`` labels the class Severe but gives no
      regression target. ``severe_floor`` labels Severe and sets the target to
      ``hb_zero_floor_value``.
    * MUAC outside ``muac_valid_range`` gets no CED label.

    The raw ``hb`` column is removed afterwards so it cannot be picked up as
    a feature by mistake. ``muac`` is kept as the validated measurement for
    EDA and is listed as forbidden for every model.
    """
    cfg = config["cleaning"]
    lab = config["labels"]
    policy = str(cfg["hb_zero_policy"])
    if policy not in HB_ZERO_POLICIES:
        raise ValueError(f"hb_zero_policy must be one of {HB_ZERO_POLICIES}, got '{policy}'")

    out = df.copy()
    hb = out["hb"].astype("float64")
    is_zero = hb == 0
    too_high = hb > float(cfg["hb_max_valid"])

    hb_valid = hb.where(~(is_zero | too_high))
    thresholds = lab["who_anemia_thresholds"]
    klass = pd.Series(
        who_anemia_class(hb_valid, thresholds["normal"], thresholds["mild"], thresholds["moderate"]),
        index=out.index,
        dtype=object,
    )
    hb_target = hb_valid.copy()

    if policy in ("severe_drop", "severe_floor"):
        klass[is_zero] = "Severe"
    if policy == "severe_floor":
        hb_target[is_zero] = float(cfg["hb_zero_floor_value"])

    out["hb_zero_flag"] = is_zero.astype("float64")
    out["hb_target"] = hb_target
    out["anemia_class"] = klass
    out["anemia_binary"] = np.where(klass.isna(), np.nan, (klass != "Normal").astype("float64"))

    low, high = cfg["muac_valid_range"]
    muac = out["muac"].astype("float64")
    muac_invalid = muac.notna() & ((muac < low) | (muac > high))
    out["muac"] = muac.where(~muac_invalid)
    out["ced"] = ced_label(out["muac"], float(lab["ced_muac_threshold"]))

    untested = out["hb_ever_tested"] == 0 if "hb_ever_tested" in out.columns else pd.Series(False, index=out.index)
    severe = out["anemia_class"] == "Severe"
    report: dict[str, Any] = {
        "hb_zero_policy": policy,
        "n_hb_zero": int(is_zero.sum()),
        "n_hb_zero_reporting_never_tested": int((is_zero & untested).sum()),
        "n_hb_above_max_dropped_from_hb_model": int(too_high.sum()),
        "n_hb_missing_in_file": int(hb.isna().sum()),
        "n_hb_target_available": int(out["hb_target"].notna().sum()),
        "n_severe_total": int(severe.sum()),
        "n_severe_from_hb_zero": int((severe & is_zero).sum()),
        "n_muac_invalid": int(muac_invalid.sum()),
        "n_ced_label_available": int(out["ced"].notna().sum()),
        "n_ced_risk": int((out["ced"] == 1).sum()),
        "anemia_class_counts": {c: int((out["anemia_class"] == c).sum()) for c in WHO_CLASSES},
        "hb_values_are_integers": bool(np.allclose(hb.dropna() % 1, 0)),
        "muac_values_are_integers": bool(np.allclose(muac.dropna() % 1, 0)),
    }
    out = out.drop(columns=["hb"])
    return out, report


def ced_label_at_most(muac: npt.ArrayLike, cutoff_cm: float) -> npt.NDArray[np.float64]:
    """1.0 when recorded MUAC is at most ``cutoff_cm`` (sensitivity analysis), NaN when unknown.

    The main label (MUAC below 23.5) equals "23 or less" for whole-centimetre
    values. The 22 cm and 24 cm sensitivity labels use the same "or less" form.
    """
    values = np.asarray(muac, dtype="float64")
    out = (values <= cutoff_cm).astype("float64")
    out[np.isnan(values)] = np.nan
    return out


def trimester_anemia(
    hb: npt.ArrayLike, gestational_weeks: npt.ArrayLike, first_max_week: float, second_max_week: float,
    cutoffs: Mapping[str, float],
) -> npt.NDArray[np.float64]:
    """Anemia by the WHO 2024 trimester-specific cutoffs, NaN when Hb or gestational age is unknown.

    First trimester up to ``first_max_week``, second up to ``second_max_week``,
    third afterwards. Women without a usable gestational age get NaN and are
    excluded from that sensitivity analysis.
    """
    h = np.asarray(hb, dtype="float64")
    w = np.asarray(gestational_weeks, dtype="float64")
    cut = np.where(w <= first_max_week, cutoffs["first"],
                   np.where(w <= second_max_week, cutoffs["second"], cutoffs["third"]))
    out = (h < cut).astype("float64")
    out[np.isnan(h) | np.isnan(w)] = np.nan
    return out
