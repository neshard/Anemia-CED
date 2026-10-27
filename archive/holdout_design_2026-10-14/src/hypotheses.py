"""Reading, validating and deciding the pre-specified hypotheses.

config/hypotheses.yaml states that only structured keys decide and that the
``success_criterion`` text is documentation. This module follows that rule:
every decision below reads a named key, and an unknown key or value raises
instead of being ignored, so a typo cannot silently weaken a criterion.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import holm_adjust

TOP_LEVEL_KEYS = {
    "status", "locked_on", "alpha", "multiple_comparison_correction", "holm_family", "paired_test",
    "effect_size", "decision_setup", "ablation_tuning", "primary_hypotheses", "hypotheses",
}
HYPOTHESIS_KEYS = {
    "statement", "metric", "comparator", "direction", "success_criterion", "min_meaningful_difference",
    "operating_point", "min_sensitivity", "min_specificity", "evaluation", "ci_level", "non_inferiority_margin",
    "min_auc", "auc_ci_lower_min", "pass_rule", "nominal", "coverage_min", "coverage_max",
}
ALLOWED_VALUES: dict[str, set[str]] = {
    "multiple_comparison_correction": {"holm", "bonferroni", "none"},
    "holm_family": {"per_hypothesis", "all_primary_superiority"},
    "paired_test": {"corrected_resampled_t", "wilcoxon"},
    "effect_size": {"paired_dz"},
    "ablation_tuning": {"retune", "reuse"},
    "pass_rule": {"any", "all"},
    "evaluation": {"cv", "pooled_out_of_fold", "holdout"},
    "direction": {"lower", "higher", "not_worse"},
}

# What each hypothesis compares. "reference" is the model the hypothesis is about.
# Comparator names from the YAML are resolved by ``resolve_comparator``.
REGISTRY: dict[str, dict[str, Any]] = {
    "H1_hb_regression": {"kind": "superiority", "target": "hb_model"},
    "H2_hb_vs_baselines": {"kind": "superiority", "target": "hb_model"},
    "H3_anemia_screening": {"kind": "holdout_screen", "target": "hb_model"},
    "H4_regression_vs_direct": {"kind": "non_inferiority", "target": "hb_model"},
    "H5_ced_classification": {"kind": "superiority", "target": "ced_model"},
    "H6_resampling_ced": {"kind": "ablation", "target": "ced_model"},
    "H7_resampling_hb": {"kind": "ablation", "target": "hb_model"},
    "H8_hb_interval_coverage": {"kind": "coverage", "target": "hb_model"},
}

Ref = tuple[str, str, str]  # (target, model, variant)


class HypothesisSpecError(ValueError):
    """Raised when hypotheses.yaml contains something the code cannot evaluate."""


def parse_ci_level(value: Any) -> float:
    """Turn ``two_sided_90`` (or 0.9) into a two-sided confidence level."""
    if isinstance(value, (int, float)):
        return float(value)
    match = re.fullmatch(r"two_sided_(\d{2})", str(value))
    if not match:
        raise HypothesisSpecError(f"ci_level '{value}' is not understood. Use two_sided_NN or a number.")
    return int(match.group(1)) / 100


def validate(hyp: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Refuse unknown keys, unknown values and settings that contradict the config."""
    problems: list[str] = []
    unknown = set(hyp) - TOP_LEVEL_KEYS
    if unknown:
        problems.append(f"unknown top-level keys {sorted(unknown)}")
    for key, allowed in ALLOWED_VALUES.items():
        if key in hyp and hyp[key] not in allowed:
            problems.append(f"{key}: '{hyp[key]}' not in {sorted(allowed)}")
    setup = hyp.get("decision_setup")
    if setup not in config["feature_setups"]:
        problems.append(f"decision_setup '{setup}' is not a feature setup in config")
    elif setup not in config["run_setups"]:
        problems.append(f"decision_setup '{setup}' is not in config run_setups, so it would never be trained")
    elif setup not in config["ablation_setups"]:
        problems.append(f"decision_setup '{setup}' is not in config ablation_setups, so H6 and H7 cannot be judged")
    for hid in hyp.get("primary_hypotheses", []):
        if hid not in hyp.get("hypotheses", {}):
            problems.append(f"primary_hypotheses lists '{hid}', which is not defined")
    for hid, spec in hyp.get("hypotheses", {}).items():
        if hid not in REGISTRY:
            problems.append(f"{hid}: no evaluation rule exists in src/hypotheses.py")
            continue
        extra = set(spec) - HYPOTHESIS_KEYS
        if extra:
            problems.append(f"{hid}: unknown keys {sorted(extra)}")
        for key in ("direction", "pass_rule", "evaluation"):
            if key in spec and spec[key] not in ALLOWED_VALUES[key]:
                problems.append(f"{hid}.{key}: '{spec[key]}' not in {sorted(ALLOWED_VALUES[key])}")
        if "ci_level" in spec:
            try:
                parse_ci_level(spec["ci_level"])
            except HypothesisSpecError as err:
                problems.append(f"{hid}: {err}")
        if REGISTRY[hid]["kind"] == "coverage":
            nominal = float(config["intervals"]["nominal"])
            if abs(float(spec.get("nominal", nominal)) - nominal) > 1e-9:
                problems.append(f"{hid}.nominal {spec['nominal']} differs from config intervals.nominal {nominal}")
    if problems:
        raise HypothesisSpecError("hypotheses.yaml cannot be evaluated as written:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# Comparison plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Comparison:
    """One paired comparison on CV folds."""

    hypothesis: str
    kind: str
    reference: Ref
    comparator: Ref
    metric: str
    ci_level: float
    evaluation: str


def _as_list(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def comparison_plan(hyp: Mapping[str, Any], config: Mapping[str, Any]) -> list[Comparison]:
    """Every CV comparison the hypotheses ask for, with names resolved to saved runs."""
    primary = config["models"]["primary_resampling"]
    direct = config["models"]["direct_comparator_resampling"]
    alpha = float(hyp["alpha"])
    plan: list[Comparison] = []
    for hid, spec in hyp["hypotheses"].items():
        kind = REGISTRY[hid]["kind"]
        metric = spec["metric"] if isinstance(spec.get("metric"), str) else None
        evaluation = spec.get("evaluation", "cv")
        if kind == "superiority":
            target = REGISTRY[hid]["target"]
            ref = (target, "xgboost", primary[target])
            for name in _as_list(spec["comparator"]):
                plan.append(Comparison(hid, kind, ref, (target, name, primary[target]), metric, 1 - alpha, evaluation))
        elif kind == "non_inferiority":
            level = parse_ci_level(spec.get("ci_level", 1 - 2 * alpha))
            for name in _as_list(spec["comparator"]):
                if name != "direct_xgb_classifier":
                    raise HypothesisSpecError(f"{hid}: comparator '{name}' has no saved run")
                plan.append(Comparison(hid, kind, ("hb_model", "xgboost", primary["hb_model"]),
                                       ("anemia_binary", name, direct), metric, level, evaluation))
        elif kind == "ablation":
            target = REGISTRY[hid]["target"]
            baseline = (target, "xgboost", "ablation_none")
            if target == "ced_model":
                # H6 lists the treatments as "comparator"; each is tested against no resampling.
                for name in _as_list(spec["comparator"]):
                    if name not in ("smotenc", "scale_pos_weight"):
                        raise HypothesisSpecError(f"{hid}: unknown resampling variant '{name}'")
                    plan.append(Comparison(hid, kind, (target, "xgboost", f"ablation_{name}"), baseline, metric,
                                           1 - alpha, evaluation))
            else:
                if _as_list(spec["comparator"]) != ["no_resampling"]:
                    raise HypothesisSpecError(f"{hid}: only 'no_resampling' is supported as comparator")
                plan.append(Comparison(hid, kind, (target, "xgboost", "ablation_smogn"), baseline, metric,
                                       1 - alpha, evaluation))
    return plan


def adjust_pvalues(table: pd.DataFrame, hyp: Mapping[str, Any], p_col: str) -> pd.DataFrame:
    """Add ``p_adjusted`` (decision) and ``p_adjusted_sensitivity`` columns.

    Decision family follows ``holm_family``. The sensitivity column uses the
    other definition so both are always reported. Families are formed within
    each feature setup. Non-inferiority rows are judged by their CI, not by a
    p value, so they are never in a family.
    """
    out = table.copy()
    method = hyp["multiple_comparison_correction"]

    def adjust(p: np.ndarray) -> np.ndarray:
        if method == "holm":
            return holm_adjust(p)
        if method == "bonferroni":
            return np.minimum(1.0, p * len(p))
        return p

    testable = out["kind"].isin(["superiority", "ablation"])
    out["p_per_hypothesis"] = np.nan
    out["p_all_superiority"] = np.nan
    for setup, block in out[testable].groupby("setup"):
        out.loc[block.index, "p_all_superiority"] = adjust(block[p_col].to_numpy())
        for _, fam in block.groupby("hypothesis"):
            out.loc[fam.index, "p_per_hypothesis"] = adjust(fam[p_col].to_numpy())
    decision, sensitivity = (("p_per_hypothesis", "p_all_superiority") if hyp["holm_family"] == "per_hypothesis"
                             else ("p_all_superiority", "p_per_hypothesis"))
    out["p_adjusted"] = out[decision]
    out["p_adjusted_sensitivity"] = out[sensitivity]
    out["holm_family_decision"] = hyp["holm_family"]
    return out


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def _comparison_passes(row: pd.Series, alpha: float, min_diff: float | None, p_column: str) -> bool:
    improvement = row["decision_improvement"]
    if not (row[p_column] < alpha and improvement > 0):
        return False
    return min_diff is None or improvement >= min_diff


def decide(
    hyp: Mapping[str, Any],
    comparisons: pd.DataFrame,
    holdout: Mapping[str, Mapping[str, float]],
    p_column: str = "p_adjusted",
) -> pd.DataFrame:
    """One verdict per hypothesis for the decision setup.

    ``comparisons`` must hold the decision setup's rows with a
    ``decision_improvement`` column (pooled out-of-fold for H7, fold mean
    otherwise). ``holdout`` maps hypothesis id to hold-out values:
    H3 {auc_ci_low, sensitivity, specificity}, H5 {auc, auc_ci_low},
    H8 {coverage}.
    """
    alpha = float(hyp["alpha"])
    primary = set(hyp.get("primary_hypotheses", []))
    rows = []
    for hid, spec in hyp["hypotheses"].items():
        kind = REGISTRY[hid]["kind"]
        sub = comparisons[comparisons["hypothesis"] == hid]
        checks: list[tuple[str, bool]] = []
        if kind in ("superiority", "ablation"):
            min_diff = spec.get("min_meaningful_difference")
            min_diff = float(min_diff) if min_diff is not None else None
            results = []
            for _, r in sub.iterrows():
                ok = _comparison_passes(r, alpha, min_diff, p_column)
                results.append(ok)
                label = r["reference"].split("/", 1)[1] + " vs " + r["comparator"].split("/", 1)[1]
                checks.append((f"{label}: improvement {r['decision_improvement']:.4f}"
                               f"{f' (needs >= {min_diff})' if min_diff is not None else ''}, "
                               f"p_adj {r[p_column]:.4f}, d_z {r['paired_dz']:.2f}", ok))
            rule = spec.get("pass_rule", "all")
            passed = bool(results) and (any(results) if rule == "any" else all(results))
            if hid == "H5_ced_classification":
                h = holdout[hid]
                if "min_auc" in spec:
                    ok = h["auc"] >= float(spec["min_auc"])
                    checks.append((f"hold-out AUC {h['auc']:.3f} (needs >= {spec['min_auc']})", ok))
                    passed = passed and ok
                if "auc_ci_lower_min" in spec:
                    ok = h["auc_ci_low"] >= float(spec["auc_ci_lower_min"])
                    checks.append((f"hold-out AUC CI lower {h['auc_ci_low']:.3f} (needs >= {spec['auc_ci_lower_min']})", ok))
                    passed = passed and ok
            rule_text = f"pass_rule {rule}"
        elif kind == "non_inferiority":
            r = sub.iloc[0]
            margin = float(spec["non_inferiority_margin"])
            passed = bool(r["improvement_ci_low"] > -margin)
            checks.append((f"AUC regression minus direct {r['mean_improvement']:.3f}, {r['ci_level']:.0%} CI lower "
                           f"{r['improvement_ci_low']:.3f} (needs > {-margin})", passed))
            rule_text = "non-inferiority on CV folds"
        elif kind == "holdout_screen":
            h = holdout[hid]
            passed = True
            for key, value_key, label in (("min_sensitivity", "sensitivity", "sensitivity"),
                                          ("min_specificity", "specificity", "specificity"),
                                          ("auc_ci_lower_min", "auc_ci_low", "AUC CI lower")):
                if key in spec:
                    ok = h[value_key] >= float(spec[key])
                    checks.append((f"hold-out {label} {h[value_key]:.3f} (needs >= {spec[key]})", ok))
                    passed = passed and ok
            rule_text = f"cut-off {h['cutoff']:.3f} g/dL fixed on training out-of-fold predictions"
        elif kind == "coverage":
            cov = holdout[hid]["coverage"]
            passed = float(spec["coverage_min"]) <= cov <= float(spec["coverage_max"])
            checks.append((f"hold-out coverage {cov:.3f} (needs {spec['coverage_min']} to {spec['coverage_max']}, "
                           f"half-width {holdout[hid]['half_width']:.2f} g/dL)", passed))
            rule_text = f"nominal {spec.get('nominal')}"
        else:
            raise HypothesisSpecError(f"{hid}: unsupported kind {kind}")
        rows.append({
            "hypothesis": hid,
            "role": "primary" if hid in primary else "secondary",
            "supported": bool(passed),
            "rule": rule_text,
            "detail": " | ".join(f"{'PASS' if ok else 'FAIL'} {text}" for text, ok in checks),
            "statement": spec["statement"],
        })
    return pd.DataFrame(rows)


def pooled_oof_metric(oof: pd.DataFrame, metric: str, anemia_threshold: float = 11.0) -> pd.Series:
    """Metric computed on all out-of-fold predictions of each CV repeat (one value per repeat)."""
    if metric != "mae_hb_below_11":
        raise HypothesisSpecError(f"pooled_out_of_fold is only implemented for mae_hb_below_11, not {metric}")
    low = oof[oof["y_true"] < anemia_threshold]
    return low.assign(err=(low["y_true"] - low["pred"]).abs()).groupby("repeat")["err"].mean()


def pooled_improvement(ref_oof: pd.DataFrame, cmp_oof: pd.DataFrame, metric: str, direction: int) -> float:
    """Mean over repeats of the pooled metric difference, positive when the reference is better."""
    ref = pooled_oof_metric(ref_oof, metric)
    cmp_ = pooled_oof_metric(cmp_oof, metric)
    return float(np.mean(direction * (ref - cmp_.reindex(ref.index)).to_numpy()))


def summarize_keys(hyp: Mapping[str, Any]) -> Sequence[str]:
    """Human-readable list of how each top-level key is used, for the run log."""
    return [
        f"decision_setup={hyp['decision_setup']}",
        f"holm_family={hyp['holm_family']} (sensitivity column uses the other family)",
        f"paired_test={hyp['paired_test']}, effect_size={hyp['effect_size']}",
        f"ablation_tuning={hyp['ablation_tuning']}",
        f"primary_hypotheses={hyp.get('primary_hypotheses', [])}",
    ]
