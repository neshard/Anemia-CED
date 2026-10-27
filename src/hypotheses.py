"""Validating and deciding the four pre-specified hypotheses (proposal Table 1.1).

Only the structured ``components`` of config/hypotheses.yaml decide. Unknown
keys or values raise, so a typo cannot silently weaken a criterion.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.protocol_stats import RepBootstrap, one_sided_p, quantity_samples, summarize

EXPECTED_IDS = ("P1_anemia_screening", "P2_ced_classification", "S1_hb_vs_mean", "S2_hb_vs_ridge")
TOP_KEYS = {"status", "locked_on", "lock_reason", "alpha", "n_bootstrap", "secondary_ci_level", "holm_family",
            "holm_first_ci_level", "holm_second_ci_level_if_first_supported", "paired_test_approximate",
            "primary_hypotheses", "secondary_hypotheses", "hypotheses", "exploratory"}
HYP_KEYS = {"statement", "success_criterion", "target", "model", "threshold_rule", "components"}
COMPONENT_KEYS = {"metric", "quantity", "versus", "op", "value", "rule"}
RULES = {"point_and_ci", "ci_lower_bound"}
OPS = {">=", ">"}
QUANTITIES = {"value", "difference", "reduction"}


class HypothesisSpecError(ValueError):
    """Raised when hypotheses.yaml cannot be evaluated as written."""


def validate(hyp: Mapping[str, Any]) -> None:
    """Refuse anything other than the four proposal hypotheses with well-formed components."""
    problems: list[str] = []
    extra = set(hyp) - TOP_KEYS
    if extra:
        problems.append(f"unknown top-level keys {sorted(extra)}")
    ids = tuple(hyp.get("hypotheses", {}))
    if sorted(ids) != sorted(EXPECTED_IDS):
        problems.append(f"hypotheses must be exactly {list(EXPECTED_IDS)}, found {list(ids)}")
    if sorted(hyp.get("holm_family", [])) != sorted(hyp.get("primary_hypotheses", [])):
        problems.append("holm_family must equal primary_hypotheses")
    for hid, spec in hyp.get("hypotheses", {}).items():
        extra = set(spec) - HYP_KEYS
        if extra:
            problems.append(f"{hid}: unknown keys {sorted(extra)}")
        if spec.get("target") not in ("hb", "ced"):
            problems.append(f"{hid}: target must be hb or ced")
        for i, comp in enumerate(spec.get("components", [])):
            where = f"{hid} component {i}"
            if set(comp) - COMPONENT_KEYS:
                problems.append(f"{where}: unknown keys {sorted(set(comp) - COMPONENT_KEYS)}")
            if comp.get("rule") not in RULES:
                problems.append(f"{where}: rule must be one of {sorted(RULES)}")
            if comp.get("op") not in OPS:
                problems.append(f"{where}: op must be one of {sorted(OPS)}")
            if comp.get("quantity", "value") not in QUANTITIES:
                problems.append(f"{where}: quantity must be one of {sorted(QUANTITIES)}")
            if comp.get("quantity", "value") != "value" and not comp.get("versus"):
                problems.append(f"{where}: a difference or reduction needs 'versus'")
    if problems:
        raise HypothesisSpecError("hypotheses.yaml cannot be evaluated as written:\n  " + "\n  ".join(problems))


def _passes(x: float, op: str, value: float) -> bool:
    return x >= value if op == ">=" else x > value


def judge_component(comp: Mapping[str, Any], rb: RepBootstrap, model: str, level: float) -> dict[str, Any]:
    """Outcome of one component at the given CI level."""
    quantity = comp.get("quantity", "value")
    points, boots = quantity_samples(rb, model, comp["metric"], comp.get("versus"), quantity)
    s = summarize(points, boots, level)
    op, value = comp["op"], float(comp["value"])
    if comp["rule"] == "point_and_ci":
        if _passes(s["point"], op, value) and _passes(s["ci_low"], op, value):
            outcome = "pass"
        elif not _passes(s["point"], op, value) and not _passes(s["ci_high"], op, value):
            outcome = "fail"
        else:
            outcome = "inconclusive"
    else:
        outcome = "met" if _passes(s["ci_low"], op, value) else "not_met"
    label = comp["metric"] if quantity == "value" else f"{comp['metric']} {quantity} vs {comp['versus']}"
    return {"component": label, "criterion": f"{op} {value}", "rule": comp["rule"], **s,
            "p_one_sided": one_sided_p(boots, value, op), "outcome": outcome}


def _verdict(outcomes: list[str]) -> str:
    if any(o in ("fail", "not_met") for o in outcomes):
        return "not_supported"
    if all(o in ("pass", "met") for o in outcomes):
        return "supported"
    return "inconclusive"


def decide(hyp: Mapping[str, Any], boot_by_target: Mapping[str, RepBootstrap]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verdicts and component details, with the Holm order applied to the primary hypotheses.

    Returns (verdicts, components). Every hypothesis row logs its p-value and
    the CI level used.
    """
    validate(hyp)
    specs = hyp["hypotheses"]

    def p_value(hid: str) -> float:
        spec = specs[hid]
        rb = boot_by_target[spec["target"]]
        return max(judge_component(c, rb, spec["model"], 0.95)["p_one_sided"] for c in spec["components"])

    p_values = {hid: p_value(hid) for hid in specs}
    levels: dict[str, float] = {}
    order: dict[str, int] = {}
    family = sorted(hyp["holm_family"], key=lambda h: (p_values[h], h))
    first, second = family[0], family[1]
    levels[first] = float(hyp["holm_first_ci_level"])
    order[first] = 1

    comp_rows: list[dict[str, Any]] = []
    verdict_rows: list[dict[str, Any]] = []

    def run(hid: str, level: float) -> str:
        spec = specs[hid]
        rb = boot_by_target[spec["target"]]
        outcomes = []
        for comp in spec["components"]:
            row = judge_component(comp, rb, spec["model"], level)
            outcomes.append(row["outcome"])
            comp_rows.append({"hypothesis": hid, **row})
        verdict = _verdict(outcomes)
        verdict_rows.append({
            "hypothesis": hid,
            "role": "primary" if hid in hyp["primary_hypotheses"] else "secondary",
            "verdict": verdict,
            "ci_level_used": level,
            "holm_order": order.get(hid),
            "hypothesis_p_value": p_values[hid],
            "statement": spec["statement"],
            "success_criterion": spec["success_criterion"],
        })
        return verdict

    first_verdict = run(first, levels[first])
    levels[second] = (float(hyp["holm_second_ci_level_if_first_supported"]) if first_verdict == "supported"
                      else float(hyp["holm_first_ci_level"]))
    order[second] = 2
    run(second, levels[second])
    for hid in hyp["secondary_hypotheses"]:
        run(hid, float(hyp["secondary_ci_level"]))

    verdicts = pd.DataFrame(verdict_rows).set_index("hypothesis").loc[list(EXPECTED_IDS)].reset_index()
    return verdicts, pd.DataFrame(comp_rows)
