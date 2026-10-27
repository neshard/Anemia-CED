"""Deterministic choice of the dashboard model per target (config/model_selection.yaml).

A pure function of the numbers it receives: the same inputs always give the same
winner. Wall-clock times never enter it.

Steps (the YAML file has the authoritative wording):

1. Drop candidates without a valid TreeExplainer or with a failed additivity check.
2. Provisional winner: the eligible candidate with the best point estimate.
3. Equivalence group: the provisional winner plus every eligible candidate whose
   paired 95 percent CI of the difference against the provisional winner
   contains 0. Defined relative to the provisional winner only, not transitive.
4. Choice, by ``rule["choice"]``:
   * ``best_point_estimate`` (version 2, 2026-10-28): the provisional winner. No
     model is preferred. The group is still reported.
   * ``equivalence_group_preferred`` (version 1, locked 2026-10-27, superseded): the
     preferred model if it is in the group, otherwise the smallest inference cost
     proxy (tree nodes), remaining ties by the order of ``candidates``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.hypotheses import _verdict, judge_component
from src.protocol_stats import RepBootstrap
from src.utils import PROJECT_ROOT

DIRECTIONS = {"lower_is_better": -1, "higher_is_better": 1}
# Exploratory comparators and baselines are never dashboard candidates.
NEVER_CANDIDATES = frozenset({"decision_tree", "random_forest", "dummy_mean", "dummy_prior", "ridge",
                              "logistic_regression"})
CHOICES = frozenset({"best_point_estimate", "equivalence_group_preferred"})


class SelectionRuleError(RuntimeError):
    """Raised when the selection rule is not locked or cannot be applied as written."""


def load_rule(path: str | Path | None = None, *, require_locked: bool = True) -> dict[str, Any]:
    rule = yaml.safe_load(Path(path or PROJECT_ROOT / "config" / "model_selection.yaml").read_text(encoding="utf-8"))
    if require_locked and rule.get("status") != "locked":
        raise SelectionRuleError("config/model_selection.yaml is not locked. Lock it before selecting a model.")
    for target, spec in rule["targets"].items():
        if spec["direction"] not in DIRECTIONS:
            raise SelectionRuleError(f"{target}: direction must be one of {sorted(DIRECTIONS)}")
    banned = NEVER_CANDIDATES & set(rule["candidates"])
    if banned:
        raise SelectionRuleError(f"Decision Tree, Random Forest and baselines cannot be candidates: {sorted(banned)}")
    choice = rule.get("choice", "equivalence_group_preferred")
    if choice not in CHOICES:
        raise SelectionRuleError(f"choice must be one of {sorted(CHOICES)}")
    if choice == "equivalence_group_preferred" and rule.get("preferred") not in rule["candidates"]:
        raise SelectionRuleError("preferred must be one of the candidates")
    if choice == "best_point_estimate" and rule.get("preferred") is not None:
        raise SelectionRuleError("best_point_estimate prefers no model: preferred must be null")
    return rule


@dataclass(frozen=True)
class SelectionResult:
    target: str
    metric: str
    selected: str
    provisional_winner: str
    equivalence_group: tuple[str, ...]
    reason: str
    table: pd.DataFrame = field(compare=False)


def ci_contains_zero(ci: tuple[float, float]) -> bool:
    low, high = ci
    return low <= 0.0 <= high


def select_model(
    rule: Mapping[str, Any],
    target: str,
    points: Mapping[str, float],
    paired_ci: Mapping[tuple[str, str], tuple[float, float]],
    eligible: Mapping[str, bool],
    cost_proxy: Mapping[str, float],
) -> SelectionResult:
    """Apply the locked rule.

    ``paired_ci[(a, b)]`` is the 95 percent CI of metric(a) - metric(b) on the same
    resampled women. Step 3 uses candidate minus provisional winner. Only whether that
    CI contains 0 decides membership, so the sign never changes the outcome.
    """
    spec = rule["targets"][target]
    metric, sign = spec["metric"], DIRECTIONS[spec["direction"]]
    candidates = list(rule["candidates"])
    missing = [c for c in candidates if c not in points or c not in eligible or c not in cost_proxy]
    if missing:
        raise SelectionRuleError(f"{target}: no inputs for {missing}")

    pool = [c for c in candidates if eligible[c]]
    if not pool:
        raise SelectionRuleError(
            f"{target}: no eligible candidate. Every candidate failed the SHAP eligibility check "
            f"({', '.join(candidates)}), so no dashboard model can be selected. Report this and decide "
            "with the supervisor. The rule never falls back to an ineligible model.")
    order = {c: i for i, c in enumerate(candidates)}
    provisional = max(pool, key=lambda c: (sign * points[c], -order[c]))

    def ci_vs(c: str) -> tuple[float, float]:
        """95 percent CI of metric(c) - metric(provisional winner)."""
        if (c, provisional) in paired_ci:
            return paired_ci[(c, provisional)]
        low, high = paired_ci[(provisional, c)]
        return -high, -low

    group = [provisional] + [c for c in pool if c != provisional and ci_contains_zero(ci_vs(c))]
    group.sort(key=lambda c: order[c])
    preferred = rule.get("preferred")
    if rule.get("choice", "equivalence_group_preferred") == "best_point_estimate":
        selected = provisional
        others = [c for c in group if c != provisional]
        why = f"{provisional} has the best eligible {metric} (no preferred model)"
        if others:
            why += (f". The 95 percent paired CI against {', '.join(others)} contains 0, "
                    "so the advantage is not statistically established")
    elif preferred in group:
        selected = preferred
        why = f"{preferred} is in the equivalence group of the provisional winner {provisional}"
        if preferred == provisional:
            why = f"{preferred} has the best eligible {metric}"
    else:
        selected = min(group, key=lambda c: (cost_proxy[c], order[c]))
        why = (f"{preferred} is not in the equivalence group of {provisional}. "
               f"{selected} has the smallest inference cost proxy in the group")

    rows = []
    for c in candidates:
        ci = ci_vs(c) if c != provisional and eligible[c] else (float("nan"), float("nan"))
        rows.append({"target": target, "candidate": c, "metric": metric, "point": points[c],
                     "eligible_shap": bool(eligible[c]),
                     "diff_candidate_minus_winner_ci_low": ci[0], "diff_candidate_minus_winner_ci_high": ci[1],
                     "provisional_winner": c == provisional, "in_equivalence_group": c in group,
                     "inference_cost_proxy": cost_proxy[c], "selected": c == selected})
    return SelectionResult(target, metric, selected, provisional, tuple(group), why, pd.DataFrame(rows))


def hb_explanation_gate(rule: Mapping[str, Any], rb_hb: RepBootstrap, model: str) -> dict[str, Any]:
    """Proposal 3.6 gate for the selected Hb model, judged exactly like component S1_hb_vs_mean."""
    gate = rule["hb_explanation_gate"]
    comp = {k: gate[k] for k in ("metric", "quantity", "versus", "op", "value", "rule")}
    row = judge_component(comp, rb_hb, model, float(gate["ci_level"]))
    verdict = _verdict([row["outcome"]])
    return {**row, "model": model, "verdict": verdict, "hb_explanation_enabled": verdict in gate["enable_on"]}
