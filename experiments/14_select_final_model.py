# %% [markdown]
# # 14 Dashboard model selection (supervisor revision 2026-10-27)
# Applies the rule locked in config/model_selection.yaml with the deterministic function
# src/model_selection.select_model. Inputs: the pooled out-of-fold metrics and pairwise
# differences from 12 (m01, m02, m04), the SHAP eligibility from 13 (m06) and the median
# tree-node count per model from the fold tables. Wall-clock times are not used.
# Then the proposal 3.6 gate is judged for the selected Hb model, exactly like S1.
#
# Writes artifacts/model_selection.json (read by 09, 10 and 11) and the tables m10/m11.
# The selected model was chosen from five candidates on the same data, so its
# cross-validated performance is optimistic. All five are reported side by side in m01/m02.
#
# Run from the project root:  python experiments/14_select_final_model.py

# %%
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from _common import ROOT, start

from src.model_selection import hb_explanation_gate, load_rule, select_model
from src.protocol_stats import hb_metrics, paired_bootstrap
from src.runs import load_folds, load_oof, save_result_table
from src.utils import library_versions, resolve_path

config, hyp = start("14_select_final_model")
assert hyp is not None
rule = load_rule(require_locked=not config["project"].get("smoke_test"))
# The superseded version 1 rule (commit 0cc4098) is applied side by side for transparency only.
RULE_V1 = {**rule, "choice": "equivalence_group_preferred", "preferred": "xgboost"}
CANDS = list(rule["candidates"])
tables = resolve_path(config, "results_dir") / "tables"
metrics = {"hb": pd.read_csv(tables / "m01_hb_boosting_metrics.csv"), "ced": pd.read_csv(tables / "m02_ced_boosting_metrics.csv")}
pairs = pd.read_csv(tables / "m04_pairwise_primary_metric_differences.csv")
for target, seed in rule["paired_bootstrap_seed"].items():
    used = set(pairs.loc[pairs["target"] == target, "bootstrap_seed"])
    if used != {int(seed)}:
        raise SystemExit(f"m04 for {target} was drawn with seed {sorted(used)}, the locked rule says {seed}. Rerun 12.")
elig = pd.read_csv(tables / "m06_shap_eligibility.csv")

results, summary_rows = {}, []
for target in ("hb", "ced"):
    metric = rule["targets"][target]["metric"]
    m = metrics[target].set_index("model")
    points = {c: float(m.loc[c, metric]) for c in CANDS}
    p = pairs[(pairs["target"] == target) & (pairs["metric"] == metric)]
    paired_ci = {(r.model_a, r.model_b): (float(r.ci_low_95), float(r.ci_high_95)) for r in p.itertuples()}
    e = elig[elig["target"] == target].set_index("model")
    eligible = {c: bool(e.loc[c, "eligible"]) for c in CANDS}
    cost = {c: float(load_folds(config, f"main__{target}__{c}__main")["n_tree_nodes"].median()) for c in CANDS}
    res = select_model(rule, target, points, paired_ci, eligible, cost)
    v1 = select_model(RULE_V1, target, points, paired_ci, eligible, cost)
    results[target] = res
    table = res.table.assign(selection_reason=res.reason, rule_version=rule.get("version", 1),
                             selected_by_version1_rule=res.table["candidate"] == v1.selected,
                             optimism_note=rule["optimism_note"].strip())
    table["eligibility_reason"] = [e.loc[c, "reason"] for c in CANDS]
    summary_rows.append(table)
    print(f"{target}: selected {res.selected} (provisional winner {res.provisional_winner}, "
          f"group {list(res.equivalence_group)}). {res.reason}. Version 1 rule would select {v1.selected}.")

save_result_table(pd.concat(summary_rows, ignore_index=True), "m10_model_selection_summary", config)

# %% Proposal 3.6 gate for the selected Hb model (same bootstrap seed as S1 in 08)
hb_sel = results["hb"].selected
frames = {hb_sel: load_oof(config, f"main__hb__{hb_sel}__main"),
          "dummy_mean": load_oof(config, "main__hb__dummy_mean__main")}
rb = paired_bootstrap(frames, hb_metrics, int(hyp["n_bootstrap"]), int(rule["hb_explanation_gate"]["bootstrap_seed"]))
gate = hb_explanation_gate(rule, rb, hb_sel)
save_result_table(pd.DataFrame([gate]), "m11_hb_explanation_gate", config)
print(f"Hb gate ({hb_sel} vs dummy_mean, MAE reduction >= 0.10): {gate['outcome']} -> {gate['verdict']}, "
      f"explanation enabled: {gate['hb_explanation_enabled']}")

selection = {
    "rule_file": "config/model_selection.yaml",
    "rule_locked_on": str(rule.get("locked_on")),
    "rule_version": rule.get("version", 1),
    "rule_note": "version 2 set on 2026-10-28 after results were known (deviation, see CHANGELOG)",
    "computed_at": datetime.now().isoformat(timespec="seconds"),
    "seed": int(config["project"]["random_seed"]),
    "paired_bootstrap_seed": {t: int(s) for t, s in rule["paired_bootstrap_seed"].items()},
    "hb_gate_bootstrap_seed": int(rule["hb_explanation_gate"]["bootstrap_seed"]),
    "library_versions": library_versions(),
    "optimism_note": rule["optimism_note"].strip(),
    **{t: {"algorithm": r.selected, "metric": r.metric,
           "metric_point": float(r.table.set_index("candidate").loc[r.selected, "point"]),
           "provisional_winner": r.provisional_winner, "equivalence_group": list(r.equivalence_group),
           "reason": r.reason} for t, r in results.items()},
}
selection["hb"]["explanation_gate"] = {k: gate[k] for k in ("point", "ci_low", "ci_high", "ci_level", "outcome", "verdict",
                                                             "hb_explanation_enabled")}
path = resolve_path(config, "artifacts_dir") / "model_selection.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(selection, indent=2, default=str), encoding="utf-8")
print(f"Saved {path}")
if results["hb"].selected != "xgboost" or results["ced"].selected != "xgboost":
    print("NOTE: a selected model is not XGBoost. Ask the researcher whether 07 should be run for it.")
