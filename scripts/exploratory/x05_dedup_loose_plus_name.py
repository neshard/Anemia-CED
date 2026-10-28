# %% [markdown]
# # x05 Near-duplicate sensitivity: loose rule plus the name-matched repeat (EXPLORATORY, POST HOC)
# Follow-up to x02, requested on 2026-10-29 after all results were known. x02 removed probable
# repeats by the strict (2 records) and loose (5 records) rules on 21 fields. x03 (table x25)
# found one further probable repeat by respondent name (same name, interview date, Hb, MUAC and
# pre-pregnancy weight, 3 of 21 fields different) that neither rule catches. This analysis
# removes all six: the five loose records plus that one.
#
# The name is read from the raw export only to find that record and is never stored; no
# anonymous ID is written to any output. Protocol, models, bootstrap seeds and output layout
# are those of x02 (nested 5 x 5 CV, inner 3-fold, 30 Optuna trials, per-fold selection,
# thresholds from inner folds, 2,000 bootstrap resamples; XGBoost with the confirmatory
# baselines). No verdict, nothing in results/tables changes.
#
# Run from the project root:  python scripts/exploratory/x05_dedup_loose_plus_name.py

# %%
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cleaning import clean_dataset  # noqa: E402
from src.hypotheses import _verdict, decide, judge_component  # noqa: E402
from src.models import CED_MODELS, HB_MODELS  # noqa: E402
from src.nested_cv import build_data  # noqa: E402
from src.protocol_stats import ced_metrics, hb_metrics, paired_bootstrap, paired_table, summary_table  # noqa: E402
from src.runs import main_run, run_and_save  # noqa: E402
from src.utils import assert_no_identifiers, load_config, load_hypotheses, resolve_path  # noqa: E402

LABEL = "exploratory, post hoc"
RULE = "loose plus name match (6 records)"
OUT = ROOT / "results" / "exploratory"
base = load_config()
config = copy.deepcopy(base)
config["paths"]["results_dir"] = str(OUT / "x05_dedup_runs")  # checkpoints and per-fold files, git-ignored
seed, B = int(config["project"]["random_seed"]), 2000
CORE = ["age", "height", "prepreg_weight", "anc_visits", "n_children", "n_under5", "education_mother", "income",
        "egg_freq", "food_taboo", "hb", "muac"]
HFIAS = [f"hfias_q{i}" for i in range(1, 10)]
SAME_VISIT = ["hb", "muac", "prepreg_weight", "interview_date_raw"]
NAME_COLUMN = "Nama Lengkap"
HB_SET = ["xgboost", "dummy_mean", "ridge"]
CED_SET = ["xgboost", "dummy_prior", "logistic_regression"]


def save(df: pd.DataFrame, name: str) -> None:
    assert_no_identifiers(df, base)
    df.insert(0, "label", LABEL)
    path = OUT / f"{name}_exploratory_post_hoc.csv"
    df.to_csv(path, index=False)
    print(f"written {path.relative_to(ROOT)}")


# %% Rebuild the 419 deduplicated records exactly as src/cleaning.load_raw (row i is anon_id P{i+1:04d})
raw = pd.read_csv(resolve_path(base, "raw_data"), sep=base["data"]["csv_sep"], encoding=base["data"]["csv_encoding"],
                  dtype=str)
mapped = {k: v for k, v in base["columns"].items() if k != "respondent_code"}
frame = raw[list(mapped.values())].copy()
frame.columns = list(mapped)
kept_raw_rows = frame.index[~frame.duplicated(keep="first")]
frame = frame.loc[kept_raw_rows].reset_index(drop=True)
assert len(frame) == 419, "expected 419 records after the protocol deduplication"
anon = [f"P{i:04d}" for i in range(1, len(frame) + 1)]
names = raw.loc[kept_raw_rows, NAME_COLUMN].fillna("").map(lambda s: " ".join(str(s).lower().split())).to_numpy()
values = frame[CORE + HFIAS].astype(str).apply(lambda s: s.str.strip()).to_numpy()
visit = frame[SAME_VISIT].astype(str).apply(lambda s: s.str.strip()).to_numpy()
del raw

# Loose rule of x02: link pairs that differ in at most 1 of the 21 fields, keep the first of each cluster.
parent = list(range(len(values)))


def find(i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i in range(len(values)):
    for off in np.where((values[i + 1:] != values[i]).sum(axis=1) <= 1)[0]:
        a, b = find(i), find(i + 1 + int(off))
        parent[max(a, b)] = min(a, b)
clusters: dict[int, list[int]] = {}
for i in range(len(values)):
    clusters.setdefault(find(i), []).append(i)
loose = {anon[i] for g in clusters.values() if len(g) > 1 for i in sorted(g)[1:]}

# Name match of x03: same name and the same Hb, MUAC, weight and interview date, not linked by the loose rule.
extra = set()
for i in range(len(values)):
    for j in range(i + 1, len(values)):
        if (names[i] and names[i] == names[j] and (visit[i] == visit[j]).all()
                and int((values[i] != values[j]).sum()) > 1):
            extra.add(anon[j])  # keep the first record, like the other rules
drop = loose | extra
del names
print(f"loose rule: {len(loose)} records, name match not caught by the rules: {len(extra)}, removed in total: {len(drop)}")
if (len(loose), len(extra)) != (5, 1):
    raise SystemExit("Counts differ from x08 (loose 5) and x25 (1 not caught). Stop and check.")

# %% Rerun the protocol without the six records
full = clean_dataset(config).frame
df = full[~full["anon_id"].isin(drop)].reset_index(drop=True)
rule_row = pd.DataFrame([{"rule": RULE, "records_removed": len(drop),
                          "removed_from_hb_model": int(full[full["anon_id"].isin(drop)]["hb_target"].notna().sum()),
                          "removed_from_ced_model": int(full[full["anon_id"].isin(drop)]["ced"].notna().sum())}])
save(rule_row.copy(), "x28_dedup_loose_plus_name_rule")

metric_rows, diff_rows, rbs = [], [], {}
for target, model_names, specs, fn, bseed in (("hb", HB_SET, HB_MODELS, hb_metrics, seed),
                                              ("ced", CED_SET, CED_MODELS, ced_metrics, seed + 100)):
    data = build_data(config, df, target, analysis="dedup_loose_plus_name")
    frames = {m: run_and_save(main_run(specs[m], config), data, config).oof for m in model_names}
    rb = paired_bootstrap(frames, fn, B, bseed)
    rbs[target] = rb
    metrics = (["mae", "rmse", "r2", "sensitivity", "specificity", "auc_roc", "referral_rate"] if target == "hb"
               else ["auc_roc", "pr_auc", "brier", "sensitivity", "specificity", "referral_rate"])
    s = summary_table(rb, model_names, metrics, 0.95)
    s.insert(0, "n", len(data.y))
    s.insert(0, "target", target)
    s.insert(0, "rule", RULE)
    metric_rows.append(s)
    d = (paired_table(rb, "xgboost", ["dummy_mean", "ridge"], "mae", "reduction", 0.95) if target == "hb"
         else paired_table(rb, "xgboost", ["dummy_prior", "logistic_regression"], "auc_roc", "difference", 0.95))
    d.insert(0, "target", target)
    d.insert(0, "rule", RULE)
    diff_rows.append(d)

# %% Side by side with the main analysis and the x02 rules (read only)
cols_m = ["rule", "target", "model", "metric", "point", "ci_low", "ci_high"]
x09 = pd.read_csv(OUT / "x09_dedup_metrics_vs_main_exploratory_post_hoc.csv")[cols_m]
x10 = pd.read_csv(OUT / "x10_dedup_paired_differences_vs_main_exploratory_post_hoc.csv")
cols_d = ["rule", "target", "versus", "metric", "quantity", "point", "ci_low", "ci_high"]
metrics_all = pd.concat([x09, *[m[cols_m] for m in metric_rows]], ignore_index=True)
diffs_all = pd.concat([x10[cols_d], *[d[cols_d] for d in diff_rows]], ignore_index=True)
save(metrics_all.round(4), "x29_dedup_metrics_vs_main_with_name_match")
save(diffs_all.round(4), "x30_dedup_paired_differences_with_name_match")
print(rule_row.to_string(index=False))
print(diffs_all.round(3).to_string(index=False))

# %% Hypothesis components at the CI levels of the main analysis (paper Table II)
# P1 and P2 at 97.5 percent (Holm: the first primary hypothesis was not supported, so both stay
# at the strict level), S1 and S2 at 95 percent. Same component rules as src/hypotheses.py and
# the same bootstrap seeds as 08 (Hb 42, CED 142). The full Holm procedure (decide) is run as a
# check that it assigns the same levels on this sample.
hyp = load_hypotheses(base)
LEVELS = {"P1_anemia_screening": 0.975, "P2_ced_classification": 0.975, "S1_hb_vs_mean": 0.95, "S2_hb_vs_ridge": 0.95}
rows = []
for hid, level in LEVELS.items():
    spec = hyp["hypotheses"][hid]
    outcomes = []
    for comp in spec["components"]:
        r = judge_component(comp, rbs[spec["target"]], spec["model"], level)
        outcomes.append(r["outcome"])
        rows.append({"hypothesis": hid, **{k: r[k] for k in ("component", "criterion", "rule", "point", "ci_low",
                                                               "ci_high", "ci_level", "p_one_sided", "outcome")}})
    for row in rows:
        if row["hypothesis"] == hid:
            row["verdict"] = _verdict(outcomes)
comp_table = pd.DataFrame(rows)
comp_table.insert(0, "rule_applied", RULE)
holm_verdicts, _ = decide(hyp, {"hb": rbs["hb"], "ced": rbs["ced"]})
check = holm_verdicts.set_index("hypothesis")
for hid, level in LEVELS.items():
    if float(check.loc[hid, "ci_level_used"]) != level:
        raise SystemExit(f"Holm assigns {check.loc[hid, 'ci_level_used']} to {hid} on this sample, not {level}. Stop.")
    if check.loc[hid, "verdict"] != comp_table.loc[comp_table["hypothesis"] == hid, "verdict"].iloc[0]:
        raise SystemExit(f"{hid}: verdict differs between the fixed levels and the Holm procedure. Stop.")
comp_table["holm_order"] = comp_table["hypothesis"].map(check["holm_order"])
comp_table["holm_p_value"] = comp_table["hypothesis"].map(check["hypothesis_p_value"])
save(comp_table.round(4), "x31_hypothesis_components_without_six_records")
print(comp_table.drop(columns="rule_applied").round(3).to_string(index=False))
