# %% [markdown]
# # 00 Prepare data
# Deterministic cleaning (proposal Table 3.1), sample-size assertions, the
# stratification keys of both models with their final cell counts, the
# date-gap counts compared with the proposal, and the gestational age parsing
# counts. There is no hold-out split in this protocol. Outputs are aggregate
# tables only. No patient row is printed or written.
#
# Run from the project root:  python experiments/00_prepare_data.py

# %%
from __future__ import annotations

import copy

import pandas as pd
from _common import start

from src.cleaning import clean_dataset, flatten_report
from src.labels import WHO_CLASSES
from src.nested_cv import build_data, describe_strata
from src.runs import save_result_table

config, _ = start("00_prepare_data", require_hypotheses=False)

# %% Cleaning (asserts 420 raw, 419 deduplicated, 396 Hb model, 410 CED model)
result = clean_dataset(config)
df, rep = result.frame, result.report
save_result_table(flatten_report(rep), "a01_cleaning_report", config)
print(flatten_report(rep).to_string(index=False))

# %% Hb == 0 policy counts
rows = []
for policy in ("missing", "severe_drop", "severe_floor"):
    cfg = copy.deepcopy(config)
    cfg["cleaning"]["hb_zero_policy"] = policy
    cfg["data"]["expected_counts"] = None
    r = clean_dataset(cfg).report
    rows.append({"policy": policy, "n_hb_target": r["n_hb_target_available"],
                 **{f"n_{c.lower()}": r["anemia_class_counts"][c] for c in WHO_CLASSES},
                 "n_hb_zero_never_tested": r["n_hb_zero_reporting_never_tested"]})
save_result_table(pd.DataFrame(rows), "a02_hb_zero_policy_counts", config)

# %% Stratification keys (built before any split) with final cell counts and merges
strata, merges = [], []
for target in ("hb", "ced"):
    data = build_data(config, df, target)
    strata.append(describe_strata(data))
    merges += [{"target": target, **m} for m in data.strat_log]
strata_table = pd.concat(strata, ignore_index=True)
save_result_table(strata_table, "a03_strat_cells", config)
save_result_table(pd.DataFrame(merges, columns=["target", "from", "to", "n_rows"]), "a04_strat_merges", config)
print(strata_table.to_string(index=False))
print(pd.DataFrame(merges).to_string(index=False) if merges else "no merges needed")

# %% Date gaps (sensitivity analysis input) compared with the proposal
hb_rows = df[df["hb_target"].notna() & df["anemia_class"].notna()]
gap = hb_rows["hb_date_gap_days"]
dates = pd.DataFrame([
    {"quantity": "usable dates in Hb model", "this_run": int(gap.notna().sum()), "proposal": 381},
    {"quantity": "median gap (days)", "this_run": float(gap.median()), "proposal": 31},
    {"quantity": "gap above 30 days", "this_run": int((gap > 30).sum()), "proposal": 196},
    {"quantity": "gap above 90 days", "this_run": int((gap > 90).sum()), "proposal": 88},
])
dates["difference"] = dates["this_run"] - dates["proposal"]
save_result_table(dates, "a05_date_gap_vs_proposal", config)
print(dates.to_string(index=False))

# %% Gestational age parsing (trimester sensitivity analysis input)
ga = pd.DataFrame([
    {"item": "rule", "value": "text with minggu or week is weeks, with bulan or month is months x 4.345, bare numbers are weeks"},
    {"item": "bare numbers", "value": rep["ga_bare_numbers"]},
    {"item": "bare numbers of 9 or less (ambiguous, could be months)", "value": rep["ga_bare_numbers_le_9"]},
    {"item": "unparsed", "value": rep["ga_unparsed"]},
    {"item": "Hb-model women without usable gestational age",
     "value": int(hb_rows["gestational_age_weeks"].isna().sum())},
])
save_result_table(ga, "a06_gestational_age_parsing", config)
print(ga.to_string(index=False))
