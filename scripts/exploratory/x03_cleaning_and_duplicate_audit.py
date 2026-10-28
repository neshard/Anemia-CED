# %% [markdown]
# # x03 Cleaning flow and duplicate respondent codes (EXPLORATORY, POST HOC)
# Answers supervisor questions of 2026-10-28. Nothing here changes a model, a result
# or the cleaning rules. Only aggregate counts are written; no code, name or row is
# printed or saved.
#
# 1. The record removed for having no usable target: is it among the Hb = 0 records
#    and the invalid MUAC records? How many women are in both cohorts, and how many
#    of the Hb = 0 women are in the CED cohort?
# 2. The 39 repeated respondent codes: one woman entered twice, or one code used for
#    different women? Each repeated record is compared with the first record of its
#    code on the respondent's name (read from the raw export for this comparison only,
#    never stored) and on the analysed questionnaire fields.
#
# Run from the project root:  python scripts/exploratory/x03_cleaning_and_duplicate_audit.py

# %%
from __future__ import annotations

import difflib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cleaning import NUMERIC_TARGETS, build_labels, clean_features, load_raw, to_numeric_flexible  # noqa: E402
from src.utils import assert_no_identifiers, load_config, resolve_path  # noqa: E402

LABEL = "exploratory, post hoc"
config = load_config()
OUT = ROOT / "results" / "exploratory"
NAME_COLUMN = "Nama Lengkap"  # not in config columns: read here only to compare, never kept


def save(df: pd.DataFrame, name: str) -> None:
    assert_no_identifiers(df, config)
    df.insert(0, "label", LABEL)
    path = OUT / f"{name}_exploratory_post_hoc.csv"
    df.to_csv(path, index=False)
    print(f"written {path.relative_to(ROOT)}")


# %% 1. Cleaning flow: the record without any target and the overlap of the cohorts
frame, info = load_raw(config)
features, _ = clean_features(frame, config)
raw_muac = to_numeric_flexible(frame["muac"])
raw_hb = to_numeric_flexible(frame["hb"])
for col in NUMERIC_TARGETS:
    features[col] = to_numeric_flexible(features[col])
labelled, _ = build_labels(features, config)

no_target = labelled["hb_target"].isna() & labelled["ced"].isna() & labelled["anemia_class"].isna()
hb_zero = labelled["hb_zero_flag"].fillna(0).astype(bool)
lo, hi = config["cleaning"]["muac_valid_range"]
muac_invalid = raw_muac.notna() & ~raw_muac.between(lo, hi)
muac_missing = raw_muac.isna()
hb_above = raw_hb > float(config["cleaning"]["hb_max_valid"])
in_hb = labelled["hb_target"].notna() & labelled["anemia_class"].notna()
in_ced = labelled["ced"].notna()
kept = ~no_target

flow = pd.DataFrame([
    {"item": "records after removing exact duplicates", "n": int(len(labelled))},
    {"item": "records without any usable target (removed)", "n": int(no_target.sum())},
    {"item": "  of which Hb recorded as 0", "n": int((no_target & hb_zero).sum())},
    {"item": "  of which Hb above the valid maximum", "n": int((no_target & hb_above).sum())},
    {"item": "  of which MUAC outside the valid range", "n": int((no_target & muac_invalid).sum())},
    {"item": "  of which MUAC missing", "n": int((no_target & muac_missing).sum())},
    {"item": "records analysed", "n": int(kept.sum())},
    {"item": "Hb recorded as 0 (all records)", "n": int(hb_zero.sum())},
    {"item": "MUAC outside the valid range (all records)", "n": int(muac_invalid.sum())},
    {"item": "Hb above the valid maximum (all records)", "n": int(hb_above.sum())},
    {"item": "Hb cohort", "n": int(in_hb.sum())},
    {"item": "CED cohort", "n": int(in_ced.sum())},
    {"item": "in both cohorts", "n": int((in_hb & in_ced).sum())},
    {"item": "Hb cohort only", "n": int((in_hb & ~in_ced).sum())},
    {"item": "CED cohort only", "n": int((in_ced & ~in_hb).sum())},
    {"item": "Hb recorded as 0 and in the CED cohort", "n": int((hb_zero & in_ced).sum())},
    {"item": "Hb above the valid maximum and in the CED cohort", "n": int((hb_above & in_ced).sum())},
    {"item": "MUAC outside the valid range and in the Hb cohort", "n": int((muac_invalid & in_hb).sum())},
])
save(flow, "x23_cleaning_flow_and_cohort_overlap")
print(flow.to_string(index=False))

# %% 2. Repeated respondent codes
csv_path = resolve_path(config, "raw_data")
raw = pd.read_csv(csv_path, sep=config["data"]["csv_sep"], encoding=config["data"]["csv_encoding"], dtype=str,
                  keep_default_na=True)
cols = config["columns"]
code = raw[cols["respondent_code"]].fillna("").str.strip().str.upper()
name = raw[NAME_COLUMN].fillna("").map(lambda s: " ".join(str(s).lower().split()))
center = raw[cols["health_center"]].fillna("").str.strip()
analysed = [cols[k] for k in cols if k != "respondent_code"]
values = raw[analysed].fillna("").apply(lambda s: s.str.strip())

groups = code[code.ne("")].groupby(code).groups
repeated = {c: list(ix) for c, ix in groups.items() if len(ix) > 1}
sizes = pd.Series([len(ix) for ix in repeated.values()], dtype=int)
pairs = []
for ix in repeated.values():
    first = ix[0]
    for other in ix[1:]:
        diff = int((values.loc[first] != values.loc[other]).sum())
        similarity = difflib.SequenceMatcher(None, name[first], name[other]).ratio()
        if diff == 0:
            kind = "identical on every analysed field (same woman entered twice)"
        elif name[first] and name[first] == name[other]:
            kind = "same name, some answers differ (same woman entered twice with changes)"
        elif similarity >= 0.85:
            kind = "near-identical name (likely the same woman, spelling differs)"
        else:
            kind = "different name (code reused for a different woman)"
        pairs.append({"kind": kind, "differing_fields": diff, "same_health_center": center[first] == center[other]})
pairs = pd.DataFrame(pairs)

summary = [
    {"item": "records in the raw export", "value": int(len(raw))},
    {"item": "records with an empty respondent code", "value": int(code.eq("").sum())},
    {"item": "distinct respondent codes", "value": int(code[code.ne("")].nunique())},
    {"item": "codes used more than once", "value": int(len(repeated))},
    {"item": "records beyond the first per code (the 39 of the cleaning report)", "value": int(len(pairs))},
]
for size, n in sizes.value_counts().sort_index().items():
    summary.append({"item": f"codes used {size} times", "value": int(n)})
for kind, block in pairs.groupby("kind"):
    summary.append({"item": f"repeated records: {kind}", "value": int(len(block))})
    summary.append({"item": f"  same health center as the first record ({kind.split(' (')[0]})",
                    "value": int(block["same_health_center"].sum())})
    summary.append({"item": f"  median differing analysed fields ({kind.split(' (')[0]})",
                    "value": float(np.median(block["differing_fields"]))})
save(pd.DataFrame(summary), "x24_repeated_respondent_codes")
print(pd.DataFrame(summary).to_string(index=False))

# %% 3. Records sharing a full name across any code (the same woman under two codes?)
# Compared on the 21 fields of the x02 near-duplicate rules and on five attributes that
# do not change between two visits of the same pregnancy (age, height, education, number
# of children, health center). A pair equal on Hb, MUAC, pre-pregnancy weight and the
# interview date is counted as a probable repeat of the same woman.
F21 = [cols[k] for k in ("age", "height", "prepreg_weight", "anc_visits", "n_children", "n_under5", "education_mother",
                         "income", "egg_freq", "food_taboo", "hb", "muac")] + [cols[f"hfias_q{i}"] for i in range(1, 10)]
STABLE = [cols[k] for k in ("age", "height", "education_mother", "n_children", "health_center")]
SAME_VISIT = [cols[k] for k in ("hb", "muac", "prepreg_weight", "interview_date_raw")]
txt = raw.fillna("").apply(lambda s: s.str.strip())
name_pairs = []
for _, ix in txt[name.ne("")].groupby(name[name.ne("")]).groups.items():
    ix = list(ix)
    for i, a in enumerate(ix):
        for b in ix[i + 1:]:
            d21 = int(sum(txt.at[a, c] != txt.at[b, c] for c in F21))
            name_pairs.append({
                "same_code": code[a] == code[b],
                "differing_of_21": d21,
                "differing_stable": int(sum(txt.at[a, c] != txt.at[b, c] for c in STABLE)),
                "same_visit_values": all(txt.at[a, c] == txt.at[b, c] for c in SAME_VISIT),
                "x02_rule": "strict" if d21 == 0 else ("loose" if d21 <= 1 else "not caught"),
            })
np_ = pd.DataFrame(name_pairs)
probable = np_[np_["same_visit_values"]]
names_table = pd.DataFrame([
    {"item": "records sharing a full name with another record", "value": int(name[name.ne("")].duplicated(keep=False).sum())},
    {"item": "pairs of records with the same full name", "value": int(len(np_))},
    {"item": "  of which with the same respondent code", "value": int(np_["same_code"].sum())},
    {"item": "  median differing fields of the 21 x02 fields", "value": float(np_["differing_of_21"].median())},
    {"item": "  equal on all five stable attributes", "value": int((np_["differing_stable"] == 0).sum())},
    {"item": "probable repeats (same Hb, MUAC, weight and interview date)", "value": int(len(probable))},
    {"item": "  caught by the x02 strict rule", "value": int((probable["x02_rule"] == "strict").sum())},
    {"item": "  caught by the x02 loose rule", "value": int((probable["x02_rule"] == "loose").sum())},
    {"item": "  not caught by either rule (still in every analysis)", "value": int((probable["x02_rule"] == "not caught").sum())},
    {"item": "pairs with the same name but different visit values (different women or visits)",
     "value": int((~np_["same_visit_values"]).sum())},
])
save(names_table, "x25_records_sharing_a_name")
print(names_table.to_string(index=False))
