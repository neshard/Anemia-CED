# %% [markdown]
# # 07 Sensitivity analyses (proposal 3.8 item 3, no verdicts)
# Each analysis reruns the main protocol (XGBoost and its confirmatory
# baselines) with one change:
# 1. CED cut-offs: at risk when recorded MUAC is 22 or less, and 24 or less.
#    Labels and the stratification key are rebuilt for each.
# 2. Trimester-specific Hb cut-offs (WHO 2024) on women with a usable gestational age.
# 3. Date gaps: Hb records whose date differs from the interview date by more
#    than 90 days, and separately by more than 30 days, are excluded. Records
#    without a usable date are excluded from both.
# 4. Cross-validation grouped by subdistrict (GroupKFold, one pass).
# 5. All 11 candidate features without selection, and the fixed preliminary set
#    (optimistic, because it came from an earlier analysis on the same women).
# 6. Three-class anemia analysis: computed by 08 from the main Hb predictions.
#
# Run from the project root:  python experiments/07_sensitivity.py

# %%
from __future__ import annotations

from _common import start

from src.cleaning import clean_dataset
from src.models import CED_MODELS, HB_MODELS
from src.nested_cv import build_data
from src.runs import main_run, run_and_save

config, _ = start("07_sensitivity")
df = clean_dataset(config).frame
HB_SET = [HB_MODELS[m] for m in ("xgboost", "dummy_mean", "ridge")]
CED_SET = [CED_MODELS[m] for m in ("xgboost", "dummy_prior", "logistic_regression")]

analyses = [
    (build_data(config, df, "ced", analysis="sens_ced_le22", ced_cutoff_cm=22), CED_SET, {}),
    (build_data(config, df, "ced", analysis="sens_ced_le24", ced_cutoff_cm=24), CED_SET, {}),
    (build_data(config, df, "hb", analysis="sens_trimester", trimester=True), HB_SET, {}),
    (build_data(config, df, "hb", analysis="sens_gap_le90", max_gap_days=90), HB_SET, {}),
    (build_data(config, df, "hb", analysis="sens_gap_le30", max_gap_days=30), HB_SET, {}),
    (build_data(config, df, "hb", analysis="sens_grouped"), HB_SET, {"outer": "group", "n_repeats": 1}),
    (build_data(config, df, "ced", analysis="sens_grouped"), CED_SET, {"outer": "group", "n_repeats": 1}),
    (build_data(config, df, "hb", analysis="sens_all11"), HB_SET, {"feature_mode": "all_candidates"}),
    (build_data(config, df, "ced", analysis="sens_all11"), CED_SET, {"feature_mode": "all_candidates"}),
    (build_data(config, df, "hb", analysis="sens_fixed_preliminary"), HB_SET, {"feature_mode": "fixed_preliminary"}),
    (build_data(config, df, "ced", analysis="sens_fixed_preliminary"), CED_SET, {"feature_mode": "fixed_preliminary"}),
]

# %% Run every analysis
for data, specs, overrides in analyses:
    print(f"[{data.analysis}] {data.target}: n={len(data.y)}, meta={ {k: v for k, v in data.meta.items() if k != 'cell_counts'} }")
    for spec in specs:
        run_and_save(main_run(spec, config, **overrides), data, config)
