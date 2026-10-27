# %% [markdown]
# # 06 Exploratory comparisons (proposal 3.5, no verdicts)
# Each variant is retuned with the same trial budget, the same folds and the
# same per-fold selection as the main models. Only the named element changes.
# * Hb: XGBoost with SMOGN.
# * CED: XGBoost with SMOTE-NC, and XGBoost with class weighting (scale_pos_weight).
# * Anemia: XGBClassifier on the binary label (Hb below 11.0), compared in 08
#   with regression followed by the threshold.
# Split-conformal intervals and the reliability curve need no extra run. They
# come from the main runs and are reported by 08.
#
# Run from the project root:  python experiments/06_exploratory.py

# %%
from __future__ import annotations

from _common import start

from src.cleaning import clean_dataset
from src.models import CED_CLASS_WEIGHTED, CED_MODELS, DIRECT_ANEMIA, HB_MODELS
from src.nested_cv import build_data
from src.runs import main_run, run_and_save

config, _ = start("06_exploratory")
df = clean_dataset(config).frame
hb = build_data(config, df, "hb")
ced = build_data(config, df, "ced")
anemia = build_data(config, df, "anemia_binary")

# %% Variants
variants = [
    (main_run(HB_MODELS["xgboost"], config, variant="smogn", resampler="smogn"), hb),
    (main_run(CED_MODELS["xgboost"], config, variant="smotenc", resampler="smotenc"), ced),
    (main_run(CED_CLASS_WEIGHTED, config, variant="class_weighting"), ced),
    (main_run(DIRECT_ANEMIA, config, variant="direct"), anemia),
]
for run, data in variants:
    result = run_and_save(run, data, config)
    print(f"{data.target} {run.spec.name} {run.variant}: {len(result.folds)} outer folds done")
