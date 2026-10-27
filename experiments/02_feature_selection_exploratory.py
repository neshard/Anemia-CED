# %% [markdown]
# # 02 Feature ranking (EXPLORATORY)
# One MI plus RF ranking of the 11 candidates on all eligible records of each
# model, labelled EXPLORATORY on every row. It is never used to build a model.
# The per-fold selection and its frequency over the 25 outer folds come from
# the nested runs (scripts 04 and 05) and are reported by script 08.
#
# Run from the project root:  python experiments/02_feature_selection_exploratory.py

# %%
from __future__ import annotations

from _common import start

from src.cleaning import clean_dataset
from src.feature_selection import exploratory_ranking
from src.nested_cv import build_data
from src.preprocessing import build_imputer
from src.runs import save_result_table

config, _ = start("02_feature_selection_exploratory", require_hypotheses=False)
fs = config["feature_selection"]
seed = int(config["project"]["random_seed"])
df = clean_dataset(config).frame

# %% Exploratory ranking per model
for target, name in (("hb", "b01_exploratory_ranking_hb"), ("ced", "b02_exploratory_ranking_ced")):
    data = build_data(config, df, target)
    table = exploratory_ranking(data.X, data.y, build_imputer(config), data.task, int(fs["rf_n_estimators"]),
                                fs["discrete_features"], seed)
    table["forced_in_protocol"] = table["feature"].isin(fs["forced_include"])
    save_result_table(table, name, config)
    print(f"\n{target} EXPLORATORY ranking (n={len(data.y)}, never used to build a model)")
    print(table[["feature", "mi", "rf_importance", "mi_rank", "rf_rank", "mean_rank", "forced_in_protocol"]]
          .round(4).to_string(index=False))
