# %% [markdown]
# # 01 Exploratory data analysis
# Aggregate description of the cleaned data. No patient row is printed or saved.
#
# The protocol evaluates with nested repeated cross-validation on all eligible
# records and has no hold-out set (proposal 3.2), so EDA describes every
# analysed woman. Nothing here is used to build a model.
#
# Run from the project root:  python experiments/01_eda.py

# %%
from __future__ import annotations

from collections.abc import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import start  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.evaluation import save_figure  # noqa: E402
from src.labels import WHO_CLASSES  # noqa: E402
from src.cleaning import clean_dataset  # noqa: E402
from src.utils import resolve_path, save_table  # noqa: E402

config, _ = start("01_eda", require_hypotheses=False)
figures = resolve_path(config, "reports_figures")
df = clean_dataset(config).frame
print(f"Analysis sample: {len(df)} women. The protocol has no hold-out set (proposal 3.2), so EDA describes all of them.")

# Chart style: one hue for magnitude, recessive grid and axes, text in ink tokens.
BAR = "#3987e5"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
REF_LINE = "#52514e"
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4b6b0", "#e34948", "#a32b2b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10, "axes.titlelocation": "left",
})

# Readable labels for coded variables, taken from the same config maps used in cleaning.
enc = config["encodings"]


def decode(mapping: Mapping[str, int]) -> dict[float, str]:
    out: dict[float, str] = {}
    for text, code in mapping.items():
        out.setdefault(float(code), text.replace("?", ">="))
    return out


LEVEL_LABELS: dict[str, dict[float, str]] = {
    "education_mother": decode(enc["education"]),
    "income": decode(enc["income"]),
    "egg_freq": decode(enc["food_frequency"]),
    "food_taboo": {0.0: "No", 1.0: "Yes"},
    "hfias_category": {0.0: "Food secure", 1.0: "Mildly insecure", 2.0: "Moderately insecure", 3.0: "Severely insecure"},
    "ced": {0.0: "No CED risk", 1.0: "CED risk (MUAC < 23.5)"},
}
NUMERIC = ["age", "height", "prepreg_weight", "anc_visits", "n_children", "n_under5", "hfias_score",
           "gestational_age_weeks", "muac", "hb_target"]
CATEGORICAL = ["education_mother", "income", "egg_freq", "food_taboo", "hfias_category", "anemia_class", "ced",
               "health_center"]
NICE = {
    "age": "Age (years)", "height": "Height (cm)", "prepreg_weight": "Pre-pregnancy weight (kg)",
    "anc_visits": "ANC visits", "n_children": "Number of children", "n_under5": "Children under five",
    "hfias_score": "HFIAS score (0 to 27)", "gestational_age_weeks": "Gestational age (weeks)",
    "muac": "MUAC (cm)", "hb_target": "Hb (g/dL)", "education_mother": "Education (mother)",
    "income": "Monthly household income", "egg_freq": "Egg consumption", "food_taboo": "Food taboo in pregnancy",
    "hfias_category": "HFIAS category", "anemia_class": "WHO anemia class", "ced": "CED status",
    "health_center": "Health centre",
}

# %% e01 Table 1: cohort characteristics overall, by CED status and by anemia
groups: dict[str, pd.Series] = {
    "Overall": pd.Series(True, index=df.index),
    "No CED risk": df["ced"] == 0,
    "CED risk": df["ced"] == 1,
    "Hb >= 11": df["anemia_binary"] == 0,
    "Hb < 11": df["anemia_binary"] == 1,
}


def numeric_cell(s: pd.Series) -> tuple[str, str]:
    s = s.dropna()
    if s.empty:
        return "", ""
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{s.mean():.1f} ({s.std(ddof=1):.1f})", f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


rows = [{"variable": "n", "level": "", **{g: str(int(m.sum())) for g, m in groups.items()}}]
for col in NUMERIC:
    mean_sd = {g: numeric_cell(df.loc[m, col])[0] for g, m in groups.items()}
    med_iqr = {g: numeric_cell(df.loc[m, col])[1] for g, m in groups.items()}
    missing = {g: str(int(df.loc[m, col].isna().sum())) for g, m in groups.items()}
    rows.append({"variable": NICE[col], "level": "mean (SD)", **mean_sd})
    rows.append({"variable": "", "level": "median [IQR]", **med_iqr})
    rows.append({"variable": "", "level": "missing, n", **missing})
for col in CATEGORICAL:
    labels = LEVEL_LABELS.get(col)
    series = df[col].map(labels) if labels else df[col]
    order = list(labels.values()) if labels else (list(WHO_CLASSES) if col == "anemia_class" else sorted(series.dropna().unique()))
    first = True
    for level in [*order, "Missing"]:
        cells = {}
        for g, m in groups.items():
            sub = series[m]
            n = int(sub.isna().sum()) if level == "Missing" else int((sub == level).sum())
            cells[g] = f"{n} ({n / max(len(sub), 1):.1%})"
        if level == "Missing" and all(c.startswith("0 ") for c in cells.values()):
            continue
        rows.append({"variable": NICE[col] if first else "", "level": level, **cells})
        first = False
table1 = pd.DataFrame(rows)
save_table(table1, "e01_table1_characteristics", config)
print(table1.to_string(index=False))

# %% e02 Missingness after cleaning (implausible values were set to missing)
feature_cols = sorted(config["features"]["candidate_pool"])
miss = pd.DataFrame({
    "column": feature_cols + ["hb_target", "ced"],
    "n_missing": [int(df[c].isna().sum()) for c in feature_cols + ["hb_target", "ced"]],
})
miss["pct_missing"] = (100 * miss["n_missing"] / len(df)).round(2)
save_table(miss, "e02_missingness", config)
print(miss.to_string(index=False))

# %% e03 Target distributions
hb_counts = df["hb_target"].value_counts().sort_index().rename_axis("hb_g_dl").reset_index(name="n")
save_table(hb_counts, "e03a_hb_value_counts", config)
class_counts = pd.DataFrame({
    "target": ["anemia_class"] * len(WHO_CLASSES) + ["ced"] * 2,
    "level": [*WHO_CLASSES, "No CED risk", "CED risk"],
    "n": [int((df["anemia_class"] == c).sum()) for c in WHO_CLASSES] + [int((df["ced"] == 0).sum()), int((df["ced"] == 1).sum())],
})
save_table(class_counts, "e03b_target_class_counts", config)
print(class_counts.to_string(index=False))

# %% Figure 01: Hb distribution with WHO cut-offs


def discrete_or_hist(ax: plt.Axes, values: pd.Series, bins: int = 20) -> None:
    """Bars per value for discrete data, a histogram otherwise. Thin surface gap between bars."""
    v = values.dropna()
    if v.nunique() <= 15 and np.allclose(v % 1, 0):
        counts = v.value_counts().sort_index()
        ax.bar(counts.index, counts.to_numpy(), width=0.8, color=BAR, edgecolor=SURFACE, linewidth=1.5)
        ticks = counts.index if len(counts) <= 9 else counts.index[::2]
        ax.set_xticks(ticks)
    else:
        ax.hist(v, bins=bins, color=BAR, edgecolor=SURFACE, linewidth=1.5)


fig, ax = plt.subplots(figsize=(6.4, 3.4))
discrete_or_hist(ax, df["hb_target"])
# Whole-number Hb: a cut-off of 11 separates the bars at 10 and 11, so the line sits at 10.5.
shift = 0.5 if np.allclose(df["hb_target"].dropna() % 1, 0) else 0.0
top = ax.get_ylim()[1]
ax.set_ylim(0, top * 1.25)
# Each label gets its own spot: (cut-off, text, height as share of the top, side of the line).
for cut, label, height, side in ((7.0, "Severe: below 7", 1.20, "right"), (10.0, "Moderate: below 10", 1.11, "left"),
                                 (11.0, "Anemia: below 11", 1.20, "right")):
    ax.axvline(cut - shift, color=REF_LINE, lw=1, ls="--")
    ax.text(cut - shift + (0.08 if side == "right" else -0.08), top * height, label, color=INK_2, fontsize=8,
            va="top", ha="left" if side == "right" else "right")
ax.set_xlabel("Measured Hb (g/dL)")
ax.set_ylabel("Women")
ax.set_title(f"Hb distribution (n={int(df['hb_target'].notna().sum())}, whole numbers)"
             if np.allclose(df["hb_target"].dropna() % 1, 0) else f"Hb distribution (n={int(df['hb_target'].notna().sum())})")
save_figure(fig, "e_fig01_hb_distribution", figures)

# %% Figure 02: MUAC distribution with the CED cut-off
fig, ax = plt.subplots(figsize=(6.4, 3.4))
discrete_or_hist(ax, df["muac"], bins=24)
ax.set_xticks(sorted(df["muac"].dropna().unique())[::2]) if df["muac"].nunique() > 15 else None
ax.axvline(23.5, color=REF_LINE, lw=1, ls="--")
ax.text(23.4, ax.get_ylim()[1] * 0.95, "CED risk below 23.5", color=INK_2, fontsize=8, va="top", ha="right")
ax.set_xlabel("MUAC (cm)")
ax.set_ylabel("Women")
ax.set_title(f"MUAC distribution (n={int(df['muac'].notna().sum())})")
save_figure(fig, "e_fig02_muac_distribution", figures)

# %% Figure 03: questionnaire feature distributions (small multiples)
fig, axes = plt.subplots(4, 3, figsize=(9, 9.6))
for ax, col in zip(axes.ravel(), feature_cols, strict=False):
    values = df[col]
    labels = LEVEL_LABELS.get(col)
    if labels:
        counts = values.value_counts().reindex(sorted(labels), fill_value=0)
        ax.bar(range(len(counts)), counts.to_numpy(), width=0.8, color=BAR, edgecolor=SURFACE, linewidth=1.5)
        ax.set_xticks(range(len(counts)), [labels[k] for k in counts.index], rotation=30, ha="right", fontsize=7)
    else:
        discrete_or_hist(ax, values)
    ax.set_title(f"{NICE[col]}  (missing {int(values.isna().sum())})", fontsize=8.5)
for ax in axes.ravel()[len(feature_cols):]:
    ax.set_visible(False)
fig.suptitle("Questionnaire features", x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout()
save_figure(fig, "e_fig03_feature_distributions", figures)

# %% e04 and Figure 04: Spearman correlation among the questionnaire features (no targets)
corr = df[feature_cols].corr(method="spearman")
save_table(corr.round(3).reset_index().rename(columns={"index": "feature"}), "e04_feature_correlation_spearman", config)
cmap = LinearSegmentedColormap.from_list("div_blue_red", DIVERGING)
fig, ax = plt.subplots(figsize=(6.6, 5.6))
ax.grid(False)
ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        r = corr.iat[i, j]
        ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if abs(r) > 0.6 else INK)
ticks = [NICE[c] for c in feature_cols]
ax.set_xticks(range(len(ticks)), ticks, rotation=40, ha="right")
ax.set_yticks(range(len(ticks)), ticks)
ax.set_title("Spearman correlation between questionnaire features (blue negative, red positive)")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
save_figure(fig, "e_fig04_feature_correlation", figures)
high = [(a, b, round(corr.loc[a, b], 2)) for i, a in enumerate(feature_cols) for b in feature_cols[i + 1:]
        if abs(corr.loc[a, b]) >= 0.5]
print("Feature pairs with |rho| >= 0.5:", high or "none")

# %% e05 and Figure 05: feature to target association, descriptive
train = df
assoc_rows = []
for target, label in (("hb_target", "Hb (g/dL)"), ("ced", "CED risk")):
    for col in feature_cols:
        pair = train[[col, target]].dropna()
        rho = pair[col].corr(pair[target], method="spearman") if len(pair) > 2 else np.nan
        assoc_rows.append({"target": label, "feature": NICE[col], "spearman_rho": round(float(rho), 3), "n": len(pair),
                           "note": "descriptive"})
assoc = pd.DataFrame(assoc_rows)
save_table(assoc, "e05_feature_target_spearman_training_only", config)
print(assoc.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, label in zip(axes, ("Hb (g/dL)", "CED risk"), strict=True):
    sub = assoc[assoc["target"] == label].set_index("feature").reindex([NICE[c] for c in feature_cols])
    ax.barh(range(len(sub)), sub["spearman_rho"], height=0.7, color=BAR, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(sub)), sub.index)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.4, 0.4)
    ax.set_xlabel("Spearman rho")
    ax.set_title(label)
fig.suptitle("Association of each feature with the target, all analysed women", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
save_figure(fig, "e_fig05_feature_target_association_training", figures)
print(f"Tables in {resolve_path(config, 'reports_tables')}, figures in {figures}")
