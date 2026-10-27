"""SHAP explanations for fitted pipelines and the Indonesian narrative for a health cadre.

SHAP values describe how a model reached its output for a given input. They do
not show what causes anemia or CED, and a model that learned little gives
explanations that carry little meaning (proposal 3.6). The narrative states
both points explicitly and never uses causal wording.

TreeExplainer runs on the fitted tree model inside the pipeline (XGBoost, GBM,
LightGBM or CatBoost). The matrix it explains is produced by the same fitted
preprocessing steps (imputation, selection, encoding) as at prediction time,
with the resampler excluded because it never runs at prediction time.
There is no fallback to KernelExplainer: a model TreeExplainer cannot handle
(AdaBoost in shap 0.46) raises ``ExplainerNotSupportedError``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src.preprocessing import preprocessing_only  # noqa: E402

NON_CAUSAL_SENTENCE = (
    "Penjelasan ini menunjukkan bagaimana model sampai pada perkiraannya, bukan penyebab kondisi ibu. "
    "Faktor yang disebut tidak berarti menyebabkan risiko tersebut."
)
DISCLAIMER = (
    "Hasil ini adalah perkiraan untuk membantu skrining awal, bukan diagnosis. "
    "Hasil ini tidak menggantikan pengukuran LILA atau pemeriksaan Hb di fasilitas kesehatan."
)


ALGORITHM_LABELS: dict[str, str] = {
    "xgboost": "XGBoost", "gbm": "Gradient Boosting Machine", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "adaboost": "AdaBoost",
}


class ExplainerNotSupportedError(TypeError):
    """Raised when shap.TreeExplainer cannot explain the model. No other explainer is substituted."""


def model_matrix(pipe: Any, X: pd.DataFrame) -> pd.DataFrame:
    """The exact matrix the fitted model receives for ``X``."""
    return preprocessing_only(pipe).transform(X)


def tree_explainer(pipe: Any) -> shap.TreeExplainer:
    model = pipe.named_steps["model"]
    try:
        return shap.TreeExplainer(model)
    except Exception as exc:  # shap raises InvalidModelError (an Exception subclass) for unsupported models
        raise ExplainerNotSupportedError(f"shap.TreeExplainer does not support {type(model).__name__}: {exc}") from exc


def explainer_supported(pipe: Any) -> bool:
    try:
        tree_explainer(pipe)
    except ExplainerNotSupportedError:
        return False
    return True


def shap_table(pipe: Any, X: pd.DataFrame, explainer: shap.TreeExplainer | None = None) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """SHAP values (raw model output: g/dL for Hb, log-odds for CED), the base value and the explained matrix."""
    explainer = explainer or tree_explainer(pipe)
    Xt = model_matrix(pipe, X)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
        raw = explainer.shap_values(Xt)
    if isinstance(raw, list):  # one array per class: keep the positive class
        raw = raw[-1]
    values = np.asarray(raw)
    if values.ndim == 3:  # (rows, features, classes) in some shap versions
        values = values[..., -1]
    base = float(np.ravel(explainer.expected_value)[-1])
    return pd.DataFrame(values, columns=Xt.columns, index=X.index), base, Xt


def model_margin(pipe: Any, X: pd.DataFrame) -> np.ndarray:
    """Raw model output that SHAP values add up to (prediction for Hb, log-odds for CED)."""
    model = pipe.named_steps["model"]
    Xt = model_matrix(pipe, X)
    if not hasattr(model, "predict_proba"):
        return np.asarray(model.predict(Xt), dtype=float)
    name = type(model).__name__
    if name.startswith("XGB"):
        out = model.predict(Xt, output_margin=True)
    elif name.startswith("LGBM"):
        out = model.predict(Xt, raw_score=True)
    elif name.startswith("CatBoost"):
        out = model.predict(Xt, prediction_type="RawFormulaVal")
    elif hasattr(model, "decision_function"):  # scikit-learn GradientBoostingClassifier
        out = model.decision_function(Xt)
    else:
        raise ExplainerNotSupportedError(f"No raw margin for {name}")
    return np.asarray(out, dtype=float)


def additivity_error(pipe: Any, X: pd.DataFrame, explainer: shap.TreeExplainer | None = None) -> float:
    """Largest absolute gap between SHAP sum plus base value and the model output."""
    values, base, _ = shap_table(pipe, X, explainer)
    return float(np.max(np.abs(values.sum(axis=1).to_numpy() + base - model_margin(pipe, X))))


def builtin_importance(pipe: Any) -> pd.DataFrame:
    """Built-in importance for the features the model uses.

    XGBoost: gain, cover and weight. Other libraries: their ``feature_importances_``
    (impurity decrease for GBM, split counts for LightGBM, PredictionValuesChange for CatBoost).
    """
    model = pipe.named_steps["model"]
    names = list(model_matrix_columns(pipe))
    if type(model).__name__.startswith("XGB"):
        booster = model.get_booster()
        rows = {kind: booster.get_score(importance_type=kind) for kind in ("gain", "cover", "weight")}
        return pd.DataFrame({kind: [rows[kind].get(n, 0.0) for n in names] for kind in rows}, index=names)
    return pd.DataFrame({"builtin": np.asarray(model.feature_importances_, dtype=float)}, index=names)


def model_matrix_columns(pipe: Any) -> list[str]:
    """Columns of the matrix the fitted model was trained on."""
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(model.feature_names_)  # CatBoost


def _decode(feature: str, value: float, config: Mapping[str, Any]) -> str:
    """Readable value for the narrative, using the questionnaire answer text where one exists."""
    if pd.isna(value):
        return "tidak diisi"
    enc = config["encodings"]
    maps = {"education_mother": enc["education"], "income": enc["income"], "egg_freq": enc["food_frequency"]}
    if feature in maps:
        inverse = {}
        for text, code in maps[feature].items():
            inverse.setdefault(float(code), text.replace("?", ">="))
        return inverse.get(float(value), f"{value:g}")
    if feature == "food_taboo":
        return "ada" if value >= 0.5 else "tidak ada"
    return f"{value:g}".replace(".", ",")


def top_contributions(values: pd.Series, row_model_matrix: pd.Series, row_input: pd.Series, k: int,
                      explain_cfg: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The k largest absolute contributions with labels and readable values."""
    labels = explain_cfg["feature_labels_id"]
    units = explain_cfg.get("feature_units_id", {})
    order = values.abs().sort_values(ascending=False).index[:k]
    out = []
    for feature in order:
        raw_value = row_input.get(feature, row_model_matrix.get(feature))
        text = _decode(feature, raw_value, config)
        unit = units.get(feature)
        out.append({"feature": feature, "label": labels.get(feature, feature),
                    "value_text": f"{text} {unit}" if unit and text != "tidak diisi" else text,
                    "shap": float(values[feature])})
    return out


def _fmt_pct(p: float) -> str:
    return f"{100 * p:.0f}%"


def _fmt_num(x: float, digits: int = 1) -> str:
    """Indonesian decimal comma, for example 11,6."""
    return f"{x:.{digits}f}".replace(".", ",")


def model_sentence(algorithm: str | None) -> str:
    """Which algorithm produced the estimate, shown with every narrative."""
    return f"Perkiraan dibuat dengan model {ALGORITHM_LABELS.get(algorithm, algorithm)}." if algorithm else ""


def narrative_ced(probability: float, threshold: float, contributions: Sequence[Mapping[str, Any]],
                  algorithm: str | None = None) -> str:
    """Plain Indonesian narrative for the CED risk estimate. No causal wording."""
    if probability >= threshold:
        status = (f"Angka ini di atas ambang skrining {_fmt_pct(threshold)}, sehingga ibu ini dianjurkan "
                  "untuk diukur LILA-nya oleh petugas.")
    else:
        status = f"Angka ini di bawah ambang skrining {_fmt_pct(threshold)}."
    up = [c for c in contributions if c["shap"] > 0]
    down = [c for c in contributions if c["shap"] < 0]
    parts = [f"Perkiraan peluang risiko KEK untuk ibu ini adalah {_fmt_pct(probability)}. {status}"]
    if up:
        parts.append("Faktor yang paling menaikkan perkiraan model: "
                     + ", ".join(f"{c['label']} ({c['value_text']})" for c in up) + ".")
    if down:
        parts.append("Faktor yang paling menurunkan perkiraan model: "
                     + ", ".join(f"{c['label']} ({c['value_text']})" for c in down) + ".")
    parts += [p for p in (model_sentence(algorithm),) if p] + [NON_CAUSAL_SENTENCE, DISCLAIMER]
    return " ".join(parts)


def narrative_hb(prediction: float, half_width: float, explanation_enabled: bool, show_range: bool = True,
                 algorithm: str | None = None) -> str:
    """Hb narrative. When the Hb model did not beat the mean-only baseline, no factors are named.

    With ``show_range`` false (researcher decision of 2026-10-21) only the point estimate is
    shown, because the interval under-covered anemic women.
    """
    if show_range:
        text = (f"Perkiraan Hb: {_fmt_num(prediction)} g/dL (rentang {_fmt_num(prediction - half_width)} sampai "
                f"{_fmt_num(prediction + half_width)} g/dL).")
    else:
        text = f"Perkiraan Hb (indikatif): {_fmt_num(prediction)} g/dL."
    if not explanation_enabled:
        text += (" Perkiraan ini hanya indikatif. Dalam evaluasi, model Hb tidak terbukti lebih akurat daripada "
                 "menebak nilai rata-rata dan tidak dapat diandalkan untuk mendeteksi anemia, sehingga "
                 "penjelasan faktor tidak ditampilkan.")
    if algorithm:
        text += " " + model_sentence(algorithm)
    return text + " " + DISCLAIMER


def waterfall_figure(values: pd.Series, base: float, row_model_matrix: pd.Series, explain_cfg: Mapping[str, Any],
                     title: str) -> Any:
    """SHAP waterfall for one input. Returns the matplotlib figure."""
    labels = explain_cfg["feature_labels_id"]
    explanation = shap.Explanation(values=values.to_numpy(), base_values=base,
                                   data=row_model_matrix[values.index].to_numpy(),
                                   feature_names=[labels.get(f, f) for f in values.index])
    plt.close("all")
    shap.plots.waterfall(explanation, max_display=len(values), show=False)
    fig = plt.gcf()
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.set_size_inches(7.5, 4.5)
    fig.tight_layout()
    return fig


def explain_ced_row(pipe: Any, row: pd.DataFrame, threshold: float, explain_cfg: Mapping[str, Any],
                    config: Mapping[str, Any], explainer: shap.TreeExplainer | None = None,
                    algorithm: str | None = None) -> tuple[Any, str, float]:
    """Waterfall figure, narrative and probability for one CED input row."""
    values, base, Xt = shap_table(pipe, row, explainer)
    probability = float(pipe.predict_proba(row)[0, 1])
    contrib = top_contributions(values.iloc[0], Xt.iloc[0], row.iloc[0], int(explain_cfg["shap"]["top_k_narrative"]),
                                explain_cfg, config)
    fig = waterfall_figure(values.iloc[0], base, Xt.iloc[0], explain_cfg,
                           "Kontribusi tiap faktor terhadap perkiraan model (skala log-odds), bukan penyebab")
    return fig, narrative_ced(probability, threshold, contrib, algorithm), probability


def explain_hb_row(pipe: Any, row: pd.DataFrame, half_width: float, explain_cfg: Mapping[str, Any],
                   enabled: bool | None = None, algorithm: str | None = None) -> tuple[Any, str, float]:
    """Narrative and prediction for one Hb input row. No figure when the explanation is disabled."""
    prediction = float(pipe.predict(row)[0])
    # ``enabled`` comes from the model selection gate (artifacts/final_models.json) when given.
    enabled = bool(explain_cfg["hb_explanation_enabled"]) if enabled is None else bool(enabled)
    fig = None
    if enabled:
        values, base, Xt = shap_table(pipe, row)
        fig = waterfall_figure(values.iloc[0], base, Xt.iloc[0], explain_cfg,
                               "Kontribusi tiap faktor terhadap perkiraan Hb (g/dL), bukan penyebab")
    show_range = bool(explain_cfg.get("hb_show_range", True))
    return fig, narrative_hb(prediction, half_width, enabled, show_range, algorithm), prediction
