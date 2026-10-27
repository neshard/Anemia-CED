"""Dashboard logic layer: validation, preprocessing replication and prediction.

The questionnaire answers go through ``src.cleaning.clean_features``, the same
deterministic rules used for training (category codes, HFIAS scoring,
plausibility limits). Prediction then uses the exact fitted pipeline objects
saved by experiments/09_final_models.py, so imputation, column selection and
encoding are never re-implemented by hand.

Which models are loaded is read from config/dashboard.yaml and the metadata file
it names (artifacts/final_models.json), never hard-coded. A model is refused
unless it has a valid shap.TreeExplainer and passes the SHAP additivity check,
both as recorded by the training script and again at load time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from src.cleaning import clean_features
from src.hfias import items_combined
from src.labels import who_anemia_class
from src.postmodel.explain import (
    ALGORITHM_LABELS,
    ExplainerNotSupportedError,
    additivity_error,
    explain_ced_row,
    explain_hb_row,
    shap_table,
    top_contributions,
    tree_explainer,
)
from src.postmodel.recommendations import Advice, build_advice
from src.utils import PROJECT_ROOT, load_config

# Every questionnaire field the dashboard can ask. The form shows only those the two
# loaded models use (the union of their feature lists), plus the nine HFIAS items.
ALL_NUMERIC_FIELDS: dict[str, str] = {
    "age": "Umur ibu (tahun)",
    "prepreg_weight": "Berat badan sebelum hamil (kg)",
    "height": "Tinggi badan (cm)",
    "anc_visits": "Jumlah kunjungan pemeriksaan kehamilan (ANC) sejauh ini",
    "n_children": "Jumlah anak",
    "n_under5": "Jumlah anak balita (<5 tahun)",
}
ALL_CHOICE_FIELDS: dict[str, str] = {
    "education_mother": "Pendidikan terakhir ibu",
    "egg_freq": "Seberapa sering ibu makan telur ayam dalam 1 bulan terakhir",
    "income": "Kisaran pendapatan bulanan keluarga",
    "food_taboo": "Apakah ibu memiliki pantangan makanan khusus selama hamil?",
}
INTEGER_FIELDS = ("anc_visits", "n_children", "n_under5", "age")
HFIAS_FIELDS = [f"hfias_q{i}" for i in range(1, 10)]


class ModelNotEligibleError(RuntimeError):
    """Raised when a saved model has no valid TreeExplainer or fails the SHAP additivity check."""


@dataclass
class ScreeningResult:
    """Everything the presentation layer shows and the data layer stores."""

    hb_prediction: float
    hb_class: str
    hb_text: str
    ced_probability: float
    ced_label: int
    ced_text: str
    ced_figure: Any
    ced_top: list[dict[str, Any]]
    advice: Advice
    extra: dict[str, Any] = field(default_factory=dict)
    hb_figure: Any = None


def model_features_to_fields(features: list[str]) -> list[str]:
    """Questionnaire fields behind a feature list: the HFIAS score is asked as its nine items."""
    out: list[str] = []
    for f in features:
        out += HFIAS_FIELDS if f == "hfias_score" else [f]
    return list(dict.fromkeys(out))


class ScreeningService:
    """Loads the saved models once and turns questionnaire answers into a screening result."""

    def __init__(self, root: Path = PROJECT_ROOT, config: Mapping[str, Any] | None = None) -> None:
        self.root = Path(root)
        self.config = dict(config) if config is not None else load_config()
        dash_path = self.root / "config" / "dashboard.yaml"
        self.dashboard_cfg = (yaml.safe_load(dash_path.read_text(encoding="utf-8")) if dash_path.exists()
                              else {"models_file": "artifacts/final_models.json", "default_algorithm": "xgboost",
                                    "additivity_tolerance": 1e-6})
        self.meta = json.loads((self.root / self.dashboard_cfg["models_file"]).read_text(encoding="utf-8"))
        default = self.dashboard_cfg["default_algorithm"]
        self.algorithms = {t: self.meta[t].get("algorithm", default) for t in ("hb", "ced")}
        self.hb_pipe = joblib.load(self.root / self.meta["hb"].get("artifact", f"artifacts/final_hb_{self.algorithms['hb']}.joblib"))
        self.ced_pipe = joblib.load(self.root / self.meta["ced"].get("artifact", f"artifacts/final_ced_{self.algorithms['ced']}.joblib"))
        explainer_path = self.root / self.meta["ced"].get("explainer", "artifacts/explainer_ced.joblib")
        self.ced_explainer = joblib.load(explainer_path) if explainer_path.exists() else None
        self.explain_cfg = yaml.safe_load((self.root / "config" / "explain.yaml").read_text(encoding="utf-8"))
        self.rec_cfg = yaml.safe_load((self.root / "config" / "recommendations.yaml").read_text(encoding="utf-8"))
        self.hb_explanation_enabled = bool(self.meta["hb"].get("hb_explanation_enabled",
                                                               self.explain_cfg["hb_explanation_enabled"]))
        fields = model_features_to_fields(list(self.meta["hb"]["features"]) + list(self.meta["ced"]["features"]))
        self.numeric_fields = {f: label for f, label in ALL_NUMERIC_FIELDS.items() if f in fields}
        self.choice_fields = {f: label for f, label in ALL_CHOICE_FIELDS.items() if f in fields}
        unknown = [f for f in fields if f not in {**ALL_NUMERIC_FIELDS, **ALL_CHOICE_FIELDS} and f not in HFIAS_FIELDS]
        if unknown:
            raise KeyError(f"No questionnaire field for model features {unknown}")
        self._check_eligible()

    # -- eligibility --------------------------------------------------------
    def _check_eligible(self) -> None:
        """Refuse models without a TreeExplainer or with a failed additivity check (recorded or now)."""
        tol = float(self.dashboard_cfg["additivity_tolerance"])
        X = self.to_frame(median_answers(self))
        for target, pipe, explainer in (("hb", self.hb_pipe, None), ("ced", self.ced_pipe, self.ced_explainer)):
            name = self.algorithms[target]
            if self.meta[target].get("shap_additivity_passed") is False:
                raise ModelNotEligibleError(f"{target} model ({name}) failed the SHAP additivity check at training")
            try:
                explainer = explainer or tree_explainer(pipe)
                err = additivity_error(pipe, X, explainer)
            except ExplainerNotSupportedError as exc:
                raise ModelNotEligibleError(f"{target} model ({name}) has no valid TreeExplainer: {exc}") from exc
            if not err <= tol:
                raise ModelNotEligibleError(f"{target} model ({name}): SHAP additivity error {err:.2e} above {tol:g}")

    def algorithm_label(self, target: str) -> str:
        return ALGORITHM_LABELS.get(self.algorithms[target], self.algorithms[target])

    # -- form definition ---------------------------------------------------
    def choices(self, field_name: str) -> list[str]:
        enc = self.config["encodings"]
        if field_name == "education_mother":
            return list(enc["education"])
        if field_name == "income":
            return list(enc["income"])
        if field_name == "food_taboo":
            return list(enc["yes_no"])
        if field_name == "egg_freq":
            seen, out = set(), []
            for text, code in enc["food_frequency"].items():
                if code not in seen and not text.startswith("?") and not text.startswith("≥"):
                    seen.add(code)
                    out.append(text)
            return out
        if field_name in HFIAS_FIELDS:
            return list(self.config["hfias"]["combined_mapping"])
        raise KeyError(field_name)

    def hfias_questions(self) -> dict[str, str]:
        cols = self.config["columns"]
        return {f: cols[f] for f in HFIAS_FIELDS}

    def numeric_range(self, field_name: str) -> tuple[float, float]:
        low, high = self.config["cleaning"]["plausible_ranges"][field_name]
        return float(low), float(high)

    # -- validation ---------------------------------------------------------
    def validate(self, answers: Mapping[str, Any]) -> list[str]:
        """Indonesian error messages. An empty list means the answers can be used."""
        errors: list[str] = []
        for f, label in self.numeric_fields.items():
            raw = answers.get(f)
            try:
                value = float(str(raw).replace(",", "."))
            except (TypeError, ValueError):
                errors.append(f"{label}: wajib diisi dengan angka.")
                continue
            low, high = self.numeric_range(f)
            if not low <= value <= high:
                errors.append(f"{label}: nilai {value:g} di luar batas wajar ({low:g} sampai {high:g}).")
            if f in INTEGER_FIELDS and value != int(value):
                errors.append(f"{label}: harus bilangan bulat.")
        for f, label in self.choice_fields.items():
            if answers.get(f) not in self.choices(f):
                errors.append(f"{label}: pilih salah satu jawaban yang tersedia.")
        for i, f in enumerate(HFIAS_FIELDS, start=1):
            if answers.get(f) not in self.choices(f):
                errors.append(f"Pertanyaan kerawanan pangan nomor {i}: pilih salah satu jawaban.")
        return errors

    # -- prediction -----------------------------------------------------------
    def to_frame(self, answers: Mapping[str, Any]) -> pd.DataFrame:
        """One-row canonical frame cleaned by the training rules.

        The HFIAS score uses the same item scoring as training (``items_combined`` with the
        configured answer map, summed over the nine items). Only the dataset-level format
        check of ``score_hfias`` is skipped: it asks whether a whole export looks binary,
        which a single woman answering "Tidak" to every item would wrongly trigger.
        """
        row = {f: str(answers[f]).replace(",", ".") for f in self.numeric_fields}
        row.update({f: answers[f] for f in self.choice_fields})
        frame, _ = clean_features(pd.DataFrame([row], index=["input"]), self.config)
        hf = self.config["hfias"]
        items = items_combined(pd.DataFrame([{f: answers[f] for f in HFIAS_FIELDS}], index=["input"]),
                               hf["items"], hf["combined_mapping"])
        frame["hfias_score"] = items.sum(axis=1, min_count=len(hf["items"])).astype("float64")
        return frame

    def predict(self, answers: Mapping[str, Any]) -> ScreeningResult:
        errors = self.validate(answers)
        if errors:
            raise ValueError("\n".join(errors))
        X = self.to_frame(answers)
        threshold = float(self.meta["ced"]["threshold_youden"])
        fig, ced_text, prob = explain_ced_row(self.ced_pipe, X, threshold, self.explain_cfg, self.config,
                                              explainer=self.ced_explainer, algorithm=self.algorithms["ced"])
        values, _, Xt = shap_table(self.ced_pipe, X, self.ced_explainer)
        top = top_contributions(values.iloc[0], Xt.iloc[0], X.iloc[0], 5, self.explain_cfg, self.config)
        hb_fig, hb_text, hb_pred = explain_hb_row(self.hb_pipe, X, float(self.meta["hb"]["interval_half_width"]),
                                                  self.explain_cfg, enabled=self.hb_explanation_enabled,
                                                  algorithm=self.algorithms["hb"])
        k = int(self.explain_cfg["shap"]["top_k_narrative"])
        advice = build_advice(prob, threshold, top[:k], self.rec_cfg)
        return ScreeningResult(
            hb_prediction=hb_pred, hb_class=str(who_anemia_class([hb_pred])[0]), hb_text=hb_text,
            ced_probability=prob, ced_label=int(prob >= threshold), ced_text=ced_text, ced_figure=fig,
            ced_top=top, advice=advice, hb_figure=hb_fig,
            extra={"threshold": threshold, "hb_algorithm": self.algorithms["hb"],
                   "ced_algorithm": self.algorithms["ced"], "hb_explanation_enabled": self.hb_explanation_enabled},
        )

    def storage_records(self, answers: Mapping[str, Any], result: ScreeningResult) -> tuple[dict[str, Any], dict[str, Any]]:
        """Inputs and prediction as stored in SQLite. No Hb range is stored (option C)."""
        inputs = {f: float(str(answers[f]).replace(",", ".")) for f in self.numeric_fields}
        inputs.update({f: answers[f] for f in (*self.choice_fields, *HFIAS_FIELDS)})
        hb_note = ("Hb explanation shown (selection gate supported)." if self.hb_explanation_enabled
                   else "Hb indicative only, explanation hidden (selection gate not supported).")
        prediction = {
            "predicted_hb": round(result.hb_prediction, 2), "hb_lower": None, "hb_upper": None,
            "anemia_class": result.hb_class, "ced_probability": round(result.ced_probability, 4),
            "ced_label": result.ced_label, "top5_shap_hb": None,
            "top5_shap_ced": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in result.ced_top],
            "model_note": f"{hb_note} CED uncalibrated probability.",
            "hb_algorithm": self.algorithms["hb"], "ced_algorithm": self.algorithms["ced"],
        }
        return inputs, prediction


def median_answers(service: ScreeningService) -> dict[str, Any]:
    """Neutral default answers for the form (no real woman)."""
    defaults = {"age": 28, "prepreg_weight": 53, "height": 152, "anc_visits": 4, "n_children": 1, "n_under5": 0,
                "education_mother": "SMA/SMK", "egg_freq": "2-3x/minggu", "income": "1-2 juta", "food_taboo": "Tidak"}
    defaults = {k: v for k, v in defaults.items() if k in service.numeric_fields or k in service.choice_fields}
    defaults.update({f: service.choices(f)[0] for f in HFIAS_FIELDS})
    return defaults
