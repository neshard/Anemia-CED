"""Dashboard logic layer: validation, preprocessing replication and prediction.

The questionnaire answers go through ``src.cleaning.clean_features``, the same
deterministic rules used for training (category codes, HFIAS scoring,
plausibility limits). Prediction then uses the exact fitted pipeline objects
saved by experiments/09_final_models.py, so imputation, column selection and
encoding are never re-implemented by hand.
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
from src.postmodel.explain import explain_ced_row, explain_hb_row, shap_table, top_contributions
from src.postmodel.recommendations import Advice, build_advice
from src.utils import PROJECT_ROOT, load_config

NUMERIC_FIELDS: dict[str, str] = {
    "age": "Umur ibu (tahun)",
    "prepreg_weight": "Berat badan sebelum hamil (kg)",
    "height": "Tinggi badan (cm)",
    "anc_visits": "Jumlah kunjungan pemeriksaan kehamilan (ANC) sejauh ini",
    "n_children": "Jumlah anak",
}
CHOICE_FIELDS: dict[str, str] = {
    "education_mother": "Pendidikan terakhir ibu",
    "egg_freq": "Seberapa sering ibu makan telur ayam dalam 1 bulan terakhir",
}
HFIAS_FIELDS = [f"hfias_q{i}" for i in range(1, 10)]


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


class ScreeningService:
    """Loads the saved models once and turns questionnaire answers into a screening result."""

    def __init__(self, root: Path = PROJECT_ROOT, config: Mapping[str, Any] | None = None) -> None:
        self.root = Path(root)
        self.config = dict(config) if config is not None else load_config()
        art = self.root / "artifacts"
        self.meta = json.loads((art / "final_models.json").read_text(encoding="utf-8"))
        self.hb_pipe = joblib.load(art / "final_hb_xgboost.joblib")
        self.ced_pipe = joblib.load(art / "final_ced_xgboost.joblib")
        explainer_path = art / "explainer_ced.joblib"
        self.ced_explainer = joblib.load(explainer_path) if explainer_path.exists() else None
        self.explain_cfg = yaml.safe_load((self.root / "config" / "explain.yaml").read_text(encoding="utf-8"))
        self.rec_cfg = yaml.safe_load((self.root / "config" / "recommendations.yaml").read_text(encoding="utf-8"))

    # -- form definition ---------------------------------------------------
    def choices(self, field_name: str) -> list[str]:
        enc = self.config["encodings"]
        if field_name == "education_mother":
            return list(enc["education"])
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
        for f, label in NUMERIC_FIELDS.items():
            raw = answers.get(f)
            try:
                value = float(str(raw).replace(",", "."))
            except (TypeError, ValueError):
                errors.append(f"{label}: wajib diisi dengan angka.")
                continue
            low, high = self.numeric_range(f)
            if not low <= value <= high:
                errors.append(f"{label}: nilai {value:g} di luar batas wajar ({low:g} sampai {high:g}).")
            if f in ("anc_visits", "n_children", "age") and value != int(value):
                errors.append(f"{label}: harus bilangan bulat.")
        for f, label in CHOICE_FIELDS.items():
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
        row = {f: str(answers[f]).replace(",", ".") for f in NUMERIC_FIELDS}
        row.update({f: answers[f] for f in CHOICE_FIELDS})
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
                                              explainer=self.ced_explainer)
        values, _, Xt = shap_table(self.ced_pipe, X, self.ced_explainer)
        top = top_contributions(values.iloc[0], Xt.iloc[0], X.iloc[0], 5, self.explain_cfg, self.config)
        _, hb_text, hb_pred = explain_hb_row(self.hb_pipe, X, float(self.meta["hb"]["interval_half_width"]),
                                             self.explain_cfg)
        k = int(self.explain_cfg["shap"]["top_k_narrative"])
        advice = build_advice(prob, threshold, top[:k], self.rec_cfg)
        return ScreeningResult(
            hb_prediction=hb_pred, hb_class=str(who_anemia_class([hb_pred])[0]), hb_text=hb_text,
            ced_probability=prob, ced_label=int(prob >= threshold), ced_text=ced_text, ced_figure=fig,
            ced_top=top, advice=advice, extra={"threshold": threshold},
        )

    def storage_records(self, answers: Mapping[str, Any], result: ScreeningResult) -> tuple[dict[str, Any], dict[str, Any]]:
        """Inputs and prediction as stored in SQLite. No Hb range is stored (option C)."""
        inputs = {f: float(str(answers[f]).replace(",", ".")) for f in NUMERIC_FIELDS}
        inputs.update({f: answers[f] for f in (*CHOICE_FIELDS, *HFIAS_FIELDS)})
        prediction = {
            "predicted_hb": round(result.hb_prediction, 2), "hb_lower": None, "hb_upper": None,
            "anemia_class": result.hb_class, "ced_probability": round(result.ced_probability, 4),
            "ced_label": result.ced_label, "top5_shap_hb": None,
            "top5_shap_ced": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in result.ced_top],
            "model_note": "Hb indicative only, explanation hidden (S1 not supported). CED uncalibrated probability.",
        }
        return inputs, prediction


def median_answers(service: ScreeningService) -> dict[str, Any]:
    """Neutral default answers for the form (no real woman)."""
    defaults = {"age": 28, "prepreg_weight": 53, "height": 152, "anc_visits": 4, "n_children": 1,
                "education_mother": "SMA/SMK", "egg_freq": "2-3x/minggu"}
    defaults.update({f: service.choices(f)[0] for f in HFIAS_FIELDS})
    return defaults

