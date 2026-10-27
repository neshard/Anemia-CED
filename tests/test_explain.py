"""Phase D: SHAP additivity, narrative rules and cross-fitted calibration."""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
import pytest
import yaml

from src.cleaning import clean_dataset
from src.models import CED_MODELS, HB_MODELS
from src.nested_cv import build_data, make_pipeline_factory
from src.postmodel import calibration as cal
from src.postmodel.explain import (
    DISCLAIMER,
    NON_CAUSAL_SENTENCE,
    additivity_error,
    explain_ced_row,
    explain_hb_row,
    narrative_ced,
    narrative_hb,
    shap_table,
)
from src.runs import main_run
from src.utils import PROJECT_ROOT

EXPLAIN_CFG = yaml.safe_load((PROJECT_ROOT / "config" / "explain.yaml").read_text(encoding="utf-8"))


@pytest.fixture()
def fitted(isolated_config: dict):  # type: ignore[no-untyped-def]
    cfg = copy.deepcopy(isolated_config)
    cfg["feature_selection"]["rf_n_estimators"] = 20
    df = clean_dataset(cfg).frame
    out = {}
    for target, spec in (("hb", HB_MODELS["xgboost"]), ("ced", CED_MODELS["xgboost"])):
        data = build_data(cfg, df, target)
        pipe = make_pipeline_factory(main_run(spec, cfg), cfg)({"n_estimators": 30, "max_depth": 3})
        out[target] = (pipe.fit(data.X, data.y), data, cfg)
    return out


@pytest.mark.parametrize("target", ["hb", "ced"])
def test_shap_additivity(fitted, target: str) -> None:  # type: ignore[no-untyped-def]
    pipe, data, _ = fitted[target]
    assert additivity_error(pipe, data.X.iloc[:40]) < 1e-4


def test_shap_columns_are_the_selected_features(fitted) -> None:  # type: ignore[no-untyped-def]
    pipe, data, _ = fitted["ced"]
    values, _, Xt = shap_table(pipe, data.X.iloc[:5])
    assert list(values.columns) == list(Xt.columns) and len(values.columns) == 7


def test_ced_narrative_is_non_causal_and_has_disclaimer(fitted) -> None:  # type: ignore[no-untyped-def]
    pipe, data, cfg = fitted["ced"]
    fig, text, prob = explain_ced_row(pipe, data.X.iloc[[0]], 0.3, EXPLAIN_CFG, cfg)
    assert fig is not None and 0 <= prob <= 1
    assert NON_CAUSAL_SENTENCE in text and DISCLAIMER in text
    without_disclaimers = text.replace(NON_CAUSAL_SENTENCE, "").replace(DISCLAIMER, "")
    assert not re.search(r"menyebabkan|penyebab|karena", without_disclaimers)
    assert "—" not in text and ";" not in text


def test_hb_explanation_hidden_when_disabled(fitted) -> None:  # type: ignore[no-untyped-def]
    pipe, data, _ = fitted["hb"]
    cfg = copy.deepcopy(EXPLAIN_CFG)
    cfg["hb_explanation_enabled"] = False
    fig, text, _ = explain_hb_row(pipe, data.X.iloc[[0]], 2.1, cfg)
    assert fig is None and "indikatif" in text and "Faktor" not in text
    assert EXPLAIN_CFG["hb_explanation_enabled"] is False  # S1 was not supported


def test_narrative_numbers_use_decimal_comma() -> None:
    assert "11,6 g/dL" in narrative_hb(11.63, 2.0, False)
    point_only = narrative_hb(11.63, 2.0, False, show_range=False)
    assert "rentang" not in point_only and "indikatif" in point_only
    assert EXPLAIN_CFG["hb_show_range"] is False  # researcher decision, option C
    text = narrative_ced(0.12, 0.29, [{"label": "berat badan sebelum hamil", "value_text": "60 kg", "shap": -0.5}])
    assert "di bawah ambang skrining 29%" in text and "menurunkan" in text


def test_calibrator_never_scores_its_own_rows(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    oof = pd.DataFrame({"fold": np.repeat(range(5), 40), "proba": rng.random(200), "y_true": rng.integers(0, 2, 200)})
    seen = []
    original = cal.fit_calibrator

    def spy(p, y, method):  # type: ignore[no-untyped-def]
        seen.append(len(p))
        return original(p, y, method)

    monkeypatch.setattr(cal, "fit_calibrator", spy)
    out = cal.cross_fitted(oof, "platt")
    assert seen == [160] * 5 and not np.isnan(out).any()


def test_calibration_slope_of_perfect_probabilities() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    slope, intercept = cal.calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.1) and intercept == pytest.approx(0.0, abs=0.1)
