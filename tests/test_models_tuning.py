"""Model factories, fold-weighted XGBoost, tuning backends, and the hypotheses gate."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.models import CED_MODELS, HB_MODELS, FoldWeightedXGBClassifier
from src.splits import KeyStratifiedRepeatedKFold
from src.utils import HypothesesNotLockedError, require_locked_hypotheses


def test_fold_weighted_xgb_uses_training_fold_ratio() -> None:
    X = np.random.default_rng(0).normal(size=(40, 2))
    y = np.array([1] * 10 + [0] * 30)
    model = FoldWeightedXGBClassifier(n_estimators=5).fit(X, y)
    assert model.get_params()["scale_pos_weight"] == pytest.approx(3.0)
    cloned = clone(model)
    assert isinstance(cloned, FoldWeightedXGBClassifier)
    cloned.fit(X[:20], y[:20])  # 10 positives, 10 negatives in this fold
    assert cloned.get_params()["scale_pos_weight"] == pytest.approx(1.0)


def test_all_specs_build(config: dict) -> None:
    for spec in [*HB_MODELS.values(), *CED_MODELS.values()]:
        est = spec.factory({}, 0, config)
        assert hasattr(est, "fit")
    assert HB_MODELS["ridge"].scale and CED_MODELS["logistic_regression"].scale
    assert HB_MODELS["dummy_mean"].space is None and CED_MODELS["dummy_prior"].space is None
    assert {s.role for s in HB_MODELS.values()} == {"main", "baseline", "exploratory_baseline"}
    assert not HB_MODELS["xgboost"].scale


def test_hypotheses_gate(config: dict, tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = copy.deepcopy(config)
    draft = tmp_path / "h.yaml"
    draft.write_text("status: draft\nhypotheses: {}\n", encoding="utf-8")
    cfg["paths"]["hypotheses"] = str(draft)
    with pytest.raises(HypothesesNotLockedError):
        require_locked_hypotheses(cfg)
    cfg["project"]["smoke_test"] = True
    assert require_locked_hypotheses(cfg)["status"] == "draft"
    cfg["project"]["smoke_test"] = False
    draft.write_text("status: locked\nhypotheses: {}\n", encoding="utf-8")
    assert require_locked_hypotheses(cfg)["status"] == "locked"
