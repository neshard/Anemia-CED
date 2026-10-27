"""Model specifications (proposal Section 3.5).

Every model is wrapped by ``src.preprocessing.build_pipeline`` with the same
imputation, per-fold selection and folds. Every tuned model receives the same
Optuna budget (config tuning.n_trials). Dummy models are not tuned.

Roles:
* main: XGBRegressor for Hb, XGBClassifier for CED.
* baseline: the comparators named in the hypotheses (mean-only and Ridge for
  Hb, class proportion and Logistic Regression for CED).
* exploratory_baseline: Decision Tree and Random Forest, reported without a verdict.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

EstimatorFactory = Callable[[Mapping[str, Any], int, Mapping[str, Any]], BaseEstimator]


class FoldWeightedXGBClassifier(XGBClassifier):
    """XGBClassifier whose ``scale_pos_weight`` is the negative to positive ratio of the training data it gets.

    Exploratory class-weighting variant. The ratio is recomputed at every fit,
    so it never reflects the class balance of rows it is evaluated on.
    """

    def fit(self, X: Any, y: Any, **kwargs: Any) -> FoldWeightedXGBClassifier:
        y_arr = np.asarray(y)
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr == 0).sum())
        self.set_params(scale_pos_weight=(n_neg / n_pos) if n_pos else 1.0)
        return super().fit(X, y, **kwargs)


@dataclass(frozen=True)
class ModelSpec:
    """How to build and tune one model."""

    name: str
    task: str                 # regression | classification
    model_key: str            # hb_model or ced_model (feature guard)
    factory: EstimatorFactory
    space: str | None         # key in tuning.search_spaces, None means not tuned
    role: str                 # main | baseline | exploratory_baseline | exploratory_variant
    scale: bool = False


def _xgb_common(seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    return {"random_state": seed, "n_jobs": int(config["models"]["xgb_n_jobs"]), "tree_method": "hist"}


def xgb_regressor(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> XGBRegressor:
    return XGBRegressor(objective="reg:squarederror", **_xgb_common(seed, config), **params)


def xgb_classifier(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> XGBClassifier:
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **_xgb_common(seed, config), **params)


def xgb_classifier_weighted(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> XGBClassifier:
    return FoldWeightedXGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                     **_xgb_common(seed, config), **params)


def _dummy_mean(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def _dummy_prior(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> DummyClassifier:
    return DummyClassifier(strategy="prior")


def _ridge(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> Ridge:
    return Ridge(**params)


def _logreg(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> LogisticRegression:
    return LogisticRegression(max_iter=5000, **params)


def _tree_reg(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(random_state=seed, **params)


def _tree_clf(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=seed, **params)


def _rf_reg(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=int(config["models"]["rf_n_jobs"]), **params)


def _rf_clf(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=int(config["models"]["rf_n_jobs"]), **params)


HB_MODELS: dict[str, ModelSpec] = {
    "xgboost": ModelSpec("xgboost", "regression", "hb_model", xgb_regressor, "xgb_regressor", "main"),
    "dummy_mean": ModelSpec("dummy_mean", "regression", "hb_model", _dummy_mean, None, "baseline"),
    "ridge": ModelSpec("ridge", "regression", "hb_model", _ridge, "ridge", "baseline", scale=True),
    "decision_tree": ModelSpec("decision_tree", "regression", "hb_model", _tree_reg, "decision_tree", "exploratory_baseline"),
    "random_forest": ModelSpec("random_forest", "regression", "hb_model", _rf_reg, "random_forest", "exploratory_baseline"),
}

CED_MODELS: dict[str, ModelSpec] = {
    "xgboost": ModelSpec("xgboost", "classification", "ced_model", xgb_classifier, "xgb_classifier", "main"),
    "dummy_prior": ModelSpec("dummy_prior", "classification", "ced_model", _dummy_prior, None, "baseline"),
    "logistic_regression": ModelSpec("logistic_regression", "classification", "ced_model", _logreg,
                                     "logistic_regression", "baseline", scale=True),
    "decision_tree": ModelSpec("decision_tree", "classification", "ced_model", _tree_clf, "decision_tree",
                               "exploratory_baseline"),
    "random_forest": ModelSpec("random_forest", "classification", "ced_model", _rf_clf, "random_forest",
                               "exploratory_baseline"),
}

# Exploratory comparisons (proposal 3.5), each retuned with the same budget.
CED_CLASS_WEIGHTED = ModelSpec("xgboost_class_weighted", "classification", "ced_model", xgb_classifier_weighted,
                               "xgb_classifier", "exploratory_variant")
DIRECT_ANEMIA = ModelSpec("direct_xgb_classifier", "classification", "hb_model", xgb_classifier, "xgb_classifier",
                          "exploratory_variant")

ALL_SPECS: tuple[ModelSpec, ...] = (*HB_MODELS.values(), *CED_MODELS.values(), CED_CLASS_WEIGHTED, DIRECT_ANEMIA)
