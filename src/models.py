"""Model specifications (proposal Section 3.5).

Every model is wrapped by ``src.preprocessing.build_pipeline`` with the same
imputation, per-fold selection and folds. Every tuned model receives the same
Optuna budget (config tuning.n_trials). Dummy models are not tuned.

Roles:
* main: XGBRegressor for Hb, XGBClassifier for CED.
* baseline: the comparators named in the hypotheses (mean-only and Ridge for
  Hb, class proportion and Logistic Regression for CED).
* exploratory_baseline: Decision Tree and Random Forest, reported without a verdict.
* boosting_comparator: GBM, LightGBM, CatBoost and AdaBoost (supervisor revision of
  2026-10-27), compared descriptively with XGBoost and candidates for the dashboard.
  CatBoost gets the same integer-coded matrix as every other model (no native
  categorical handling), so the comparison stays fair.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
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


def _gbm_reg(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=seed, **params)


def _gbm_clf(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=seed, **params)


def _lgbm_common(seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    # subsample_freq=1 makes the tuned subsample take effect (LightGBM ignores it at 0).
    # deterministic plus force_row_wise and one thread give identical results on every run.
    return {"random_state": seed, "n_jobs": int(config["models"]["lgbm_n_jobs"]), "deterministic": True,
            "force_row_wise": True, "subsample_freq": 1, "verbosity": -1}


def _lgbm_reg(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> LGBMRegressor:
    return LGBMRegressor(objective="regression", **_lgbm_common(seed, config), **params)


def _lgbm_clf(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> LGBMClassifier:
    return LGBMClassifier(objective="binary", **_lgbm_common(seed, config), **params)


def _catboost_common(seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    return {"random_seed": seed, "thread_count": int(config["models"]["catboost_thread_count"]), "verbose": 0,
            "allow_writing_files": False}


def _catboost_reg(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> CatBoostRegressor:
    return CatBoostRegressor(loss_function="RMSE", **_catboost_common(seed, config), **params)


def _catboost_clf(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> CatBoostClassifier:
    return CatBoostClassifier(loss_function="Logloss", **_catboost_common(seed, config), **params)


def _ada_split(params: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    """The tuned max_depth belongs to the base tree, everything else to AdaBoost."""
    rest = dict(params)
    return int(rest.pop("max_depth", 1)), rest


def _ada_reg(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> AdaBoostRegressor:
    depth, rest = _ada_split(params)
    return AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=depth, random_state=seed), random_state=seed,
                             **rest)


def _ada_clf(params: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> AdaBoostClassifier:
    # SAMME set explicitly: SAMME.R, the default in scikit-learn 1.5, is deprecated.
    depth, rest = _ada_split(params)
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth, random_state=seed), algorithm="SAMME",
                              random_state=seed, **rest)


def tree_node_count(model: Any) -> int | None:
    """Inference cost proxy: nodes (splits plus leaves) over all trees of a fitted model. None for non-tree models.

    CatBoost trees are oblivious, so a tree with L leaves is counted as the full binary
    tree of 2L - 1 nodes, the same convention as LightGBM.
    """
    if isinstance(model, (XGBClassifier, XGBRegressor)):
        return int(len(model.get_booster().trees_to_dataframe()))
    if isinstance(model, (LGBMClassifier, LGBMRegressor)):
        return int(sum(2 * t["num_leaves"] - 1 for t in model.booster_.dump_model()["tree_info"]))
    if isinstance(model, (CatBoostClassifier, CatBoostRegressor)):
        return int(sum(2 * int(n) - 1 for n in model.get_tree_leaf_counts()))
    if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        return int(model.tree_.node_count)
    if hasattr(model, "estimators_") and not isinstance(model, (DummyClassifier, DummyRegressor)):
        trees = np.ravel(np.asarray(model.estimators_, dtype=object))
        return int(sum(t.tree_.node_count for t in trees))
    return None


HB_MODELS: dict[str, ModelSpec] = {
    "xgboost": ModelSpec("xgboost", "regression", "hb_model", xgb_regressor, "xgb_regressor", "main"),
    "dummy_mean": ModelSpec("dummy_mean", "regression", "hb_model", _dummy_mean, None, "baseline"),
    "ridge": ModelSpec("ridge", "regression", "hb_model", _ridge, "ridge", "baseline", scale=True),
    "decision_tree": ModelSpec("decision_tree", "regression", "hb_model", _tree_reg, "decision_tree", "exploratory_baseline"),
    "random_forest": ModelSpec("random_forest", "regression", "hb_model", _rf_reg, "random_forest", "exploratory_baseline"),
    "gbm": ModelSpec("gbm", "regression", "hb_model", _gbm_reg, "gbm_regressor", "boosting_comparator"),
    "lightgbm": ModelSpec("lightgbm", "regression", "hb_model", _lgbm_reg, "lgbm_regressor", "boosting_comparator"),
    "catboost": ModelSpec("catboost", "regression", "hb_model", _catboost_reg, "catboost_regressor", "boosting_comparator"),
    "adaboost": ModelSpec("adaboost", "regression", "hb_model", _ada_reg, "adaboost_regressor", "boosting_comparator"),
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
    "gbm": ModelSpec("gbm", "classification", "ced_model", _gbm_clf, "gbm_classifier", "boosting_comparator"),
    "lightgbm": ModelSpec("lightgbm", "classification", "ced_model", _lgbm_clf, "lgbm_classifier",
                          "boosting_comparator"),
    "catboost": ModelSpec("catboost", "classification", "ced_model", _catboost_clf, "catboost_classifier",
                          "boosting_comparator"),
    "adaboost": ModelSpec("adaboost", "classification", "ced_model", _ada_clf, "adaboost_classifier",
                          "boosting_comparator"),
}

# XGBoost plus the four comparators, in the fixed order used for reports and tie-breaks.
BOOSTING_CANDIDATES: tuple[str, ...] = ("xgboost", "gbm", "lightgbm", "catboost", "adaboost")

# Exploratory comparisons (proposal 3.5), each retuned with the same budget.
CED_CLASS_WEIGHTED = ModelSpec("xgboost_class_weighted", "classification", "ced_model", xgb_classifier_weighted,
                               "xgb_classifier", "exploratory_variant")
DIRECT_ANEMIA = ModelSpec("direct_xgb_classifier", "classification", "hb_model", xgb_classifier, "xgb_classifier",
                          "exploratory_variant")

ALL_SPECS: tuple[ModelSpec, ...] = (*HB_MODELS.values(), *CED_MODELS.values(), CED_CLASS_WEIGHTED, DIRECT_ANEMIA)
