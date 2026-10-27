"""In-fold preprocessing and the full model pipeline.

Every step that learns from data lives inside one imblearn ``Pipeline``.
When the pipeline is fitted on a training fold, the imputer medians and
modes, the feature ranking, the resampling neighbours and the scaler means
all come from that fold alone. At prediction time the same fitted objects are
applied to the validation fold, the test set, or a dashboard input, and the
resampler is skipped automatically by imblearn.

Step order and why:

1. ``guard``      refuses targets and identifiers before anything else runs.
2. ``columns``    keeps the model's feature list (or the candidate pool).
3. ``impute``     median for numeric, mode for ordinal, binary and nominal.
                  It runs before selection because MI and random forests do not
                  accept NaN.
4. ``select``     optional in-fold MI plus RF rank aggregation.
5. ``resample``   optional SMOTENC or SMOGN, applied to training folds only.
                  It runs on unscaled, unencoded columns so categories are
                  still categories and integer columns can be rounded back.
6. ``encode``     one-hot for nominal columns, and StandardScaler only when
                  the model needs it (Logistic Regression, Ridge). Placed after
                  resampling so synthetic rows are scaled with the same
                  parameters as real rows.
7. ``guard_final`` re-checks the exact matrix that reaches the model.
8. ``model``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import assert_no_target_leakage, forbidden_columns_for

MODEL_KEYS: tuple[str, ...] = ("hb_model", "ced_model")


class ColumnsPresent:
    """Picklable column selector for ``ColumnTransformer``.

    It returns the configured columns that exist in the incoming frame. The
    set of columns can differ between folds when in-fold feature selection is
    on, so the selection must be resolved at fit time, not at construction.
    A plain lambda would do the same but cannot be saved with joblib.
    """

    def __init__(self, names: Sequence[str], exclude: bool = False) -> None:
        self.names = tuple(names)
        self.exclude = exclude

    def __call__(self, X: pd.DataFrame) -> list[str]:
        if self.exclude:
            return [c for c in X.columns if c not in self.names]
        return [c for c in X.columns if c in self.names]

    def __repr__(self) -> str:
        return f"ColumnsPresent({list(self.names)}, exclude={self.exclude})"


class LeakageGuard(BaseEstimator, TransformerMixin):
    """Stateless check that raises ``LeakageError`` if a forbidden column is present.

    It checks at fit and at transform, so the guard also protects the
    inference path used by the dashboard.
    """

    def __init__(self, forbidden: Sequence[str] = (), context: str = "pipeline") -> None:
        self.forbidden = forbidden
        self.context = context

    def fit(self, X: pd.DataFrame, y: Any = None) -> LeakageGuard:
        assert_no_target_leakage(X, self.forbidden, self.context)
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert_no_target_leakage(X, self.forbidden, self.context)
        return X

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a fixed list of columns and fail loudly if one is missing."""

    def __init__(self, columns: Sequence[str] = ()) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> ColumnSelector:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"Columns missing from input: {missing}")
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, list(self.columns)]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.columns)


def column_types(config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return the configured numeric, ordinal, binary and nominal column lists."""
    feats = config["features"]
    return {kind: list(feats.get(kind) or []) for kind in ("numeric", "ordinal", "binary", "nominal")}


def build_imputer(config: Mapping[str, Any]) -> ColumnTransformer:
    """Median imputation for numeric columns and mode imputation for the rest.

    Fitted inside each training fold through the pipeline, so validation and
    test rows are filled with statistics they did not contribute to.
    """
    types = column_types(config)
    categorical_like = types["ordinal"] + types["binary"] + types["nominal"]
    imputer = ColumnTransformer(
        [
            ("median", SimpleImputer(strategy="median", keep_empty_features=True), ColumnsPresent(types["numeric"])),
            ("mode", SimpleImputer(strategy="most_frequent", keep_empty_features=True), ColumnsPresent(categorical_like)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return imputer.set_output(transform="pandas")


def build_encoder(config: Mapping[str, Any], scale: bool) -> ColumnTransformer:
    """One-hot encode nominal columns and optionally standardise the others.

    Tree models get ``scale=False`` because splits are invariant to monotone
    rescaling. Ordinal and binary columns are left as integer codes, which is
    valid for ordered categories and for 0/1 indicators in linear models.
    """
    nominal = column_types(config)["nominal"]
    encoder = ColumnTransformer(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ColumnsPresent(nominal),
            ),
            ("rest", StandardScaler() if scale else "passthrough", ColumnsPresent(nominal, exclude=True)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return encoder.set_output(transform="pandas")


def setup_spec(config: Mapping[str, Any], setup: str | None = None) -> tuple[str, dict[str, Any]]:
    """Return the setup name and its spec. ``None`` means the primary setup."""
    name = setup or str(config["primary_setup"])
    if name not in config["feature_setups"]:
        raise KeyError(f"Unknown feature setup '{name}'. Known: {list(config['feature_setups'])}")
    return name, dict(config["feature_setups"][name])


def model_input_columns(config: Mapping[str, Any], model_key: str, setup: str | None = None) -> list[str]:
    """Columns entering the pipeline: the setup's fixed list, or the candidate pool for in-fold selection."""
    name, spec = setup_spec(config, setup)
    if spec["mode"] == "fixed":
        return list(spec[model_key])
    if spec["mode"] == "in_fold":
        return list(config["features"]["candidate_pool"])
    raise ValueError(f"Unknown mode '{spec['mode']}' in feature setup '{name}'")


def build_pipeline(
    config: Mapping[str, Any],
    model_key: str,
    estimator: BaseEstimator,
    *,
    task: str,
    scale: bool = False,
    resampler: BaseEstimator | None = None,
    setup: str | None = None,
) -> Pipeline:
    """Assemble the leakage-safe pipeline for one model and one feature setup.

    ``task`` is "regression" or "classification" and is passed to the
    in-fold selector. ``resampler`` is any object with ``fit_resample`` (see
    ``src.resampling``). Baselines and ablations call this function with
    different estimators and resamplers, so they share identical
    preprocessing by construction.
    """
    from src.feature_selection import RankAggregationSelector  # local import avoids a cycle

    if model_key not in MODEL_KEYS:
        raise ValueError(f"model_key must be one of {MODEL_KEYS}")
    setup_name, spec = setup_spec(config, setup)
    forbidden = forbidden_columns_for(config, model_key)
    columns = model_input_columns(config, model_key, setup_name)
    assert_no_target_leakage(columns, forbidden, f"{model_key} input columns ({setup_name})")

    steps: list[tuple[str, Any]] = [
        ("guard", LeakageGuard(forbidden, f"{model_key} input")),
        ("columns", ColumnSelector(columns)),
        ("impute", build_imputer(config)),
    ]
    if spec["mode"] == "in_fold":
        fs = config["feature_selection"]
        forced = spec.get("forced_include", [])
        if isinstance(forced, Mapping):
            forced = forced.get(model_key, [])
        steps.append(
            (
                "select",
                RankAggregationSelector(
                    k=int(fs["k"]),
                    task=task,
                    n_estimators=int(fs["rf_n_estimators"]),
                    forced_include=tuple(forced),
                    discrete_features=tuple(fs["discrete_features"]),
                    random_state=int(config["project"]["random_seed"]),
                    inner_folds=int(fs.get("inner_folds", 0)),
                    min_selection_rate=float(fs.get("min_selection_rate", 0.5)),
                    max_features=int(fs.get("max_features", fs["k"])),
                ),
            )
        )
    if resampler is not None:
        steps.append(("resample", resampler))
    steps += [
        ("encode", build_encoder(config, scale)),
        ("guard_final", LeakageGuard(forbidden, f"{model_key} model matrix")),
        ("model", estimator),
    ]
    return Pipeline(steps)


def selected_features(pipeline: Pipeline) -> list[str]:
    """Columns the fitted pipeline actually passes to the model."""
    steps = getattr(pipeline, "named_steps", {})
    if "select" in steps:
        return list(steps["select"].selected_features_)
    return list(steps["columns"].columns)


def preprocessing_only(pipeline: Pipeline) -> Pipeline:
    """Return the fitted steps before the model, without the resampler.

    Used later to build the matrix SHAP explains. Dropping the resampler is
    correct because it never runs at prediction time.
    """
    kept = [(name, step) for name, step in pipeline.steps[:-1] if name != "resample"]
    return Pipeline(kept)
