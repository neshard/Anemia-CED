"""In-fold behaviour of the full pipeline.

These tests check the leakage properties directly: fitted statistics come from
the training fold only, no step is fitted on validation rows, the resampler
is not called at prediction time, and targets cannot enter the model matrix.
"""

from __future__ import annotations

import io
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor

from src.cleaning import clean_dataset
from src.preprocessing import build_imputer, build_pipeline, preprocessing_only
from src.resampling import make_smogn, make_smotenc
from src.splits import KeyStratifiedRepeatedKFold, build_strat_key
from src.utils import LeakageError


class SpyTransformer(BaseEstimator, TransformerMixin):
    """Records the DataFrame index seen by every fit call."""

    fitted_on: list[set[Any]] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> SpyTransformer:
        SpyTransformer.fitted_on.append(set(X.index))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class CountingResampler(BaseEstimator):
    """Pass-through sampler that counts calls and records the rows it saw."""

    calls: list[set[Any]] = []

    def fit_resample(self, X: pd.DataFrame, y: Any) -> tuple[pd.DataFrame, Any]:
        CountingResampler.calls.append(set(X.index))
        return X, y


@pytest.fixture()
def frame(isolated_config: dict) -> pd.DataFrame:
    df = clean_dataset(isolated_config).frame
    return df.set_index("anon_id", drop=False)


def _hb_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sub = frame.loc[frame["hb_target"].notna()]
    key, _, _ = build_strat_key(sub["anemia_class"], sub["ced"], 5)
    return sub, sub["hb_target"].to_numpy(), key.to_numpy()


def test_pipeline_fits_only_on_training_rows(config: dict, frame: pd.DataFrame) -> None:
    X, y, key = _hb_data(frame)
    pipe = build_pipeline(config, "hb_model", DummyRegressor(), task="regression", feature_mode="fixed_preliminary", resampler=CountingResampler())
    pipe.steps.insert(0, ("spy", SpyTransformer()))
    cv = KeyStratifiedRepeatedKFold(key, n_splits=5, n_repeats=1, random_state=0)

    X = X[config["features"]["fixed_preliminary"]["hb_model"]]
    SpyTransformer.fitted_on, CountingResampler.calls = [], []
    cross_validate(pipe, X, y, cv=cv)
    folds = list(cv.split(X))
    assert len(SpyTransformer.fitted_on) == len(folds) == len(CountingResampler.calls)
    for (tr, va), seen, resampled in zip(folds, SpyTransformer.fitted_on, CountingResampler.calls, strict=True):
        train_ids, val_ids = set(X.index[tr]), set(X.index[va])
        assert seen == train_ids and not seen & val_ids
        assert resampled <= train_ids  # resampler never sees validation rows


def test_resampler_not_called_at_predict(config: dict, frame: pd.DataFrame) -> None:
    X, y, _ = _hb_data(frame)
    X = X[config["features"]["fixed_preliminary"]["hb_model"]]
    pipe = build_pipeline(config, "hb_model", DummyRegressor(), task="regression", feature_mode="fixed_preliminary", resampler=CountingResampler())
    pipe.fit(X, y)
    CountingResampler.calls = []
    pipe.predict(X.iloc[:10])
    assert CountingResampler.calls == []


def test_imputer_uses_training_median_only(config: dict) -> None:
    train = pd.DataFrame({"age": [20.0, 22.0, np.nan, 24.0], "egg_freq": [1.0, 1.0, 3.0, np.nan]})
    val = pd.DataFrame({"age": [np.nan, 90.0], "egg_freq": [np.nan, 5.0]})
    imp = build_imputer(config).fit(train)
    out = imp.transform(val)
    assert out.loc[0, "age"] == 22.0          # median of the training fold, not of train plus val
    assert out.loc[0, "egg_freq"] == 1.0      # mode of the training fold


def test_target_in_input_is_rejected(config: dict, frame: pd.DataFrame) -> None:
    X, y, _ = _hb_data(frame)
    pipe = build_pipeline(config, "hb_model", DummyRegressor(), task="regression", feature_mode="fixed_preliminary")
    with pytest.raises(LeakageError):
        pipe.fit(X, y)  # X still carries hb_target, anemia_class and so on
    clean_X = X[config["features"]["fixed_preliminary"]["hb_model"]]
    pipe.fit(clean_X, y)
    with pytest.raises(LeakageError):
        pipe.predict(clean_X.assign(muac=1.0))


def test_model_matrix_has_exactly_the_configured_features(config: dict, frame: pd.DataFrame) -> None:
    X, y, _ = _hb_data(frame)
    feats = config["features"]["fixed_preliminary"]["hb_model"]
    pipe = build_pipeline(config, "hb_model", XGBRegressor(n_estimators=10), task="regression", feature_mode="fixed_preliminary").fit(X[feats], y)
    matrix = preprocessing_only(pipe).transform(X[feats])
    assert sorted(matrix.columns) == sorted(feats) and not matrix.isna().any().any()


def test_scaling_only_when_requested(config: dict, frame: pd.DataFrame) -> None:
    sub = frame.loc[frame["ced"].notna()]
    feats = config["features"]["fixed_preliminary"]["ced_model"]
    X, y = sub[feats], sub["ced"].astype(int).to_numpy()
    scaled = build_pipeline(config, "ced_model", LogisticRegression(), task="classification", feature_mode="fixed_preliminary", scale=True).fit(X, y)
    raw = build_pipeline(config, "ced_model", DummyClassifier(), task="classification", feature_mode="fixed_preliminary").fit(X, y)
    m_scaled = preprocessing_only(scaled).transform(X)
    m_raw = preprocessing_only(raw).transform(X)
    assert abs(m_scaled["age"].mean()) < 1e-6
    assert m_raw["age"].mean() > 15


def test_in_fold_selection_mode_picks_k_from_pool(config: dict, frame: pd.DataFrame) -> None:
    X, y, _ = _hb_data(frame)
    pool = config["features"]["candidate_pool"]
    config["feature_selection"]["rf_n_estimators"] = 30
    pipe = build_pipeline(config, "hb_model", DummyRegressor(), task="regression", feature_mode="per_fold")
    pipe.fit(X[pool], y)
    chosen = pipe.named_steps["select"].selected_features_
    assert len(chosen) == config["feature_selection"]["k"] and set(chosen) <= set(pool)


def test_resamplers_run_inside_pipeline(config: dict, frame: pd.DataFrame) -> None:
    X, y, _ = _hb_data(frame)
    feats = config["features"]["fixed_preliminary"]["hb_model"]
    pipe = build_pipeline(config, "hb_model", DummyRegressor(), task="regression", feature_mode="fixed_preliminary", resampler=make_smogn(config))
    pipe.fit(X[feats], y)
    assert pipe.named_steps["resample"].report_["n_in"] == len(X)

    sub = frame.loc[frame["ced"].notna()]
    ced_feats = config["features"]["candidate_pool"]
    pipe = build_pipeline(config, "ced_model", DummyClassifier(), task="classification", feature_mode="all_candidates", resampler=make_smotenc(config))
    pipe.fit(sub[ced_feats], sub["ced"].astype(int).to_numpy())
    rep = pipe.named_steps["resample"].report_
    assert rep["method"] == "SMOTENC" and len(set(rep["class_counts_after"].values())) == 1


def test_fitted_pipeline_round_trips_through_joblib(config: dict, frame: pd.DataFrame) -> None:
    X, y, _ = _hb_data(frame)
    feats = config["features"]["fixed_preliminary"]["hb_model"]
    pipe = build_pipeline(config, "hb_model", XGBRegressor(n_estimators=10), task="regression", feature_mode="fixed_preliminary").fit(X[feats], y)
    buffer = io.BytesIO()
    joblib.dump(pipe, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    np.testing.assert_allclose(restored.predict(X[feats]), pipe.predict(X[feats]))
