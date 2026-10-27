"""Orchestration shared by the experiment scripts.

Flow for one model, all on the hold-out TRAINING set:

1. Tune on tuning folds (seed offset, 1 repeat).
2. Evaluate the tuned configuration with 5 x 5 stratified repeated CV and,
   when a group column exists, with GroupKFold by health centre.
3. Refit on the whole training set and save the fitted pipeline.

The hold-out TEST set is only loaded by ``load_test_data``, which needs a
``HoldoutToken`` from ``claim_holdout``. A token is issued once per data
version and model set: claiming writes a record to artifacts/, and a second
claim is refused. That enforces "the hold-out set is touched once".
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline

from src.cleaning import clean_dataset
from src.evaluation import CVResult, choose_hb_cutoff, cv_evaluate
from src.intervals import fit_oof_conformal
from src.labels import WHO_CLASSES
from src.models import ModelSpec
from src.preprocessing import build_pipeline, model_input_columns, selected_features, setup_spec
from src.resampling import make_smogn, make_smotenc
from src.splits import (
    BoundGroupKFold,
    KeyStratifiedRepeatedKFold,
    load_or_create_holdout,
    make_strat_key,
)
from src.tuning import TuningResult, tune
from src.utils import file_fingerprint, get_logger, load_hypotheses, resolve_path, save_json, save_table

LOGGER = get_logger(__name__)

# target key -> (label column, task, feature set used)
TARGETS: dict[str, tuple[str, str, str]] = {
    "hb_model": ("hb_target", "regression", "hb_model"),
    "ced_model": ("ced", "classification", "ced_model"),
    "anemia_binary": ("anemia_binary", "classification", "hb_model"),
    "anemia_who": ("anemia_class", "multiclass", "hb_model"),
}


@dataclass
class ModelData:
    """Features, labels and splitters for one target on one part of the hold-out split."""

    target_key: str
    task: str
    feature_set: str
    setup: str
    X: pd.DataFrame
    y: np.ndarray
    anon_ids: pd.Series
    true_class: pd.Series
    class_names: list[str] | None = None
    cv: Any = None
    cv_group: Any = None
    cv_tune: Any = None


@dataclass
class ModelRun:
    """Everything produced for one model and one resampling variant."""

    name: str
    variant: str
    params: dict[str, Any]
    tuning: TuningResult | None
    cv: CVResult
    cv_group: CVResult | None
    pipeline: Pipeline
    extra: dict[str, Any] = field(default_factory=dict)


def load_split_frame(config: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Clean the data and return the frame with train and test masks from the saved split."""
    df = clean_dataset(config).frame
    split_cfg = config["splits"]
    key, _ = make_strat_key(df["anemia_class"], df["ced"], split_cfg["strat_key_merge_map"], int(split_cfg["min_cell_count"]))
    split = load_or_create_holdout(df, key, config)
    train_mask, test_mask = split.masks(df["anon_id"])
    return df, train_mask, test_mask


def _rows_for(df: pd.DataFrame, target_key: str) -> np.ndarray:
    """Rows with a usable label. Anemia comparators use the regressor's rows so comparisons are paired."""
    if target_key in ("anemia_binary", "anemia_who", "hb_model"):
        return df["hb_target"].notna().to_numpy() & df["anemia_class"].notna().to_numpy()
    return df["ced"].notna().to_numpy()


def _encode_target(sub: pd.DataFrame, target_key: str, class_names: list[str] | None) -> tuple[np.ndarray, list[str] | None]:
    label, task, _ = TARGETS[target_key]
    if task == "regression":
        return sub[label].to_numpy(dtype=float), None
    if task == "classification":
        return sub[label].to_numpy().astype(int), None
    names = class_names or [c for c in WHO_CLASSES if (sub[label] == c).any()]
    unknown = set(sub[label]) - set(names)
    if unknown:
        raise ValueError(f"Classes {unknown} not seen in training. Cannot encode.")
    return sub[label].map({c: i for i, c in enumerate(names)}).to_numpy().astype(int), names


def _make_data(config: Mapping[str, Any], df: pd.DataFrame, mask: np.ndarray, target_key: str, setup: str,
               class_names: list[str] | None = None) -> ModelData:
    _, task, feature_set = TARGETS[target_key]
    sub = df.loc[mask & _rows_for(df, target_key)].reset_index(drop=True)
    y, names = _encode_target(sub, target_key, class_names)
    X = sub[model_input_columns(config, feature_set, setup)]
    return ModelData(target_key, task, feature_set, setup, X, y, sub["anon_id"], sub["anemia_class"], names)


def load_training_data(config: Mapping[str, Any], target_key: str, setup: str | None = None) -> ModelData:
    """Training rows for ``target_key`` with evaluation, group and tuning splitters attached."""
    setup_name, _ = setup_spec(config, setup)
    df, train_mask, _ = load_split_frame(config)
    data = _make_data(config, df, train_mask, target_key, setup_name)
    sub = df.loc[train_mask & _rows_for(df, target_key)].reset_index(drop=True)

    cfg = config["splits"]
    seed = int(config["project"]["random_seed"])
    key, _ = make_strat_key(sub["anemia_class"], sub["ced"], cfg["strat_key_merge_map"], int(cfg["cv_n_splits"]))
    data.cv = KeyStratifiedRepeatedKFold(key, int(cfg["cv_n_splits"]), int(cfg["cv_n_repeats"]), seed)
    tcfg = config["tuning"]
    data.cv_tune = KeyStratifiedRepeatedKFold(
        key, int(tcfg["cv_n_splits"]), int(tcfg["cv_n_repeats"]), seed + int(tcfg["cv_seed_offset"])
    )
    group_col = cfg.get("group_column")
    if group_col and group_col in sub.columns:
        data.cv_group = BoundGroupKFold(sub[group_col].to_numpy(), int(cfg["cv_n_splits"]))
    return data


class HoldoutAlreadyUsedError(RuntimeError):
    """Raised when the hold-out test set is claimed a second time."""


@dataclass(frozen=True)
class HoldoutToken:
    """Proof that the caller holds the single hold-out evaluation for this data version."""

    record_path: str
    claimed_at: str


def holdout_record_path(config: Mapping[str, Any]) -> Path:
    """Where the one-time hold-out claim is recorded."""
    return resolve_path(config, "artifacts_dir") / "holdout_evaluation_record.json"


def claim_holdout(config: Mapping[str, Any], purpose: str) -> HoldoutToken:
    """Issue the one-time permission to read the hold-out test set.

    The record stores the data, hypotheses and split fingerprints and the
    time. A second claim raises, because looking at the test set again after
    changing anything would turn it into a tuning set. If it really must be
    re-evaluated (for example a bug fix requested by a reviewer), delete the
    record by hand and report the re-evaluation as a deviation in the thesis.
    """
    from datetime import datetime, timezone

    path = holdout_record_path(config)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        raise HoldoutAlreadyUsedError(
            f"The hold-out set was already evaluated on {previous['claimed_at']} for '{previous['purpose']}'. "
            f"It may be touched once only. Record: {path}"
        )
    claimed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = {
        "claimed_at": claimed_at,
        "purpose": purpose,
        "raw_data_sha256": file_fingerprint(resolve_path(config, "raw_data")),
        "hypotheses_sha256": file_fingerprint(resolve_path(config, "hypotheses")),
        "split_file_sha256": file_fingerprint(resolve_path(config, "split_file")),
    }
    save_json(record, path)
    LOGGER.info("Hold-out claimed once for '%s' at %s", purpose, claimed_at)
    return HoldoutToken(str(path), claimed_at)


def load_test_data(config: Mapping[str, Any], target_key: str, token: HoldoutToken, setup: str | None = None,
                   class_names: list[str] | None = None) -> ModelData:
    """Hold-out test rows. Requires the token from ``claim_holdout``."""
    if not isinstance(token, HoldoutToken) or not Path(token.record_path).exists():
        raise HoldoutAlreadyUsedError("load_test_data needs a valid HoldoutToken from claim_holdout")
    setup_name, _ = setup_spec(config, setup)
    df, _, test_mask = load_split_frame(config)
    return _make_data(config, df, test_mask, target_key, setup_name, class_names)


def make_resampler(name: str, config: Mapping[str, Any]) -> Any:
    """Resampler by variant name."""
    if name == "none":
        return None
    if name == "smotenc":
        return make_smotenc(dict(config))
    if name == "smogn":
        return make_smogn(dict(config))
    if name == "smogn_no_undersampling":
        sampler = make_smogn(dict(config))
        sampler.set_params(undersample_normal=False)
        return sampler
    raise ValueError(f"Unknown resampler '{name}'")


def pipeline_factory(spec: ModelSpec, config: Mapping[str, Any], resampler: str,
                     setup: str) -> Callable[[Mapping[str, Any]], Pipeline]:
    """Return ``params -> unfitted pipeline`` for this model, resampling variant and feature setup."""
    seed = int(config["project"]["random_seed"])
    fs_task = "classification" if spec.task in ("classification", "multiclass") else "regression"

    def make(params: Mapping[str, Any]) -> Pipeline:
        return build_pipeline(
            config, spec.model_key, spec.factory(params, seed, config),
            task=fs_task, scale=spec.scale, resampler=make_resampler(resampler, config), setup=setup,
        )

    return make


def run_model(
    spec: ModelSpec,
    data: ModelData,
    config: Mapping[str, Any],
    resampler: str,
    *,
    params: Mapping[str, Any] | None = None,
    variant: str | None = None,
    group_cv: bool = True,
) -> ModelRun:
    """Tune (unless ``params`` is given), cross-validate, and refit on all training rows."""
    make = pipeline_factory(spec, config, resampler, data.setup)
    tuning_result: TuningResult | None = None
    if params is None:
        tcfg = config["tuning"]
        space = tcfg["search_spaces"].get(spec.space) if spec.space else None
        tuning_result = tune(
            make, space, data.X, data.y, data.cv_tune, data.task, tcfg["metric"][data.target_key],
            int(tcfg["n_trials"][spec.budget]), int(config["project"]["random_seed"]), tcfg["backend"],
            int(tcfg["grid_points"]), data.class_names,
        )
        params = tuning_result.best_params
        LOGGER.info("[%s] %s %s tuned: %s = %.4f", data.setup, data.target_key, spec.name,
                    tcfg["metric"][data.target_key], tuning_result.best_score)

    threshold = float(config["models"]["ced_threshold"])
    anemia_thr = float(config["evaluation"]["anemia_threshold"])
    cv_result = cv_evaluate(lambda: make(params), data.X, data.y, data.cv, data.task,
                            class_names=data.class_names, threshold=threshold, anemia_threshold=anemia_thr)
    group_result = None
    if group_cv and data.cv_group is not None:
        group_result = cv_evaluate(lambda: make(params), data.X, data.y, data.cv_group, data.task,
                                   class_names=data.class_names, threshold=threshold, anemia_threshold=anemia_thr)
    final = make(params).fit(data.X, data.y)
    return ModelRun(spec.name, variant or resampler, dict(params), tuning_result, cv_result, group_result, final)


def run_tag(setup: str, target_key: str, model: str, variant: str) -> str:
    """Name used for every file a run writes."""
    return f"{setup}__{target_key}__{model}__{variant}"


def save_run(run: ModelRun, data: ModelData, config: Mapping[str, Any], save_model: bool = True) -> None:
    """Write fold tables, out-of-fold predictions, tuned parameters and the fitted pipeline."""
    tag = run_tag(data.setup, data.target_key, run.name, run.variant)
    for scheme, result in (("stratified", run.cv), ("group", run.cv_group)):
        if result is None:
            continue
        folds = result.folds.copy()
        folds.insert(0, "cv_scheme", scheme)
        folds.insert(0, "variant", run.variant)
        folds.insert(0, "model", run.name)
        folds.insert(0, "target", data.target_key)
        folds.insert(0, "setup", data.setup)
        save_table(folds, f"cv_folds/{tag}__{scheme}", config)
        oof = result.oof.copy()
        oof.insert(0, "anon_id", data.anon_ids.iloc[oof["row"]].to_numpy())
        oof_path = resolve_path(config, "artifacts_dir") / "oof" / f"{tag}__{scheme}.csv"
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof.to_csv(oof_path, index=False)

    artifacts = resolve_path(config, "artifacts_dir")
    params_path = artifacts / "tuned_params.json"
    all_params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
    all_params[tag] = {
        "params": run.params,
        "tuning_backend": run.tuning.backend if run.tuning else "reused",
        "tuning_best_score": run.tuning.best_score if run.tuning else None,
        "class_names": data.class_names,
        "setup": data.setup,
        "input_columns": list(data.X.columns),
        "features_used_by_final_model": selected_features(run.pipeline),
        **regression_extras(run, data, config),
    }
    save_json(all_params, params_path)
    if run.tuning is not None and len(run.tuning.trials):
        trials = run.tuning.trials.copy()
        trials.columns = [str(c) for c in trials.columns]
        save_table(trials.astype(str), f"tuning/{tag}", config)
    if save_model:
        joblib.dump(run.pipeline, artifacts / f"{tag}.joblib")


def regression_extras(run: ModelRun, data: ModelData, config: Mapping[str, Any]) -> dict[str, Any]:
    """H3 cut-off and H8 interval, both fixed from training out-of-fold predictions.

    Only regression runs on the Hb target get them. The stratified CV
    out-of-fold predictions are the only input, so neither value has seen
    the hold-out set. 07 applies them to the hold-out set once.
    """
    if data.task != "regression" or data.target_key != "hb_model":
        return {}
    hyp = load_hypotheses(config)
    oof = run.cv.oof
    extras: dict[str, Any] = {}
    h3 = hyp.get("hypotheses", {}).get("H3_anemia_screening")
    if h3 and "min_sensitivity" in h3:
        extras["operating_point"] = choose_hb_cutoff(
            oof["y_true"].to_numpy(), oof["pred"].to_numpy(), float(config["evaluation"]["anemia_threshold"]),
            float(h3["min_sensitivity"]),
        )
    if config.get("intervals", {}).get("method") == "oof_conformal":
        extras["interval"] = fit_oof_conformal(oof, float(config["intervals"]["nominal"])).to_dict()
    return extras


def load_run_record(config: Mapping[str, Any], tag: str) -> dict[str, Any]:
    """Everything save_run stored for a tag (parameters, features, cut-off, interval)."""
    path = resolve_path(config, "artifacts_dir") / "tuned_params.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the training scripts first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if tag not in data:
        raise KeyError(f"No saved run for {tag}. Run the script that trains it first.")
    return dict(data[tag])


def load_params(config: Mapping[str, Any], tag: str) -> dict[str, Any]:
    """Tuned parameters saved by an earlier script."""
    path = resolve_path(config, "artifacts_dir") / "tuned_params.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the training scripts first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if tag not in data:
        raise KeyError(f"No tuned parameters for {tag}. Run the script that trains it first.")
    return dict(data[tag]["params"])


def cv_summary_row(run: ModelRun, data: ModelData, metrics: list[str]) -> dict[str, Any]:
    """One summary row (mean and SD over folds) for the stratified and group schemes."""
    row: dict[str, Any] = {"setup": data.setup, "target": data.target_key, "model": run.name,
                           "variant": run.variant, "n_train_rows": len(data.y),
                           "final_model_features": ", ".join(selected_features(run.pipeline))}
    for scheme, result in (("strat", run.cv), ("group", run.cv_group)):
        if result is None:
            continue
        for m in metrics:
            if m in result.folds.columns:
                row[f"{scheme}_{m}_mean"] = round(float(result.folds[m].mean()), 4)
                row[f"{scheme}_{m}_sd"] = round(float(result.folds[m].std(ddof=1)), 4)
    return row


def artifact_path(config: Mapping[str, Any], tag: str) -> Path:
    """Path of a saved fitted pipeline."""
    return resolve_path(config, "artifacts_dir") / f"{tag}.joblib"
