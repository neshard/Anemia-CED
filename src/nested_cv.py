"""Nested repeated cross-validation engine (proposal Section 3.5).

Outer loop: repeated stratified 5 x 5 CV on all eligible records (or one
GroupKFold pass for the subdistrict sensitivity analysis). For every outer
training set:

1. Inner stratified 3-fold CV. Optuna tunes the model on it (same trial
   budget for every tuned model). Per-fold feature selection is refit inside
   every inner training split and cached, so it is computed once per split
   and reused across trials.
2. The inner-fold predictions of the best trial set the decision threshold
   (Hb: the cut-off flagging 70 percent of true anemia cases, CED: Youden's
   index plus higher-sensitivity cut-offs) and the split-conformal half-width.
   Predictions of a model on its own training rows are never used, and the
   outer test fold is never used.
3. The pipeline is refit on the whole outer training set with the best
   parameters and scores the outer test fold once.

Every fold result is checkpointed under a key that hashes the config, the
hypotheses file, the raw data and the source code, so a changed setup never
resumes stale results.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, roc_auc_score

from src.labels import ced_label_at_most, trimester_anemia
from src.models import ModelSpec, tree_node_count
from src.preprocessing import build_pipeline, model_input_columns
from src.resampling import make_smogn, make_smotenc
from src.splits import BoundGroupKFold, KeyStratifiedRepeatedKFold, assert_disjoint, build_strat_key, inner_splits
from src.tuning import make_sampler, suggest_params
from src.utils import PROJECT_ROOT, file_fingerprint, get_logger, resolve_path

LOGGER = get_logger(__name__)


class ThresholdLeakageError(AssertionError):
    """Raised when a threshold or interval would be set from outer test rows or from training predictions."""


# ---------------------------------------------------------------------------
# Analysis data
# ---------------------------------------------------------------------------

@dataclass
class AnalysisData:
    """Eligible records of one analysis, indexed by anonymous ID."""

    analysis: str
    target: str               # hb | ced | anemia_binary
    task: str                 # regression | classification
    model_key: str            # hb_model | ced_model
    X: pd.DataFrame
    y: np.ndarray
    y_binary: np.ndarray | None
    strat_key: pd.Series
    strat_log: list[dict[str, Any]]
    groups: np.ndarray | None
    meta: dict[str, Any] = field(default_factory=dict)


def build_data(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    target: str,
    *,
    analysis: str = "main",
    ced_cutoff_cm: float | None = None,
    trimester: bool = False,
    max_gap_days: float | None = None,
) -> AnalysisData:
    """Select eligible rows, build labels and the stratification key before any split."""
    df = frame.set_index("anon_id", drop=False)
    columns = list(dict.fromkeys(config["features"]["candidate_pool"]))
    thr = float(config["evaluation"]["anemia_threshold"])
    meta: dict[str, Any] = {}

    if target in ("hb", "anemia_binary"):
        rows = df["hb_target"].notna() & df["anemia_class"].notna()
        if trimester:
            t = config["labels"]["trimester"]
            tri = pd.Series(trimester_anemia(df["hb_target"], df["gestational_age_weeks"], t["first_max_week"],
                                             t["second_max_week"], t["cutoffs"]), index=df.index)
            meta["excluded_no_gestational_age"] = int((rows & tri.isna()).sum())
            rows &= tri.notna()
        if max_gap_days is not None:
            gap = df["hb_date_gap_days"]
            meta["excluded_no_usable_date"] = int((rows & gap.isna()).sum())
            meta["excluded_gap_above_limit"] = int((rows & gap.notna() & (gap > max_gap_days)).sum())
            rows &= gap.notna() & (gap <= max_gap_days)
        sub = df.loc[rows]
        y_bin = (tri.loc[rows].to_numpy() if trimester else (sub["hb_target"] < thr).astype(float).to_numpy())
        key, log, cells = build_strat_key(sub["anemia_class"], sub["ced"], int(config["cv"]["min_cell_count"]))
        if target == "hb":
            y, task = sub["hb_target"].to_numpy(dtype=float), "regression"
        else:
            y, task = y_bin.astype(int), "classification"
        model_key = "hb_model"
    elif target == "ced":
        if ced_cutoff_cm is None:
            label = df["ced"]
        else:
            label = pd.Series(ced_label_at_most(df["muac"], ced_cutoff_cm), index=df.index)
        rows = label.notna()
        sub = df.loc[rows]
        y = label.loc[rows].to_numpy().astype(int)
        y_bin = None
        key, log, cells = build_strat_key(sub["anemia_class"], label.loc[rows], int(config["cv"]["min_cell_count"]))
        task, model_key = "classification", "ced_model"
    else:
        raise ValueError(f"Unknown target '{target}'")

    group_col = config["cv"].get("group_column")
    groups = sub[group_col].astype(str).str.strip().to_numpy() if group_col and group_col in sub.columns else None
    meta.update({"n": int(len(sub)), "cell_counts": {k: int(v) for k, v in cells.items()}})
    return AnalysisData(analysis, target, task, model_key, sub[columns].copy(), y, y_bin, key, log, groups, meta)


# ---------------------------------------------------------------------------
# Thresholds and intervals from inner-fold predictions only
# ---------------------------------------------------------------------------

def cutoff_flagging_share(pred: np.ndarray, truth: np.ndarray, share: float) -> float:
    """Lowest cut-off on predicted Hb (flag when pred < cut-off) that flags ``share`` of true cases."""
    pred = np.asarray(pred, dtype=float)
    pos = np.asarray(truth).astype(bool)
    if not pos.any():
        raise ValueError("No true anemia case in the inner-fold predictions")
    values = np.unique(pred)
    candidates = np.concatenate([(values[:-1] + values[1:]) / 2, [values[-1] + 1e-6]])
    for cut in candidates:
        if (pred[pos] < cut).mean() >= share:
            return float(cut)
    return float(candidates[-1])


def youden_threshold(proba: np.ndarray, truth: np.ndarray) -> float:
    """Probability cut-off maximising sensitivity + specificity - 1. Ties go to the larger cut-off."""
    proba = np.asarray(proba, dtype=float)
    pos = np.asarray(truth).astype(bool)
    best, best_j = 0.5, -np.inf
    for cut in np.unique(proba):
        flag = proba >= cut
        j = flag[pos].mean() + (~flag[~pos]).mean() - 1
        if j > best_j or (j == best_j and cut > best):
            best, best_j = float(cut), j
    return best


def sensitivity_threshold(proba: np.ndarray, truth: np.ndarray, target: float) -> float:
    """Largest probability cut-off whose sensitivity reaches ``target``."""
    proba = np.asarray(proba, dtype=float)
    pos = np.asarray(truth).astype(bool)
    for cut in np.sort(np.unique(proba))[::-1]:
        if (proba[pos] >= cut).mean() >= target:
            return float(cut)
    return float(np.min(proba))


def conformal_half_width(abs_residuals: np.ndarray, nominal: float) -> float:
    """Split-conformal quantile: the ceil((n + 1) * nominal)-th smallest absolute residual."""
    r = np.sort(np.abs(np.asarray(abs_residuals, dtype=float)))
    n = len(r)
    return float(r[min(math.ceil((n + 1) * nominal), n) - 1])


# ---------------------------------------------------------------------------
# Fit spy (canary 2 and tests)
# ---------------------------------------------------------------------------

class FitSpy(BaseEstimator, TransformerMixin):
    """First pipeline step that records the anonymous IDs of every fit call."""

    def __init__(self, log: list | None = None, stage: str = "") -> None:
        self.log = log
        self.stage = stage

    def fit(self, X: pd.DataFrame, y: Any = None) -> FitSpy:
        if self.log is not None:
            self.log.append((self.stage, frozenset(X.index)))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


# ---------------------------------------------------------------------------
# One outer fold
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """Everything that defines one nested run besides the data."""

    spec: ModelSpec
    variant: str = "main"             # main or the name of an exploratory variant
    resampler: str = "none"           # none | smogn | smotenc (exploratory only)
    feature_mode: str = "per_fold"
    fixed_features: tuple[str, ...] = ()
    n_trials: int = 30
    outer: str = "stratified"         # stratified | group
    n_repeats: int = 5


def _make_resampler(name: str, config: Mapping[str, Any]) -> Any:
    if name == "none":
        return None
    if name == "smogn":
        return make_smogn(dict(config))
    if name == "smotenc":
        return make_smotenc(dict(config))
    raise ValueError(f"Unknown resampler '{name}'")


def make_pipeline_factory(run: RunSpec, config: Mapping[str, Any], spy_log: list | None = None,
                          spy_stage: str = "") -> Callable[[Mapping[str, Any]], Any]:
    seed = int(config["project"]["random_seed"])
    fs_task = "classification" if run.spec.task == "classification" else "regression"

    def make(params: Mapping[str, Any]) -> Any:
        pipe = build_pipeline(config, run.spec.model_key, run.spec.factory(params, seed, config), task=fs_task,
                              scale=run.spec.scale, resampler=_make_resampler(run.resampler, config),
                              feature_mode=run.feature_mode, fixed_features=run.fixed_features or None)
        if spy_log is not None:
            pipe.steps.insert(0, ("spy", FitSpy(spy_log, spy_stage)))
        return pipe

    return make


def _predict(pipe: Any, X: pd.DataFrame, task: str) -> np.ndarray:
    return pipe.predict(X) if task == "regression" else pipe.predict_proba(X)[:, 1]


def _inner_score(task: str, y: np.ndarray, pred: np.ndarray) -> float:
    if task == "regression":
        return float(mean_absolute_error(y, pred))
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, pred))


def tune_inner(run: RunSpec, data: AnalysisData, config: Mapping[str, Any], train_idx: np.ndarray, seed: int,
               spy_log: list | None = None) -> tuple[dict[str, Any], np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Tune on inner folds of the outer training set. Returns best params and inner-fold predictions.

    Inner predictions are out-of-fold within the outer training set: each
    row is predicted by a model fitted on the other inner folds only.
    """
    X_tr, y_tr = data.X.iloc[train_idx], data.y[train_idx]
    splits = inner_splits(data.strat_key.iloc[train_idx].to_numpy(), int(config["cv"]["inner_n_splits"]), seed)
    for a, b in splits:
        assert_disjoint(a, b, "inner fold")
    task = run.spec.task

    def evaluate(params: Mapping[str, Any], stage: str) -> tuple[float, np.ndarray]:
        oof = np.full(len(train_idx), np.nan)
        scores = []
        for k, (a, b) in enumerate(splits):
            make = make_pipeline_factory(run, config, spy_log, f"{stage}:inner{k}")
            pipe = make(params).fit(X_tr.iloc[a], y_tr[a])
            oof[b] = _predict(pipe, X_tr.iloc[b], task)
            scores.append(_inner_score(task, y_tr[b], oof[b]))
        return float(np.nanmean(scores)), oof

    if run.spec.space is None:
        _, oof = evaluate({}, "untuned")
        return {}, oof, splits

    space = config["tuning"]["search_spaces"][run.spec.space]
    oofs: dict[int, np.ndarray] = {}

    def objective(trial: optuna.Trial) -> float:
        value, oof = evaluate(suggest_params(trial, space), f"trial{trial.number}")
        oofs[trial.number] = oof
        return value

    study = optuna.create_study(direction="minimize" if task == "regression" else "maximize", sampler=make_sampler(seed))
    study.optimize(objective, n_trials=run.n_trials, show_progress_bar=False)
    return dict(study.best_params), oofs[study.best_trial.number], splits


def run_fold(run: RunSpec, data: AnalysisData, config: Mapping[str, Any], repeat: int, fold: int,
             train_idx: np.ndarray, test_idx: np.ndarray, spy: bool = False) -> dict[str, Any]:
    """Tune, set thresholds from inner predictions, refit on the outer training set and score the test fold."""
    assert_disjoint(train_idx, test_idx, f"outer fold r{repeat} f{fold}")
    seed = int(config["project"]["random_seed"]) + 1000 * repeat + fold
    spy_log: list | None = [] if spy else None
    started = time.perf_counter()
    params, inner_pred, splits = tune_inner(run, data, config, train_idx, seed, spy_log)
    tuning_seconds = time.perf_counter() - started

    train_ids = data.X.index[train_idx]
    test_ids = data.X.index[test_idx]
    inner_ids = train_ids  # inner predictions cover exactly the outer training rows
    if set(inner_ids) & set(test_ids):
        raise ThresholdLeakageError("Inner predictions include outer test rows")
    if np.isnan(inner_pred).any():
        raise ThresholdLeakageError("Some outer training rows have no inner out-of-fold prediction")

    record: dict[str, Any] = {"repeat": repeat, "fold": fold, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                              "params": params, "inner_seed": seed}
    th = config["thresholds"]
    if data.target == "hb":
        truth_inner = data.y_binary[train_idx].astype(bool)
        record["hb_cutoff"] = cutoff_flagging_share(inner_pred, truth_inner, float(th["hb_flag_share_of_cases"]))
        residuals = np.abs(data.y[train_idx] - inner_pred)
        record["interval_half_width"] = conformal_half_width(residuals, float(config["intervals"]["nominal"]))
    else:
        truth_inner = data.y[train_idx].astype(bool)
        record["threshold_youden"] = youden_threshold(inner_pred, truth_inner)
        for target in th["ced_extra_sensitivity_targets"]:
            record[f"threshold_sens_{int(round(target * 100))}"] = sensitivity_threshold(inner_pred, truth_inner, target)
    record["threshold_source"] = "inner_out_of_fold"
    record["threshold_source_ids_hash"] = hashlib.sha256("|".join(sorted(map(str, inner_ids))).encode()).hexdigest()

    make = make_pipeline_factory(run, config, spy_log, "outer_final")
    started = time.perf_counter()
    pipe = make(params).fit(data.X.iloc[train_idx], data.y[train_idx])
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    pred = _predict(pipe, data.X.iloc[test_idx], data.task)
    predict_seconds = time.perf_counter() - started
    # Wall-clock times are descriptive only: they vary between runs and never enter a decision.
    record.update({"tuning_seconds": tuning_seconds, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                   "predict_seconds_per_row": predict_seconds / len(test_idx)})
    steps = pipe.named_steps
    record["n_tree_nodes"] = tree_node_count(steps["model"])  # deterministic inference cost proxy
    record["selected_features"] = (list(steps["select"].selected_features_) if "select" in steps
                                   else list(steps["columns"].columns))

    out = pd.DataFrame({"anon_id": test_ids, "repeat": repeat, "fold": fold, "y_true": data.y[test_idx]})
    if data.target == "hb":
        out["pred"] = pred
        out["y_binary"] = data.y_binary[test_idx]
        out["flag"] = (pred < record["hb_cutoff"]).astype(int)
        out["interval_lo"] = pred - record["interval_half_width"]
        out["interval_hi"] = pred + record["interval_half_width"]
    else:
        out["proba"] = pred
        out["flag"] = (pred >= record["threshold_youden"]).astype(int)
        for target in th["ced_extra_sensitivity_targets"]:
            name = f"threshold_sens_{int(round(target * 100))}"
            out[f"flag_sens_{int(round(target * 100))}"] = (pred >= record[name]).astype(int)
    if data.groups is not None:
        out["group"] = data.groups[test_idx]
    record["oof"] = out

    if spy_log is not None:
        record["spy"] = _check_spy(spy_log, set(train_ids), set(test_ids), splits, train_ids)
    return record


def _check_spy(log: list, train_ids: set, test_ids: set, splits: list, train_index: pd.Index) -> dict[str, Any]:
    """Every fit saw training rows only. Inner fits saw only their inner training split."""
    inner_train = [set(train_index[a]) for a, _ in splits]
    n_fits = 0
    for stage, ids in log:
        n_fits += 1
        if ids & test_ids:
            raise ThresholdLeakageError(f"{stage}: a fit saw outer test rows")
        if stage == "outer_final" and set(ids) != train_ids:
            raise ThresholdLeakageError("Final outer fit did not use exactly the outer training rows")
        if ":inner" in stage:
            k = int(stage.rsplit("inner", 1)[1])
            if not set(ids) <= inner_train[k]:
                raise ThresholdLeakageError(f"{stage}: an inner fit saw rows outside its inner training split")
    return {"n_fits_checked": n_fits, "ok": True}


# ---------------------------------------------------------------------------
# Whole nested run with checkpoints, parallel folds and progress
# ---------------------------------------------------------------------------

def run_hash(config: Mapping[str, Any]) -> str:
    """Hash of everything that can change a result: config, hypotheses, raw data and source code."""
    cfg = json.loads(json.dumps(config, default=str))
    cfg.get("runtime", {}).pop("n_jobs", None)  # parallelism does not change results
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    for path in (resolve_path(config, "hypotheses"), resolve_path(config, "raw_data")):
        digest.update(file_fingerprint(path).encode())
    for src in sorted((PROJECT_ROOT / "src").glob("*.py")):
        digest.update(src.read_bytes())
    return digest.hexdigest()[:16]


@dataclass
class NestedResult:
    """Fold records and pooled out-of-fold predictions of one nested run."""

    analysis: str
    target: str
    model: str
    variant: str
    folds: list[dict[str, Any]]
    oof: pd.DataFrame


def outer_splitter(run: RunSpec, data: AnalysisData, config: Mapping[str, Any]) -> Any:
    if run.outer == "group":
        if data.groups is None:
            raise ValueError("Grouped CV needs a group column")
        return BoundGroupKFold(data.groups, int(config["cv"]["outer_n_splits"]))
    return KeyStratifiedRepeatedKFold(data.strat_key.to_numpy(), int(config["cv"]["outer_n_splits"]), run.n_repeats,
                                      int(config["project"]["random_seed"]))


def _fold_job(run: RunSpec, data: AnalysisData, config: Mapping[str, Any], repeat: int, fold: int,
              train_idx: np.ndarray, test_idx: np.ndarray, path: Path, spy: bool) -> dict[str, Any]:
    record = run_fold(run, data, config, repeat, fold, train_idx, test_idx, spy)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(record, path)
    return record


def run_nested(run: RunSpec, data: AnalysisData, config: Mapping[str, Any], *, spy: bool = False,
               n_jobs: int | None = None, label: str | None = None) -> NestedResult:
    """Run (or resume) every outer fold of one nested analysis."""
    splitter = outer_splitter(run, data, config)
    n_splits = int(getattr(splitter, "n_splits"))
    folds = list(splitter.split(data.X))
    tag = label or f"{data.analysis}__{data.target}__{run.spec.name}__{run.variant}"
    base = resolve_path(config, "results_dir") / "checkpoints" / run_hash(config) / tag
    paths = [base / f"r{i // n_splits}_f{i % n_splits}.joblib" for i in range(len(folds))]

    records: dict[int, dict[str, Any]] = {}
    for i, path in enumerate(paths):
        if path.exists():
            records[i] = joblib.load(path)
    todo = [i for i in range(len(folds)) if i not in records]
    if records:
        LOGGER.info("[%s] resuming: %d of %d folds already done", tag, len(records), len(folds))

    start = time.time()
    jobs = n_jobs if n_jobs is not None else int(config["runtime"]["n_jobs"])
    gen = joblib.Parallel(n_jobs=jobs, return_as="generator")(
        joblib.delayed(_fold_job)(run, data, config, i // n_splits, i % n_splits, folds[i][0], folds[i][1], paths[i], spy)
        for i in todo
    )
    for done, record in enumerate(gen, start=1):
        i = record["repeat"] * n_splits + record["fold"]
        records[i] = record
        elapsed = time.time() - start
        eta = elapsed / done * (len(todo) - done)
        LOGGER.info("[%s] fold %d/%d done (r%d f%d), elapsed %.0fs, about %.0fs left",
                    tag, len(records), len(folds), record["repeat"], record["fold"], elapsed, eta)
        rep = record["repeat"]
        rep_ids = range(rep * n_splits, (rep + 1) * n_splits)
        if all(j in records for j in rep_ids):
            rep_path = base / f"repetition_{rep}_oof.csv"
            if not rep_path.exists():
                pd.concat([records[j]["oof"] for j in rep_ids]).to_csv(rep_path, index=False)

    ordered = [records[i] for i in range(len(folds))]
    oof = pd.concat([r["oof"] for r in ordered], ignore_index=True)
    for rep, block in oof.groupby("repeat"):
        if block["anon_id"].duplicated().any() or len(block) != len(data.y):
            raise ThresholdLeakageError(f"[{tag}] repetition {rep} does not predict every woman exactly once")
    oof.insert(0, "variant", run.variant)
    oof.insert(0, "model", run.spec.name)
    oof.insert(0, "target", data.target)
    oof.insert(0, "analysis", data.analysis)
    return NestedResult(data.analysis, data.target, run.spec.name, run.variant, ordered, oof)


def fold_table(result: NestedResult) -> pd.DataFrame:
    """One row per outer fold: sizes, parameters, thresholds and the selected features."""
    rows = []
    for r in result.folds:
        row = {k: v for k, v in r.items() if k not in ("oof", "spy", "params", "selected_features")}
        row["params"] = json.dumps(r["params"], sort_keys=True)
        row["selected_features"] = "|".join(r["selected_features"])
        rows.append(row)
    table = pd.DataFrame(rows)
    for key, value in (("variant", result.variant), ("model", result.model), ("target", result.target),
                       ("analysis", result.analysis)):
        table.insert(0, key, value)
    return table


def selected_per_fold(result: NestedResult) -> list[list[str]]:
    return [r["selected_features"] for r in result.folds]


def input_columns(config: Mapping[str, Any], run: RunSpec, model_key: str) -> list[str]:
    return model_input_columns(config, model_key, run.feature_mode, run.fixed_features or None)


def describe_strata(data: AnalysisData) -> pd.DataFrame:
    """Final stratification cell counts and the merge log for the report."""
    cells = pd.DataFrame({"cell": list(data.meta["cell_counts"]), "n": list(data.meta["cell_counts"].values())})
    cells.insert(0, "target", data.target)
    cells.insert(0, "analysis", data.analysis)
    return cells


def label_counts(values: Sequence[Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in pd.Series(list(values)).value_counts().items()}
