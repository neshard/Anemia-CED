"""Shared start-up for experiment scripts.

Loads the config, fixes seeds, logs library versions, validates feature lists
and hypotheses, enforces the hypothesis lock, prints the exact protocol
settings and opens a timestamped log file in results/logs/.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import (  # noqa: E402
    assert_feature_lists_safe,
    load_config,
    log_library_versions,
    require_locked_hypotheses,
    resolve_path,
    set_global_seeds,
)


def protocol_summary(config: dict[str, Any], versions: dict[str, str]) -> list[str]:
    cv, tu, fs = config["cv"], config["tuning"], config["feature_selection"]
    return [
        f"seed={config['project']['random_seed']}",
        f"outer CV={cv['outer_n_splits']} folds x {cv['outer_n_repeats']} repeats (stratified on anemia class x CED)",
        f"inner CV={cv['inner_n_splits']} folds (tuning and thresholds), min stratum={cv['min_cell_count']}",
        f"tuning=Optuna {tu['sampler']}, {tu['n_trials']} trials for every tuned model, objective={tu['objective']}",
        f"features=per fold, forced {fs['forced_include']} plus best of the rest, k={fs['k']}, RF {fs['rf_n_estimators']} trees",
        f"thresholds: Hb flags {config['thresholds']['hb_flag_share_of_cases']:.0%} of true cases, "
        f"CED {config['thresholds']['ced_rule']} plus sensitivity {config['thresholds']['ced_extra_sensitivity_targets']}",
        f"bootstrap={config['evaluation']['n_bootstrap']} resamples per repetition, CI={config['evaluation']['ci_level']}",
        "versions: " + ", ".join(f"{k} {v}" for k, v in versions.items() if k != "platform"),
    ]


def start(script: str, require_hypotheses: bool = True) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Prepare a run and return (config, hypotheses)."""
    config = load_config()
    set_global_seeds(int(config["project"]["random_seed"]))
    versions = log_library_versions(config)
    assert_feature_lists_safe(config)
    hypotheses = require_locked_hypotheses(config) if require_hypotheses else None
    if hypotheses is not None:
        from src.hypotheses import validate

        validate(hypotheses)

    log_dir = resolve_path(config, "results_dir") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    handler = logging.FileHandler(log_dir / f"{stamp}_{script}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in ("src", "anemia_ced"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    print(f"=== {script} | config used ===")
    for line in protocol_summary(config, versions):
        print(f"  {line}")
        logging.getLogger("anemia_ced").info(line)
    if config["project"].get("smoke_test"):
        print("SMOKE TEST on synthetic data. Numbers are meaningless.")
    return config, hypotheses
