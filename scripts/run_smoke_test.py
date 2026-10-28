"""End-to-end smoke run of the experiment scripts on SYNTHETIC data.

Purpose: prove that scripts 00, 02 to 09 and 12 to 14 run from start to finish
on data/synthetic/synthetic_dataset.csv without producing any real result. Running them on the real data would show test
results before config/hypotheses.yaml is locked.

Everything (fake CSV, config copy, split, artifacts, tables, figures) goes
into the output directory given on the command line, so the project's own
reports/ and artifacts/ are not touched.

Usage (from the project root):
    python scripts/run_smoke_test.py <output_dir>
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402
from tests.conftest import synthetic_raw_frame  # noqa: E402

SCRIPTS = [
    "00_prepare_data.py",
    "02_feature_selection_exploratory.py",
    "03_canaries.py",
    "04_main_hb.py",
    "05_main_ced.py",
    "06_exploratory.py",
    "07_sensitivity.py",
    "08_evaluation_and_report.py",
    "12_boosting_comparison.py",
    "13_shap_boosting.py",
    "14_select_final_model.py",
    "09_final_models.py",
]
# Fictitious records in the raw export format, shipped with the code (no real woman).
SYNTHETIC = ROOT / "data" / "synthetic" / "synthetic_dataset.csv"


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_config(ROOT / "config" / "config.yaml")
    cfg = copy.deepcopy(base)

    raw = SYNTHETIC
    if not raw.exists():  # regenerate the shipped file (same generator, size and seed)
        raw.parent.mkdir(parents=True, exist_ok=True)
        synthetic_raw_frame(cfg, n=400, seed=11).to_csv(raw, sep=cfg["data"]["csv_sep"], encoding=cfg["data"]["csv_encoding"],
                                                       index=False)

    cfg["project"]["smoke_test"] = True
    cfg["data"]["expected_row_count"] = None
    cfg["data"]["expected_counts"] = None
    cfg["paths"].update(
        raw_data=str(raw),
        artifacts_dir=str(out_dir / "artifacts"),
        reports_tables=str(out_dir / "tables"),
        reports_figures=str(out_dir / "figures"),
        versions_file=str(out_dir / "library_versions.json"),
        results_dir=str(out_dir / "results"),
        hypotheses=str(out_dir / "smoke_hypotheses.yaml"),
    )
    hyp = yaml.safe_load((ROOT / "config" / "hypotheses.yaml").read_text(encoding="utf-8"))
    hyp["n_bootstrap"] = 60  # speed only. The real file is never modified.
    (out_dir / "smoke_hypotheses.yaml").write_text(yaml.safe_dump(hyp, allow_unicode=True, sort_keys=False), encoding="utf-8")
    cfg["tuning"]["n_trials"] = 2
    cfg["cv"]["outer_n_repeats"] = 2
    cfg["evaluation"]["n_bootstrap"] = 60
    cfg["feature_selection"]["rf_n_estimators"] = 20
    cfg["canaries"]["n_trials"] = 2
    cfg["runtime"]["n_jobs"] = 8
    config_path = out_dir / "smoke_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    env = {**os.environ, "ANEMIA_CED_CONFIG": str(config_path)}
    for script in SCRIPTS:
        print(f"=== {script}", flush=True)
        result = subprocess.run([sys.executable, str(ROOT / "experiments" / script)], env=env, cwd=ROOT,
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-3000:])
            print(result.stderr[-5000:])
            raise SystemExit(f"{script} failed")
        print(f"ok ({len(result.stdout.splitlines())} lines of output)")
    print(f"Smoke test passed. Outputs in {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/run_smoke_test.py <output_dir>")
    main(Path(sys.argv[1]))
