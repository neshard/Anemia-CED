"""Write fictitious questionnaire records in the exact format of the raw Google Forms export.

The records are drawn at random (tests/conftest.py synthetic_raw_frame): plausible ages,
heights, weights, Hb, MUAC, answer categories and HFIAS answers, with fake respondent codes
(X-0, X-1, ...). They describe no real woman, and numbers computed from them are meaningless.
They exist so the pipeline and the tests can run without the real data.

The defaults reproduce data/synthetic/synthetic_dataset.csv, the file used by
scripts/run_smoke_test.py, byte for byte.

Usage (from the project root):
    python scripts/make_synthetic_dataset.py [--n 400] [--seed 11] [--out data/synthetic/synthetic_dataset.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402
from tests.conftest import synthetic_raw_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=400, help="number of records (default 400)")
    parser.add_argument("--seed", type=int, default=11, help="random seed (default 11)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic" / "synthetic_dataset.csv")
    args = parser.parse_args()

    config = load_config()
    frame = synthetic_raw_frame(config, n=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, sep=config["data"]["csv_sep"], encoding=config["data"]["csv_encoding"], index=False)
    print(f"Wrote {len(frame)} fictitious records to {args.out}")


if __name__ == "__main__":
    main()
