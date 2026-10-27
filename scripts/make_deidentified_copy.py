"""Make a de-identified copy of the raw questionnaire export.

Removes direct identifiers (name, phone or WhatsApp number, email, enumerator
name, GPS coordinates) and empty unnamed columns. Then scans every remaining
cell for text that looks like a phone number (8 or more digits, optionally
with + or spaces) or an email address, and replaces only that text with
"[REMOVED]". The original file is not modified or deleted.

The script then checks that the cleaned analysis frame built from the copy is
identical to the one built from the original, so the copy can replace the
original without changing any model input.

Usage (from the project root):
    python scripts/make_deidentified_copy.py
"""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaning import clean_dataset  # noqa: E402
from src.utils import load_config  # noqa: E402

SOURCE = ROOT / "data" / "raw" / "dataset.csv"
TARGET = ROOT / "data" / "raw" / "dataset_deidentified.csv"
DROP = [
    "Nama Lengkap",
    "No. HP/WhatsApp",
    "Email Address",
    "Nama Enumetor",
    "Latitude",
    "Longitude",
    "Titik Koordinat GPS (Longitude)",
    "Titik Koordinat GPS (Latitude)",
]
PHONE = re.compile(r"(?:\+?\d[\d\s\-]{7,}\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def main() -> None:
    config = load_config()
    sep, enc = config["data"]["csv_sep"], config["data"]["csv_encoding"]
    raw = pd.read_csv(SOURCE, sep=sep, encoding=enc, dtype=str, keep_default_na=False)

    present = [c for c in DROP if c in raw.columns]
    empty_unnamed = [c for c in raw.columns if c.startswith("Unnamed") and (raw[c].str.strip() == "").all()]
    out = raw.drop(columns=present + empty_unnamed)

    used = set(config["columns"].values())
    masked: dict[str, int] = {}
    for col in out.columns:
        values = out[col].astype(str)
        hit = values.str.contains(PHONE) | values.str.contains(EMAIL)
        if col in used:
            # Model inputs are checked separately below. Dates and codes are short, so a
            # hit here would mean a real problem and stops the script.
            if hit.any():
                raise SystemExit(f"Identifier-like text found in a model input column: {col!r} ({int(hit.sum())} cells)")
            continue
        if hit.any():
            out.loc[hit, col] = values[hit].str.replace(PHONE, "[REMOVED]", regex=True).str.replace(EMAIL, "[REMOVED]", regex=True)
            masked[col] = int(hit.sum())

    out.to_csv(TARGET, sep=sep, encoding=enc, index=False)

    original = clean_dataset(config).frame
    cfg = copy.deepcopy(config)
    cfg["paths"]["raw_data"] = str(TARGET)
    deidentified = clean_dataset(cfg).frame
    pd.testing.assert_frame_equal(original, deidentified)

    print(f"Written: {TARGET.relative_to(ROOT)}")
    print(f"Rows {len(out)}, columns {raw.shape[1]} -> {out.shape[1]}")
    print(f"Dropped identifier columns: {present}")
    print(f"Dropped empty unnamed columns: {empty_unnamed}")
    print(f"Cells masked in free-text columns (column: count): {masked or 'none'}")
    print("Check passed: the cleaned analysis data from the copy is identical to the original.")


if __name__ == "__main__":
    main()
