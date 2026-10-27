"""SQLite storage for dashboard sessions (proposal 3.7, data layer).

Three tables: sessions, inputs and predictions. Only anonymous identifiers are
stored, never names, phone numbers, coordinates or respondent codes. Every
query is parameterized. Deleting a session removes its inputs and prediction
through foreign keys with ON DELETE CASCADE.

Each session also gets a readable code such as 20261021-003 (local date plus a
daily sequence number) that a cadre can write in the paper register. The link
between the code and a woman's name stays on paper, never in the app. A
counter table makes sure a code is never reused, even after a deletion, so a
register entry cannot later point to a different session.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

INPUT_COLUMNS: tuple[str, ...] = (
    "age", "anc_visits", "prepreg_weight", "height", "n_children", "education_mother", "egg_freq",
    "n_under5", "income", "food_taboo",  # asked only when a loaded model uses them
    *(f"hfias_q{i}" for i in range(1, 10)),
)
PREDICTION_COLUMNS: tuple[str, ...] = (
    "predicted_hb", "hb_lower", "hb_upper", "anemia_class", "ced_probability", "ced_label",
    "top5_shap_hb", "top5_shap_ced", "model_note", "hb_algorithm", "ced_algorithm",
)
REAL_INPUTS = ("age", "anc_visits", "prepreg_weight", "height", "n_children", "n_under5")
FORBIDDEN_STORAGE = {"name", "nama", "phone", "hp", "whatsapp", "email", "latitude", "longitude", "respondent_code"}

SCHEMA = f"""
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    anonymous_user_id TEXT NOT NULL,
    session_code TEXT
);
CREATE TABLE IF NOT EXISTS code_counter (
    day TEXT PRIMARY KEY,
    last_number INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS inputs (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    {", ".join(f"{c} {'REAL' if c in REAL_INPUTS else 'TEXT'}" for c in INPUT_COLUMNS)},
    self_reported_fields TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    predicted_hb REAL,
    hb_lower REAL,
    hb_upper REAL,
    anemia_class TEXT,
    ced_probability REAL NOT NULL,
    ced_label INTEGER NOT NULL,
    top5_shap_hb TEXT,
    top5_shap_ced TEXT,
    model_note TEXT,
    hb_algorithm TEXT,
    ced_algorithm TEXT
);
"""

SELF_REPORTED = ["prepreg_weight", "anc_visits"]  # recalled by the mother or read from her own records


class SessionStore:
    """Create, read, update and delete dashboard sessions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)
            columns = [r[1] for r in con.execute("PRAGMA table_info(sessions)").fetchall()]
            if "session_code" not in columns:  # database created before readable codes existed
                con.execute("ALTER TABLE sessions ADD COLUMN session_code TEXT")
            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_session_code ON sessions(session_code)")
            # databases created before optional inputs and algorithm names were stored
            for table, wanted in (("inputs", {c: "REAL" if c in REAL_INPUTS else "TEXT" for c in INPUT_COLUMNS}),
                                  ("predictions", {"hb_algorithm": "TEXT", "ced_algorithm": "TEXT"})):
                have = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
                for col, kind in wanted.items():
                    if col not in have:
                        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {kind}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA foreign_keys = ON")
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    @staticmethod
    def _check(values: Mapping[str, Any], allowed: tuple[str, ...]) -> None:
        unknown = set(values) - set(allowed)
        if unknown:
            raise ValueError(f"Unexpected fields: {sorted(unknown)}")
        if any(any(bad in k.lower() for bad in FORBIDDEN_STORAGE) for k in values):
            raise ValueError("Identifying fields cannot be stored")

    @staticmethod
    def _pred_values(prediction: Mapping[str, Any]) -> list[Any]:
        out = []
        for c in PREDICTION_COLUMNS:
            v = prediction.get(c)
            out.append(json.dumps(v, ensure_ascii=False) if c.startswith("top5") and v is not None else v)
        return out

    @staticmethod
    def _next_code(con: sqlite3.Connection, now: datetime) -> str:
        """Next readable code for the local date of ``now``. Numbers are never reused."""
        day = now.strftime("%Y%m%d")
        row = con.execute("SELECT last_number FROM code_counter WHERE day = ?", (day,)).fetchone()
        number = (int(row[0]) if row else 0) + 1
        con.execute("INSERT INTO code_counter (day, last_number) VALUES (?, ?) "
                    "ON CONFLICT(day) DO UPDATE SET last_number = excluded.last_number", (day, number))
        return f"{day}-{number:03d}"

    def create(self, inputs: Mapping[str, Any], prediction: Mapping[str, Any], anonymous_user_id: str,
               now: datetime | None = None) -> str:
        """Store one session and return its new id. ``now`` (local time) is only set by tests."""
        self._check(inputs, INPUT_COLUMNS)
        self._check(prediction, PREDICTION_COLUMNS)
        session_id = uuid.uuid4().hex
        local_now = now or datetime.now()
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as con:
            code = self._next_code(con, local_now)
            con.execute("INSERT INTO sessions (session_id, timestamp, anonymous_user_id, session_code) "
                        "VALUES (?, ?, ?, ?)", (session_id, stamp, anonymous_user_id, code))
            con.execute(f"INSERT INTO inputs (session_id, {', '.join(INPUT_COLUMNS)}, self_reported_fields) "
                        f"VALUES (?, {', '.join('?' for _ in INPUT_COLUMNS)}, ?)",
                        (session_id, *[inputs.get(c) for c in INPUT_COLUMNS], json.dumps(SELF_REPORTED)))
            con.execute(f"INSERT INTO predictions (session_id, {', '.join(PREDICTION_COLUMNS)}) "
                        f"VALUES (?, {', '.join('?' for _ in PREDICTION_COLUMNS)})",
                        (session_id, *self._pred_values(prediction)))
        return session_id

    def read(self, session_id: str) -> dict[str, Any] | None:
        """One session with its inputs and prediction, or None."""
        with self._connect() as con:
            s = con.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if s is None:
                return None
            i = con.execute("SELECT * FROM inputs WHERE session_id = ?", (session_id,)).fetchone()
            p = con.execute("SELECT * FROM predictions WHERE session_id = ?", (session_id,)).fetchone()
        pred = dict(p) if p else {}
        for c in ("top5_shap_hb", "top5_shap_ced"):
            if pred.get(c):
                pred[c] = json.loads(pred[c])
        return {"session": dict(s), "inputs": dict(i) if i else {}, "prediction": pred}

    def list(self, limit: int = 200) -> pd.DataFrame:
        """Most recent sessions first, summary columns only."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT s.session_id, s.session_code, s.timestamp, p.ced_probability, p.ced_label, p.predicted_hb, "
                "p.anemia_class, p.hb_algorithm, p.ced_algorithm "
                "FROM sessions s LEFT JOIN predictions p ON p.session_id = s.session_id "
                "ORDER BY s.timestamp DESC, s.rowid DESC LIMIT ?", (int(limit),)).fetchall()
        return pd.DataFrame([dict(r) for r in rows],
                            columns=["session_id", "session_code", "timestamp", "ced_probability", "ced_label",
                                     "predicted_hb", "anemia_class", "hb_algorithm", "ced_algorithm"])

    def session_code(self, session_id: str) -> str | None:
        """The readable code of a session, or None."""
        with self._connect() as con:
            row = con.execute("SELECT session_code FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row[0] if row else None

    def update(self, session_id: str, inputs: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
        """Replace the inputs and prediction of an existing session."""
        self._check(inputs, INPUT_COLUMNS)
        self._check(prediction, PREDICTION_COLUMNS)
        with self._connect() as con:
            if con.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is None:
                return False
            con.execute(f"UPDATE inputs SET {', '.join(f'{c} = ?' for c in INPUT_COLUMNS)} WHERE session_id = ?",
                        (*[inputs.get(c) for c in INPUT_COLUMNS], session_id))
            con.execute(f"UPDATE predictions SET {', '.join(f'{c} = ?' for c in PREDICTION_COLUMNS)} WHERE session_id = ?",
                        (*self._pred_values(prediction), session_id))
        return True

    def delete(self, session_id: str) -> bool:
        """Delete a session and, by cascade, its inputs and prediction."""
        with self._connect() as con:
            cur = con.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return cur.rowcount > 0

    def counts(self) -> dict[str, int]:
        with self._connect() as con:
            return {t: int(con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                    for t in ("sessions", "inputs", "predictions")}

    def column_names(self) -> dict[str, list[str]]:
        with self._connect() as con:
            return {t: [r[1] for r in con.execute(f"PRAGMA table_info({t})").fetchall()]
                    for t in ("sessions", "inputs", "predictions")}
