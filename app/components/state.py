"""Shared resources for the Streamlit pages: path setup, cached service and session store."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postmodel.db import SessionStore  # noqa: E402
from src.postmodel.inference import ScreeningService  # noqa: E402

import os  # noqa: E402

DEFAULT_DB_PATH = ROOT / "data" / "app" / "sessions.sqlite"


def db_path() -> Path:
    """Database location, read at every call (not frozen at import time).

    ANEMIA_CED_APP_DB lets tests point the app at a temporary database. Reading it at
    call time matters: a module imported earlier in the same process must not keep
    pointing at the real database.
    """
    return Path(os.environ.get("ANEMIA_CED_APP_DB", DEFAULT_DB_PATH))
FOOTER = ("Alat ini adalah prototipe untuk membantu skrining awal, bukan diagnosis. Hasilnya tidak menggantikan "
          "pengukuran LILA atau pemeriksaan Hb. Data yang disimpan hanya berupa jawaban kuesioner dan kode sesi "
          "anonim, tanpa nama atau nomor telepon.")


@st.cache_resource(show_spinner="Memuat model...")
def get_service() -> ScreeningService:
    """The exact pipelines and explainer saved by the training scripts, loaded once."""
    return ScreeningService(ROOT)


@st.cache_resource
def _open_store(path: str) -> SessionStore:
    return SessionStore(path)


def get_store() -> SessionStore:
    """One cached store per database path."""
    return _open_store(str(db_path()))


def anonymous_user_id() -> str:
    """Random identifier for this browser session. It identifies no person."""
    if "anonymous_user_id" not in st.session_state:
        st.session_state["anonymous_user_id"] = uuid.uuid4().hex[:12]
    return st.session_state["anonymous_user_id"]


def footer(service: ScreeningService | None = None) -> None:
    """Disclaimer, plus the algorithm behind each estimate when the service is given."""
    st.divider()
    text = FOOTER
    if service is not None:
        text += (f" Model yang dipakai: KEK {service.algorithm_label('ced')}, Hb {service.algorithm_label('hb')}, "
                 "dipilih dengan aturan yang ditetapkan sebelum hasil perbandingan dihitung.")
    st.caption(text)
