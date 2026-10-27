"""Skrining risiko KEK dan Hb ibu hamil (prototipe).

Three layers (proposal 3.7):
* presentation: this file and app/components (form, panels),
* logic: src/postmodel/inference.py and recommendations.py (validation, the saved
  pipelines, SHAP explanation, advice),
* data: src/postmodel/db.py (SQLite, anonymous sessions only).

Run from the project root:  streamlit run app/Skrining.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Must run before any project import. `streamlit run` only puts app/ on sys.path,
# so without the project root `import src` fails.
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for _p in (str(ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Skrining", page_icon=":material/monitor_heart:", layout="centered")

from components.form import questionnaire  # noqa: E402
from components.results import results_tabs  # noqa: E402
from components.state import anonymous_user_id, footer, get_service, get_store  # noqa: E402

st.title("Skrining risiko KEK ibu hamil")
st.caption("Prototipe penelitian. Hasil utama adalah perkiraan risiko KEK. Perkiraan Hb hanya indikatif.")

service = get_service()
store = get_store()

answers = questionnaire(service, st.session_state.get("last_answers"))
if answers is not None:
    errors = service.validate(answers)
    if errors:
        st.error("Mohon periksa kembali isian berikut:\n\n" + "\n".join(f"- {e}" for e in errors))
        st.session_state.pop("result", None)
    else:
        st.session_state["last_answers"] = answers
        st.session_state["result"] = service.predict(answers)
        st.session_state.pop("saved_id", None)

result = st.session_state.get("result")
if result is not None:
    results_tabs(result)
    if st.session_state.get("saved_id"):
        code = store.session_code(st.session_state["saved_id"])
        st.success(f"Sesi tersimpan dengan kode **{code}**. Catat kode ini di buku register posyandu. "
                   "Aplikasi tidak menyimpan nama ibu.")
    elif st.button("Simpan sesi", use_container_width=True):
        inputs, prediction = service.storage_records(st.session_state["last_answers"], result)
        st.session_state["saved_id"] = store.create(inputs, prediction, anonymous_user_id())
        st.rerun()

footer()
