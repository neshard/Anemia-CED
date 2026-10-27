"""Riwayat sesi: daftar, lihat rincian, hitung ulang (update) dan hapus (CRUD)."""

from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

# Must run before any project import: app/ holds the components package, the root holds src.
APP_DIR = Path(__file__).resolve().parents[1]
for _p in (str(APP_DIR.parent), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Riwayat sesi", page_icon=":material/history:", layout="centered")

from components.results import fmt_num  # noqa: E402
from components.state import footer, get_service, get_store  # noqa: E402

st.title("Riwayat sesi")
store = get_store()
service = get_service()
sessions = store.list()

if sessions.empty:
    st.info("Belum ada sesi yang tersimpan.")
    footer()
    st.stop()

st.caption("Setiap sesi dikenali dengan kode tanggal dan nomor urut. Cocokkan kode dengan buku register posyandu. "
           "Aplikasi tidak menyimpan nama ibu.")
codes = sessions["session_code"].fillna(sessions["session_id"].str[:8])
table = sessions.assign(
    kode=codes,
    waktu=pd.to_datetime(sessions["timestamp"]).dt.strftime("%Y-%m-%d %H:%M"),
    risiko_KEK=(100 * sessions["ced_probability"]).round(0).astype(int).astype(str) + "%",
    di_atas_ambang=sessions["ced_label"].map({1: "ya", 0: "tidak"}),
)[["kode", "waktu", "risiko_KEK", "di_atas_ambang"]]
st.dataframe(table, hide_index=True, use_container_width=True)

labels = dict(zip(sessions["session_id"], codes + "  (" + pd.to_datetime(sessions["timestamp"]).dt.strftime("%H:%M") + ")",
                  strict=True))
choice = st.selectbox("Pilih sesi (kode)", sessions["session_id"], format_func=lambda s: labels[s])
record = store.read(choice)
if record is None:
    st.warning("Sesi tidak ditemukan.")
    st.stop()

st.subheader("Rincian")
st.markdown("**Jawaban kuesioner**")
st.dataframe(pd.DataFrame([{k: v for k, v in record["inputs"].items() if k not in ("session_id", "self_reported_fields")}]).T
             .rename(columns={0: "jawaban"}), use_container_width=True)
pred = record["prediction"]
st.markdown("**Hasil yang tersimpan**")
st.write(f"Peluang risiko KEK: {100 * pred['ced_probability']:.0f}%. "
         f"Perkiraan Hb (indikatif): {fmt_num(pred['predicted_hb'])} g/dL.")
if pred.get("top5_shap_ced"):
    st.dataframe(pd.DataFrame(pred["top5_shap_ced"])[["label", "value_text", "shap"]]
                 .rename(columns={"label": "faktor", "value_text": "jawaban", "shap": "kontribusi (log-odds)"}),
                 hide_index=True, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("Hitung ulang dengan model saat ini", use_container_width=True):
        answers = {k: v for k, v in record["inputs"].items() if k not in ("session_id", "self_reported_fields")}
        result = service.predict(answers)
        inputs, prediction = service.storage_records(answers, result)
        store.update(choice, inputs, prediction)
        st.success("Hasil sesi diperbarui.")
        st.rerun()
with col2:
    confirm = st.checkbox("Saya yakin ingin menghapus sesi ini")
    if st.button("Hapus sesi", disabled=not confirm, use_container_width=True):
        store.delete(choice)
        st.success("Sesi dihapus.")
        st.rerun()

footer()
