"""Questionnaire form in two columns, HFIAS in an expander (presentation layer). No field for Hb or MUAC."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.postmodel.inference import HFIAS_FIELDS, ScreeningService, median_answers

HELP = {
    "age": "Umur ibu saat ini dalam tahun.",
    "prepreg_weight": "Berat badan sebelum hamil menurut ingatan ibu atau catatan buku KIA. Ini data yang dilaporkan sendiri.",
    "height": "Tinggi badan ibu dalam cm.",
    "anc_visits": "Jumlah kunjungan pemeriksaan kehamilan sejauh ini, sesuai buku KIA bila ada. Ini data yang dilaporkan sendiri.",
    "n_children": "Jumlah anak yang masih hidup.",
}
LABELS = {
    "age": "Umur ibu (tahun)",
    "prepreg_weight": "Berat badan sebelum hamil (kg)",
    "height": "Tinggi badan (cm)",
    "anc_visits": "Kunjungan pemeriksaan kehamilan (ANC)",
    "n_children": "Jumlah anak",
}
LEFT = ("age", "height", "prepreg_weight")
RIGHT = ("anc_visits", "n_children")


def _number(service: ScreeningService, field: str, default: Any) -> float:
    low, high = service.numeric_range(field)
    return st.number_input(LABELS[field], min_value=low, max_value=high, value=float(default), step=1.0,
                           help=HELP[field], format="%.0f")


def questionnaire(service: ScreeningService, defaults: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Render the form. Returns the answers when submitted, otherwise None."""
    d = defaults or median_answers(service)
    answers: dict[str, Any] = {}
    with st.form("kuesioner"):
        st.markdown("**Data ibu**")
        left, right = st.columns(2)
        with left:
            for f in LEFT:
                answers[f] = _number(service, f, d[f])
        with right:
            for f in RIGHT:
                answers[f] = _number(service, f, d[f])
            edu = service.choices("education_mother")
            answers["education_mother"] = st.selectbox("Pendidikan terakhir ibu", edu,
                                                       index=edu.index(d["education_mother"]))
        egg = service.choices("egg_freq")
        answers["egg_freq"] = st.selectbox("Seberapa sering ibu makan telur ayam dalam 1 bulan terakhir", egg,
                                           index=egg.index(d["egg_freq"]))
        with st.expander("Ketersediaan pangan rumah tangga, 4 minggu terakhir (9 pertanyaan HFIAS)"):
            st.caption("Pilih seberapa sering kejadian itu dialami rumah tangga. Jawaban awal adalah \"Tidak\".")
            options = service.choices(HFIAS_FIELDS[0])
            for i, (f, question) in enumerate(service.hfias_questions().items(), start=1):
                answers[f] = st.selectbox(f"{i}. {question}", options, index=options.index(d[f]), key=f)
        submitted = st.form_submit_button("Hitung perkiraan", type="primary", use_container_width=True)
    return answers if submitted else None
