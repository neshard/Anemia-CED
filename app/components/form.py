"""Questionnaire form in two columns, HFIAS in an expander (presentation layer). No field for Hb or MUAC.

The fields shown are the union of the two loaded models' feature lists (ScreeningService),
so the form follows the selected models instead of a fixed list.
"""

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
    "n_under5": "Jumlah anak yang berumur di bawah 5 tahun.",
}
LABELS = {
    "age": "Umur ibu (tahun)",
    "prepreg_weight": "Berat badan sebelum hamil (kg)",
    "height": "Tinggi badan (cm)",
    "anc_visits": "Kunjungan pemeriksaan kehamilan (ANC)",
    "n_children": "Jumlah anak",
    "n_under5": "Jumlah anak balita",
}
LEFT = ("age", "height", "prepreg_weight")
RIGHT = ("anc_visits", "n_children", "n_under5")
RIGHT_CHOICES = ("education_mother",)


def _number(service: ScreeningService, field: str, default: Any) -> float:
    low, high = service.numeric_range(field)
    return st.number_input(LABELS[field], min_value=low, max_value=high, value=float(default), step=1.0,
                           help=HELP[field], format="%.0f")


def _choice(service: ScreeningService, field: str, default: Any) -> str:
    options = service.choices(field)
    return st.selectbox(service.choice_fields[field], options, index=options.index(default))


def questionnaire(service: ScreeningService, defaults: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Render the form. Returns the answers when submitted, otherwise None."""
    d = {**median_answers(service), **(defaults or {})}
    answers: dict[str, Any] = {}
    with st.form("kuesioner"):
        st.markdown("**Data ibu**")
        left, right = st.columns(2)
        with left:
            for f in LEFT:
                if f in service.numeric_fields:
                    answers[f] = _number(service, f, d[f])
        with right:
            for f in RIGHT:
                if f in service.numeric_fields:
                    answers[f] = _number(service, f, d[f])
            for f in RIGHT_CHOICES:
                if f in service.choice_fields:
                    answers[f] = _choice(service, f, d[f])
        for f in service.choice_fields:
            if f not in RIGHT_CHOICES:
                answers[f] = _choice(service, f, d[f])
        with st.expander("Ketersediaan pangan rumah tangga, 4 minggu terakhir (9 pertanyaan HFIAS)"):
            st.caption("Pilih seberapa sering kejadian itu dialami rumah tangga. Jawaban awal adalah \"Tidak\".")
            options = service.choices(HFIAS_FIELDS[0])
            for i, (f, question) in enumerate(service.hfias_questions().items(), start=1):
                answers[f] = st.selectbox(f"{i}. {question}", options, index=options.index(d[f]), key=f)
        submitted = st.form_submit_button("Hitung perkiraan", type="primary", use_container_width=True)
    return answers if submitted else None
