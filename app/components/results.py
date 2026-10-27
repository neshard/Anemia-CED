"""Result tabs: Hasil, Penjelasan, Saran (presentation layer only, no inference logic)."""

from __future__ import annotations

import streamlit as st

from src.postmodel.explain import ALGORITHM_LABELS
from src.postmodel.inference import ScreeningResult

TAB_NAMES = ("Hasil", "Penjelasan", "Saran")
HB_NOT_SHOWN = ("Model Hb tidak terbukti lebih akurat daripada menebak nilai rata-rata. Angka ini tidak ditampilkan "
                "dengan rentang dan tidak dapat diandalkan untuk mendeteksi anemia.")


def _pct(p: float) -> str:
    return f"{100 * p:.0f}%"


def fmt_num(x: float, digits: int = 1) -> str:
    """Indonesian decimal comma."""
    return f"{x:.{digits}f}".replace(".", ",")


def _label(result: ScreeningResult, target: str) -> str:
    name = result.extra.get(f"{target}_algorithm", "xgboost")
    return ALGORITHM_LABELS.get(name, name)


def hasil_tab(result: ScreeningResult) -> None:
    threshold = result.extra["threshold"]
    hb_enabled = bool(result.extra.get("hb_explanation_enabled", False))
    with st.container(border=True):
        st.metric("Peluang risiko KEK (hasil utama)", _pct(result.ced_probability),
                  delta="di atas ambang skrining" if result.ced_label else "di bawah ambang skrining",
                  delta_color="inverse" if result.ced_label else "off")
        st.progress(min(max(result.ced_probability, 0.0), 1.0))
        st.caption(f"Ambang skrining {_pct(threshold)}. Di atas ambang berarti ibu dianjurkan untuk diukur LILA-nya. "
                   f"Model: {_label(result, 'ced')}.")
    with st.container(border=True):
        st.metric("Perkiraan Hb" if hb_enabled else "Perkiraan Hb (hanya indikatif)",
                  f"{fmt_num(result.hb_prediction)} g/dL", help=None if hb_enabled else HB_NOT_SHOWN)
        st.caption(f"Kelas WHO dari perkiraan: {result.hb_class}{'' if hb_enabled else ' (indikatif)'}. "
                   f"Model: {_label(result, 'hb')}.")
        st.warning("Angka Hb ini tidak dapat diandalkan untuk mendeteksi anemia. Pemeriksaan Hb tetap diperlukan.")


def penjelasan_tab(result: ScreeningResult) -> None:
    st.markdown("**Faktor yang memengaruhi perkiraan risiko KEK**")
    st.pyplot(result.ced_figure, use_container_width=True)
    st.write(result.ced_text)
    if result.hb_figure is not None:
        st.markdown("**Faktor yang memengaruhi perkiraan Hb**")
        st.pyplot(result.hb_figure, use_container_width=True)
        st.write(result.hb_text)
    else:
        st.info("Penjelasan faktor untuk perkiraan Hb tidak ditampilkan karena model Hb tidak terbukti lebih akurat "
                "daripada menebak nilai rata-rata.")


def saran_tab(result: ScreeningResult) -> None:
    if result.advice.draft_banner:
        st.warning(result.advice.draft_banner)
    for message in [result.advice.status_message, *result.advice.driver_messages, result.advice.hb_message]:
        st.markdown(f"- {message}")
    st.caption(result.advice.disclaimer)


def results_tabs(result: ScreeningResult) -> None:
    """Render the three result tabs."""
    hasil, penjelasan, saran = st.tabs(list(TAB_NAMES))
    with hasil:
        hasil_tab(result)
    with penjelasan:
        penjelasan_tab(result)
    with saran:
        saran_tab(result)
