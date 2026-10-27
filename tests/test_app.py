"""Phase E: input validation, dashboard equals offline prediction, SQLite CRUD, advice rules, app smoke test."""

from __future__ import annotations

import copy
import json
import shutil
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import shap
import yaml

from src.cleaning import clean_dataset
from src.models import CED_MODELS, HB_MODELS
from src.nested_cv import build_data, make_pipeline_factory
from src.postmodel.db import SessionStore
from src.postmodel.inference import HFIAS_FIELDS, ScreeningService, median_answers
from src.postmodel.recommendations import build_advice
from src.runs import main_run
from src.utils import PROJECT_ROOT
from tests.conftest import synthetic_raw_frame

HB_FEATS = ["age", "anc_visits", "hfias_score", "prepreg_weight", "height", "education_mother", "n_children"]
CED_FEATS = [*HB_FEATS, "egg_freq"]


@pytest.fixture()
def service_root(tmp_path: Path, isolated_config: dict) -> tuple[Path, dict]:
    """Train small final pipelines on synthetic data and lay them out like artifacts/."""
    cfg = copy.deepcopy(isolated_config)
    cfg["feature_selection"]["rf_n_estimators"] = 20
    df = clean_dataset(cfg).frame
    art = tmp_path / "root" / "artifacts"
    art.mkdir(parents=True)
    meta = {}
    for target, spec, feats in (("hb", HB_MODELS["xgboost"], HB_FEATS), ("ced", CED_MODELS["xgboost"], CED_FEATS)):
        data = build_data(cfg, df, target)
        run = main_run(spec, cfg, feature_mode="fixed", fixed_features=tuple(feats))
        pipe = make_pipeline_factory(run, cfg)({"n_estimators": 40, "max_depth": 3}).fit(data.X, data.y)
        joblib.dump(pipe, art / f"final_{target}_xgboost.joblib")
        meta[target] = {"features": feats, "threshold_youden": 0.3, "interval_half_width": 2.0}
    joblib.dump(shap.TreeExplainer(joblib.load(art / "final_ced_xgboost.joblib").named_steps["model"]),
                art / "explainer_ced.joblib")
    (art / "final_models.json").write_text(json.dumps(meta), encoding="utf-8")
    shutil.copytree(PROJECT_ROOT / "config", tmp_path / "root" / "config")
    return tmp_path / "root", cfg


def _answers_from_raw(cfg: dict, n: int = 25) -> tuple[list[dict], pd.DataFrame]:
    raw = synthetic_raw_frame(cfg)  # same generator and seed as the isolated_config CSV
    cols = cfg["columns"]
    answers = []
    for _, r in raw.head(n).iterrows():
        a = {f: r[cols[f]] for f in ("age", "prepreg_weight", "height", "anc_visits", "n_children",
                                     "education_mother", "egg_freq", *HFIAS_FIELDS)}
        a["egg_freq"] = ">=1x/hari" if a["egg_freq"] == "?1x/hari" else a["egg_freq"]  # same code 5
        answers.append(a)
    return answers, raw


def test_dashboard_equals_offline_prediction(service_root) -> None:  # type: ignore[no-untyped-def]
    root, cfg = service_root
    service = ScreeningService(root, cfg)
    answers, _ = _answers_from_raw(cfg)
    offline = clean_dataset(cfg).frame.set_index("anon_id")[cfg["features"]["candidate_pool"]]  # model inputs only
    for i, a in enumerate(answers, start=1):
        assert service.validate(a) == []
        row = offline.loc[[f"P{i:04d}"]]
        X = service.to_frame(a)
        assert float(X["hfias_score"].iloc[0]) == float(row["hfias_score"].iloc[0])  # same HFIAS scoring
        result = service.predict(a)
        np.testing.assert_allclose(result.ced_probability, service.ced_pipe.predict_proba(row)[0, 1], rtol=1e-9)
        np.testing.assert_allclose(result.hb_prediction, service.hb_pipe.predict(row)[0], rtol=1e-9)


def test_food_secure_woman_is_accepted(service_root) -> None:  # type: ignore[no-untyped-def]
    root, cfg = service_root
    service = ScreeningService(root, cfg)
    a = median_answers(service)  # every HFIAS answer is "Tidak"
    assert float(service.to_frame(a)["hfias_score"].iloc[0]) == 0.0
    assert 0 <= service.predict(a).ced_probability <= 1


@pytest.mark.parametrize("field,value", [("age", 3), ("age", 51), ("height", 0), ("height", 256), ("prepreg_weight", 11),
                                         ("prepreg_weight", 130), ("anc_visits", 25), ("anc_visits", 2.5),
                                         ("n_children", -1), ("age", "abc"), ("age", None)])
def test_validation_rejects_out_of_range(service_root, field: str, value: object) -> None:  # type: ignore[no-untyped-def]
    root, cfg = service_root
    service = ScreeningService(root, cfg)
    a = {**median_answers(service), field: value}
    assert service.validate(a)
    with pytest.raises(ValueError):
        service.predict(a)


def test_validation_rejects_unknown_choices(service_root) -> None:  # type: ignore[no-untyped-def]
    root, cfg = service_root
    service = ScreeningService(root, cfg)
    assert service.validate({**median_answers(service), "egg_freq": "setiap jam"})
    assert service.validate({**median_answers(service), "hfias_q3": "mungkin"})


def test_hb_shown_without_range(service_root) -> None:  # type: ignore[no-untyped-def]
    root, cfg = service_root
    service = ScreeningService(root, cfg)
    result = service.predict(median_answers(service))
    assert "rentang" not in result.hb_text and "indikatif" in result.hb_text
    inputs, prediction = service.storage_records(median_answers(service), result)
    assert prediction["hb_lower"] is None and prediction["hb_upper"] is None and prediction["top5_shap_hb"] is None


def test_sqlite_crud_across_sessions(tmp_path: Path, service_root) -> None:  # type: ignore[no-untyped-def]
    root, cfg = service_root
    service = ScreeningService(root, cfg)
    store = SessionStore(tmp_path / "s.sqlite")
    answers, _ = _answers_from_raw(cfg, 3)
    ids = []
    for a in answers:
        inputs, prediction = service.storage_records(a, service.predict(a))
        ids.append(store.create(inputs, prediction, "anon-test"))
    assert len(set(ids)) == 3 and store.counts() == {"sessions": 3, "inputs": 3, "predictions": 3}
    rec = store.read(ids[1])
    assert rec["inputs"]["age"] == float(answers[1]["age"]) and isinstance(rec["prediction"]["top5_shap_ced"], list)
    assert len(store.list()) == 3
    new_inputs, new_pred = service.storage_records(answers[2], service.predict(answers[2]))
    assert store.update(ids[1], new_inputs, new_pred)
    assert store.read(ids[1])["inputs"]["age"] == float(answers[2]["age"])
    assert store.read(ids[0])["inputs"]["age"] == float(answers[0]["age"])  # other sessions untouched
    assert store.delete(ids[0]) and store.read(ids[0]) is None
    assert store.counts() == {"sessions": 2, "inputs": 2, "predictions": 2}  # cascade
    assert not store.delete("does-not-exist") and not store.update("does-not-exist", new_inputs, new_pred)


def test_sqlite_is_parameterized_and_stores_no_identifiers(tmp_path: Path, service_root) -> None:  # type: ignore[no-untyped-def]
    root, cfg = service_root
    service = ScreeningService(root, cfg)
    store = SessionStore(tmp_path / "s.sqlite")
    a = median_answers(service)
    inputs, prediction = service.storage_records(a, service.predict(a))
    nasty = {**inputs, "education_mother": "x'); DROP TABLE sessions; --"}
    sid = store.create(nasty, prediction, "anon")
    assert store.read(sid)["inputs"]["education_mother"] == nasty["education_mother"]
    assert store.counts()["sessions"] == 1
    columns = " ".join(c for cols in store.column_names().values() for c in cols).lower()
    for bad in ("nama", "name", "phone", "whatsapp", "email", "latitude", "longitude", "respondent"):
        assert bad not in columns
    with pytest.raises(ValueError):
        store.create({**inputs, "nama_lengkap": "x"}, prediction, "anon")
    assert sqlite3.connect(tmp_path / "s.sqlite").execute("PRAGMA foreign_keys").fetchone() is not None


def test_advice_rules() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config" / "recommendations.yaml").read_text(encoding="utf-8"))
    contrib = [{"feature": "prepreg_weight", "shap": 0.9}, {"feature": "education_mother", "shap": 0.5},
               {"feature": "anc_visits", "shap": -0.3}, {"feature": "egg_freq", "shap": 0.1},
               {"feature": "hfias_score", "shap": 0.2}, {"feature": "age", "shap": 0.05}]
    advice = build_advice(0.6, 0.3, contrib, cfg)
    assert advice.status_message.startswith("Perkiraan risiko KEK di atas")
    assert len(advice.driver_messages) == 3  # max_driver_messages
    assert advice.driver_messages[0].startswith("Berat badan")  # largest positive first
    assert not any("Jumlah kunjungan" in m for m in advice.driver_messages)  # negative SHAP gives no message
    assert not any("pendidikan" in m.lower() for m in advice.driver_messages)  # no education message by design
    assert advice.disclaimer and advice.draft_banner  # draft until reviewed
    assert build_advice(0.1, 0.3, contrib, cfg).status_message.startswith("Perkiraan risiko KEK di bawah")
    for m in advice.all_messages():
        assert "—" not in m and ";" not in m and "menyebabkan" not in m


REAL_ARTIFACTS = (PROJECT_ROOT / "artifacts" / "final_ced_xgboost.joblib").exists()
needs_real = pytest.mark.skipif(not REAL_ARTIFACTS, reason="final model artifacts not present")


@pytest.mark.parametrize("page", ["Skrining.py", "pages/1_Riwayat_sesi.py"])
@needs_real
def test_pages_import_without_project_root_on_path(tmp_path: Path, page: str) -> None:
    """Regression for ModuleNotFoundError 'src': run each page as `streamlit run` does, in a fresh
    interpreter whose working directory and sys.path do not contain the project root."""
    import subprocess
    import sys

    code = "\n".join([
        "import sys",
        "import os",
        f"root = os.path.normcase({str(PROJECT_ROOT)!r})",
        # drop only the project root itself (the venv site-packages inside it must stay)
        "sys.path = [p for p in sys.path if os.path.normcase(os.path.abspath(p or '.')) != root]",
        "assert all(os.path.normcase(os.path.abspath(p or '.')) != root for p in sys.path)",
        "from streamlit.testing.v1 import AppTest",
        f"at = AppTest.from_file({str(PROJECT_ROOT / 'app' / page)!r}, default_timeout=120).run()",
        "print('EXCEPTION' if at.exception else 'OK')",
        "print([e.message for e in at.exception])",
    ])
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert "OK" in out.stdout.splitlines(), out.stdout + out.stderr[-2000:]


@needs_real
def test_skrining_page_layout_and_result() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "Skrining.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.columns) >= 2  # two-column form
    assert any("HFIAS" in e.label for e in at.expander)  # HFIAS questions in an expander
    assert len(at.selectbox) == 11  # education, egg and the nine HFIAS items
    at.button[0].click().run()  # form submit
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Hasil", "Penjelasan", "Saran"]
    labels = [m.label for m in at.metric]
    assert "Peluang risiko KEK (hasil utama)" in labels and "Perkiraan Hb (hanya indikatif)" in labels
    text = " ".join(str(x.value) for x in [*at.markdown, *at.caption, *at.warning, *at.info])
    assert "rentang" not in text  # option C: no Hb range anywhere
    assert "bukan penyebab" in " ".join(str(x.value) for x in at.markdown)  # non-causal narrative shown


@needs_real
def test_dashboard_shap_equals_direct_tree_explainer() -> None:
    service = ScreeningService()
    a = median_answers(service)
    result = service.predict(a)
    X = service.to_frame(a)
    model = service.ced_pipe.named_steps["model"]
    from src.postmodel.explain import model_matrix

    direct = shap.TreeExplainer(model).shap_values(model_matrix(service.ced_pipe, X))[0]
    direct = dict(zip(model_matrix(service.ced_pipe, X).columns, direct, strict=True))
    for c in result.ced_top:
        assert c["shap"] == pytest.approx(float(direct[c["feature"]]), abs=1e-9)


@needs_real
def test_advice_for_manually_built_risk_profiles() -> None:
    """Hand-built profiles: a thin, young mother with one ANC visit versus a heavier mother with many visits."""
    service = ScreeningService()
    base = median_answers(service)
    high = service.predict({**base, "prepreg_weight": 38, "age": 18, "anc_visits": 1})
    low = service.predict({**base, "prepreg_weight": 75, "age": 32, "anc_visits": 8})
    assert high.ced_probability > low.ced_probability
    assert high.ced_label == 1 and high.advice.status_message.startswith("Perkiraan risiko KEK di atas")
    assert low.ced_label == 0 and low.advice.status_message.startswith("Perkiraan risiko KEK di bawah")
    assert any(m.startswith("Berat badan sebelum hamil") for m in high.advice.driver_messages)
    assert not any(m.startswith("Berat badan sebelum hamil") for m in low.advice.driver_messages)
    for r in (high, low):
        assert r.advice.disclaimer and "indikatif" in r.hb_text and "rentang" not in r.hb_text


# --- readable session codes (researcher request 2026-10-21: no names, readable history) ---

def _store_with(tmp_path: Path, service_root):  # type: ignore[no-untyped-def]
    root, cfg = service_root
    service = ScreeningService(root, cfg)
    a = median_answers(service)
    inputs, prediction = service.storage_records(a, service.predict(a))
    return SessionStore(tmp_path / "codes.sqlite"), inputs, prediction


def test_session_codes_are_readable_sequential_and_never_reused(tmp_path: Path, service_root) -> None:  # type: ignore[no-untyped-def]
    import re
    from datetime import datetime

    store, inputs, prediction = _store_with(tmp_path, service_root)
    day1, day2 = datetime(2026, 10, 21, 9, 0), datetime(2026, 10, 22, 8, 0)
    ids = [store.create(inputs, prediction, "anon", now=day1) for _ in range(3)]
    codes = [store.session_code(i) for i in ids]
    assert codes == ["20261021-001", "20261021-002", "20261021-003"]
    assert all(re.fullmatch(r"\d{8}-\d{3}", c) for c in codes)
    store.delete(ids[2])
    nxt = store.create(inputs, prediction, "anon", now=day1)
    assert store.session_code(nxt) == "20261021-004"  # 003 is never given to another session
    assert store.session_code(store.create(inputs, prediction, "anon", now=day2)) == "20261022-001"
    listed = store.list()
    assert listed["session_code"].is_unique and "20261021-003" not in set(listed["session_code"])


def test_codes_contain_no_identifying_text(tmp_path: Path, service_root) -> None:  # type: ignore[no-untyped-def]
    store, inputs, prediction = _store_with(tmp_path, service_root)
    sid = store.create(inputs, prediction, "anon")
    code = store.session_code(sid)
    assert code.replace("-", "").isdigit()  # only date and number
    columns = " ".join(c for cols in store.column_names().values() for c in cols).lower()
    assert "nama" not in columns and "name" not in columns


def test_old_database_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, "
                "anonymous_user_id TEXT NOT NULL)")
    con.execute("INSERT INTO sessions VALUES ('old1', '2026-10-20T10:00:00+00:00', 'anon')")
    con.commit()
    con.close()
    store = SessionStore(path)
    assert "session_code" in store.column_names()["sessions"]
    assert store.session_code("old1") is None  # earlier rows keep working and show their short id


@needs_real
def test_app_save_shows_readable_code(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import re

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ANEMIA_CED_APP_DB", str(tmp_path / "app.sqlite"))
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "Skrining.py"), default_timeout=120)
    at.run()
    at.button[0].click().run()  # submit the form
    save = [b for b in at.button if b.label == "Simpan sesi"]
    assert save
    save[0].click().run()
    assert not at.exception
    msg = " ".join(s.value for s in at.success)
    assert re.search(r"\*\*\d{8}-001\*\*", msg) and "tidak menyimpan nama" in msg
    assert SessionStore(tmp_path / "app.sqlite").counts()["sessions"] == 1
    assert SessionStore(PROJECT_ROOT / "data" / "app" / "sessions.sqlite").counts()["sessions"] == 0  # real DB untouched


@needs_real
def test_dashboard_names_the_selected_models(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The result tabs, the 'Tentang model' panel and the footer follow artifacts/final_models.json."""
    from streamlit.testing.v1 import AppTest

    from src.postmodel.explain import ALGORITHM_LABELS

    meta = json.loads((PROJECT_ROOT / "artifacts" / "final_models.json").read_text(encoding="utf-8"))
    labels = {t: ALGORITHM_LABELS[meta[t]["algorithm"]] for t in ("hb", "ced")}
    monkeypatch.setenv("ANEMIA_CED_APP_DB", str(tmp_path / "app.sqlite"))
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "Skrining.py"), default_timeout=120)
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert any(e.label == "Tentang model yang dipakai" for e in at.expander)
    text = " ".join(str(x.value) for x in [*at.markdown, *at.caption])
    assert f"Model: {labels['ced']}" in text and f"Model: {labels['hb']}" in text
    assert f"KEK {labels['ced']}, Hb {labels['hb']}" in text  # footer
    assert "belum terbukti" in text  # equivalence group stated, no claim of superiority


@needs_real
def test_history_shows_the_model_and_flags_outdated_sessions(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from streamlit.testing.v1 import AppTest

    db = tmp_path / "app.sqlite"
    monkeypatch.setenv("ANEMIA_CED_APP_DB", str(db))
    service = ScreeningService()
    answers = median_answers(service)
    inputs, prediction = service.storage_records(answers, service.predict(answers))
    store = SessionStore(db)
    store.create(inputs, {**prediction, "hb_algorithm": None, "ced_algorithm": None}, "u1")  # saved before names existed
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "pages" / "1_Riwayat_sesi.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert {"model_KEK", "model_Hb"} <= set(at.dataframe[0].value.columns)
    assert any("berbeda dari model saat ini" in str(i.value) for i in at.info)
    [b for b in at.button if b.label.startswith("Hitung ulang")][0].click().run()
    assert not at.exception
    saved = store.read(store.list()["session_id"][0])["prediction"]
    assert (saved["hb_algorithm"], saved["ced_algorithm"]) == (service.algorithms["hb"], service.algorithms["ced"])
    at.run()
    assert not any("berbeda dari model saat ini" in str(i.value) for i in at.info)
