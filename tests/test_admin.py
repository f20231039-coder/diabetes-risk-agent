"""Admin API: auth, CSV validation, merge, version listing and rollback.
Runs against a temporary copy of models/ and data/ so real files are untouched."""

import os
import shutil

import pandas as pd
import pytest

os.environ["LLM_PROVIDER"] = "rules"
os.environ["RATE_LIMIT_PER_HOUR"] = "1000"

from fastapi.testclient import TestClient  # noqa: E402

from health_agent import model_tool, training  # noqa: E402
from health_agent.server import app  # noqa: E402

TOKEN = "test-admin-token"


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    models, data = tmp_path / "models", tmp_path / "data"
    (models / "versions").mkdir(parents=True)
    data.mkdir()
    # The original v1 pickles are only read (from legacy/models), never written.
    shutil.copy(training.DATASET, data / "diabetes_full.csv")
    monkeypatch.setattr(training, "MODELS_DIR", models)
    monkeypatch.setattr(training, "VERSIONS_DIR", models / "versions")
    monkeypatch.setattr(training, "REGISTRY", models / "registry.json")
    monkeypatch.setattr(training, "UPLOADS_DIR", data / "uploads")
    monkeypatch.setattr(training, "DATASETS", {"pima": {"path": data / "diabetes_full.csv", "label": "Pima test copy"}})
    monkeypatch.setattr(training, "DEFAULT_DATASET", "pima")
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)
    model_tool._cache.clear()
    yield data
    model_tool._cache.clear()


def headers(token=TOKEN):
    return {"X-Admin-Token": token, "Content-Type": "text/csv"}


def test_admin_requires_token(sandbox):
    with TestClient(app) as c:
        assert c.get("/admin/models").status_code == 401
        assert c.get("/admin/models", headers=headers("wrong")).status_code == 401
        assert c.get("/admin/models", headers=headers()).status_code == 200


def test_admin_disabled_without_token_env(sandbox, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN")
    with TestClient(app) as c:
        assert c.get("/admin/models", headers=headers()).status_code == 403


def test_invalid_csv_rejected(sandbox):
    with TestClient(app) as c:
        r = c.post("/admin/dataset?retrain=false", content=b"Age,Glucose\n30,100\n", headers=headers())
        assert r.status_code == 400 and "missing required columns" in r.json()["detail"]
        r = c.post("/admin/dataset?retrain=false", content=b"Age,Glucose,BMI,Outcome\n30,100,22,7\n", headers=headers())
        assert r.status_code == 400


def test_upload_merges_and_dedupes(sandbox):
    csv = "age,glucose,bmi,outcome\n30,100,22.5,0\n30,100,22.5,0\n61,190,35,1\n"  # lower-case names accepted
    with TestClient(app) as c:
        r = c.post("/admin/dataset?retrain=false", content=csv.encode(), headers=headers())
        assert r.status_code == 200, r.text
        m = r.json()["merge"]
        assert m["rows_before"] == 768 and m["rows_after"] == 770 and m["duplicates_skipped"] == 1
    assert len(pd.read_csv(sandbox / "diabetes_full.csv")) == 770
    assert any(p.name.startswith("pima-before-") for p in (sandbox / "uploads").iterdir())  # backup kept


def test_list_and_activate_rollback(sandbox):
    with TestClient(app) as c:
        listed = c.get("/admin/models", headers=headers()).json()
        assert listed["active"]["pima"] == "v1"
        assert c.post("/admin/models/v99/activate", headers=headers()).status_code == 404
        assert c.post("/admin/models/v1/activate", headers=headers()).json() == {"dataset": "pima", "active": "v1"}


def test_config_lists_options(sandbox):
    with TestClient(app) as c:
        cfg = c.get("/config").json()
        assert any(l["id"] == "rules:regex-baseline" for l in cfg["llms"])
        assert [d["id"] for d in cfg["datasets"]] == ["pima"]
        assert {f["id"] for f in cfg["fields"] if f["required"]} == {"age", "glucose", "bmi"}


def test_chat_settings_and_form_values(sandbox):
    with TestClient(app) as c:
        body = {
            "message": "Age 50, glucose 148, BMI 33.6",
            "values": {"age": 50, "glucose": 148, "bmi": 33.6},
            "settings": {"dataset": "pima", "ml_model": "compare", "llm": "rules:regex-baseline", "prompt_version": "v2", "temperature": 0},
        }
        r = c.post("/chat", json=body).json()
        assert r["reply"]["status"] == "result" and r["reply"]["comparison"] is not None
        assert r["reply"]["dataset"] == "pima"
        bad = c.post("/chat", json={**body, "settings": {"llm": "evil:model"}})
        assert bad.status_code == 400
