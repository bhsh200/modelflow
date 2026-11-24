import pandas as pd

from app.db import SessionLocal
from app.models import RunStatus, TrainingRun
from app.services.training import detect_task, fail_interrupted_runs
from tests.helpers import train, upload


def test_training_succeeds(trained):
    client, _, run = trained
    # TestClient runs background tasks before returning, so the run is already finished
    result = client.get(f"/runs/{run['id']}").json()
    assert result["status"] == "succeeded"
    assert result["metrics"]["accuracy"] > 0.9
    assert result["model_version_id"] is not None


def test_versions_increment(trained):
    client, ds, _ = trained
    train(client, ds["id"], algorithm="linear", params={"max_iter": 1000})
    versions = [m["version"] for m in client.get("/models?name=cancer").json()]
    assert versions == [2, 1]
    assert client.get("/models/cancer/latest").json()["version"] == 2


def test_unknown_target_is_400(client, cancer_df):
    ds = upload(client, cancer_df).json()
    r = client.post("/runs", json={"dataset_id": ds["id"], "model_name": "m", "target": "nope"})
    assert r.status_code == 400


def test_bad_params_are_400(client, cancer_df):
    ds = upload(client, cancer_df).json()
    assert train(client, ds["id"], params={"not_a_param": 1}).status_code == 400


def test_invalid_model_name_is_422(client, cancer_df):
    ds = upload(client, cancer_df).json()
    assert train(client, ds["id"], model_name="has spaces").status_code == 422


def test_failed_run_is_recorded(client):
    df = pd.DataFrame({"x": range(20), "target": ["a", "b"] * 10})
    ds = upload(client, df).json()
    run = train(client, ds["id"], model_name="bad", task="regression").json()  # text target can't regress
    result = client.get(f"/runs/{run['id']}").json()
    assert result["status"] == "failed"
    assert result["error"]
    assert client.get("/models?name=bad").json() == []


def test_detect_task():
    assert detect_task(pd.Series([0, 1, 1, 0])) == "classification"
    assert detect_task(pd.Series(["yes", "no"])) == "classification"
    assert detect_task(pd.Series([1.5, 2.7, 3.1, 8.9])) == "regression"


def test_interrupted_runs_are_failed_on_startup(client, cancer_df):
    ds = upload(client, cancer_df).json()
    with SessionLocal() as db:
        db.add(TrainingRun(dataset_id=ds["id"], model_name="m", algorithm="random_forest",
                           target="target", params={}, status=RunStatus.running))
        db.commit()
    fail_interrupted_runs()
    run = client.get("/runs").json()[0]
    assert run["status"] == "failed"
    assert "restart" in run["error"]