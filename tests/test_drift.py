import numpy as np
import pandas as pd

from app.services.drift import categorical_drift, numeric_drift, psi
from tests.helpers import predict


def test_psi_identical_is_zero():
    p = np.array([0.25, 0.25, 0.5])
    assert psi(p, p) == 0


def test_numeric_shift_is_detected():
    rng = np.random.default_rng(0)
    train = pd.Series(rng.normal(0, 1, 2000))
    same = pd.Series(rng.normal(0, 1, 500))
    shifted = pd.Series(rng.normal(2, 1, 500))
    assert numeric_drift(train, same)["psi"] < 0.1
    assert numeric_drift(train, shifted)["psi"] > 0.25


def test_new_category_is_reported():
    result = categorical_drift(pd.Series(["a", "b"] * 50), pd.Series(["a", "c"] * 50))
    assert result["new_categories"] == ["c"]
    assert result["psi"] > 0.25


def test_drift_report_after_prediction(trained, cancer_df):
    client, _, _ = trained
    X = cancer_df.drop(columns=["target"])

    ok_batch = predict(client, X.sample(150, random_state=1)).json()
    shifted = X.sample(150, random_state=2).copy()
    shifted["mean radius"] *= 1.5
    drift_batch = predict(client, shifted).json()

    ok = client.get(f"/monitoring/batches/{ok_batch['id']}/drift").json()
    drift = client.get(f"/monitoring/batches/{drift_batch['id']}/drift").json()
    assert ok["status"] == "ok"
    assert drift["status"] == "drift"
    assert next(iter(drift["features"])) == "mean radius"     # most drifted feature first
    assert len(client.get("/monitoring/models/cancer/drift").json()) == 2