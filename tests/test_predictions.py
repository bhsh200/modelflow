import io

import pandas as pd

from tests.helpers import predict


def test_predict_latest(trained, cancer_df):
    client, _, _ = trained
    X = cancer_df.drop(columns=["target"]).head(20)
    r = predict(client, X)
    assert r.status_code == 201
    assert r.json()["n_rows"] == 20

    rows = client.get(f"/predictions/{r.json()['id']}/preview?rows=5").json()
    assert {"prediction", "proba_0", "proba_1"} <= rows[0].keys()


def test_column_order_does_not_matter(trained, cancer_df):
    client, _, _ = trained
    X = cancer_df.drop(columns=["target"]).head(20)
    a = predict(client, X).json()["id"]
    b = predict(client, X[X.columns[::-1]]).json()["id"]
    preds_a = [row["prediction"] for row in client.get(f"/predictions/{a}/preview?rows=20").json()]
    preds_b = [row["prediction"] for row in client.get(f"/predictions/{b}/preview?rows=20").json()]
    assert preds_a == preds_b


def test_missing_column_is_400(trained, cancer_df):
    client, _, _ = trained
    X = cancer_df.drop(columns=["target", "mean radius"]).head(5)
    r = predict(client, X)
    assert r.status_code == 400
    assert "mean radius" in r.json()["detail"]


def test_unknown_model_is_404(trained, cancer_df):
    client, _, _ = trained
    X = cancer_df.drop(columns=["target"]).head(5)
    assert predict(client, X, model_name="nope").status_code == 404
    assert predict(client, X, version=99).status_code == 404


def test_download_csv(trained, cancer_df):
    client, _, _ = trained
    batch = predict(client, cancer_df.drop(columns=["target"]).head(10)).json()
    r = client.get(f"/predictions/{batch['id']}/download")
    assert r.status_code == 200
    df = pd.read_csv(io.StringIO(r.text))
    assert len(df) == 10 and "prediction" in df.columns