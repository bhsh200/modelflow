from tests.helpers import upload


def test_upload_csv(client, cancer_df):
    r = upload(client, cancer_df, "cancer")
    assert r.status_code == 201
    body = r.json()
    assert body["n_rows"] == 569
    assert body["n_cols"] == 31
    assert "path" not in body            # internal storage key is never exposed


def test_stats_are_stored(client, cancer_df):
    ds = upload(client, cancer_df).json()
    stats = client.get(f"/datasets/{ds['id']}").json()["stats"]
    assert stats["mean radius"]["kind"] == "numeric"
    assert len(stats["mean radius"]["deciles"]) == 9


def test_rejects_non_csv(client):
    r = client.post("/datasets", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_rejects_empty_file(client):
    r = client.post("/datasets", files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_preview(client, cancer_df):
    ds = upload(client, cancer_df).json()
    rows = client.get(f"/datasets/{ds['id']}/preview?rows=3").json()
    assert len(rows) == 3


def test_missing_dataset_is_404(client):
    assert client.get("/datasets/999").status_code == 404