"""Small helpers shared by the tests."""
import io

import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def upload(client, df: pd.DataFrame, name: str = "data"):
    return client.post(
        "/datasets",
        files={"file": (f"{name}.csv", to_csv_bytes(df), "text/csv")},
        data={"name": name},
    )


def train(client, dataset_id: int, model_name: str = "cancer", **extra):
    body = {"dataset_id": dataset_id, "model_name": model_name, "target": "target", **extra}
    return client.post("/runs", json=body)


def predict(client, df: pd.DataFrame, model_name: str = "cancer", version: int | None = None):
    data = {"model_name": model_name}
    if version is not None:
        data["version"] = str(version)
    return client.post(
        "/predictions",
        files={"file": ("score.csv", to_csv_bytes(df), "text/csv")},
        data=data,
    )