import io
from pathlib import Path
from typing import Any, Protocol

import joblib
import pandas as pd

from app.config import settings


class Storage(Protocol):
    """What every storage backend must provide. Keys look like 'datasets/abc.parquet'."""

    def save_df(self, df: pd.DataFrame, key: str) -> str: ...
    def load_df(self, key: str) -> pd.DataFrame: ...
    def save_obj(self, obj: Any, key: str) -> str: ...
    def load_obj(self, key: str) -> Any: ...
    def read_bytes(self, key: str) -> bytes: ...


class LocalStorage:
    """Files under a folder on disk (development, tests, single server)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_df(self, df: pd.DataFrame, key: str) -> str:
        df.to_parquet(self._path(key), index=False)
        return key

    def load_df(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(key))

    def save_obj(self, obj: Any, key: str) -> str:
        joblib.dump(obj, self._path(key))
        return key

    def load_obj(self, key: str) -> Any:
        return joblib.load(self._path(key))

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()


class S3Storage:
    """Files in an S3 bucket. Credentials come from boto3's default chain:
    env vars / ~/.aws locally, the instance's IAM role on EC2 (no keys in code)."""

    def __init__(self, bucket: str, prefix: str = "", region: str | None = None):
        import boto3  # imported here so local mode doesn't need AWS set up

        self.s3 = boto3.client("s3", region_name=region)
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _put(self, key: str, data: bytes) -> str:
        self.s3.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)
        return key

    def read_bytes(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()

    def save_df(self, df: pd.DataFrame, key: str) -> str:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return self._put(key, buf.getvalue())

    def load_df(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.read_bytes(key)))

    def save_obj(self, obj: Any, key: str) -> str:
        buf = io.BytesIO()
        joblib.dump(obj, buf)
        return self._put(key, buf.getvalue())

    def load_obj(self, key: str) -> Any:
        return joblib.load(io.BytesIO(self.read_bytes(key)))


def create_storage() -> Storage:
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise RuntimeError("STORAGE_BACKEND=s3 needs S3_BUCKET to be set")
        return S3Storage(settings.s3_bucket, settings.s3_prefix, settings.aws_region)
    return LocalStorage(settings.data_dir)


storage = create_storage()