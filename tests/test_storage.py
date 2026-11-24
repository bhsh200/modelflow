import boto3
import pandas as pd
import pytest
from moto import mock_aws
from sklearn.linear_model import LogisticRegression

from app.services.storage import LocalStorage, S3Storage


@pytest.fixture
def s3_storage():
    # moto fakes S3 in memory: no AWS account or network needed
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        yield S3Storage("test-bucket", prefix="modelflow", region="us-east-1")


@pytest.fixture
def local_storage(tmp_path):
    return LocalStorage(tmp_path)


@pytest.mark.parametrize("backend", ["local_storage", "s3_storage"])
def test_dataframe_roundtrip(backend, request):
    storage = request.getfixturevalue(backend)
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]})
    storage.save_df(df, "datasets/t.parquet")
    pd.testing.assert_frame_equal(storage.load_df("datasets/t.parquet"), df)


@pytest.mark.parametrize("backend", ["local_storage", "s3_storage"])
def test_model_roundtrip(backend, request):
    storage = request.getfixturevalue(backend)
    model = LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    storage.save_obj(model, "models/m/v1.joblib")
    assert list(storage.load_obj("models/m/v1.joblib").predict([[0], [3]])) == [0, 1]


def test_s3_keys_use_prefix(s3_storage):
    s3_storage.save_df(pd.DataFrame({"a": [1]}), "datasets/x.parquet")
    keys = [o["Key"] for o in s3_storage.s3.list_objects_v2(Bucket="test-bucket")["Contents"]]
    assert keys == ["modelflow/datasets/x.parquet"]