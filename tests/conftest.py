import os
import tempfile

# Point the app at a throwaway database and data folder BEFORE importing it,
# because the engine and storage are created at import time from settings.
_tmp = tempfile.mkdtemp(prefix="modelflow-test-")
os.environ["DATA_DIR"] = _tmp
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["STORAGE_BACKEND"] = "local"     # tests never touch real S3, whatever .env says

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.services.inference import load_model  # noqa: E402
from tests.helpers import train, upload  # noqa: E402


@pytest.fixture
def client():
    """A fresh, empty database for every test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    load_model.cache_clear()
    with TestClient(app) as c:      # "with" runs the lifespan (startup) code
        yield c


@pytest.fixture(scope="session")
def cancer_df() -> pd.DataFrame:
    return load_breast_cancer(as_frame=True).frame


@pytest.fixture
def trained(client, cancer_df):
    """A client with the cancer dataset uploaded and model 'cancer' v1 trained."""
    ds = upload(client, cancer_df, "cancer").json()
    run = train(client, ds["id"]).json()
    return client, ds, run