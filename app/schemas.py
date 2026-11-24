from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models import RunStatus

# this file defines Pydantic schemas: the shapes of the JSON your endpoints accept and return

class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class DatasetOut(ORMModel):
    id: int
    name: str
    n_rows: int
    n_cols: int
    columns: dict[str, str]
    created_at: datetime


class DatasetDetail(DatasetOut):
    stats: dict


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    dataset_id: int
    model_name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_\-]+$")
    target: str
    algorithm: Literal["random_forest", "gradient_boosting", "linear"] = "random_forest"
    task: Literal["classification", "regression"] | None = None  # None = detect automatically
    params: dict[str, Any] = {}
    test_size: float = Field(default=0.2, gt=0, lt=0.5)


class TrainingRunOut(ORMModel):
    id: int
    dataset_id: int
    model_name: str
    algorithm: str
    target: str
    params: dict
    status: RunStatus
    error: str | None
    metrics: dict | None
    model_version_id: int | None
    created_at: datetime
    finished_at: datetime | None


class ModelVersionOut(ORMModel):
    id: int
    name: str
    version: int
    dataset_id: int
    algorithm: str
    target: str
    features: list[str]
    metrics: dict
    created_at: datetime



class PredictionBatchOut(ORMModel):
    id: int
    model_version_id: int
    n_rows: int
    created_at: datetime

class DriftReportOut(ORMModel):
    id: int
    prediction_batch_id: int
    model_version_id: int
    status: str
    n_features: int
    n_drifted: int
    features: dict
    created_at: datetime