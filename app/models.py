from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import JSON, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)

# allowed states of a training run
class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    path: Mapped[str]                                  # where the Parquet file lives
    n_rows: Mapped[int]
    n_cols: Mapped[int]
    columns: Mapped[dict] = mapped_column(JSON)        # {column: dtype}
    stats: Mapped[dict] = mapped_column(JSON)          # per-column summary, used later for drift
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    model_name: Mapped[str] = mapped_column(String(200))
    algorithm: Mapped[str] = mapped_column(String(100))
    target: Mapped[str] = mapped_column(String(200))
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[RunStatus] = mapped_column(default=RunStatus.pending)
    error: Mapped[str | None]
    metrics: Mapped[dict | None] = mapped_column(JSON)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    finished_at: Mapped[datetime | None]


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[int]
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    algorithm: Mapped[str] = mapped_column(String(100))
    target: Mapped[str] = mapped_column(String(200))
    features: Mapped[list] = mapped_column(JSON)       # exact column order the model expects
    metrics: Mapped[dict] = mapped_column(JSON)
    artifact_path: Mapped[str]                         # joblib file
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class PredictionBatch(Base):
    __tablename__ = "prediction_batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    input_path: Mapped[str]
    output_path: Mapped[str]
    n_rows: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

class DriftReport(Base):
    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_batch_id: Mapped[int] = mapped_column(ForeignKey("prediction_batches.id"))
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    status: Mapped[str] = mapped_column(String(20))       # "ok" | "warning" | "drift"
    n_features: Mapped[int]
    n_drifted: Mapped[int]
    features: Mapped[dict] = mapped_column(JSON)          # per-feature results
    created_at: Mapped[datetime] = mapped_column(default=utcnow)