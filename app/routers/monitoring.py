from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.db import DbSession
from app.models import DriftReport, ModelVersion, PredictionBatch
from app.schemas import DriftReportOut
from app.services.drift import build_report

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.post("/batches/{batch_id}/drift", response_model=DriftReportOut, status_code=201)
def run_drift_check(batch_id: int, db: DbSession):
    """Compute (or recompute) a drift report for one prediction batch."""
    batch = db.get(PredictionBatch, batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Prediction batch {batch_id} not found")
    return build_report(db, batch)


@router.get("/batches/{batch_id}/drift", response_model=DriftReportOut)
def latest_report_for_batch(batch_id: int, db: DbSession):
    report = db.scalars(
        select(DriftReport)
        .where(DriftReport.prediction_batch_id == batch_id)
        .order_by(DriftReport.id.desc())
    ).first()
    if report is None:
        raise HTTPException(status_code=404, detail=f"No drift report for batch {batch_id} yet")
    return report


@router.get("/models/{name}/drift", response_model=list[DriftReportOut])
def drift_history(
    name: str,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """Drift reports for every version of a model, newest first: the trend over time."""
    return db.scalars(
        select(DriftReport)
        .join(ModelVersion, DriftReport.model_version_id == ModelVersion.id)
        .where(ModelVersion.name == name)
        .order_by(DriftReport.id.desc())
        .limit(limit)
    ).all()