import io
import json
from typing import Annotated, Literal
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select

from app.db import DbSession
from app.models import ModelVersion, PredictionBatch
from app.schemas import PredictionBatchOut
from app.services.drift import build_report_in_background
from app.services.inference import InputError, predict
from app.services.storage import storage

router = APIRouter(prefix="/predictions", tags=["predictions"])


def find_model_version(db, name: str, version: int | None) -> ModelVersion:
    query = select(ModelVersion).where(ModelVersion.name == name)
    if version is None:
        query = query.order_by(ModelVersion.version.desc())       # latest
    else:
        query = query.where(ModelVersion.version == version)
    mv = db.scalars(query).first()
    if mv is None:
        label = f"'{name}' v{version}" if version else f"'{name}'"
        raise HTTPException(status_code=404, detail=f"Model {label} not found")
    return mv


def get_batch_or_404(db, batch_id: int) -> PredictionBatch:
    batch = db.get(PredictionBatch, batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Prediction batch {batch_id} not found")
    return batch


@router.post("", response_model=PredictionBatchOut, status_code=201)
def create_predictions(
    db: DbSession,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="CSV with the model's feature columns")],
    model_name: Annotated[str, Form()],
    version: Annotated[int | None, Form(description="Leave empty for the latest version")] = None,
):
    mv = find_model_version(db, model_name, version)

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")
    try:
        df = pd.read_csv(file.file)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows")

    try:
        out = predict(mv, df)
    except InputError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # keep both input and output: the input is what drift monitoring compares later
    folder = f"predictions/{uuid4().hex}"
    input_key = storage.save_df(df, f"{folder}/input.parquet")
    output_key = storage.save_df(out, f"{folder}/output.parquet")

    batch = PredictionBatch(
        model_version_id=mv.id,
        input_path=input_key,
        output_path=output_key,
        n_rows=len(df),
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    background_tasks.add_task(build_report_in_background, batch.id)
    return batch


@router.get("", response_model=list[PredictionBatchOut])
def list_batches(db: DbSession):
    return db.scalars(select(PredictionBatch).order_by(PredictionBatch.id.desc())).all()


@router.get("/{batch_id}", response_model=PredictionBatchOut)
def get_batch(batch_id: int, db: DbSession):
    return get_batch_or_404(db, batch_id)


@router.get("/{batch_id}/preview")
def preview_batch(
    batch_id: int,
    db: DbSession,
    rows: Annotated[int, Query(ge=1, le=100)] = 10,
):
    batch = get_batch_or_404(db, batch_id)
    df = storage.load_df(batch.output_path).head(rows)
    return json.loads(df.to_json(orient="records"))


@router.get("/{batch_id}/download")
def download_batch(
    batch_id: int,
    db: DbSession,
    format: Annotated[Literal["csv", "parquet"], Query()] = "csv",
):
    batch = get_batch_or_404(db, batch_id)
    filename = f"predictions_{batch_id}.{format}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    if format == "parquet":
        # works for both local and S3 storage
        return Response(storage.read_bytes(batch.output_path),
                        media_type="application/octet-stream", headers=headers)

    df = storage.load_df(batch.output_path)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="text/csv", headers=headers)