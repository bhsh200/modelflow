import json
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import select

from app.db import DbSession
from app.models import Dataset
from app.schemas import DatasetDetail, DatasetOut
from app.services.profiling import profile
from app.services.storage import storage

# prefix
router = APIRouter(prefix="/datasets", tags=["datasets"])


def get_dataset_or_404(db, dataset_id: int) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return ds


@router.post("", response_model=DatasetOut, status_code=201)
def upload_dataset(
    db: DbSession,
    file: Annotated[UploadFile, File(description="A CSV file")],
    name: Annotated[str | None, Form()] = None,
):
    # 1. validate
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")
    try:
        df = pd.read_csv(file.file)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows")

    # 2. profile + store as Parquet
    columns, stats = profile(df)
    key = f"datasets/{uuid4().hex}.parquet"
    storage.save_df(df, key)

    # 3. record in the database
    ds = Dataset(
        name=name or Path(file.filename).stem,
        path=key,
        n_rows=len(df),
        n_cols=df.shape[1],
        columns=columns,
        stats=stats,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: DbSession):
    return db.scalars(select(Dataset).order_by(Dataset.id.desc())).all()


@router.get("/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: int, db: DbSession):
    return get_dataset_or_404(db, dataset_id)


@router.get("/{dataset_id}/preview")
def preview_dataset(
    dataset_id: int,
    db: DbSession,
    rows: Annotated[int, Query(ge=1, le=100)] = 10,
):
    ds = get_dataset_or_404(db, dataset_id)
    df = storage.load_df(ds.path).head(rows)
    return json.loads(df.to_json(orient="records"))