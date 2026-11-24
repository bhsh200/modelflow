from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.db import DbSession
from app.models import ModelVersion
from app.schemas import ModelVersionOut

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelVersionOut])
def list_models(db: DbSession, name: Annotated[str | None, Query()] = None):
    query = select(ModelVersion).order_by(ModelVersion.name, ModelVersion.version.desc())
    if name:
        query = query.where(ModelVersion.name == name)
    return db.scalars(query).all()


@router.get("/{name}/latest", response_model=ModelVersionOut)
def latest_version(name: str, db: DbSession):
    mv = db.scalars(
        select(ModelVersion).where(ModelVersion.name == name).order_by(ModelVersion.version.desc())
    ).first()
    if mv is None:
        raise HTTPException(status_code=404, detail=f"No versions of model '{name}'")
    return mv


@router.get("/{name}/versions/{version}", response_model=ModelVersionOut)
def get_version(name: str, version: int, db: DbSession):
    mv = db.scalars(
        select(ModelVersion).where(ModelVersion.name == name, ModelVersion.version == version)
    ).first()
    if mv is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' v{version} not found")
    return mv