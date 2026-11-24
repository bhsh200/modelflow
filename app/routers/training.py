from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import select

from app.db import DbSession
from app.models import Dataset, TrainingRun
from app.schemas import TrainingRunOut, TrainRequest
from app.services.training import ALGORITHMS, run_training

router = APIRouter(prefix="/runs", tags=["training"])


@router.post("", response_model=TrainingRunOut, status_code=202)
def start_training(req: TrainRequest, background_tasks: BackgroundTasks, db: DbSession):
    # validate everything we can before accepting the job
    dataset = db.get(Dataset, req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {req.dataset_id} not found")
    if req.target not in dataset.columns:
        raise HTTPException(status_code=400, detail=f"Target '{req.target}' is not a column of this dataset")
    if dataset.n_cols < 2:
        raise HTTPException(status_code=400, detail="Dataset needs at least one feature besides the target")

    # catch bad hyperparameters now, not minutes later in the background
    task_for_check = req.task or "classification"
    try:
        ALGORITHMS[task_for_check][req.algorithm](**req.params)
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid params: {e}")

    run = TrainingRun(
        dataset_id=req.dataset_id,
        model_name=req.model_name,
        algorithm=req.algorithm,
        target=req.target,
        params=req.params,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(run_training, run.id, req.task, req.test_size)
    return run


@router.get("", response_model=list[TrainingRunOut])
def list_runs(db: DbSession):
    return db.scalars(select(TrainingRun).order_by(TrainingRun.id.desc())).all()


@router.get("/{run_id}", response_model=TrainingRunOut)
def get_run(run_id: int, db: DbSession):
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run