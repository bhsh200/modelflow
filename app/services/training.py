import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import func, select

from app.db import SessionLocal
from app.models import Dataset, ModelVersion, RunStatus, TrainingRun
from app.services.profiling import is_numeric
from app.services.storage import storage

ALGORITHMS = {
    "classification": {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "linear": LogisticRegression,
    },
    "regression": {
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "linear": Ridge,
    },
}


def detect_task(y: pd.Series) -> str:
    """Text targets, or numbers with few distinct values (like 0/1), are classification."""
    if not is_numeric(y):
        return "classification"
    is_integer_like = bool(np.all(np.mod(y.dropna(), 1) == 0))
    return "classification" if is_integer_like and y.nunique() <= 20 else "regression"


def build_pipeline(X: pd.DataFrame, task: str, algorithm: str, params: dict) -> Pipeline:
    numeric = [c for c in X.columns if is_numeric(X[c])]
    categorical = [c for c in X.columns if c not in numeric]

    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20)),
            ]), categorical),
        ]
    )
    estimator = ALGORITHMS[task][algorithm](**params)
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, task: str) -> dict:
    preds = model.predict(X_test)
    if task == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_weighted": f1_score(y_test, preds, average="weighted"),
        }
        if y_test.nunique() == 2 and hasattr(model, "predict_proba"):
            metrics["roc_auc"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        metrics = {
            "rmse": root_mean_squared_error(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds),
        }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    metrics["task"] = task
    return metrics


def run_training(run_id: int, task: str | None, test_size: float) -> None:
    """Runs in the background after the API has responded. Uses its own DB session."""
    db = SessionLocal()
    run = db.get(TrainingRun, run_id)
    try:
        run.status = RunStatus.running
        db.commit()

        # 1. load data, split features / target
        dataset = db.get(Dataset, run.dataset_id)
        df = storage.load_df(dataset.path)
        df = df.dropna(subset=[run.target])      
        X = df.drop(columns=[run.target])
        y = df[run.target]
        task = task or detect_task(y)

        # 2. split, train, evaluate
        stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        model = build_pipeline(X_train, task, run.algorithm, run.params)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, task)

        # 3. register a new version: previous max + 1
        last = db.scalar(select(func.max(ModelVersion.version)).where(ModelVersion.name == run.model_name))
        version = (last or 0) + 1
        key = storage.save_obj(model, f"models/{run.model_name}/v{version}.joblib")

        mv = ModelVersion(
            name=run.model_name,
            version=version,
            dataset_id=run.dataset_id,
            algorithm=run.algorithm,
            target=run.target,
            features=list(X.columns),
            metrics=metrics,
            artifact_path=key,
        )
        db.add(mv)
        db.flush()                             

        run.status = RunStatus.succeeded
        run.metrics = metrics
        run.model_version_id = mv.id
    except Exception as e:
        db.rollback()
        run = db.get(TrainingRun, run_id)
        run.status = RunStatus.failed
        run.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()                      
    finally:
        run.finished_at = datetime.now(timezone.utc)
        db.commit()
        db.close()


def fail_interrupted_runs() -> None:
    """At startup: runs left pending/running by a crash or restart can never finish."""
    with SessionLocal() as db:
        stuck = db.scalars(
            select(TrainingRun).where(TrainingRun.status.in_([RunStatus.pending, RunStatus.running]))
        ).all()
        for run in stuck:
            run.status = RunStatus.failed
            run.error = "Interrupted by a server restart"
            run.finished_at = datetime.now(timezone.utc)
        db.commit()