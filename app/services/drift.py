import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Dataset, DriftReport, ModelVersion, PredictionBatch
from app.services.profiling import is_numeric
from app.services.storage import storage

# Common PSI rule of thumb
PSI_WARNING = 0.1     # 0.1 - 0.25: moderate shift, keep an eye on it
PSI_DRIFT = 0.25      # > 0.25: significant shift, consider retraining
EPS = 1e-4            # avoids log(0) when a bin is empty


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two sets of bin proportions."""
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_drift(train: pd.Series, new: pd.Series) -> dict:
    train, new = train.dropna(), new.dropna()
    if len(train) == 0 or len(new) == 0:
        return {"kind": "numeric", "psi": None, "note": "no values to compare"}

    # bins = training deciles, so each bin holds ~10% of the training data
    edges = np.unique(np.quantile(train, np.linspace(0.1, 0.9, 9)))
    n_bins = len(edges) + 1
    expected = np.bincount(np.searchsorted(edges, train, side="right"), minlength=n_bins) / len(train)
    actual = np.bincount(np.searchsorted(edges, new, side="right"), minlength=n_bins) / len(new)

    ks = ks_2samp(train, new)
    return {
        "kind": "numeric",
        "psi": round(psi(expected, actual), 4),
        "ks_statistic": round(float(ks.statistic), 4),
        "ks_pvalue": round(float(ks.pvalue), 4),
        "train_mean": round(float(train.mean()), 4),
        "new_mean": round(float(new.mean()), 4),
    }


def categorical_drift(train: pd.Series, new: pd.Series) -> dict:
    train = train.astype("string").fillna("<missing>")
    new = new.astype("string").fillna("<missing>")
    categories = sorted(set(train) | set(new))
    expected = train.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    actual = new.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return {
        "kind": "categorical",
        "psi": round(psi(expected, actual), 4),
        "new_categories": sorted(set(new) - set(train))[:20],
    }


def level(value: float | None) -> str:
    if value is None:
        return "ok"
    if value >= PSI_DRIFT:
        return "drift"
    if value >= PSI_WARNING:
        return "warning"
    return "ok"


def compare(train_df: pd.DataFrame, new_df: pd.DataFrame, features: list[str]) -> dict[str, dict]:
    results = {}
    for col in features:
        train, new = train_df[col], new_df[col]
        result = numeric_drift(train, new) if is_numeric(train) else categorical_drift(train, new)
        result["missing_rate_train"] = round(float(train.isna().mean()), 4)
        result["missing_rate_new"] = round(float(new.isna().mean()), 4)
        result["level"] = level(result["psi"])
        results[col] = result
    return results


def build_report(db, batch: PredictionBatch) -> DriftReport:
    """Compare a prediction batch's input with the data its model was trained on."""
    mv = db.get(ModelVersion, batch.model_version_id)
    dataset = db.get(Dataset, mv.dataset_id)

    train_df = storage.load_df(dataset.path)
    new_df = storage.load_df(batch.input_path)
    features = compare(train_df, new_df, mv.features)

    levels = [f["level"] for f in features.values()]
    n_drifted = levels.count("drift")
    status = "drift" if n_drifted else ("warning" if "warning" in levels else "ok")

    report = DriftReport(
        prediction_batch_id=batch.id,
        model_version_id=mv.id,
        status=status,
        n_features=len(features),
        n_drifted=n_drifted,
        # most shifted features first
        features=dict(sorted(features.items(), key=lambda kv: -(kv[1]["psi"] or 0))),
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def build_report_in_background(batch_id: int) -> None:
    """Called after each prediction batch; failures are logged, never raised to the user."""
    with SessionLocal() as db:
        try:
            batch = db.get(PredictionBatch, batch_id)
            exists = db.scalar(select(DriftReport.id).where(DriftReport.prediction_batch_id == batch_id))
            if batch and not exists:
                build_report(db, batch)
        except Exception as e:
            print(f"Drift report for batch {batch_id} failed: {e}")