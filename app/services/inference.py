from functools import lru_cache

import pandas as pd
from sklearn.pipeline import Pipeline

from app.models import ModelVersion
from app.services.storage import storage


class InputError(ValueError):
    """The uploaded data doesn't fit the model (missing columns, wrong types)."""


@lru_cache(maxsize=16)
def load_model(artifact_path: str) -> Pipeline:
    """Load a model from storage once and keep it in memory; artifacts never change."""
    return storage.load_obj(artifact_path)


def predict(mv: ModelVersion, df: pd.DataFrame) -> pd.DataFrame:
    """Return the input rows plus a prediction column (and class probabilities)."""
    missing = [c for c in mv.features if c not in df.columns]
    if missing:
        raise InputError(f"Missing columns required by the model: {missing}")

    X = df[mv.features]                      # exact training column order; extras dropped
    model = load_model(mv.artifact_path)

    try:
        preds = model.predict(X)
    except (ValueError, TypeError) as e:     # e.g. text in a column that was numeric
        raise InputError(f"Could not score this data: {e}")

    out = df.copy()
    out["prediction"] = preds
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        for i, cls in enumerate(model.classes_):
            out[f"proba_{cls}"] = probs[:, i].round(4)
    return out