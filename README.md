# ModelFlow

An ML model lifecycle platform built with FastAPI: dataset upload, background training with model versioning, batch inference, and data drift monitoring (PSI / KS test).

**Stack:** FastAPI, Pydantic v2, SQLAlchemy 2.0 + SQLite, pandas + Parquet, scikit-learn, Docker, AWS S3 + EC2, GitHub Actions.

## Run locally
    uv sync
    uv run fastapi dev app/main.py      # http://127.0.0.1:8000/docs

## Tests
    uv run pytest