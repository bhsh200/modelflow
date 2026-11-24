from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.config import settings
from app.db import init_db
from app.routers import datasets, models, monitoring, predictions, training
from app.services.training import fail_interrupted_runs


# defines what happens when a server starts up and shuts down (yield runs at startup and code after it runs at shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    fail_interrupted_runs()
    yield

# creates application
app = FastAPI(title=settings.app_name, lifespan=lifespan)

# adds router end points to app
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(predictions.router)
app.include_router(monitoring.router)


# basic url redirects to /docs
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# health check that docker can use
@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}