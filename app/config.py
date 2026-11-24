from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "ModelFlow"
    database_url: str = "sqlite:///./modelflow.db"
    data_dir: Path = Path("data")

    # file storage: "local" (data_dir on disk) or "s3"
    storage_backend: Literal["local", "s3"] = "local"
    s3_bucket: str | None = None
    s3_prefix: str = "modelflow"          # folder inside the bucket
    aws_region: str = "ap-south-1"


settings = Settings()