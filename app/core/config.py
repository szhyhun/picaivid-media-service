"""Application configuration loaded from environment variables."""
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # API
    API_KEY: str = "dev-api-key"
    CORS_ORIGINS: List[str] = ["http://localhost:3001", "http://localhost:3003"]

    # Database (Postgres is system of record)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/picaivid_development"

    # AWS S3 (set credentials/endpoints explicitly for local MinIO; omit for AWS IAM roles)
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    S3_BUCKET: str = "picaivid-dev"
    S3_ENDPOINT: str | None = None

    # AWS SQS (set endpoint explicitly for local LocalStack; omit for AWS)
    SQS_ENDPOINT: str | None = None
    SQS_QUEUE_URL: str = "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/picaivid-jobs"

    # Rails webhook URL for status updates
    RAILS_WEBHOOK_URL: str | None = "http://localhost:3000"

    # Worker Type
    WORKER_TYPE: str = "cpu"
    WORKER_PHASES: str | None = None

    # ML Models
    MODEL_CACHE_DIR: str = "./ml_models"
    ANALYSIS_MATCH_ENGINE: str = "vggt_scene_graph"
    # VGGT-Omega is the private reconstruction runtime. Its repository and
    # checkpoint are supplied outside source control on each worker.
    VGGT_REPO_DIR: str | None = None
    VGGT_REPO_COMMIT: str | None = None
    VGGT_REPO_ARCHIVE_S3_URI: str | None = None
    VGGT_MODEL_CHECKPOINT: str | None = None
    VGGT_MODEL_CHECKPOINT_S3_URI: str | None = None
    VGGT_DEVICE: str = "auto"
    VGGT_PRECISION: str = "auto"
    VGGT_IMAGE_SIZE: int = 512
    VGGT_IMAGE_MODE: str = "balanced"
    # Raw threshold-free pair evidence. Shared with the calibration sweep so a
    # re-analysis can reuse model inference and only rerun graph policy.
    VGGT_PAIR_CACHE_DIR: str = "./tmp_sweep/cache"
    OPENCLIP_MODEL: str = "ViT-B-32"
    OPENCLIP_PRETRAINED: str = "openai"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
