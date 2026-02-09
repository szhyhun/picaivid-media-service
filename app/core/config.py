"""Application configuration loaded from environment variables."""
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # API
    API_KEY: str = "dev-api-key"
    CORS_ORIGINS: List[str] = ["http://localhost:3001"]

    # Database (Postgres is system of record)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/picaivid_development"

    # AWS S3 (MinIO for local)
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadmin"
    S3_BUCKET: str = "picaivid-dev"
    S3_ENDPOINT: str | None = "http://localhost:9000"

    # AWS SQS (LocalStack for local)
    SQS_ENDPOINT: str | None = "http://localhost:4566"
    SQS_QUEUE_URL: str = "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/picaivid-jobs"
    SQS_ACCESS_KEY_ID: str = "localstack"
    SQS_SECRET_ACCESS_KEY: str = "localstack"

    # Worker Type
    WORKER_TYPE: str = "cpu"

    # ML Models
    MODEL_CACHE_DIR: str = "./ml_models"
    OPENCLIP_MODEL: str = "ViT-B-32"
    OPENCLIP_PRETRAINED: str = "openai"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
