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

    # Rails webhook URL for status updates
    RAILS_WEBHOOK_URL: str | None = "http://localhost:3000"

    # Worker Type
    WORKER_TYPE: str = "cpu"

    # ML Models
    MODEL_CACHE_DIR: str = "./ml_models"
    OPENCLIP_MODEL: str = "ViT-B-32"
    OPENCLIP_PRETRAINED: str = "openai"
    ROMA_DEBUG_DEVICE: str = "auto"
    MATCHFORMER_DEBUG_DEVICE: str = "auto"

    # ZJU LoFTR strict debug settings
    LOFTR_ZJU_REPO_DIR: str | None = None
    LOFTR_ZJU_INDOOR_CKPT: str | None = None
    LOFTR_ZJU_INDOOR_DS_CKPT: str | None = None
    LOFTR_ZJU_INDOOR_OT_CKPT: str | None = None
    LOFTR_ZJU_OUTDOOR_DS_CKPT: str | None = None
    LOFTR_ZJU_OUTDOOR_OT_CKPT: str | None = None

    # MatchFormer strict debug settings
    MATCHFORMER_REPO_DIR: str | None = None
    MATCHFORMER_INDOOR_CKPT: str | None = None
    MATCHFORMER_OUTDOOR_CKPT: str | None = None

    # Clustering behavior
    DELETE_OBVIOUS_DUPLICATES: bool = True
    REQUIRE_GEOMETRIC_TRANSITIONS: bool = True
    REQUIRE_DIRECTION_FOR_TRANSITIONS: bool = False
    HARD_TRANSITION_MIN_SIDE_OVERLAP: float = 0.06
    HARD_TRANSITION_MIN_CENTER_OVERLAP: float = 0.09
    HARD_TRANSITION_MIN_OVERLAP_RATIO: float = 0.08
    GEOMETRIC_SCORE_WEIGHT: float = 0.90
    SEMANTIC_SCORE_WEIGHT: float = 0.10
    GEOMETRY_ONLY_CLUSTER_MEMBERSHIP: bool = True
    STRICT_SEMANTIC_COMPONENT_CONNECTIVITY: bool = True
    COMPONENT_SEMANTIC_ADJ_MIN: float = 0.78
    COMPONENT_SEMANTIC_DIST2_MIN: float = 0.82
    COMPONENT_SEMANTIC_RECOVERY_MIN: float = 0.88
    COMPONENT_FRONT_RECOVERY_MIN: float = 0.80
    COMPONENT_SEMANTIC_MAX_GAP: int = 2
    COMPONENT_SAME_LABEL_ADJ_MIN: float = 0.70
    COMPONENT_SAME_LABEL_DIST2_MIN: float = 0.74
    COMPONENT_AMBIGUOUS_SAME_LABEL_MIN: float = 0.82
    COMPONENT_CROSS_LABEL_ADJ_MIN: float = 0.84
    COMPONENT_CROSS_LABEL_DIST2_MIN: float = 0.88

    # Kornia geometric oracle (A/B)
    # off: disabled, shadow: log-only, gate: enforce oracle pass for geometric edges
    KORNIA_ORACLE_MODE: str = "off"
    KORNIA_ORACLE_MIN_OVERLAP_RATIO: float = 0.08
    KORNIA_ORACLE_MIN_SIDE_OVERLAP: float = 0.06
    KORNIA_ORACLE_MIN_CENTER_OVERLAP: float = 0.09
    KORNIA_ORACLE_MIN_INLIER_RATIO: float = 0.20
    KORNIA_ORACLE_INLIER_THRESHOLD_PX: float = 2.0

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
