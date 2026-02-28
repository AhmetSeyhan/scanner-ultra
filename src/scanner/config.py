"""Scanner ULTRA — Centralized typed configuration.

All values can be overridden via environment variables with the SCANNER_ prefix,
except fields with explicit validation_alias which use their own env var name.

Usage:
    from scanner.config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Typed, validated application settings."""

    model_config = {"env_prefix": "SCANNER_", "env_file": ".env", "extra": "ignore"}

    # --- Core ---
    env: str = Field("development", description="Runtime environment")
    version: str = Field("5.0.0", description="Application version")
    debug: bool = Field(False, description="Debug mode flag")
    log_level: str = Field("INFO", description="Log level")
    log_json: bool = Field(True, description="Emit JSON-structured logs")

    # --- Authentication ---
    secret_key: str = Field("change-this-to-a-secure-random-string", description="JWT signing secret")
    api_key: str = Field("change-this-to-a-secure-random-string", description="Service API key")
    access_token_expire_minutes: int = Field(30, description="JWT token lifetime")
    admin_password: str = Field("", description="Admin user password")

    # --- Server ---
    host: str = Field("0.0.0.0", description="API bind host")
    port: int = Field(8000, description="API bind port")
    cors_origins: str = Field("*", description="Comma-separated CORS origins")
    max_upload_size: int = Field(524_288_000, description="Max upload size in bytes (500MB)")

    # --- Database ---
    database_url: str = Field(
        "sqlite:///./scanner.db",
        validation_alias=AliasChoices("DATABASE_URL", "SCANNER_DATABASE_URL"),
        description="Database connection URL",
    )

    # --- Redis ---
    redis_url: str = Field(
        "redis://localhost:6379/0",
        validation_alias=AliasChoices("REDIS_URL", "SCANNER_REDIS_URL"),
        description="Redis URL",
    )

    # --- S3 / MinIO ---
    s3_endpoint_url: str = Field(
        "http://localhost:9000",
        validation_alias=AliasChoices("S3_ENDPOINT_URL", "SCANNER_S3_ENDPOINT_URL"),
    )
    s3_bucket_name: str = Field(
        "scanner-scans",
        validation_alias=AliasChoices("S3_BUCKET_NAME", "SCANNER_S3_BUCKET_NAME"),
    )
    aws_access_key_id: str = Field(
        "",
        validation_alias=AliasChoices("AWS_ACCESS_KEY_ID", "SCANNER_AWS_ACCESS_KEY_ID"),
    )
    aws_secret_access_key: str = Field(
        "",
        validation_alias=AliasChoices("AWS_SECRET_ACCESS_KEY", "SCANNER_AWS_SECRET_ACCESS_KEY"),
    )

    # --- Vector DB ---
    qdrant_url: str = Field(
        "http://localhost:6333",
        validation_alias=AliasChoices("QDRANT_URL", "SCANNER_QDRANT_URL"),
    )

    # --- Rate Limiting ---
    rate_limit: int = Field(50, description="Requests per minute per IP")

    # --- Model / Inference ---
    weights_dir: str = Field("./weights", description="Path to model weight files")
    device: str = Field("auto", description="Inference device (auto, cpu, cuda, cuda:0)")

    # --- Scan Pipeline ---
    detector_timeout: int = Field(30, description="Per-detector timeout in seconds")
    cache_ttl: int = Field(86400, description="Redis cache TTL in seconds (24h)")

    # --- Derived helpers ---
    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
