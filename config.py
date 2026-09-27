"""
BankGuard AI — Centralised configuration.
Loaded from environment variables (via .env in dev, SSM/Parameter Store in prod).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_secret_key: str = "change-me-256-bit-random"
    log_level: str = "INFO"
    api_port: int = 8000

    # ── AWS ────────────────────────────────────────────────────────
    aws_region: str = "ap-south-1"
    aws_account_id: str = ""

    # ── Bedrock ────────────────────────────────────────────────────
    bedrock_default_model: str = "us.amazon.nova-micro-v1:0"
    bedrock_fallback_model: str = "us.amazon.nova-lite-v1:0"
    bedrock_region: str = "ap-south-1"

    # ── PostgreSQL ─────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "bankguard"
    postgres_user: str = "bankguard"
    postgres_password: str = "bankguard_dev_password"
    database_url: str = "postgresql+asyncpg://bankguard:bankguard_dev_password@localhost:5432/bankguard"

    # ── Valkey / Redis ─────────────────────────────────────────────
    valkey_url: str = "redis://localhost:6379/0"

    # ── Langfuse ───────────────────────────────────────────────────
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = "pk-lf-dev"
    langfuse_secret_key: str = "sk-lf-dev"

    # ── OpenTelemetry ──────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "bankguard-api"
    otel_environment: str = "development"

    # ── Cognito ────────────────────────────────────────────────────
    cognito_user_pool_id: str = ""
    cognito_client_id: str = ""
    cognito_region: str = "ap-south-1"

    # ── S3 ─────────────────────────────────────────────────────────
    s3_bucket_policies: str = "bankguard-policies"
    s3_bucket_backups: str = "bankguard-backups"
    s3_bucket_evals: str = "bankguard-evals"

    # ── Temporal ──────────────────────────────────────────────────
    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "bankguard"

    # ── Agent Harness Limits ───────────────────────────────────────
    max_agent_iterations: int = 8
    max_tool_calls: int = 15
    max_cost_per_case_usd: float = 0.15
    tool_timeout_seconds: int = 10
    model_timeout_seconds: int = 30
    max_retries: int = 2
    max_parallel_tools: int = 4
    token_budget_input: int = 50_000
    token_budget_output: int = 10_000

    # ── OPA ────────────────────────────────────────────────────────
    opa_url: str = "http://localhost:8181"

    # ── Embeddings ─────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"

    # ── CORS ───────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3001,http://localhost:5173"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str) -> str:
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


# Convenience alias
settings = get_settings()
