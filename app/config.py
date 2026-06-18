from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/reqflow"

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_database_url(cls, v: str) -> str:
        if v.startswith("postgres://"):
            return "postgresql+asyncpg://" + v[len("postgres://"):]
        if v.startswith("postgresql://"):
            return "postgresql+asyncpg://" + v[len("postgresql://"):]
        return v

    jwt_secret_key: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    encryption_key: str = ""
    encryption_key_previous: str = ""
    password_pepper: str = ""

    github_client_id: str = ""
    github_client_secret: str = ""
    github_redirect_uri: str = "http://localhost:8000/auth/github/callback"
    github_state_secret: str = ""

    # GitHub App — used for repo connection (installation flow)
    github_app_id: str = ""
    github_app_client_id: str = ""
    github_app_private_key: str = ""
    github_app_slug: str = ""
    github_app_redirect_uri: str = ""

    # AWS Bedrock — notation auto-detection
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    bedrock_notation_model: str = "google.gemma-3-4b-it"

    app_env: str = "development"
    app_debug: bool = False
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    auto_migrate: bool = True

    @model_validator(mode="after")
    def _enforce_production_secrets(self) -> "Settings":
        if self.app_env != "development":
            if self.jwt_secret_key == "change-this-in-production":
                raise ValueError("JWT_SECRET_KEY must be changed in non-development environments")
            if len(self.jwt_secret_key) < 32:
                raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
            if not self.encryption_key:
                raise ValueError("ENCRYPTION_KEY must be set in non-development environments")
            if not self.password_pepper:
                raise ValueError("PASSWORD_PEPPER must be set in non-development environments")
            if not self.github_client_id or not self.github_client_secret:
                raise ValueError("GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET must be set in non-development environments")
            if not self.github_state_secret:
                raise ValueError("GITHUB_STATE_SECRET must be set in non-development environments")
            if not self.cors_origins:
                raise ValueError("CORS_ORIGINS must be non-empty in non-development environments")
            if not self.github_app_id or not self.github_app_private_key or not self.github_app_slug:
                raise ValueError("GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY, and GITHUB_APP_SLUG must be set in non-development environments")
        return self


settings = Settings()
