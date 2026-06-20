"""Judge configuration for the eval harness.

Loaded from `.env.test` (NOT the application `.env`) so the eval-only judge
credentials stay separate from application config. The judge uses a fixed
strong model, decoupled from the agent session's provider.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class JudgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.test", env_file_encoding="utf-8", extra="ignore")

    judge_provider: str = "anthropic"
    judge_model: str = "claude-3-5-sonnet-20241022"
    judge_api_key: str = ""
    # Bedrock only: region is required; secret_key selects IAM-key auth (leave empty for bearer-key auth)
    judge_region: str = "us-east-1"
    judge_secret_key: str = ""


judge_settings = JudgeSettings()
