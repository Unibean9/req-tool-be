"""LLM configuration for the eval harness.

Loaded from `.env.test` (NOT application `.env`). Analyst and judge share
credentials but use separate models: the analyst can use a weak model, while
the judge uses a strong model for scoring.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class JudgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.test", env_file_encoding="utf-8", extra="ignore")

    llm_provider_type: str = "anthropic"
    llm_api_key: str = ""
    llm_secret_key: str = ""
    llm_region: str = "us-east-1"
    llm_model_name: str = "claude-3-haiku-20240307"

    judge_provider: str = "anthropic"
    judge_model: str = "claude-3-5-sonnet-20241022"

    @property
    def judge_api_key(self) -> str:
        return self.llm_api_key

    @property
    def judge_region(self) -> str:
        return self.llm_region

    @property
    def judge_secret_key(self) -> str:
        return self.llm_secret_key


judge_settings = JudgeSettings()
