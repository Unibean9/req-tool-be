from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/reqflow"

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_database_url(cls, v: str) -> str:
        if v.startswith("postgres://"):
            return "postgresql+asyncpg://" + v[len("postgres://") :]
        if v.startswith("postgresql://"):
            return "postgresql+asyncpg://" + v[len("postgresql://") :]
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

    # External web search for elicitation (comparable_products). "" disables it (graceful fallback to
    # model knowledge); "duckduckgo" uses the keyless DuckDuckGo HTML endpoint. CI leaves this empty.
    search_provider: str = ""

    app_env: str = "development"
    app_debug: bool = False
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    auto_migrate: bool = True
    # Circuit-breaker for silent internal tool-loops WITHIN one request (note/critique steps that
    # don't interrupt), NOT a conversation-length limit: turn_count resets to 0 on every human
    # resume, so user-facing exchanges are unbounded. A high backstop that should never trip in
    # healthy operation — hitting it means the model is stuck looping without interacting.
    max_agent_turns: int = 30
    llm_provider_health_timeout_seconds: float = 25.0
    agent_turn_timeout_seconds: float = 90.0
    summary_trigger_every: int = 6

    # Quality gate — reflection critic loop
    max_critique_rounds: int = 2
    critique_score_threshold: float = 0.7

    # Lazy expiry: an ACTIVE/WAITING_FOR_HUMAN session with no activity for this many hours is
    # eligible to be marked EXPIRED the next time it is read.
    session_abandoned_ttl: int = 72

    # Behavior thresholds kept as settings so eval grid sweeps can vary them via env without a
    # code edit. Values below are the calibrated defaults.
    # low_coverage_ratio: below this the diagnosis loop treats a section as weakly covered.
    low_coverage_ratio: float = 0.34
    # readiness rubric (readiness.py): dimension pass floor and overall ready threshold.
    readiness_dimension_pass: float = 0.5
    readiness_ready_threshold: float = 0.7

    # Deterministic proposal gate (validators.validate_proposal) runs before candidate readiness
    # in write_draft. Operator rollback path — flip to False to disable the deterministic gate on
    # draft proposals without a code revert. Mirrors enable_adaptive_diagnosis.
    enforce_deterministic_gate: bool = True

    # Adaptive diagnosis loop (orchestrator_node -> analyze_node): heuristic-gated thinking-mode
    # selection. Operator-facing rollback path: flip to False to disable diagnosis, prompt suffixes,
    # and judge escalation without a code revert.
    enable_adaptive_diagnosis: bool = True
    # Cap on LLM judge calls the diagnosis step may spend per turn escalating a heuristic
    # high-risk classification. Mirrors max_critique_rounds' role as a cost ceiling.
    max_diagnosis_judge_calls: int = 1

    # Analyst call token budget — must be large enough to serialize a full artifact body in JSON.
    analyze_max_tokens: int = 6000

    # "auto" → model decides whether to call a tool (enables clean terminal-text turns).
    # "required" → model must pick at least one tool (pre-M1 behaviour, for rollback).
    tool_choice_mode: str = "auto"

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
                raise ValueError(
                    "GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET must be set in non-development environments"
                )  # noqa: E501 — single message string
            if not self.github_state_secret:
                raise ValueError("GITHUB_STATE_SECRET must be set in non-development environments")
            if not self.cors_origins:
                raise ValueError("CORS_ORIGINS must be non-empty in non-development environments")
            if not self.github_app_id or not self.github_app_private_key or not self.github_app_slug:
                raise ValueError(
                    "GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY, and GITHUB_APP_SLUG must be set in "
                    "non-development environments"
                )
        return self


settings = Settings()
