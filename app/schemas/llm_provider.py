import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from app.models.llm_provider import LLMProviderStatus, ProviderType


class LLMProviderKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_type: ProviderType = ProviderType.OPENAI
    api_key: str = Field(min_length=1)
    secret_key: str | None = Field(default=None, min_length=1)
    region: str | None = Field(default=None, min_length=1, max_length=64)
    model_name: str | None = Field(default=None, min_length=1, max_length=255)
    strong_model_name: str | None = Field(default=None, min_length=1, max_length=255)


class LLMProviderUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str | None = Field(default=None, min_length=1, max_length=64)
    model_name: str | None = Field(default=None, min_length=1, max_length=255)
    strong_model_name: str | None = Field(default=None, min_length=1, max_length=255)

    @model_validator(mode="after")
    def require_update_field(self) -> "LLMProviderUpdateRequest":
        if not self.model_fields_set:
            raise ValueError("At least one model configuration field is required")
        return self


class LLMProviderConfigRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    provider_type: ProviderType
    name: str
    base_url: str | None
    region: str | None
    model_name: str | None
    strong_model_name: str | None
    encrypted_api_key: str | None = Field(default=None, exclude=True)
    encrypted_secret_key: str | None = Field(default=None, exclude=True)
    status: LLMProviderStatus
    is_default: bool
    last_checked_at: datetime | None
    last_check_error: str | None
    created_at: datetime
    updated_at: datetime

    @computed_field
    @property
    def api_key_set(self) -> bool:
        return bool(self.encrypted_api_key)

    @computed_field
    @property
    def secret_key_set(self) -> bool:
        return bool(self.encrypted_secret_key)


class LLMProviderHealthCheckResult(BaseModel):
    config: LLMProviderConfigRead
    response_time_ms: int
    provider_reply: str | None = None
    tool_calling_supported: bool | None = None
