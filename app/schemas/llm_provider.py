import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from app.models.llm_provider import LLMProviderStatus, ProviderType


class LLMProviderConfigCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_type: ProviderType
    name: str = Field(min_length=1, max_length=255)
    base_url: str | None = Field(default=None, max_length=512)
    model_name: str | None = Field(default=None, max_length=255)
    api_key: str | None = None
    secret_ref: str | None = Field(default=None, max_length=255)
    is_default: bool = False

    @model_validator(mode="after")
    def validate_secret_source(self) -> "LLMProviderConfigCreate":
        if bool(self.api_key) == bool(self.secret_ref):
            raise ValueError("Phải cung cấp đúng một trong hai trường api_key hoặc secret_ref")
        if self.secret_ref and not self.secret_ref.startswith("LLM_KEY_"):
            raise ValueError("secret_ref phải bắt đầu bằng LLM_KEY_")
        return self


class LLMProviderConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_type: ProviderType | None = None
    name: str | None = Field(default=None, min_length=1, max_length=255)
    base_url: str | None = Field(default=None, max_length=512)
    model_name: str | None = Field(default=None, max_length=255)
    api_key: str | None = None
    secret_ref: str | None = Field(default=None, max_length=255)
    is_default: bool | None = None

    @model_validator(mode="after")
    def validate_secret_ref(self) -> "LLMProviderConfigUpdate":
        if self.api_key and self.secret_ref:
            raise ValueError("Chỉ được cập nhật một loại secret trong một request")
        if self.secret_ref and not self.secret_ref.startswith("LLM_KEY_"):
            raise ValueError("secret_ref phải bắt đầu bằng LLM_KEY_")
        return self


class LLMProviderConfigRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    org_id: uuid.UUID | None
    project_id: uuid.UUID | None
    provider_type: ProviderType
    name: str
    base_url: str | None
    model_name: str | None
    secret_ref: str | None = Field(default=None, exclude=True)
    encrypted_api_key: str | None = Field(default=None, exclude=True)
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
    def has_secret_ref(self) -> bool:
        return bool(self.secret_ref)
