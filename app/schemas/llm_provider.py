import enum
import ipaddress
import uuid
from datetime import datetime
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from app.models.llm_provider import LLMProviderStatus, ProviderType


class PublicProviderType(enum.StrEnum):
    BEDROCK = "bedrock"
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    CUSTOM = "custom"


def _normalize_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return value
    parts = urlsplit(value)
    if parts.scheme.lower() != "https":
        raise ValueError("base_url must use https")
    if not parts.hostname:
        raise ValueError("base_url must include a host")
    if parts.username or parts.password:
        raise ValueError("base_url must not include credentials")
    if parts.query or parts.fragment:
        raise ValueError("base_url must not include query or fragment")

    host = parts.hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("base_url must not target localhost")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            raise ValueError("base_url must not target a private or local IP address")
    return value.rstrip("/")


class LLMProviderKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_type: PublicProviderType = PublicProviderType.OPENAI
    api_key: str = Field(min_length=1)
    secret_key: str | None = Field(default=None, min_length=1)
    region: str | None = Field(default=None, min_length=1, max_length=64)
    base_url: str | None = Field(default=None, min_length=1, max_length=512)
    model_name: str | None = Field(default=None, min_length=1, max_length=255)
    strong_model_name: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("base_url", mode="before")
    @classmethod
    def strip_base_url(cls, value: str | None) -> str | None:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        return _normalize_base_url(value)

    @model_validator(mode="after")
    def validate_provider_fields(self) -> "LLMProviderKeyRequest":
        if self.provider_type == PublicProviderType.CUSTOM:
            if not self.base_url:
                raise ValueError("base_url is required for custom provider")
            if not self.model_name:
                raise ValueError("model_name is required for custom provider")
        elif self.base_url is not None:
            raise ValueError("base_url is only supported for custom provider")
        return self


class LLMProviderUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str | None = Field(default=None, min_length=1, max_length=64)
    base_url: str | None = Field(default=None, min_length=1, max_length=512)
    model_name: str | None = Field(default=None, min_length=1, max_length=255)
    strong_model_name: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("base_url", mode="before")
    @classmethod
    def strip_base_url(cls, value: str | None) -> str | None:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        return _normalize_base_url(value)

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
