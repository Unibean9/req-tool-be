import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AuditMixin, Base


class ProviderType(enum.StrEnum):
    BEDROCK = "bedrock"
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    CUSTOM = "custom"


class LLMProviderStatus(enum.StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


def enum_column(enum_class: type[enum.Enum], **kwargs):
    return mapped_column(
        SAEnum(enum_class, values_callable=lambda values: [item.value for item in values], validate_strings=True),
        **kwargs,
    )


class LLMProviderConfig(AuditMixin, Base):
    __tablename__ = "llm_provider_configs"
    __table_args__ = (CheckConstraint("encrypted_api_key IS NOT NULL", name="ck_llm_provider_api_key_required"),)

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    provider_type: Mapped[ProviderType] = enum_column(
        ProviderType, nullable=False, default=ProviderType.OPENAI, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    region: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    strong_model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    encrypted_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_secret_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[LLMProviderStatus] = enum_column(
        LLMProviderStatus, nullable=False, default=LLMProviderStatus.DRAFT, index=True
    )
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_check_error: Mapped[str | None] = mapped_column(Text, nullable=True)


Index(
    "uq_llm_provider_default_user",
    LLMProviderConfig.user_id,
    unique=True,
    postgresql_where=(LLMProviderConfig.is_default.is_(True)),
    sqlite_where=(LLMProviderConfig.is_default.is_(True)),
)
