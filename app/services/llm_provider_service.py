import asyncio
import re
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.crypto import decrypt_token, encrypt_token
from app.models.llm_provider import LLMProviderConfig, LLMProviderStatus, ProviderType
from app.schemas.llm_provider import LLMProviderHealthCheckResult, LLMProviderKeyRequest, LLMProviderUpdateRequest
from app.services.llm_clients import DEFAULT_MODEL_BY_PROVIDER, LLMClientFactory


class CooldownError(Exception):
    pass


class ProviderUnavailableError(Exception):
    pass


class ProviderCapabilityError(Exception):
    pass


SECRET_PATTERN = re.compile(r"(?:sk|key|token)-[A-Za-z0-9_\-]+")
DEFAULT_PROVIDER_TYPE = ProviderType.OPENAI
TOOL_CALLING_REQUIRED_MESSAGE = "Model khong ho tro tool calling"


def _sanitize_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return SECRET_PATTERN.sub("[REDACTED]", message)[:500]


def _resolve_api_key(config: LLMProviderConfig) -> str:
    if config.encrypted_api_key:
        value = decrypt_token(config.encrypted_api_key)
        if value is None:
            raise ValueError("Encrypted API key could not be decrypted — possible key rotation mismatch")
        return value
    raise ValueError("Config has no API key")


def _decrypt_required(value: str | None, field_name: str) -> str:
    if not value:
        raise ValueError(f"Config missing {field_name}")
    decrypted = decrypt_token(value)
    if decrypted is None:
        raise ValueError(f"{field_name} cannot be decrypted - key rotation may be out of sync")
    return decrypted


def _resolve_secret_key(config: LLMProviderConfig) -> str | None:
    if not config.encrypted_secret_key:
        return None
    return _decrypt_required(config.encrypted_secret_key, "secret_key")


class LLMProviderService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, *, user_id: uuid.UUID, body: LLMProviderKeyRequest | dict[str, Any]) -> LLMProviderConfig:
        body = self._key_schema(body)
        existing = await self._get_first_user_config(user_id)
        if existing is not None:
            await self._disable_user_configs_except(user_id, existing.id)
            values = self._values_from_key_request(body)
            for field, value in values.items():
                setattr(existing, field, value)
            existing.is_default = True
            existing.status = LLMProviderStatus.DRAFT
            existing.last_checked_at = None
            existing.last_check_error = None
            await self.db.flush()
            return existing
        await self._unset_user_default(user_id)
        values = self._values_from_key_request(body)
        config = LLMProviderConfig(
            user_id=user_id,
            is_default=True,
            status=LLMProviderStatus.DRAFT,
            **values,
        )
        self.db.add(config)
        await self.db.flush()
        return config

    async def list(self, *, user_id: uuid.UUID) -> list[LLMProviderConfig]:
        result = await self.db.execute(
            select(LLMProviderConfig)
            .where(LLMProviderConfig.user_id == user_id, LLMProviderConfig.status != LLMProviderStatus.DISABLED)
            .order_by(LLMProviderConfig.created_at, LLMProviderConfig.id)
        )
        return list(result.scalars().all())

    async def get(self, *, user_id: uuid.UUID, config_id: uuid.UUID) -> LLMProviderConfig:
        result = await self.db.execute(
            select(LLMProviderConfig).where(
                LLMProviderConfig.id == config_id,
                LLMProviderConfig.user_id == user_id,
                LLMProviderConfig.status != LLMProviderStatus.DISABLED,
            )
        )
        config = result.scalar_one_or_none()
        if config is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="LLM provider config not found")
        return config

    async def update(
        self,
        *,
        user_id: uuid.UUID,
        config_id: uuid.UUID,
        body: LLMProviderUpdateRequest | dict[str, Any],
    ) -> LLMProviderConfig:
        schema = self._update_schema(body)
        config = await self.get(user_id=user_id, config_id=config_id)
        values = {
            "status": LLMProviderStatus.DRAFT,
            "last_checked_at": None,
            "last_check_error": None,
            "is_default": True,
        }
        values.update(self._values_from_update_request(schema, config))
        await self._unset_user_default(user_id, exclude_id=config_id)
        if values:
            await self.db.execute(
                update(LLMProviderConfig)
                .where(LLMProviderConfig.id == config_id, LLMProviderConfig.user_id == user_id)
                .values(**values)
            )
            await self.db.flush()
            await self.db.refresh(config)
        return config

    async def delete(self, *, user_id: uuid.UUID, config_id: uuid.UUID) -> None:
        config = await self.get(user_id=user_id, config_id=config_id)
        config.status = LLMProviderStatus.DISABLED
        config.is_default = False
        await self.db.flush()

    async def health_check(self, *, user_id: uuid.UUID, config_id: uuid.UUID) -> LLMProviderHealthCheckResult:
        config = await self.get(user_id=user_id, config_id=config_id)
        now = datetime.now(UTC)
        if config.last_checked_at and now - config.last_checked_at < timedelta(seconds=30):
            raise CooldownError("Health check is in cooldown")
        start = time.perf_counter()
        try:
            provider_reply, tool_calling_supported = await asyncio.wait_for(
                _ping_provider(config),
                timeout=settings.llm_provider_health_timeout_seconds,
            )
        except Exception as exc:
            config.status = LLMProviderStatus.ERROR
            config.last_checked_at = now
            config.last_check_error = _sanitize_error(exc)
            await self.db.flush()
            raise ProviderUnavailableError(config.last_check_error) from exc
        if tool_calling_supported is not True:
            config.status = LLMProviderStatus.ERROR
            config.last_checked_at = now
            config.last_check_error = TOOL_CALLING_REQUIRED_MESSAGE
            await self.db.flush()
            raise ProviderCapabilityError(TOOL_CALLING_REQUIRED_MESSAGE)
        config.status = LLMProviderStatus.ACTIVE
        config.last_checked_at = now
        config.last_check_error = None
        await self.db.flush()
        await self.db.refresh(config)
        response_time_ms = max(0, round((time.perf_counter() - start) * 1000))
        return LLMProviderHealthCheckResult(
            config=config,
            response_time_ms=response_time_ms,
            provider_reply=provider_reply,
            tool_calling_supported=tool_calling_supported,
        )

    async def _unset_user_default(self, user_id: uuid.UUID, exclude_id: uuid.UUID | None = None) -> None:
        query = update(LLMProviderConfig).where(
            LLMProviderConfig.user_id == user_id, LLMProviderConfig.is_default.is_(True)
        )
        if exclude_id:
            query = query.where(LLMProviderConfig.id != exclude_id)
        await self.db.execute(query.values(is_default=False))

    async def _get_first_user_config(self, user_id: uuid.UUID) -> LLMProviderConfig | None:
        result = await self.db.execute(
            select(LLMProviderConfig)
            .where(LLMProviderConfig.user_id == user_id, LLMProviderConfig.status != LLMProviderStatus.DISABLED)
            .order_by(LLMProviderConfig.is_default.desc(), LLMProviderConfig.created_at, LLMProviderConfig.id)
        )
        return result.scalars().first()

    async def _disable_user_configs_except(self, user_id: uuid.UUID, config_id: uuid.UUID) -> None:
        await self.db.execute(
            update(LLMProviderConfig)
            .where(
                LLMProviderConfig.user_id == user_id,
                LLMProviderConfig.id != config_id,
                LLMProviderConfig.status != LLMProviderStatus.DISABLED,
            )
            .values(status=LLMProviderStatus.DISABLED, is_default=False)
        )

    def _key_schema(self, body: LLMProviderKeyRequest | dict[str, Any]) -> LLMProviderKeyRequest:
        if isinstance(body, LLMProviderKeyRequest):
            return body
        return LLMProviderKeyRequest.model_validate(body)

    def _update_schema(self, body: LLMProviderUpdateRequest | dict[str, Any]) -> LLMProviderUpdateRequest:
        if isinstance(body, LLMProviderUpdateRequest):
            return body
        return LLMProviderUpdateRequest.model_validate(body)

    def _values_from_key_request(self, body: LLMProviderKeyRequest) -> dict[str, Any]:
        provider_type = ProviderType((body.provider_type or DEFAULT_PROVIDER_TYPE).value)
        provider_name = provider_type.value
        if provider_type == ProviderType.CUSTOM:
            if not body.base_url or not body.model_name:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="base_url and model_name are required",
                )
            model_name = body.model_name
        else:
            model_name = body.model_name or DEFAULT_MODEL_BY_PROVIDER[provider_type]
        values: dict[str, Any] = {
            "provider_type": provider_type,
            "name": provider_name,
            "base_url": body.base_url if provider_type == ProviderType.CUSTOM else None,
            "region": body.region,
            "model_name": model_name,
            "strong_model_name": body.strong_model_name,
            "encrypted_api_key": encrypt_token(body.api_key),
            "encrypted_secret_key": encrypt_token(body.secret_key) if body.secret_key else None,
        }
        return values

    def _values_from_update_request(
        self, body: LLMProviderUpdateRequest, config: LLMProviderConfig
    ) -> dict[str, Any]:
        sent_fields = body.model_fields_set
        values: dict[str, Any] = {}
        if "region" in sent_fields:
            values["region"] = body.region
        if "base_url" in sent_fields:
            if config.provider_type != ProviderType.CUSTOM:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="base_url is only supported for custom provider",
                )
            if not body.base_url:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="base_url is required")
            values["base_url"] = body.base_url
        if "model_name" in sent_fields:
            if config.provider_type == ProviderType.CUSTOM:
                if not body.model_name:
                    raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="model_name is required")
                values["model_name"] = body.model_name
            else:
                values["model_name"] = body.model_name or DEFAULT_MODEL_BY_PROVIDER[config.provider_type]
        if "strong_model_name" in sent_fields:
            values["strong_model_name"] = body.strong_model_name
        return values


async def _ping_provider(config: LLMProviderConfig) -> tuple[str | None, bool | None]:
    api_key = _resolve_api_key(config)
    secret_key = _resolve_secret_key(config)
    client = LLMClientFactory.create(
        provider_type=config.provider_type,
        api_key=api_key,
        secret_key=secret_key,
        region=config.region,
        model=config.model_name,
        base_url=config.base_url,
    )
    reply = await client.ping()
    try:
        tool_calling_supported = await client.ping_tool_calling()
    except Exception:
        tool_calling_supported = None
    return reply, tool_calling_supported
