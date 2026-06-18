import asyncio
import os
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.crypto import decrypt_token, encrypt_token
from app.models.llm_provider import LLMProviderConfig, LLMProviderStatus, ProviderType
from app.schemas.llm_provider import LLMProviderConfigCreate, LLMProviderConfigUpdate

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - optional dependency
    AsyncOpenAI = None


class CooldownError(Exception):
    pass


class ProviderUnavailableError(Exception):
    pass


SECRET_PATTERN = re.compile(r"(?:sk|key|token)-[A-Za-z0-9_\-]+")


def _sanitize_error(message: str) -> str:
    return SECRET_PATTERN.sub("[REDACTED]", message)[:500]


def _resolve_api_key(config: LLMProviderConfig) -> str:
    if config.secret_ref:
        if not config.secret_ref.startswith("LLM_KEY_"):
            raise ValueError("secret_ref phải bắt đầu bằng LLM_KEY_")
        value = os.environ.get(config.secret_ref)
        if not value:
            raise ValueError("Không tìm thấy biến môi trường chứa API key")
        return value
    if config.encrypted_api_key:
        value = decrypt_token(config.encrypted_api_key)
        if value is None:
            raise ValueError("Encrypted API key could not be decrypted — possible key rotation mismatch")
        return value
    raise ValueError("Config không có API key")


class LLMProviderService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, *, project_id: uuid.UUID, body: LLMProviderConfigCreate | dict[str, Any]) -> LLMProviderConfig:
        body = self._create_schema(body)
        if body.is_default:
            await self._unset_project_default(project_id)
        config = LLMProviderConfig(
            project_id=project_id,
            provider_type=body.provider_type,
            name=body.name,
            base_url=body.base_url,
            model_name=body.model_name,
            secret_ref=body.secret_ref,
            encrypted_api_key=encrypt_token(body.api_key) if body.api_key else None,
            is_default=body.is_default,
            status=LLMProviderStatus.DRAFT,
        )
        self.db.add(config)
        await self.db.flush()
        return config

    async def list(self, *, project_id: uuid.UUID) -> list[LLMProviderConfig]:
        result = await self.db.execute(
            select(LLMProviderConfig)
            .where(LLMProviderConfig.project_id == project_id, LLMProviderConfig.status != LLMProviderStatus.DISABLED)
            .order_by(LLMProviderConfig.created_at, LLMProviderConfig.id)
        )
        return list(result.scalars().all())

    async def get(self, *, project_id: uuid.UUID, config_id: uuid.UUID) -> LLMProviderConfig:
        result = await self.db.execute(
            select(LLMProviderConfig).where(
                LLMProviderConfig.id == config_id,
                LLMProviderConfig.project_id == project_id,
                LLMProviderConfig.status != LLMProviderStatus.DISABLED,
            )
        )
        config = result.scalar_one_or_none()
        if config is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Không tìm thấy cấu hình LLM provider")
        return config

    async def update(
        self,
        *,
        project_id: uuid.UUID,
        config_id: uuid.UUID,
        body: LLMProviderConfigUpdate | dict[str, Any],
    ) -> LLMProviderConfig:
        schema = self._update_schema(body)
        config = await self.get(project_id=project_id, config_id=config_id)
        values = schema.model_dump(exclude_unset=True)
        key_changed = False
        if values.pop("api_key", None) is not None:
            values["encrypted_api_key"] = encrypt_token(schema.api_key or "")
            values["secret_ref"] = None
            key_changed = True
        elif "secret_ref" in values:
            values["encrypted_api_key"] = None
            key_changed = True
        if key_changed:
            values["status"] = LLMProviderStatus.DRAFT
            values["last_checked_at"] = None
            values["last_check_error"] = None
        if values.get("is_default") is True:
            await self._unset_project_default(project_id, exclude_id=config_id)
        if values:
            await self.db.execute(
                update(LLMProviderConfig)
                .where(LLMProviderConfig.id == config_id, LLMProviderConfig.project_id == project_id)
                .values(**values)
            )
            await self.db.flush()
            await self.db.refresh(config)
        return config

    async def delete(self, *, project_id: uuid.UUID, config_id: uuid.UUID) -> None:
        config = await self.get(project_id=project_id, config_id=config_id)
        config.status = LLMProviderStatus.DISABLED
        config.is_default = False
        await self.db.flush()

    async def health_check(self, *, project_id: uuid.UUID, config_id: uuid.UUID) -> LLMProviderConfig:
        config = await self.get(project_id=project_id, config_id=config_id)
        now = datetime.now(UTC)
        if config.last_checked_at and now - config.last_checked_at < timedelta(seconds=30):
            raise CooldownError("Health-check đang trong thời gian cooldown")
        handler = HEALTH_CHECKS.get(config.provider_type)
        if handler is None:
            raise ProviderUnavailableError("Provider chưa được hỗ trợ")
        try:
            await asyncio.wait_for(handler(config), timeout=5.0)
        except Exception as exc:
            config.status = LLMProviderStatus.ERROR
            config.last_checked_at = now
            config.last_check_error = _sanitize_error(str(exc))
            await self.db.flush()
            raise ProviderUnavailableError(config.last_check_error)
        config.status = LLMProviderStatus.ACTIVE
        config.last_checked_at = now
        config.last_check_error = None
        await self.db.flush()
        return config

    async def _unset_project_default(self, project_id: uuid.UUID, exclude_id: uuid.UUID | None = None) -> None:
        query = update(LLMProviderConfig).where(LLMProviderConfig.project_id == project_id, LLMProviderConfig.is_default.is_(True))
        if exclude_id:
            query = query.where(LLMProviderConfig.id != exclude_id)
        await self.db.execute(query.values(is_default=False))

    def _create_schema(self, body: LLMProviderConfigCreate | dict[str, Any]) -> LLMProviderConfigCreate:
        if isinstance(body, LLMProviderConfigCreate):
            return body
        return LLMProviderConfigCreate.model_validate(body)

    def _update_schema(self, body: LLMProviderConfigUpdate | dict[str, Any]) -> LLMProviderConfigUpdate:
        if isinstance(body, LLMProviderConfigUpdate):
            return body
        return LLMProviderConfigUpdate.model_validate(body)


async def _check_openai_compatible(config: LLMProviderConfig) -> None:
    if AsyncOpenAI is None:
        raise RuntimeError("openai package is not installed")
    client = AsyncOpenAI(api_key=_resolve_api_key(config), base_url=config.base_url)
    await client.models.list()


async def _check_bedrock(config: LLMProviderConfig) -> None:
    _resolve_api_key(config)
    await asyncio.to_thread(lambda: True)


async def _check_noop(config: LLMProviderConfig) -> None:
    _resolve_api_key(config)


HEALTH_CHECKS = {
    ProviderType.OPENAI_COMPATIBLE: _check_openai_compatible,
    ProviderType.AZURE_OPENAI: _check_noop,
    ProviderType.ANTHROPIC: _check_noop,
    ProviderType.BEDROCK: _check_bedrock,
    ProviderType.GEMINI: _check_noop,
}
