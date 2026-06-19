import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

from app.config import settings
from app.core import crypto
from app.models.llm_provider import LLMProviderConfig, LLMProviderStatus, ProviderType
from app.schemas.llm_provider import LLMProviderConfigRead, LLMProviderHealthCheckResult
from app.services.llm_provider_service import (
    CooldownError,
    LLMProviderService,
    ProviderUnavailableError,
    _resolve_api_key,
)
from tests.conftest import BASE
from tests.helpers import make_auth_headers


@pytest.fixture(autouse=True)
def clear_crypto_cache():
    crypto._get_fernet.cache_clear()
    yield
    crypto._get_fernet.cache_clear()


def test_encrypt_decrypt_roundtrip(monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(settings, "encryption_key_previous", "")
    crypto._get_fernet.cache_clear()

    token = crypto.encrypt_token("sk-test-vietnam")

    assert token != "sk-test-vietnam"
    assert crypto.decrypt_token(token) == "sk-test-vietnam"


def test_multifernet_rotation_decrypts_with_old_key(monkeypatch):
    old_key = Fernet.generate_key().decode()
    new_key = Fernet.generate_key().decode()
    monkeypatch.setattr(settings, "encryption_key", old_key)
    monkeypatch.setattr(settings, "encryption_key_previous", "")
    crypto._get_fernet.cache_clear()
    token = crypto.encrypt_token("sk-old-key")

    monkeypatch.setattr(settings, "encryption_key", new_key)
    monkeypatch.setattr(settings, "encryption_key_previous", old_key)
    crypto._get_fernet.cache_clear()

    assert crypto.decrypt_token(token) == "sk-old-key"


def test_multifernet_rotation_rejects_unknown_key(monkeypatch):
    token = Fernet(Fernet.generate_key()).encrypt(b"sk-unknown").decode()
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(settings, "encryption_key_previous", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()

    assert crypto.decrypt_token(token) is None


def test_llm_provider_api_key_required_constraint_sqlite_skip(db_session):
    if db_session.bind.dialect.name != "postgresql":
        pytest.skip("Chỉ kiểm tra DB-level constraint trên PostgreSQL")
    names = {item.name for item in LLMProviderConfig.__table_args__ if getattr(item, "name", None)}
    assert "ck_llm_provider_api_key_required" in names


@pytest.mark.asyncio
async def test_create_config_stores_encrypted_key(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)

    row = await LLMProviderService(db_session).create(
        user_id=user_id,
        body=_create_body(api_key="sk-secret-value"),
    )

    assert row.encrypted_api_key
    assert row.encrypted_api_key != "sk-secret-value"
    assert row.provider_type == ProviderType.OPENAI
    assert row.name == "openai"
    assert row.model_name == "gpt-4o-mini"
    assert row.is_default is True


@pytest.mark.asyncio
async def test_create_config_stores_user_selected_model_name(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)

    row = await LLMProviderService(db_session).create(
        user_id=user_id,
        body={
            "provider_type": "bedrock",
            "api_key": "AKIATEST",
            "secret_key": "aws-secret",
            "region": "us-east-1",
            "model_name": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        },
    )

    assert row.provider_type == ProviderType.BEDROCK
    assert row.model_name == "anthropic.claude-3-5-sonnet-20241022-v2:0"


@pytest.mark.asyncio
async def test_post_config_accepts_model_name_and_returns_it(client, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)

    resp = await client.post(
        f"{BASE}/users/me/llm-provider-configs",
        json={
            "provider_type": "openai",
            "api_key": "sk-secret",
            "model_name": "gpt-4.1-mini",
        },
        headers=headers,
    )

    assert resp.status_code == 201, resp.text
    assert resp.json()["data"]["model_name"] == "gpt-4.1-mini"


@pytest.mark.asyncio
async def test_read_config_response_redacts_key(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)
    row = await LLMProviderService(db_session).create(
        user_id=user_id,
        body=_create_body(api_key="sk-redacted"),
    )

    payload = LLMProviderConfigRead.model_validate(row).model_dump()

    assert payload["api_key_set"] is True
    assert "encrypted_api_key" not in payload
    assert "api_key" not in payload


@pytest.mark.asyncio
async def test_create_rejects_secret_ref_payload(client):
    headers = await make_auth_headers(client)

    resp = await client.post(
        f"{BASE}/users/me/llm-provider-configs",
        json={"api_key": "sk-test", "secret_ref": "LLM_KEY_TEST"},
        headers=headers,
    )

    assert resp.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_type", ["bedrock", "openai", "google", "anthropic"])
async def test_supported_provider_types_are_accepted(client, provider_type):
    headers = await make_auth_headers(client)

    resp = await client.post(
        f"{BASE}/users/me/llm-provider-configs",
        json={"provider_type": provider_type, "api_key": "sk-provider"},
        headers=headers,
    )

    assert resp.status_code == 201, resp.text
    assert resp.json()["data"]["provider_type"] == provider_type


@pytest.mark.asyncio
async def test_unsupported_provider_type_is_rejected(client):
    headers = await make_auth_headers(client)

    resp = await client.post(
        f"{BASE}/users/me/llm-provider-configs",
        json={"provider_type": "openai_compatible", "api_key": "sk-provider"},
        headers=headers,
    )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_bedrock_api_key_config_stores_encrypted_key(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)

    row = await LLMProviderService(db_session).create(
        user_id=user_id,
        body={"provider_type": "bedrock", "api_key": "bedrock-api-key"},
    )

    assert row.provider_type == ProviderType.BEDROCK
    assert row.encrypted_api_key
    assert row.encrypted_secret_key is None
    assert row.name == "bedrock"
    assert row.region is None


@pytest.mark.asyncio
async def test_secret_key_config_stores_encrypted_optional_secret(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)

    row = await LLMProviderService(db_session).create(
        user_id=user_id,
        body={
            "provider_type": "bedrock",
            "api_key": "AKIATEST",
            "secret_key": "aws-secret",
            "region": "ap-southeast-1",
        },
    )

    assert row.provider_type == ProviderType.BEDROCK
    assert row.encrypted_api_key != "AKIATEST"
    assert row.encrypted_secret_key != "aws-secret"
    assert row.region == "ap-southeast-1"


@pytest.mark.asyncio
async def test_resolve_api_key_decrypt_failure_raises(monkeypatch):
    config = LLMProviderConfig(user_id=uuid.uuid4(), provider_type=ProviderType.OPENAI, name="Bad", encrypted_api_key="bad")
    monkeypatch.setattr("app.services.llm_provider_service.decrypt_token", lambda _value: None)

    with pytest.raises(ValueError, match="could not be decrypted"):
        _resolve_api_key(config)


@pytest.mark.asyncio
async def test_create_updates_existing_user_config(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)
    service = LLMProviderService(db_session)

    first = await service.create(user_id=user_id, body=_create_body(api_key="sk-one"))
    second = await service.create(user_id=user_id, body=_create_body(api_key="sk-two"))
    await db_session.flush()
    await db_session.refresh(first)
    rows = (
        await db_session.execute(
            select(LLMProviderConfig).where(
                LLMProviderConfig.user_id == user_id,
                LLMProviderConfig.status != LLMProviderStatus.DISABLED,
            )
        )
    ).scalars().all()

    assert first.id == second.id
    assert second.is_default is True
    assert len(rows) == 1
    assert _resolve_api_key(second) == "sk-two"


@pytest.mark.asyncio
async def test_health_check_api_key_success(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)
    service = LLMProviderService(db_session)
    row = await service.create(user_id=user_id, body=_create_body(api_key="sk-health"))

    async def fake_ping(config):
        assert config.id == row.id
        return "pong"

    times = iter([10.0, 10.123])
    monkeypatch.setattr("app.services.llm_provider_service._ping_provider", fake_ping)
    monkeypatch.setattr("app.services.llm_provider_service.time.perf_counter", lambda: next(times))

    checked = await service.health_check(user_id=user_id, config_id=row.id)

    assert checked.config.status == LLMProviderStatus.ACTIVE
    assert checked.config.last_checked_at is not None
    assert checked.response_time_ms == 123
    assert checked.provider_reply == "pong"


@pytest.mark.asyncio
async def test_health_check_non_default_provider_api_key_success(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)
    service = LLMProviderService(db_session)
    row = await service.create(
        user_id=user_id,
        body={"provider_type": "bedrock", "api_key": "bedrock-api-key"},
    )
    monkeypatch.setattr("app.services.llm_provider_service._ping_provider", lambda _config: _async_reply("pong"))

    checked = await service.health_check(user_id=user_id, config_id=row.id)

    assert checked.config.status == LLMProviderStatus.ACTIVE


@pytest.mark.asyncio
async def test_health_check_api_key_with_secret_key_success(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)
    service = LLMProviderService(db_session)
    row = await service.create(
        user_id=user_id,
        body={
            "provider_type": "bedrock",
            "api_key": "AKIATEST",
            "secret_key": "aws-secret",
            "region": "ap-southeast-1",
        },
    )
    monkeypatch.setattr("app.services.llm_provider_service._ping_provider", lambda _config: _async_reply("pong"))
    checked = await service.health_check(user_id=user_id, config_id=row.id)

    assert checked.config.status == LLMProviderStatus.ACTIVE


@pytest.mark.asyncio
async def test_health_check_cooldown_rejects(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)
    service = LLMProviderService(db_session)
    row = await service.create(user_id=user_id, body=_create_body(api_key="sk-cooldown"))
    row.last_checked_at = datetime.now(UTC)
    await db_session.flush()

    with pytest.raises(CooldownError):
        await service.health_check(user_id=user_id, config_id=row.id)


@pytest.mark.asyncio
async def test_rotate_key_resets_status_to_draft(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)
    service = LLMProviderService(db_session)
    row = await service.create(user_id=user_id, body=_create_body(api_key="sk-old"))
    row.status = LLMProviderStatus.ACTIVE
    await db_session.flush()

    updated = await service.update(user_id=user_id, config_id=row.id, body={"api_key": "sk-new"})

    assert updated.status == LLMProviderStatus.DRAFT
    assert _resolve_api_key(updated) == "sk-new"


@pytest.mark.asyncio
async def test_update_config_stores_user_selected_model_name(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)
    service = LLMProviderService(db_session)
    row = await service.create(user_id=user_id, body=_create_body(api_key="sk-old"))

    updated = await service.update(
        user_id=user_id,
        config_id=row.id,
        body={
            "provider_type": "openai",
            "api_key": "sk-new",
            "model_name": "gpt-4.1-mini",
        },
    )

    assert updated.status == LLMProviderStatus.DRAFT
    assert updated.model_name == "gpt-4.1-mini"
    assert _resolve_api_key(updated) == "sk-new"


@pytest.mark.asyncio
async def test_health_check_error_sanitized_before_storage(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)
    service = LLMProviderService(db_session)
    row = await service.create(user_id=user_id, body=_create_body(api_key="sk-real-secret"))

    def fail_decrypt(_value):
        raise RuntimeError("provider echoed sk-leaked-secret-token")

    monkeypatch.setattr("app.services.llm_provider_service.decrypt_token", fail_decrypt)

    with pytest.raises(ProviderUnavailableError):
        await service.health_check(user_id=user_id, config_id=row.id)
    await db_session.refresh(row)

    assert "sk-leaked-secret-token" not in row.last_check_error
    assert "[REDACTED]" in row.last_check_error


@pytest.mark.asyncio
async def test_post_config_requires_auth(client):
    resp = await client.post(
        f"{BASE}/users/me/llm-provider-configs",
        json=_create_body(api_key="sk-secret"),
    )

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_post_config_returns_201_with_redacted_body(client, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)

    resp = await client.post(
        f"{BASE}/users/me/llm-provider-configs",
        json=_create_body(api_key="sk-secret"),
        headers=headers,
    )

    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    assert data["api_key_set"] is True
    assert "encrypted_api_key" not in data
    assert "api_key" not in data


@pytest.mark.asyncio
async def test_get_config_list_scoped_to_user(client, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers_a = await make_auth_headers(client)
    headers_b = await make_auth_headers(client)
    user_id_a = str(await _user_id_from_headers(headers_a))
    await client.post(f"{BASE}/users/me/llm-provider-configs", json=_create_body(api_key="sk-1"), headers=headers_a)
    await client.post(f"{BASE}/users/me/llm-provider-configs", json=_create_body(api_key="sk-2"), headers=headers_b)

    resp = await client.get(f"{BASE}/users/me/llm-provider-configs", headers=headers_a)

    assert resp.status_code == 200
    items = resp.json()["data"]
    assert len(items) == 1
    assert items[0]["user_id"] == user_id_a


@pytest.mark.asyncio
async def test_patch_another_users_config_returns_404(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers_a = await make_auth_headers(client)
    headers_b = await make_auth_headers(client)
    create_resp = await client.post(f"{BASE}/users/me/llm-provider-configs", json=_create_body(api_key="sk-1"), headers=headers_a)
    config_id = create_resp.json()["data"]["id"]

    resp = await client.patch(
        f"{BASE}/users/me/llm-provider-configs/{config_id}",
        json={"api_key": "sk-new"},
        headers=headers_b,
    )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_config_returns_204(client, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    create_resp = await client.post(f"{BASE}/users/me/llm-provider-configs", json=_create_body(api_key="sk-1"), headers=headers)
    config_id = create_resp.json()["data"]["id"]

    delete_resp = await client.delete(f"{BASE}/users/me/llm-provider-configs/{config_id}", headers=headers)
    get_resp = await client.get(f"{BASE}/users/me/llm-provider-configs/{config_id}", headers=headers)

    assert delete_resp.status_code == 204
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_health_check_endpoint_returns_200_on_success(client, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    create_resp = await client.post(f"{BASE}/users/me/llm-provider-configs", json=_create_body(api_key="sk-1"), headers=headers)
    config_id = create_resp.json()["data"]["id"]

    async def fake_health_check(self, user_id, config_id):
        config = await self.get(user_id=user_id, config_id=config_id)
        config.status = LLMProviderStatus.ACTIVE
        return LLMProviderHealthCheckResult(config=config, response_time_ms=42, provider_reply="pong")

    monkeypatch.setattr(LLMProviderService, "health_check", fake_health_check)

    resp = await client.post(f"{BASE}/users/me/llm-provider-configs/{config_id}/health-check", headers=headers)

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["config"]["status"] == "active"
    assert data["response_time_ms"] == 42
    assert data["provider_reply"] == "pong"


@pytest.mark.asyncio
async def test_health_check_endpoint_returns_503_on_provider_failure(client, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    create_resp = await client.post(f"{BASE}/users/me/llm-provider-configs", json=_create_body(api_key="sk-1"), headers=headers)
    config_id = create_resp.json()["data"]["id"]

    async def fake_health_check(self, user_id, config_id):
        raise ProviderUnavailableError("Provider lỗi")

    monkeypatch.setattr(LLMProviderService, "health_check", fake_health_check)

    resp = await client.post(f"{BASE}/users/me/llm-provider-configs/{config_id}/health-check", headers=headers)

    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_health_check_endpoint_returns_429_on_cooldown(client, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    create_resp = await client.post(f"{BASE}/users/me/llm-provider-configs", json=_create_body(api_key="sk-1"), headers=headers)
    config_id = create_resp.json()["data"]["id"]

    async def fake_health_check(self, user_id, config_id):
        raise CooldownError("Thử lại sau")

    monkeypatch.setattr(LLMProviderService, "health_check", fake_health_check)

    resp = await client.post(f"{BASE}/users/me/llm-provider-configs/{config_id}/health-check", headers=headers)

    assert resp.status_code == 429


def test_router_does_not_reference_resolve_api_key():
    import ast
    from pathlib import Path

    tree = ast.parse(Path("app/routers/llm_providers.py").read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert "_resolve_api_key" not in names
    assert "get_provider_key_for_use" not in names


@pytest.mark.asyncio
async def test_full_lifecycle_create_check_rotate(client, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    create_resp = await client.post(f"{BASE}/users/me/llm-provider-configs", json=_create_body(api_key="sk-1"), headers=headers)
    config_id = create_resp.json()["data"]["id"]

    async def fake_health_check(self, user_id, config_id):
        config = await self.get(user_id=user_id, config_id=config_id)
        config.status = LLMProviderStatus.ACTIVE
        config.last_checked_at = datetime.now(UTC) - timedelta(seconds=31)
        return LLMProviderHealthCheckResult(config=config, response_time_ms=42)

    monkeypatch.setattr(LLMProviderService, "health_check", fake_health_check)
    health_resp = await client.post(f"{BASE}/users/me/llm-provider-configs/{config_id}/health-check", headers=headers)
    patch_resp = await client.patch(
        f"{BASE}/users/me/llm-provider-configs/{config_id}",
        json={"api_key": "sk-2"},
        headers=headers,
    )

    assert health_resp.status_code == 200
    assert patch_resp.json()["data"]["status"] == "draft"


@pytest.mark.asyncio
async def test_secret_ref_not_supported_end_to_end(client):
    headers = await make_auth_headers(client)

    resp = await client.post(
        f"{BASE}/users/me/llm-provider-configs",
        json={"api_key": "sk-test", "secret_ref": "OPENAI_API_KEY"},
        headers=headers,
    )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_response_body_never_contains_key(client, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    create_resp = await client.post(f"{BASE}/users/me/llm-provider-configs", json=_create_body(api_key="sk-no-leak"), headers=headers)
    config_id = create_resp.json()["data"]["id"]
    get_resp = await client.get(f"{BASE}/users/me/llm-provider-configs/{config_id}", headers=headers)
    patch_resp = await client.patch(f"{BASE}/users/me/llm-provider-configs/{config_id}", json={"api_key": "sk-new-no-leak"}, headers=headers)

    for resp in (create_resp, get_resp, patch_resp):
        body = resp.text
        assert "encrypted_api_key" not in body
        assert "sk-no-leak" not in body
        assert "sk-new-no-leak" not in body
        assert "api_key" not in resp.json()["data"]


@pytest.mark.asyncio
async def test_response_body_never_contains_secret_key(client, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)

    create_resp = await client.post(
        f"{BASE}/users/me/llm-provider-configs",
        json={
            "provider_type": "bedrock",
            "api_key": "AKIA-NO-LEAK",
            "secret_key": "secret-no-leak",
        },
        headers=headers,
    )

    assert create_resp.status_code == 201
    data = create_resp.json()["data"]
    assert data["secret_key_set"] is True
    body = create_resp.text
    assert "AKIA-NO-LEAK" not in body
    assert "secret-no-leak" not in body
    assert "encrypted_secret_key" not in body
    assert "secret_key" not in data


@pytest.mark.asyncio
async def test_default_flag_isolation_across_users(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers_a = await make_auth_headers(client)
    headers_b = await make_auth_headers(client)
    user_id_a = await _user_id_from_headers(headers_a)
    user_id_b = await _user_id_from_headers(headers_b)
    service = LLMProviderService(db_session)

    config_a = await service.create(user_id=user_id_a, body=_create_body(api_key="sk-a"))
    config_b = await service.create(user_id=user_id_b, body=_create_body(api_key="sk-b"))

    assert config_a.is_default is True
    assert config_b.is_default is True


@pytest.mark.asyncio
async def test_no_plaintext_key_in_db(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)
    user_id = await _user_id_from_headers(headers)

    await client.post(
        f"{BASE}/users/me/llm-provider-configs",
        json=_create_body(api_key="sk-db-secret"),
        headers=headers,
    )
    row = (
        await db_session.execute(
            select(LLMProviderConfig).where(LLMProviderConfig.user_id == user_id)
        )
    ).scalar_one()

    assert isinstance(row.encrypted_api_key, str)
    assert row.encrypted_api_key != "sk-db-secret"


@pytest.mark.asyncio
async def test_no_api_key_in_logs_after_create(client, caplog, monkeypatch):
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    crypto._get_fernet.cache_clear()
    headers = await make_auth_headers(client)

    with caplog.at_level(logging.INFO):
        resp = await client.post(
            f"{BASE}/users/me/llm-provider-configs",
            json=_create_body(api_key="sk-log-secret"),
            headers=headers,
        )

    assert resp.status_code == 201
    assert "sk-log-secret" not in caplog.text


@pytest.mark.asyncio
async def test_partial_index_one_default_per_user(db_session):
    if db_session.bind.dialect.name != "postgresql":
        pytest.skip("Partial unique index chỉ verify trên PostgreSQL thật")
    user_id = uuid.uuid4()
    db_session.add_all(
        [
            LLMProviderConfig(user_id=user_id, provider_type=ProviderType.OPENAI, name="A", encrypted_api_key="a", is_default=True),
            LLMProviderConfig(user_id=user_id, provider_type=ProviderType.OPENAI, name="B", encrypted_api_key="b", is_default=True),
        ]
    )
    with pytest.raises(Exception):
        await db_session.flush()


def _create_body(**overrides):
    body = {
        "api_key": "sk-test",
    }
    body.update(overrides)
    return body


async def _user_id_from_headers(headers) -> uuid.UUID:
    from app.core.security import decode_token

    token = headers["Authorization"].split(" ", 1)[1]
    return uuid.UUID(decode_token(token)["sub"])


async def _async_reply(value):
    return value
