import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.main import app
from tests.conftest import BASE, TestSessionFactory


@pytest.mark.asyncio
async def test_register_commits_user_for_next_request_login(db_session):
    async def get_db_without_auto_commit():
        async with TestSessionFactory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    email = f"register-{uuid.uuid4().hex[:8]}@example.com"
    password = "Secret123!"
    previous_override = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = get_db_without_auto_commit
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            register_response = await client.post(
                f"{BASE}/auth/register",
                json={"email": email, "password": password, "full_name": "E2E User"},
            )
            assert register_response.status_code == 201

            login_response = await client.post(
                f"{BASE}/auth/login",
                json={"email": email, "password": password},
            )
            assert login_response.status_code == 200
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = previous_override
