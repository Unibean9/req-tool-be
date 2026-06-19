import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.main import app
from tests.conftest import BASE, TestSessionFactory


@pytest.mark.asyncio
async def test_create_org_commits_membership_for_next_request_project_create(db_session):
    async def get_db_without_auto_commit():
        async with TestSessionFactory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    email = f"org-owner-{uuid.uuid4().hex[:8]}@example.com"
    password = "Secret123!"
    previous_override = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = get_db_without_auto_commit
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            register_response = await client.post(
                f"{BASE}/auth/register",
                json={"email": email, "password": password, "full_name": "Chủ sở hữu org"},
            )
            assert register_response.status_code == 201

            login_response = await client.post(f"{BASE}/auth/login", json={"email": email, "password": password})
            assert login_response.status_code == 200
            headers = {"Authorization": f"Bearer {login_response.json()['data']['access_token']}"}

            org = await client.post(f"{BASE}/orgs", headers=headers, json={"name": "Org regression"})
            assert org.status_code == 201
            org_id = org.json()["data"]["id"]

            project = await client.post(
                f"{BASE}/orgs/{org_id}/projects",
                headers=headers,
                json={"name": "Project regression", "description": "Kiểm tra membership đã commit"},
            )
            assert project.status_code == 201
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = previous_override
