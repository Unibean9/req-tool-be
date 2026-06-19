import uuid as uuid_mod

from tests.conftest import BASE


async def make_auth_headers(client, db_session=None) -> dict:
    from app.core.security import create_access_token, hash_password
    from app.database import get_db
    from app.main import app as _app
    from app.models.user import User

    if db_session is None:
        override = _app.dependency_overrides.get(get_db)
        if override is None:
            raise RuntimeError("Không có db_session hoặc override get_db đang hoạt động")
        gen = override()
        try:
            db_session = await gen.__anext__()
        except StopAsyncIteration:
            raise RuntimeError("Override get_db không trả về session")

    uid = uuid_mod.uuid4().hex[:8]
    user = User(
        email=f"user-{uid}@example.com",
        hashed_password=hash_password("Secret123!"),
        full_name=f"User {uid}",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()

    token = create_access_token(str(user.id), user.role)
    return {"Authorization": f"Bearer {token}"}


async def create_org(client, h: dict) -> dict:
    slug = f"org-{uuid_mod.uuid4().hex[:8]}"
    resp = await client.post(f"{BASE}/orgs", json={"name": slug, "slug": slug}, headers=h)
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]


async def create_project(client, h: dict, org_id: str) -> dict:
    slug = f"proj-{uuid_mod.uuid4().hex[:8]}"
    resp = await client.post(
        f"{BASE}/orgs/{org_id}/projects",
        json={"name": slug, "description": "test"},
        headers=h,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]
