import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.database as db_module
from app.database import get_db
from app.models.base import Base
from tests.golden_decision_helpers import graph_from, make_node_factory

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

TestSessionFactory = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

db_module.engine = test_engine
db_module.async_session_factory = TestSessionFactory

# app.main (and the routers it pulls in) must import after the patch above, since routers bind
# `async_session_factory` by name at import time — importing earlier would capture the real,
# unpatched production session factory.
from app.main import app  # noqa: E402
import app.main as main_module  # noqa: E402

main_module.engine = test_engine


@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_tables():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(autouse=True)
async def db_session():
    async with TestSessionFactory() as session:
        async def _override_get_db():
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

        app.dependency_overrides[get_db] = _override_get_db
        yield session
        app.dependency_overrides.pop(get_db, None)
        await session.rollback()


@pytest.fixture
def decision_node_factory():
    """Mint schema-complete DecisionNode dicts with auto-incremented ids.

    Usage: node = decision_node_factory(kind="decision", status="confirmed", depends_on=["N1"]).
    Pass id="N1" to pin a stable name when later assertions reference it.
    """
    return make_node_factory()


@pytest.fixture
def decision_graph_factory(decision_node_factory):
    """Build a decision_nodes dict from keyword specs, indexed by id.

    Usage:
        graph = decision_graph_factory(
            {"id": "N1", "kind": "decision", "status": "confirmed"},
            {"id": "N3", "depends_on": ["N1"]},
        )
    """

    def _build(*node_specs: dict) -> dict[str, dict]:
        return graph_from([decision_node_factory(**spec) for spec in node_specs])

    return _build


BASE = "/api/v1"


@pytest_asyncio.fixture
async def client(db_session):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def registered_user(client):
    payload = {
        "email": "alice@example.com",
        "password": "Secret123!",
        "full_name": "Alice Test",
    }
    resp = await client.post(f"{BASE}/auth/register", json=payload)
    assert resp.status_code == 201
    return payload


@pytest_asyncio.fixture
async def auth_headers(client, registered_user):
    resp = await client.post(
        f"{BASE}/auth/login",
        json={
            "email": registered_user["email"],
            "password": registered_user["password"],
        },
    )
    assert resp.status_code == 200
    token = resp.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def second_user(client):
    payload = {
        "email": "bob@example.com",
        "password": "Secret123!",
        "full_name": "Bob Test",
    }
    resp = await client.post(f"{BASE}/auth/register", json=payload)
    assert resp.status_code == 201
    return payload


@pytest_asyncio.fixture
async def second_auth_headers(client, second_user):
    resp = await client.post(
        f"{BASE}/auth/login",
        json={
            "email": second_user["email"],
            "password": second_user["password"],
        },
    )
    assert resp.status_code == 200
    token = resp.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def org(client, auth_headers):
    resp = await client.post(
        f"{BASE}/orgs",
        json={"name": "ACME Corp", "slug": "acme-corp"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    return resp.json()


@pytest_asyncio.fixture
async def project(client, auth_headers, org):
    resp = await client.post(
        f"{BASE}/orgs/{org['id']}/projects",
        json={"name": "Alpha Project", "slug": "alpha", "description": "Test project"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    return resp.json()
