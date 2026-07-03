"""Fixtures for the API-level behavior-scenario suite.

Wires a REAL compiled graph + DelegatingCheckpointer into `app.state` so the HTTP
endpoints drive genuine multi-turn conversations, and redirects every binding
that would otherwise hit the real database / a real LLM:

- A dedicated **file-based** SQLite engine (`ScenarioSessionFactory`). The parent
  conftest uses an in-memory `StaticPool` (one shared connection) — fatal here,
  because the graph, the checkpointer and each HTTP request open their own
  sessions and their commits/rollbacks would clobber each other on that single
  connection. A file DB gives every session its own connection with real
  transaction isolation while still sharing committed data.
- `app.routers.agent_sessions.async_session_factory` and the `get_db` dependency
  -> the scenario factory (the router captured the real one at import time).
- `AgentService._resolve_llm_client` -> the current scenario's ScriptedLLM.
- `agent_service.asyncio.create_task` -> a sink, so fire-and-forget graph turns
  are captured and awaited sequentially by the driver (never concurrent with an
  open request session).
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.services.agent_service as agent_service_module
from app.database import get_db
from app.graphs.checkpointer import AgentSessionCheckpointer, DelegatingCheckpointer
from app.graphs.graph import build_graph
from app.main import app
from app.models.agent import AgentSession
from app.models.base import Base
from app.services.agent_service import AgentService
from tests.helpers import create_org, create_project, make_auth_headers

# Keep the DB file in the workspace so scenario tests do not depend on Windows Temp capacity.
_DB_PATH = Path(__file__).parents[3] / ".pytest_cache" / "reqtool_scenarios.sqlite"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_SCENARIO_DB_URL = f"sqlite+aiosqlite:///{_DB_PATH.as_posix()}"
scenario_engine = create_async_engine(_SCENARIO_DB_URL)
ScenarioSessionFactory = async_sessionmaker(scenario_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _scenario_tables():
    # Re-exported into tests/eval/conftest.py, so one pytest session may run this once per
    # directory. Dispose pooled connections before deleting the file (Windows blocks unlink on an
    # open handle) and tolerate the shared teardown running twice.
    await scenario_engine.dispose()
    if _DB_PATH.exists():
        _DB_PATH.unlink()
    async with scenario_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await scenario_engine.dispose()
    if _DB_PATH.exists():
        try:
            _DB_PATH.unlink()
        except PermissionError:
            pass


@pytest_asyncio.fixture(autouse=True)
async def db_session():
    """Override the parent autouse fixture: hold NO long-lived shared session.

    Scenario tests route every DB access through the per-request `get_db`
    override (installed by `scenario_env`), so nothing pins a connection.
    """
    yield None


class _AsyncioProxy:
    """Stand-in for `asyncio` inside agent_service.

    Forwards everything to the real module EXCEPT `create_task`, which captures
    the coroutine into a sink instead of scheduling it. The driver awaits those
    sequentially, giving a deterministic turn order (real background scheduling
    races with the checkpointer's pending-writes handling).
    """

    def __init__(self, sink: list) -> None:
        self._sink = sink

    def create_task(self, coro):  # noqa: ANN001
        self._sink.append(coro)
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)


class ScenarioEnv:
    """Per-test environment: holds the active ScriptedLLM and drains the graph."""

    def __init__(self) -> None:
        self.llm: Any = None
        self.pending: list = []

    def set_llm(self, llm: Any) -> None:
        self.llm = llm

    async def session_status(self, session_id: uuid.UUID) -> str | None:
        async with ScenarioSessionFactory() as db:
            row = (
                await db.execute(select(AgentSession).where(AgentSession.id == session_id))
            ).scalar_one_or_none()
            return row.status.value if row else None

    async def get_checkpoint_field(self, session_id: uuid.UUID, field: str) -> Any:
        values = await self._checkpoint_values(session_id)
        return values.get(field)

    async def get_checkpoint_fields(self, session_id: uuid.UUID, fields: tuple[str, ...]) -> dict[str, Any]:
        """Read several checkpoint channel values in one checkpointer round trip."""
        values = await self._checkpoint_values(session_id)
        return {field: values.get(field) for field in fields}

    async def _checkpoint_values(self, session_id: uuid.UUID) -> dict[str, Any]:
        checkpointer = AgentSessionCheckpointer(
            session_id=str(session_id),
            session_factory=ScenarioSessionFactory,
        )
        checkpoint = await checkpointer.aget_tuple({"configurable": {"thread_id": str(session_id)}})
        if checkpoint is None:
            return {}
        return checkpoint.checkpoint.get("channel_values") or {}

    async def drain(self, session_id: uuid.UUID, *, max_coros: int = 50) -> str | None:
        """Run every captured graph coroutine to completion, in order."""
        ran = 0
        while self.pending and ran < max_coros:
            coro = self.pending.pop(0)
            await coro
            ran += 1
        return await self.session_status(session_id)


@pytest_asyncio.fixture
async def scenario_env(monkeypatch):
    """Install the real graph + redirected bindings; yield a ScenarioEnv."""
    env = ScenarioEnv()

    app.state.compiled_graph = build_graph(DelegatingCheckpointer(session_factory=ScenarioSessionFactory))
    monkeypatch.setattr("app.routers.agent_sessions.async_session_factory", ScenarioSessionFactory)
    monkeypatch.setattr(agent_service_module, "asyncio", _AsyncioProxy(env.pending))

    async def _fake_resolve(self, provider_config_id):  # noqa: ANN001
        return env.llm, env.llm

    monkeypatch.setattr(AgentService, "_resolve_llm_client", _fake_resolve)

    async def _scenario_get_db():
        async with ScenarioSessionFactory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    previous_override = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = _scenario_get_db
    yield env
    if previous_override is not None:
        app.dependency_overrides[get_db] = previous_override
    else:
        app.dependency_overrides.pop(get_db, None)

    if hasattr(app.state, "compiled_graph"):
        del app.state.compiled_graph


@pytest_asyncio.fixture
async def scenario_project(client, scenario_env):
    """A fresh authenticated user + org + project.

    The user is created on an explicitly-committed scenario session
    (make_auth_headers only flushes); org/project go through HTTP.
    """
    async with ScenarioSessionFactory() as session:
        headers = await make_auth_headers(client, session)
        await session.commit()
    org = await create_org(client, headers)
    proj = await create_project(client, headers, org["id"])
    return headers, proj
