import base64
import uuid
from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.base import CheckpointTuple, empty_checkpoint
from sqlalchemy import select

from app.graphs.checkpointer import AgentSessionCheckpointer
from app.models.agent import AgentSession
from tests.conftest import TestSessionFactory
from tests.helpers import create_org, create_project, make_auth_headers


@pytest.mark.asyncio
async def test_aput_persists_base64_checkpoint(client, db_session):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    checkpoint = _checkpoint()

    result_config = await checkpointer.aput(_config(session.id), checkpoint, {"source": "unit"}, {"messages": "1"})

    stored = await _load_agent_session(session.id)
    raw = stored.graph_checkpoint
    assert result_config["configurable"]["thread_id"] == str(session.id)
    assert result_config["configurable"]["checkpoint_id"] == checkpoint["id"]
    assert raw["metadata"] == {"source": "unit"}
    assert raw["new_versions"] == {"messages": "1"}
    assert isinstance(base64.b64decode(raw["data"]), bytes)
    assert raw["serde_type"]


@pytest.mark.asyncio
async def test_aget_tuple_returns_checkpoint_after_aput(client, db_session):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    checkpoint = _checkpoint()
    config = _config(session.id)

    await checkpointer.aput(config, checkpoint, {"step": 1}, {"messages": "1"})
    loaded = await checkpointer.aget_tuple(config)

    assert isinstance(loaded, CheckpointTuple)
    assert loaded.config["configurable"]["thread_id"] == str(session.id)
    assert loaded.config["configurable"]["checkpoint_id"] == checkpoint["id"]
    assert loaded.checkpoint == checkpoint
    assert loaded.metadata == {"step": 1}
    assert loaded.parent_config is None


@pytest.mark.asyncio
async def test_checkpoint_round_trip_preserves_data(client, db_session):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    checkpoint = _checkpoint()
    checkpoint["channel_values"]["analysis_result"] = {"confidence": 0.82, "next_action": "ask_human"}

    await checkpointer.aput(_config(session.id), checkpoint, {"source": "round-trip"}, {})
    loaded = await checkpointer.aget_tuple(_config(session.id))

    assert loaded is not None
    assert loaded.checkpoint == checkpoint


@pytest.mark.asyncio
async def test_aget_tuple_without_checkpoint_returns_none(client, db_session):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())

    assert await checkpointer.aget_tuple(_config(session.id)) is None


@pytest.mark.asyncio
async def test_aput_writes_persists_pending_writes_and_alist_returns_latest(client, db_session):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    checkpoint = _checkpoint()
    config = await checkpointer.aput(_config(session.id), checkpoint, {"source": "list"}, {})

    await checkpointer.aput_writes(config, [("messages", [{"role": "user", "content": "ok"}])], task_id="task-1")
    latest = [item async for item in checkpointer.alist(config)]

    assert len(latest) == 1
    assert latest[0].checkpoint == checkpoint
    assert latest[0].pending_writes == [("task-1", "messages", [{"role": "user", "content": "ok"}])]


@pytest.mark.asyncio
async def test_checkpointer_opens_new_session_for_each_call(client, db_session):
    session = await _create_agent_session(client, db_session)
    calls = 0

    @asynccontextmanager
    async def session_factory():
        nonlocal calls
        calls += 1
        async with TestSessionFactory() as db:
            yield db

    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=session_factory)

    await checkpointer.aput(_config(session.id), _checkpoint(), {}, {})
    await checkpointer.aget_tuple(_config(session.id))

    assert calls == 2


@pytest.mark.asyncio
async def test_aput_falls_back_to_session_id_when_config_has_no_thread_id(client, db_session):
    session = await _create_agent_session(client, db_session)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())

    config = await checkpointer.aput({}, _checkpoint(), {}, {})

    assert config["configurable"]["thread_id"] == str(session.id)


async def _create_agent_session(client, db_session) -> AgentSession:
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    session = AgentSession(
        project_id=uuid.UUID(project["id"]),
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
    )
    db_session.add(session)
    await db_session.commit()
    return session


def _session_factory():
    @asynccontextmanager
    async def factory():
        async with TestSessionFactory() as db:
            yield db

    return factory


async def _load_agent_session(session_id: uuid.UUID) -> AgentSession:
    async with TestSessionFactory() as db:
        return (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()


def _config(session_id: uuid.UUID) -> dict:
    return {"configurable": {"thread_id": str(session_id)}}


def _checkpoint() -> dict:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"messages": [{"role": "user", "content": "Xin chào"}]}
    checkpoint["channel_versions"] = {"messages": "1"}
    checkpoint["versions_seen"] = {"analyze": {"messages": "1"}}
    return checkpoint
