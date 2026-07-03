"""mode_hint retention, SSE heartbeat, evidence_source, edge-label rename.

Edge-label rename is verified by the existing routing tests in test_graph_nodes.py (labels now equal
their target node names); this module covers the other three sub-fixes.
"""

import uuid
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

from app.graphs.agent_tools import elicit
from app.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentSession,
    AgentSessionStatus,
)
from app.services.agent_event_service import AgentEventService
from tests.factories import _project


def _service(db_session):
    from app.services.agent_service import AgentService

    @asynccontextmanager
    async def _sf():
        yield db_session

    return AgentService(db=db_session, graph=MagicMock(), session_factory=_sf)


# --- mode_hint retention -----------------------------------------------------


@pytest.mark.asyncio
async def test_queue_message_persists_mode_hint(client, db_session):
    project_id = await _project(client)
    session = AgentSession(project_id=project_id, artifact_type="goal", workflow_area="analysis", graph_checkpoint={})
    db_session.add(session)
    await db_session.flush()

    msg = await _service(db_session)._queue_message(session.id, "làm PRD đi", "prd")

    stored = (await db_session.execute(select(AgentMessage).where(AgentMessage.id == msg.id))).scalar_one()
    assert stored.payload == {"queued": True, "mode_hint": "prd"}


@pytest.mark.asyncio
async def test_drain_queue_replays_mode_hint(client, db_session):
    project_id = await _project(client)
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.COMPLETED,
    )
    db_session.add(session)
    await db_session.flush()
    db_session.add(
        AgentMessage(
            session_id=session.id,
            role=AgentMessageRole.USER,
            content="tiếp tục",
            payload={"queued": True, "mode_hint": "prd"},
        )
    )
    await db_session.commit()

    svc = _service(db_session)
    with (
        patch("app.services.agent_service.build_initial_workflow_state", return_value={}) as build,
        patch("app.services.agent_service.asyncio.create_task", side_effect=lambda coro, *_, **__: coro.close()),
    ):
        await svc._drain_queue(
            session_id=session.id,
            project_id=project_id,
            artifact_type="goal",
            step_key=None,
            workflow_area="analysis",
            agent_role=None,
            missing_context=[],
            llm_client=None,
        )

    assert build.call_args.kwargs["mode_hint"] == "prd"


# --- SSE heartbeat -----------------------------------------------------------


class _FakeRequest:
    """Disconnects after `checks` calls so the stream loop terminates."""

    def __init__(self, checks: int):
        self._remaining = checks

    async def is_disconnected(self) -> bool:
        self._remaining -= 1
        return self._remaining < 0


@pytest.mark.asyncio
async def test_sse_heartbeat_on_idle_stream(client, db_session):
    project_id = await _project(client)
    owner_id = uuid.uuid4()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.ACTIVE,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.flush()

    frames = []
    async for frame in AgentEventService(db_session).stream_session_events(
        project_id=project_id,
        session_id=session.id,
        user_id=owner_id,
        request=_FakeRequest(checks=4),
        interval_seconds=0.01,
        heartbeat_seconds=0.02,
    ):
        frames.append(frame)

    assert any(f == ": keepalive\n\n" for f in frames)


# --- evidence_source ---------------------------------------------------------


def test_evidence_source_web_when_search_succeeds():
    def fake_client(query: str) -> list[dict]:
        return [{"title": "POS A", "snippet": "coffee shop software", "url": "https://a.example"}]

    out = elicit(technique="comparable_products", seed="coffee shop app", search_client=fake_client)
    assert out["evidence_source"] == "web"


def test_comparable_products_search_uses_seed_without_domain_prefix():
    queries = []

    def fake_client(query: str) -> list[dict]:
        queries.append(query)
        return [{"title": "A", "snippet": "result", "url": "https://a.example"}]

    elicit(technique="comparable_products", seed="warehouse robot", search_client=fake_client)

    assert queries == ["warehouse robot"]


def test_evidence_source_model_knowledge_on_fallback():
    def failing_client(query: str):
        raise RuntimeError("provider down")

    out = elicit(technique="comparable_products", seed="coffee shop app", search_client=failing_client)
    assert out["evidence_source"] == "model_knowledge"
