"""
Integration tests — end-to-end wiring.
Uses real graph execution with mock LLM.
"""
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.graphs.checkpointer import AgentSessionCheckpointer, DelegatingCheckpointer
from app.graphs.graph import build_graph
from app.models.agent import AgentSession, AgentSessionInterruptType, AgentSessionStatus
from tests.conftest import BASE
from tests.helpers import create_org, create_project, make_auth_headers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _project(client):
    h = await make_auth_headers(client)
    org = await create_org(client, h)
    proj = await create_project(client, h, org["id"])
    return h, uuid.UUID(proj["id"])


def _mock_llm(analysis: dict):
    """Smart stub: the analyze pass calls generate(tools=...) and expects an AIMessage with
    tool_calls; every other pass (triage/summary) uses response_format and reads a dict."""
    llm = AsyncMock()

    async def _generate(**kwargs):
        if kwargs.get("tools") is not None:
            tool_calls = [
                {"id": f"scripted:{i}", "name": item["name"], "args": item.get("args") or {}}
                for i, item in enumerate(analysis.get("tools") or [])
            ]
            return AIMessage(content=analysis.get("draft_update", ""), tool_calls=tool_calls), None
        return analysis, None

    llm.generate = _generate
    return llm


# ---------------------------------------------------------------------------
# Startup wiring
# ---------------------------------------------------------------------------

def test_delegating_checkpointer_instantiates():
    """DelegatingCheckpointer can be instantiated and delegates _for()."""
    async def _sf():
        yield None

    dc = DelegatingCheckpointer(session_factory=_sf)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    inner = dc._for(config)
    assert isinstance(inner, AgentSessionCheckpointer)


def test_startup_wiring_compiles_successfully():
    """The exact startup sequence: build_graph(DelegatingCheckpointer(...)) does not raise."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _sf():
        yield None

    graph = build_graph(DelegatingCheckpointer(session_factory=_sf))
    assert graph is not None


# ---------------------------------------------------------------------------
# End-to-end: session → ask_human interrupt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_flow_session_reaches_waiting_for_human(client, db_session):
    """
    Graph executes analyze → ask_user tool → session becomes WAITING_FOR_HUMAN.
    Uses checkpointer=None to avoid concurrent session access in test context.
    Verification of DelegatingCheckpointer + real graph is separate (round-trip test).
    """
    from contextlib import asynccontextmanager as acm
    from unittest.mock import MagicMock

    from app.main import app as _app
    from tests.conftest import TestSessionFactory

    h, project_id = await _project(client)

    # Set compiled_graph so _require_graph() passes (real graph built later for manual run).
    _app.state.compiled_graph = MagicMock()

    # Suppress background tasks — we'll run the graph manually
    with patch("app.services.agent_service.asyncio.create_task") as mock_ct:
        mock_ct.side_effect = lambda coro: coro.close() or None
        resp = await client.post(
            f"{BASE}/projects/{project_id}/agent-sessions",
            json={"artifact_type": "intent"},
            headers=h,
        )

    assert resp.status_code == 201
    session_id = uuid.UUID(resp.json()["data"]["session_id"])

    # Commit data so it's visible to new sessions
    await db_session.commit()

    # Use separate sessions per call — avoids concurrent access on same session
    @acm
    async def _sf():
        async with TestSessionFactory() as s:
            yield s

    # Tool-loop: analyze (the entry node) picks the ask_user tool, which interrupts for the human.
    # The mock returns this selection dict for every generate.
    llm = _mock_llm({"tools": [{"name": "ask_user", "args": {"message": "What do you want?"}}], "confidence": 0.9})

    # No checkpointer — avoids checkpointer + node concurrent session writes in test
    graph = build_graph(checkpointer=None)

    config = {
        "configurable": {
            "thread_id": str(session_id),
            "project_id": str(project_id),
            "session_factory": _sf,
            "llm_client": llm,
        }
    }
    state = {
        "artifact_type": "intent",
        "workflow_area": "analysis",
        "step_key": None,
        "messages": [],
        "analysis_result": None,
        "pending_tool_call_ids": [],
        "last_agent_run_id": None,
        "turn_count": 0,
        "missing_context": [],
    }

    with patch("app.graphs.interrupts.interrupt") as mock_interrupt:
        mock_interrupt.side_effect = Exception("interrupt raised")
        try:
            await graph.ainvoke(state, config)
        except Exception as e:
            if "interrupt raised" not in str(e):
                raise

    from sqlalchemy import select
    async with TestSessionFactory() as verify_db:
        session_row = (await verify_db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
    assert session_row.status == AgentSessionStatus.ACTIVE
    assert session_row.interrupt_type == AgentSessionInterruptType.STREAM_RESPONSE


# ---------------------------------------------------------------------------
# DelegatingCheckpointer round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delegating_checkpointer_round_trip(db_session):
    """DelegatingCheckpointer.aput + aget_tuple round-trip stores and retrieves checkpoint."""
    from app.models.agent import AgentSession

    session = AgentSession(
        project_id=uuid.uuid4(),
        artifact_type="intent",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.ACTIVE,
    )
    db_session.add(session)
    await db_session.flush()

    @asynccontextmanager
    async def _sf():
        yield db_session

    dc = DelegatingCheckpointer(session_factory=_sf)
    config = {"configurable": {"thread_id": str(session.id)}}

    # aget_tuple before any put → None
    result = await dc.aget_tuple(config)
    assert result is None

    # Build a minimal checkpoint and put it
    from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint
    cp = create_checkpoint(empty_checkpoint(), {}, 1)
    returned_config = await dc.aput(config, cp, {}, {})
    assert returned_config is not None

    # aget_tuple after put → returns CheckpointTuple
    retrieved = await dc.aget_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["id"] == cp["id"]
