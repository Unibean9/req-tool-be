import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from app.graphs.policy import GovernanceDenied
from app.graphs.state import WorkflowState
from app.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentRun,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCall,
    AgentToolCallStatus,
)
from tests.conftest import TestSessionFactory
from tests.helpers import create_org, create_project, make_auth_headers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_factory():
    @asynccontextmanager
    async def factory():
        async with TestSessionFactory() as db:
            yield db

    return factory


def _state(artifact_type: str = "goal", turn_count: int = 0, analysis_result=None) -> WorkflowState:
    return {
        "artifact_type": artifact_type,
        "workflow_area": "analysis",
        "step_key": None,
        "messages": [],
        "analysis_result": analysis_result,
        "pending_tool_call_ids": [],
        "last_agent_run_id": None,
        "turn_count": turn_count,
        "missing_context": [],
    }


def _config(session_id: str, project_id: str, llm_client=None) -> dict:
    return {
        "configurable": {
            "thread_id": session_id,
            "project_id": project_id,
            "llm_client": llm_client or AsyncMock(),
            "session_factory": _session_factory(),
        }
    }


async def _make_agent_session(client, db_session, project_id: uuid.UUID) -> AgentSession:
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
    )
    db_session.add(session)
    await db_session.commit()
    return session


async def _make_agent_run(db_session, agent_session: AgentSession) -> AgentRun:
    run = AgentRun(session_id=agent_session.id, analysis_result={})
    db_session.add(run)
    await db_session.commit()
    return run


# ---------------------------------------------------------------------------
# analyze_node tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_node_low_confidence_returns_ask_action(client, db_session):
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value={
        "next_action": "ask",
        "confidence": 0.3,
        "gaps": ["thiếu business context"],
        "message": "Bạn có thể mô tả thêm về mục tiêu không?",
        "proposals": [],
    })

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert result["analysis_result"]["next_action"] == "ask"
    assert result["turn_count"] == 1
    assert result["last_agent_run_id"] is not None


@pytest.mark.asyncio
async def test_analyze_node_high_confidence_returns_propose_action(client, db_session):
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value={
        "next_action": "propose",
        "confidence": 0.9,
        "gaps": [],
        "message": "",
        "proposals": [{"artifact_type": "goal", "title": "Tăng doanh thu", "body": "..."}],
    })

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert result["analysis_result"]["next_action"] == "propose"
    assert result["turn_count"] == 1


@pytest.mark.asyncio
async def test_analyze_node_creates_agent_run_record(client, db_session):
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value={"next_action": "done", "confidence": 0.8, "gaps": [], "proposals": []})

    state = _state()
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    run_id = uuid.UUID(result["last_agent_run_id"])
    async with TestSessionFactory() as db:
        run = (await db.execute(select(AgentRun).where(AgentRun.id == run_id))).scalar_one()
        assert run.session_id == agent_session.id


# ---------------------------------------------------------------------------
# route_node tests
# ---------------------------------------------------------------------------

def test_route_node_max_turns_routes_to_end():
    from app.graphs.nodes import route_node
    from langgraph.graph import END

    state = _state(turn_count=10, analysis_result={"next_action": "propose"})
    assert route_node(state) == END


def test_route_node_ask_routes_to_ask_human():
    from app.graphs.nodes import route_node

    state = _state(turn_count=2, analysis_result={"next_action": "ask"})
    assert route_node(state) == "ask_human"


def test_route_node_propose_routes_to_confirm():
    from app.graphs.nodes import route_node

    state = _state(turn_count=2, analysis_result={"next_action": "propose"})
    assert route_node(state) == "confirm"


def test_route_node_done_routes_to_end():
    from app.graphs.nodes import route_node
    from langgraph.graph import END

    state = _state(turn_count=2, analysis_result={"next_action": "done"})
    assert route_node(state) == END


# ---------------------------------------------------------------------------
# ask_human_node tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_ask_human_node_saves_message_and_interrupts(mock_interrupt, client, db_session):
    from app.graphs.nodes import ask_human_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state(analysis_result={"next_action": "ask", "message": "Cần thêm thông tin về người dùng"})
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await ask_human_node(state, config)

    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        msg = (
            await db.execute(
                select(AgentMessage).where(AgentMessage.session_id == agent_session.id)
            )
        ).scalar_one()
        assert msg.role == AgentMessageRole.AGENT
        assert "người dùng" in msg.content

        session_row = (await db.execute(select(AgentSession).where(AgentSession.id == agent_session.id))).scalar_one()
        assert session_row.status == AgentSessionStatus.WAITING_FOR_HUMAN
        assert session_row.interrupt_type == AgentSessionInterruptType.ASK_HUMAN


# ---------------------------------------------------------------------------
# propose_artifacts_node tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_propose_artifacts_node_creates_tool_calls_for_each_proposal(mock_interrupt, client, db_session):
    from app.graphs.nodes import propose_artifacts_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    proposals = [
        {"artifact_type": "goal", "title": "Tăng doanh thu", "body": "Mô tả 1"},
        {"artifact_type": "goal", "title": "Giảm chi phí", "body": "Mô tả 2"},
    ]
    analysis_result = {"next_action": "propose", "confidence": 0.9, "proposals": proposals}
    state = _state(analysis_result=analysis_result)
    state["last_agent_run_id"] = str(run.id)

    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    result = await propose_artifacts_node(state, config)

    assert len(result["pending_tool_call_ids"]) == 2
    mock_interrupt.assert_called_once()

    async with TestSessionFactory() as db:
        tool_calls = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
        assert len(tool_calls) == 2
        assert all(tc.status == AgentToolCallStatus.PROPOSED for tc in tool_calls)
        assert all(tc.tool_name == "create_artifact" for tc in tool_calls)

        session_row = (await db.execute(select(AgentSession).where(AgentSession.id == agent_session.id))).scalar_one()
        assert session_row.status == AgentSessionStatus.WAITING_FOR_HUMAN
        assert session_row.interrupt_type == AgentSessionInterruptType.PROPOSE_ARTIFACTS


# ---------------------------------------------------------------------------
# tools.py governed tests (P3-8)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_governed_unknown_write_tool_raises_governance_denied():
    from app.graphs.tools import create_artifact

    with pytest.raises(GovernanceDenied):
        await create_artifact(artifact_type="unknown_type", title="Test", body="", context={"allowed_types": ["goal"]})


# ---------------------------------------------------------------------------
# build_graph tests
# ---------------------------------------------------------------------------

def test_build_graph_returns_compiled_graph_without_error():
    from app.graphs.graph import build_graph

    graph = build_graph(checkpointer=None)
    assert graph is not None


@pytest.mark.asyncio
async def test_build_graph_with_checkpointer_attaches_it(client, db_session):
    from app.graphs.checkpointer import AgentSessionCheckpointer
    from app.graphs.graph import build_graph

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    agent_session = await _make_agent_session(client, db_session, project_id)
    checkpointer = AgentSessionCheckpointer(
        session_id=str(agent_session.id),
        session_factory=_session_factory(),
    )
    graph = build_graph(checkpointer=checkpointer)
    assert graph is not None
