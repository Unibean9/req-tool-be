"""Enum-to-Tool Parity Wrap.

Wraps the three enum branches as parallel tools (ask→ask_user, propose→write_draft,
done→finalize) without removing the enum branches. Guards the R1 (duplicate-message on
HTTP-resume) and R3 (idempotency-key collision) risks.

Unit tests (T1–T5) call the tool impls directly with `interrupt` patched. T6 exercises the
real ToolNode dispatch + interrupt/resume through a minimal compiled graph (analyze_node
cannot emit native tool_calls until Phase 4's bind_tools, so the HTTP driver path cannot
reach the tools yet — a seeded AIMessage is the Phase-2 precedent for tool-path coverage).
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from app.models.agent import (
    AgentMessage,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCall,
)
from tests.conftest import TestSessionFactory
from tests.helpers import create_org, create_project, make_auth_headers
from tests.test_graph_nodes import (
    _config,
    _make_agent_run,
    _make_agent_session,
    _session_factory,
    _state,
)


async def _project(client) -> uuid.UUID:
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return uuid.UUID(project["id"])


# ---------------------------------------------------------------------------
# T1 — ask_user idempotency on resume
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")  # ask_user delegates to nodes._save_and_interrupt_ask
async def test_ask_user_tool_idempotent_on_resume(mock_interrupt, client, db_session):
    from app.graphs.agent_tools import _ask_user_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    state["locale"] = "vi"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    # Resume re-executes the tool body from the top: same ToolCall.id + content twice.
    await _ask_user_impl("Bạn muốn xây gì?", state, config, "call_abc")
    await _ask_user_impl("Bạn muốn xây gì?", state, config, "call_abc")

    async with TestSessionFactory() as db:
        msgs = (
            await db.execute(
                select(AgentMessage).where(AgentMessage.session_id == agent_session.id)
            )
        ).scalars().all()
        assert len(msgs) == 1


# ---------------------------------------------------------------------------
# T2 — both paths delegate to the shared helper
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_user_tool_uses_shared_helper(client, db_session):
    from app.graphs import nodes
    from app.graphs.agent_tools import _ask_user_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    with patch.object(nodes, "_save_and_interrupt_ask", new=AsyncMock(return_value="ok")) as helper:
        await _ask_user_impl("Bạn muốn xây gì?", state, config, "call_1")

    helper.assert_awaited_once()


# ---------------------------------------------------------------------------
# T3 — write_draft idempotency key (run_id, tool_name)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")  # write_draft calls interrupt directly (not via nodes)
async def test_write_draft_tool_idempotency_key_run_id_tool_name(mock_interrupt, client, db_session):
    from app.graphs.agent_tools import _write_draft_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    state = _state(analysis_result={"next_action": "propose"})
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await _write_draft_impl("Tiêu đề", "Thân bài", state, config, "call_1")
    await _write_draft_impl("Tiêu đề", "Thân bài", state, config, "call_1")

    async with TestSessionFactory() as db:
        rows = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].tool_name == "write_draft"


# ---------------------------------------------------------------------------
# T4 — finalize interrupt gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")  # finalize calls interrupt directly (not via nodes)
async def test_finalize_tool_raises_interrupt(mock_interrupt, client, db_session):
    from app.graphs.agent_tools import _finalize_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    state["quality_report"] = {"quality_gate_result": "pass"}  # finalize now requires a passing gate
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await _finalize_impl("Đã hoàn tất.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        session_row = (
            await db.execute(select(AgentSession).where(AgentSession.id == agent_session.id))
        ).scalar_one()
        assert session_row.status == AgentSessionStatus.WAITING_FOR_HUMAN


# ---------------------------------------------------------------------------
# T5 — ask_user uses ToolCall.id (not state's last_agent_run_id) as idempotency key
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_user_uses_tool_call_id_not_state_run_id(client, db_session):
    from app.graphs import nodes
    from app.graphs.agent_tools import _ask_user_impl

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    state["last_agent_run_id"] = "old-run-id"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    with patch.object(nodes, "_save_and_interrupt_ask", new=AsyncMock(return_value="ok")) as helper:
        await _ask_user_impl("Bạn muốn xây gì?", state, config, "new-tool-call-id")

    run_id = helper.await_args.kwargs["run_id"]
    assert "new-tool-call-id" in str(run_id)
    assert run_id != "old-run-id"


# ---------------------------------------------------------------------------
# T6 — end-to-end tool path through a real ToolNode dispatch (interrupt/resume)
# ---------------------------------------------------------------------------

def _tool_graph():
    """Minimal compiled graph: START → tools (real ToolNode) → END, with a checkpointer.

    The compiled graph injects the Runtime that ToolNode + interrupt() need, so we exercise the
    real dispatch path. WorkflowState carries the fields the tools read (last_agent_run_id, etc.).
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    from app.graphs.agent_tools import ask_user, finalize, write_draft
    from app.graphs.state import WorkflowState

    builder = StateGraph(WorkflowState)
    builder.add_node("tools", ToolNode([ask_user, write_draft, finalize]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile(checkpointer=MemorySaver())


def _ai_tool_call(name: str, args: dict, call_id: str = "c1"):
    from langchain_core.messages import AIMessage

    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args}])


@pytest.mark.asyncio
async def test_ask_user_tool_call_scenario(client, db_session):
    from langgraph.types import Command

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    graph = _tool_graph()
    state = _state()
    state["messages"] = [_ai_tool_call("ask_user", {"message": "Bạn muốn xây gì?"})]
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    out = await graph.ainvoke(state, config)
    assert "__interrupt__" in out  # paused for the human

    async with TestSessionFactory() as db:
        from app.models.agent import AgentSession

        session_row = (
            await db.execute(select(AgentSession).where(AgentSession.id == agent_session.id))
        ).scalar_one()
        assert session_row.status == AgentSessionStatus.WAITING_FOR_HUMAN
        assert session_row.interrupt_type == AgentSessionInterruptType.ASK_HUMAN

    # Resume round-trip: a second invoke with the user's reply must complete, no crash.
    resumed = await graph.ainvoke(Command(resume={"content": "Một app lịch nhóm"}), config)
    assert "__interrupt__" not in resumed


@pytest.mark.asyncio
async def test_write_draft_tool_call_scenario(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    graph = _tool_graph()
    state = _state()
    state["last_agent_run_id"] = str(run.id)
    state["messages"] = [_ai_tool_call("write_draft", {"title": "Mục tiêu", "body": "Nội dung"})]
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    out = await graph.ainvoke(state, config)
    assert "__interrupt__" in out

    # Resume re-executes the tool body: the idempotency guard must keep it at one row, and the
    # Command(update={messages:[ToolMessage]}) return path (only reached on resume) must complete.
    from langgraph.types import Command

    resumed = await graph.ainvoke(Command(resume={"decision": "approve"}), config)
    assert "__interrupt__" not in resumed

    async with TestSessionFactory() as db:
        rows = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].tool_name == "write_draft"


@pytest.mark.asyncio
async def test_finalize_tool_call_scenario(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    graph = _tool_graph()
    state = _state()
    state["quality_report"] = {"quality_gate_result": "pass"}  # finalize now requires a passing gate
    state["messages"] = [_ai_tool_call("finalize", {"summary": "Đã hoàn tất."})]
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    out = await graph.ainvoke(state, config)
    assert "__interrupt__" in out

    # Resume must complete via the Command(update=...) return path (only reached on resume).
    from langgraph.types import Command

    resumed = await graph.ainvoke(Command(resume={"content": "ok"}), config)
    assert "__interrupt__" not in resumed
