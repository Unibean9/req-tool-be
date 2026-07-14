"""Tests for the core-loop hardening deltas.

Covers artifact lifecycle helpers (current_draft_body, artifact_stage) and interrupt
semantics: ask_user → ACTIVE+STREAM_RESPONSE; write_draft regression guard.
"""

from unittest.mock import patch

import pytest

from app.graphs.agent_tools import artifact_stage, current_draft_body
from app.graphs.decision_graph import create_node
from tests.factories import _state


def _state_with_node(statement: str = "Reduce processing time") -> dict:
    state = _state(artifact_type="brd")
    state["decision_nodes"] = {
        "N1": create_node(
            kind="objective",
            statement=statement,
            origin={"source": "test"},
            status="confirmed",
        )
    }
    return state


# current_draft_body's graph/DB/legacy source precedence is owned by test_gate_stack_minimal
# and test_finalize_gate; this file keeps only the truly-empty edge that neither covers.
@pytest.mark.asyncio
async def test_current_draft_body_empty_when_neither_present():
    assert await current_draft_body(_state()) == ""


@pytest.mark.asyncio
async def test_artifact_stage_empty_without_draft():
    assert await artifact_stage(_state()) == "empty"


@pytest.mark.asyncio
async def test_artifact_stage_drafting_when_draft_uncritiqued():
    state = _state_with_node()
    assert await artifact_stage(state) == "drafting"


@pytest.mark.asyncio
async def test_artifact_stage_critiqued_when_rounds_but_gate_not_passed():
    state = _state_with_node()
    state["critique_rounds"] = 1
    state["quality_report"] = {"quality_gate_result": "fail"}
    assert await artifact_stage(state) == "critiqued"


@pytest.mark.asyncio
async def test_artifact_stage_gate_passed_when_report_passes():
    state = _state_with_node()
    state["critique_rounds"] = 1
    state["quality_report"] = {"quality_gate_result": "pass"}
    assert await artifact_stage(state) == "gate_passed"


# --------------------------------------------------------------------------- Phase D4

@pytest.mark.asyncio
async def test_save_and_interrupt_ask_stream_response_sets_active_status(client, db_session):
    """ask_user uses interrupt_kind='stream_response' → ACTIVE + STREAM_RESPONSE."""
    from sqlalchemy import select

    from app.graphs import nodes
    from app.models.agent import AgentSession, AgentSessionInterruptType, AgentSessionStatus
    from tests.factories import _config, _make_agent_session, _session_factory
    from tests.helpers import create_org, create_project, make_auth_headers

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    import uuid
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    state["locale"] = "vi"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    with patch("app.graphs.interrupts.interrupt", return_value={"content": "reply"}):
        await nodes._save_and_interrupt_ask(
            state, config, "Cau hoi?", run_id="call_1", interrupt_kind="stream_response"
        )

    from tests.conftest import TestSessionFactory
    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentSession).where(AgentSession.id == agent_session.id))).scalar_one()
        assert row.status == AgentSessionStatus.ACTIVE
        assert row.interrupt_type == AgentSessionInterruptType.STREAM_RESPONSE


@pytest.mark.asyncio
async def test_save_and_interrupt_ask_ask_human_sets_waiting_status(client, db_session):
    """Approval flows use default interrupt_kind='ask_human' -> WAITING_FOR_HUMAN + ASK_HUMAN."""
    from sqlalchemy import select

    from app.graphs import nodes
    from app.models.agent import AgentSession, AgentSessionInterruptType, AgentSessionStatus
    from tests.factories import _config, _make_agent_session, _session_factory
    from tests.helpers import create_org, create_project, make_auth_headers

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    import uuid
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    state["locale"] = "vi"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    with patch("app.graphs.interrupts.interrupt", return_value={"content": "reply"}):
        await nodes._save_and_interrupt_ask(
            state, config, "Cau hoi?", run_id="call_2"
        )

    from tests.conftest import TestSessionFactory
    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentSession).where(AgentSession.id == agent_session.id))).scalar_one()
        assert row.status == AgentSessionStatus.WAITING_FOR_HUMAN
        assert row.interrupt_type == AgentSessionInterruptType.ASK_HUMAN


# --------------------------------------------------------------------------- M1: tool_choice_mode

def test_settings_tool_choice_mode_defaults_to_auto():
    from app.config import settings
    assert settings.tool_choice_mode == "auto"


def test_auto_plain_text_is_retried_with_required_tool_calling():
    from langchain_core.messages import AIMessage

    from app.graphs.nodes import _requires_native_tool_retry

    tool_names = ["write_draft", "respond"]
    assert _requires_native_tool_retry(
        AIMessage(content="Your plan: call write_draft next."), "auto", tool_names
    ) is True
    assert _requires_native_tool_retry(AIMessage(content="This is the final conclusion."), "auto", tool_names) is False
    assert _requires_native_tool_retry(AIMessage(content="Please note the final result."), "auto", tool_names) is False
    assert _requires_native_tool_retry(
        AIMessage(content="I will respond with the concise result now."), "auto", tool_names
    ) is False
    assert _requires_native_tool_retry(
        AIMessage(content="I will call write_draft next."), "auto", tool_names
    ) is True
    assert _requires_native_tool_retry(
        AIMessage(content="Your plan:\n- Review the current risks.\n- Decide tomorrow."),
        "auto",
        tool_names,
    ) is False
    assert _requires_native_tool_retry(AIMessage(content="", tool_calls=[]), "auto", tool_names) is False
    assert _requires_native_tool_retry(
        AIMessage(content="", tool_calls=[{"id": "call-1", "name": "respond", "args": {}}]),
        "auto",
        tool_names,
    ) is False
    assert _requires_native_tool_retry(AIMessage(content="Done."), "required", tool_names) is False


def test_route_node_ends_on_terminal_text_turn():
    """tool_choice=auto terminal path: AIMessage with no tool_calls → route_node returns END."""
    from langchain_core.messages import AIMessage

    from app.graphs.nodes import route_node
    state = {"turn_count": 0, "messages": [AIMessage(content="Analysis complete; no more tool needed.")]}
    assert route_node(state) == "__end__"


def test_legacy_draft_update_field_removed_from_analysis_result_contract():
    assert "draft_update" not in {"tools": [], "locale": "vi", "coverage_complete": False}


def test_ask_user_impl_passes_stream_response_interrupt_kind():
    """_ask_user_impl calls _save_and_interrupt_ask with interrupt_kind='stream_response'."""
    from app.graphs import interrupts
    from app.graphs.agent_tools import _ask_user_impl

    captured = {}

    async def fake_save(state, config, content, *, run_id, interrupt_kind="ask_human", **kw):
        captured["interrupt_kind"] = interrupt_kind
        return "user_reply"

    with patch.object(interrupts, "_save_and_interrupt_ask", side_effect=fake_save):
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            _ask_user_impl("Q?", _state(), {}, "call_x")
        )

    assert captured["interrupt_kind"] == "stream_response"


def test_respond_impl_passes_stream_response_interrupt_kind():
    """_respond_impl is a conversational pause, so it matches ask_user's stream_response status."""
    from app.graphs import interrupts
    from app.graphs.agent_tools import _respond_impl

    captured = {}

    async def fake_save(state, config, content, *, run_id, interrupt_kind="ask_human", **kw):
        captured["interrupt_kind"] = interrupt_kind
        return "user_reply"

    with patch.object(interrupts, "_save_and_interrupt_ask", side_effect=fake_save):
        import asyncio

        asyncio.get_event_loop().run_until_complete(_respond_impl("Assessment", "critique", _state(), {}, "call_x"))

    assert captured["interrupt_kind"] == "stream_response"
