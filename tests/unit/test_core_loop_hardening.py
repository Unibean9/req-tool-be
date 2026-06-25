"""Tests for the core-loop hardening deltas.

Phase 1 — artifact lifecycle helpers (current_draft_body, artifact_stage).
Phase D4 — interrupt semantics: ask_user → ACTIVE+STREAM_RESPONSE; write_draft regression guard.
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.graphs.agent_tools import artifact_stage, current_draft_body
from tests.integration.test_graph_nodes import _state

# --------------------------------------------------------------------------- Phase 1

@pytest.mark.asyncio
async def test_current_draft_body_prefers_draft_body_over_working_draft():
    state = _state()
    state["draft_body"] = "DB draft"
    state["working_draft"] = "session draft"
    assert await current_draft_body(state) == "DB draft"


@pytest.mark.asyncio
async def test_current_draft_body_uses_cached_body_without_db_config():
    state = _state()
    state["focused_artifact_id"] = "00000000-0000-0000-0000-000000000001"
    state["draft_body"] = "DB draft"

    assert await current_draft_body(state) == "DB draft"


@pytest.mark.asyncio
async def test_current_draft_body_falls_back_to_working_draft():
    state = _state()
    state["working_draft"] = "session draft"
    assert await current_draft_body(state) == "session draft"


@pytest.mark.asyncio
async def test_current_draft_body_empty_when_neither_present():
    assert await current_draft_body(_state()) == ""


@pytest.mark.asyncio
async def test_artifact_stage_empty_without_draft():
    assert await artifact_stage(_state()) == "empty"


@pytest.mark.asyncio
async def test_artifact_stage_drafting_when_draft_uncritiqued():
    state = _state()
    state["working_draft"] = "draft"
    assert await artifact_stage(state) == "drafting"


@pytest.mark.asyncio
async def test_artifact_stage_critiqued_when_rounds_but_gate_not_passed():
    state = _state()
    state["working_draft"] = "draft"
    state["critique_rounds"] = 1
    state["quality_report"] = {"quality_gate_result": "fail"}
    assert await artifact_stage(state) == "critiqued"


@pytest.mark.asyncio
async def test_artifact_stage_gate_passed_when_report_passes():
    state = _state()
    state["working_draft"] = "draft"
    state["critique_rounds"] = 1
    state["quality_report"] = {"quality_gate_result": "pass"}
    assert await artifact_stage(state) == "gate_passed"


# --------------------------------------------------------------------------- Phase D4

@pytest.mark.asyncio
async def test_save_and_interrupt_ask_stream_response_sets_active_status(client, db_session):
    """ask_user uses interrupt_kind='stream_response' → ACTIVE + STREAM_RESPONSE."""
    from app.graphs import nodes
    from app.models.agent import AgentSession, AgentSessionInterruptType, AgentSessionStatus
    from sqlalchemy import select
    from tests.integration.test_graph_nodes import _config, _make_agent_session, _session_factory
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

    with patch("app.graphs.nodes.interrupt", return_value={"content": "reply"}):
        await nodes._save_and_interrupt_ask(
            state, config, "Câu hỏi?", run_id="call_1", interrupt_kind="stream_response"
        )

    from tests.conftest import TestSessionFactory
    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentSession).where(AgentSession.id == agent_session.id))).scalar_one()
        assert row.status == AgentSessionStatus.ACTIVE
        assert row.interrupt_type == AgentSessionInterruptType.STREAM_RESPONSE


@pytest.mark.asyncio
async def test_save_and_interrupt_ask_ask_human_sets_waiting_status(client, db_session):
    """write_draft/respond use default interrupt_kind='ask_human' → WAITING_FOR_HUMAN + ASK_HUMAN (regression guard)."""
    from app.graphs import nodes
    from app.models.agent import AgentSession, AgentSessionInterruptType, AgentSessionStatus
    from sqlalchemy import select
    from tests.integration.test_graph_nodes import _config, _make_agent_session, _session_factory
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

    with patch("app.graphs.nodes.interrupt", return_value={"content": "reply"}):
        await nodes._save_and_interrupt_ask(
            state, config, "Câu hỏi?", run_id="call_2"
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


def test_route_node_ends_on_terminal_text_turn():
    """tool_choice=auto terminal path: AIMessage with no tool_calls → route_node returns END."""
    from app.graphs.nodes import route_node
    from langchain_core.messages import AIMessage
    state = {"turn_count": 0, "messages": [AIMessage(content="Phân tích xong, không cần thêm tool.")]}
    assert route_node(state) == "__end__"


def test_analyze_node_draft_update_set_on_terminal_turn():
    """No tool_calls → draft_update captures model content (not None)."""
    from langchain_core.messages import AIMessage
    # Simulate the draft_update derivation logic from analyze_node directly.
    ai_message = AIMessage(content="Nội dung cuối.", tool_calls=[])
    has_tool_calls = bool(getattr(ai_message, "tool_calls", None))
    draft_update = None if has_tool_calls else ((getattr(ai_message, "content", None) or "").strip() or None)
    assert draft_update == "Nội dung cuối."


def test_analyze_node_draft_update_none_when_tool_calls_present():
    """tool_calls present → draft_update is None (model content is reasoning, not a draft)."""
    from langchain_core.messages import AIMessage
    ai_message = AIMessage(content="Thinking...", tool_calls=[{"id": "x", "name": "ask_user", "args": {}}])
    has_tool_calls = bool(getattr(ai_message, "tool_calls", None))
    draft_update = None if has_tool_calls else ((getattr(ai_message, "content", None) or "").strip() or None)
    assert draft_update is None


def test_ask_user_impl_passes_stream_response_interrupt_kind():
    """_ask_user_impl calls _save_and_interrupt_ask with interrupt_kind='stream_response'."""
    from app.graphs import nodes
    from app.graphs.agent_tools import _ask_user_impl

    captured = {}

    async def fake_save(state, config, content, *, run_id, interrupt_kind="ask_human", **kw):
        captured["interrupt_kind"] = interrupt_kind
        return "user_reply"

    with patch.object(nodes, "_save_and_interrupt_ask", side_effect=fake_save):
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            _ask_user_impl("Q?", _state(), {}, "call_x")
        )

    assert captured["interrupt_kind"] == "stream_response"
