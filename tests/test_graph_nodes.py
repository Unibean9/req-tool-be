import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

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
        "conversation_summary": "",
        "analysis_result": analysis_result,
        "pending_tool_call_ids": [],
        "last_agent_run_id": None,
        "turn_count": turn_count,
        "missing_context": [],
        "user_confirmed": None,
        "critique_rounds": 0,
        "quality_report": None,
        "locale": None,
        "intent": None,
        "section_coverage": None,
        "coverage_ratio": None,
        "coverage_complete": None,
        "section_coverage_stall_count": None,
        "assumptions": [],
        "risks": [],
        "open_questions": [],
        "draft_body": None,
        "working_draft": None,
        "mode_hint": None,
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
    mock_llm.generate = AsyncMock(return_value=({
        "next_action": "ask",
        "confidence": 0.3,
        "gaps": ["thiếu business context"],
        "message": "Bạn có thể mô tả thêm về mục tiêu không?",
        "proposals": [],
    }, None))

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert result["analysis_result"]["next_action"] == "ask"
    assert result["turn_count"] == 1
    assert result["last_agent_run_id"] is not None


@pytest.mark.asyncio
async def test_analyze_node_feeds_predecessor_artifacts_into_prompt(client, db_session):
    """A `problem` session must see its `intent` predecessor as analyst context.

    Regression: analyze_node previously read only same-type artifacts, so a
    derived type (problem/goal/...) never saw the upstream source it derives
    from — leaving the analyst blind to intent and hurting traceability.
    """
    from app.graphs.nodes import analyze_node
    from app.models.artifact import Artifact

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    # An existing intent artifact in the project — the predecessor of `problem`.
    intent_title = "Intent: Điều phối lịch học nhóm cho sinh viên"
    db_session.add(
        Artifact(project_id=project_id, type="intent", title=intent_title, extra_metadata={}, status="draft")
    )
    await db_session.commit()

    session = AgentSession(
        project_id=project_id, artifact_type="problem", workflow_area="analysis", graph_checkpoint={}
    )
    db_session.add(session)
    await db_session.commit()

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(
        return_value=({"next_action": "done", "confidence": 0.5, "gaps": [], "proposals": []}, None)
    )

    state = _state(artifact_type="problem")
    config = _config(str(session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    await analyze_node(state, config)

    prompt = mock_llm.generate.call_args.kwargs["messages"][0]["content"]
    assert intent_title in prompt, "Predecessor intent title must appear in the analyst prompt context"


@pytest.mark.asyncio
async def test_analyze_node_feeds_transitive_ancestry_into_prompt(client, db_session):
    """A deep type (`story`) must see its full ancestry, not just the direct parent.

    `story`'s only declared predecessor is `epic`, but provenance runs all the way
    up to `intent`. The context loader uses the transitive closure, so both the
    direct parent (`epic`) and a distant ancestor (`intent`) must reach the prompt.
    """
    from app.graphs.nodes import analyze_node
    from app.models.artifact import Artifact

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    intent_title = "Intent: Điều phối lịch học nhóm"
    epic_title = "Epic: Đồng bộ và đối chiếu lịch nhóm"
    db_session.add(Artifact(project_id=project_id, type="intent", title=intent_title, extra_metadata={}, status="draft"))
    db_session.add(Artifact(project_id=project_id, type="epic", title=epic_title, extra_metadata={}, status="draft"))
    await db_session.commit()

    session = AgentSession(
        project_id=project_id, artifact_type="story", workflow_area="analysis", graph_checkpoint={}
    )
    db_session.add(session)
    await db_session.commit()

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(
        return_value=({"next_action": "done", "confidence": 0.5, "gaps": [], "proposals": []}, None)
    )

    state = _state(artifact_type="story")
    config = _config(str(session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    await analyze_node(state, config)

    prompt = mock_llm.generate.call_args.kwargs["messages"][0]["content"]
    assert epic_title in prompt, "Direct parent (epic) must appear in the prompt"
    assert intent_title in prompt, "Distant ancestor (intent) must appear via transitive closure"


@pytest.mark.asyncio
async def test_analyze_uses_strong_client_when_present(client, db_session):
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    default_llm = AsyncMock()
    default_llm.generate = AsyncMock()
    strong_llm = AsyncMock()
    strong_llm.generate = AsyncMock(return_value=({
        "next_action": "done",
        "confidence": 0.9,
        "gaps": [],
        "proposals": [],
    }, None))

    state = _state()
    config = _config(str(agent_session.id), str(project_id), default_llm)
    config["configurable"]["strong_llm_client"] = strong_llm

    await analyze_node(state, config)

    strong_llm.generate.assert_called_once()
    default_llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_falls_back_to_default_when_strong_absent(client, db_session):
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    default_llm = AsyncMock()
    default_llm.generate = AsyncMock(return_value=({
        "next_action": "done",
        "confidence": 0.9,
        "gaps": [],
        "proposals": [],
    }, None))

    state = _state()
    config = _config(str(agent_session.id), str(project_id), default_llm)
    config["configurable"]["strong_llm_client"] = None

    await analyze_node(state, config)

    default_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_triggers_at_threshold(monkeypatch):
    from app.graphs.nodes import route_before_analyze, summarize_node

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 6)
    state = _state()
    state["messages"] = [{"role": "user", "content": f"Tin nhắn {i}"} for i in range(7)]
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=("Tóm tắt mới", None))

    result = await summarize_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert route_before_analyze(state) == "summarize"
    assert result["conversation_summary"] == "Tóm tắt mới"
    llm.generate.assert_called_once()


def test_summarize_triggers_on_real_ask_loop_message_counts(monkeypatch):
    from app.graphs.nodes import route_before_analyze

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 6)

    routes = {}
    for count in [1, 3, 5, 7, 9, 11, 13]:
        state = _state()
        state["messages"] = [{"role": "user", "content": f"Tin nhắn {i}"} for i in range(count)]
        routes[count] = route_before_analyze(state)

    assert routes == {
        1: "analyze",
        3: "analyze",
        5: "analyze",
        7: "summarize",
        9: "analyze",
        11: "analyze",
        13: "summarize",
    }


@pytest.mark.asyncio
async def test_summarize_skipped_below_threshold(monkeypatch):
    from app.graphs.nodes import route_before_analyze, summarize_node

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 6)
    state = _state()
    state["conversation_summary"] = "Tóm tắt cũ"
    state["messages"] = [{"role": "user", "content": f"Tin nhắn {i}"} for i in range(5)]
    llm = AsyncMock()
    llm.generate = AsyncMock()

    result = await summarize_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert route_before_analyze(state) == "analyze"
    assert result["conversation_summary"] == "Tóm tắt cũ"
    llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_summary_preserves_constraints_verbatim(monkeypatch):
    from app.graphs.nodes import summarize_node

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 2)
    state = _state()
    state["messages"] = [
        {"role": "user", "content": "Tạo MVP"},
        {"role": "user", "content": "Ngân sách tối đa 50 triệu"},
        {"role": "assistant", "content": "Đã ghi nhận"},
    ]
    summary = (
        "Yêu cầu đã xác nhận\n- Làm MVP\n"
        "Ràng buộc — KHÔNG paraphrase\n- Ngân sách tối đa 50 triệu\n"
        "Khoảng trống chưa rõ\n- Thời hạn\n"
        "Quyết định đã thống nhất\n- Chưa có"
    )
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(summary, None))

    result = await summarize_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert "Ràng buộc — KHÔNG paraphrase" in result["conversation_summary"]
    assert "Ngân sách tối đa 50 triệu" in result["conversation_summary"]


@pytest.mark.asyncio
async def test_summarize_node_uses_default_client(monkeypatch):
    from app.graphs.nodes import summarize_node

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 2)
    state = _state()
    state["messages"] = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
    ]
    default_llm = AsyncMock()
    default_llm.generate = AsyncMock(return_value=("Tóm tắt", None))
    strong_llm = AsyncMock()
    strong_llm.generate = AsyncMock()
    config = _config(str(uuid.uuid4()), str(uuid.uuid4()), default_llm)
    config["configurable"]["strong_llm_client"] = strong_llm

    await summarize_node(state, config)

    default_llm.generate.assert_called_once()
    strong_llm.generate.assert_not_called()


def test_build_prompt_uses_summary_when_present():
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state()
    state["conversation_summary"] = "Ràng buộc — KHÔNG paraphrase\n- Ngân sách tối đa 50 triệu"
    state["messages"] = [{"role": "user", "content": f"Tin nhắn {i}"} for i in range(5)]

    prompt = _build_tool_selection_prompt(state, [])

    assert "Tóm tắt hội thoại đã tích lũy" in prompt
    assert "Ngân sách tối đa 50 triệu" in prompt
    assert "Tin nhắn 1" not in prompt
    assert "Tin nhắn 2" in prompt


def test_build_prompt_falls_back_to_5_messages_when_empty():
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state()
    state["messages"] = [{"role": "user", "content": f"Tin nhắn {i}"} for i in range(6)]

    prompt = _build_tool_selection_prompt(state, [])

    assert "Tóm tắt hội thoại đã tích lũy" not in prompt
    assert "Tin nhắn 0" not in prompt
    assert "Tin nhắn 1" in prompt


def test_build_prompt_includes_synthesis_directive():
    """The propose path must instruct the LLM to mine the full context into a rich body,
    not emit a thin one-paragraph artifact. Without this directive the prompt only says
    'proposals (if propose)', which produces shallow artifacts even after full elicitation.
    """
    from app.graphs.nodes import _build_tool_selection_prompt

    prompt = _build_tool_selection_prompt(_state(), [])

    assert "ĐỘ SÂU NỘI DUNG" in prompt
    # Must steer toward exploiting all gathered information, not summarizing thinly.
    assert "toàn bộ" in prompt.lower() or "khai thác" in prompt
    # Must forbid fabricating detail beyond what was gathered.
    assert "không bịa" in prompt.lower()


def test_coverage_hint_injected_in_prompt_when_incomplete():
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="problem")
    state["section_coverage"] = {
        "vision_objectives": "filled",
        "problem_statement": "missing",
        "stakeholder_register": "partial",
    }
    state["coverage_complete"] = False

    prompt = _build_tool_selection_prompt(state, [])

    assert "Độ phủ section" in prompt
    # Gap-inventory marker — lists weak sections, not a single pinned question.
    assert "các khía cạnh còn thiếu" in prompt
    assert "trả lời cụt chỉ bằng câu hỏi" in prompt
    assert "một câu hỏi chính" in prompt


def test_no_coverage_hint_when_complete():
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="problem")
    state["section_coverage"] = {"problem_statement": "filled"}
    state["coverage_complete"] = True

    prompt = _build_tool_selection_prompt(state, [])

    assert "Độ phủ section" not in prompt


@pytest.mark.asyncio
async def test_messages_not_truncated_by_summarize(monkeypatch):
    from app.graphs.nodes import summarize_node

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 3)
    state = _state()
    state["messages"] = [{"role": "user", "content": f"Tin nhắn {i}"} for i in range(3)]
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=("Tóm tắt", None))

    await summarize_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert len(state["messages"]) == 3


@pytest.mark.asyncio
async def test_analyze_node_high_confidence_returns_propose_action(client, db_session):
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=({
        "next_action": "propose",
        "confidence": 0.9,
        "gaps": [],
        "message": "",
        "proposals": [{"artifact_type": "goal", "title": "Tăng doanh thu", "body": "..."}],
    }, None))

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
    mock_llm.generate = AsyncMock(return_value=({"next_action": "done", "confidence": 0.8, "gaps": [], "proposals": []}, None))

    state = _state()
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    run_id = uuid.UUID(result["last_agent_run_id"])
    async with TestSessionFactory() as db:
        run = (await db.execute(select(AgentRun).where(AgentRun.id == run_id))).scalar_one()
        assert run.session_id == agent_session.id
        # usage = None -> token_usage None, but latency is always measured as a non-negative int.
        assert run.token_usage is None
        assert isinstance(run.latency_ms, int)
        assert run.latency_ms >= 0


@pytest.mark.asyncio
async def test_analyze_node_records_token_usage_and_latency(client, db_session):
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(
        {"next_action": "done", "confidence": 0.8, "gaps": [], "proposals": []},
        {"input": 5, "output": 10, "total": 15},
    ))

    state = _state()
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    run_id = uuid.UUID(result["last_agent_run_id"])
    async with TestSessionFactory() as db:
        run = (await db.execute(select(AgentRun).where(AgentRun.id == run_id))).scalar_one()
        assert run.token_usage == {"input": 5, "output": 10, "total": 15}
        assert isinstance(run.latency_ms, int)
        assert run.latency_ms >= 0


# ---------------------------------------------------------------------------
# route_node tests
# ---------------------------------------------------------------------------

def test_route_node_max_turns_routes_to_end():
    from langgraph.graph import END

    from app.graphs.nodes import route_node

    state = _state(turn_count=10, analysis_result={"next_action": "propose"})
    assert route_node(state) == END


# ---------------------------------------------------------------------------
# tools.py governed tests (P3-8)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_governed_unknown_write_tool_raises_governance_denied():
    from app.graphs.tools import create_artifact

    with pytest.raises(GovernanceDenied):
        await create_artifact(artifact_type="unknown_type", title="Test", body="", context={"allowed_types": ["goal"]})


# ---------------------------------------------------------------------------
# Phase 4 — intent_router + greeting + language-lock (S4, S5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_intent_router_classifies_greeting():
    from app.graphs.nodes import intent_router_node

    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=({"intent": "greeting", "locale": "vi"}, None))
    state = _state()
    state["messages"] = [{"role": "user", "content": "hello"}]

    result = await intent_router_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert result["intent"] == "greeting"
    assert result["locale"] == "vi"


@pytest.mark.asyncio
async def test_intent_router_classifies_task():
    from app.graphs.nodes import intent_router_node

    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=({"intent": "task", "locale": "en"}, None))
    state = _state()
    state["messages"] = [{"role": "user", "content": "I need a user story"}]

    result = await intent_router_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert result["intent"] == "task"
    assert result["locale"] == "en"


@pytest.mark.asyncio
async def test_intent_router_raises_when_llm_client_none():
    from app.graphs.nodes import intent_router_node

    config = _config(str(uuid.uuid4()), str(uuid.uuid4()), llm_client=None)
    config["configurable"]["llm_client"] = None
    state = _state()
    state["messages"] = [{"role": "user", "content": "hello"}]

    with pytest.raises(ValueError):
        await intent_router_node(state, config)


@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_greeting_node_saves_message_and_no_agent_run(mock_interrupt, client, db_session):
    from app.graphs.nodes import greeting_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    state["locale"] = "vi"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await greeting_node(state, config)

    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        msg = (
            await db.execute(select(AgentMessage).where(AgentMessage.session_id == agent_session.id))
        ).scalar_one()
        assert msg.role == AgentMessageRole.AGENT
        assert msg.content
        assert msg.payload["kind"] == "greeting"

        runs = (await db.execute(select(AgentRun).where(AgentRun.session_id == agent_session.id))).scalars().all()
        assert runs == []

        session_row = (await db.execute(select(AgentSession).where(AgentSession.id == agent_session.id))).scalar_one()
        assert session_row.status == AgentSessionStatus.WAITING_FOR_HUMAN
        assert session_row.interrupt_type == AgentSessionInterruptType.ASK_HUMAN


@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_greeting_node_english_locale(mock_interrupt, client, db_session):
    from app.graphs.nodes import greeting_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    state["locale"] = "en"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await greeting_node(state, config)

    async with TestSessionFactory() as db:
        msg = (
            await db.execute(select(AgentMessage).where(AgentMessage.session_id == agent_session.id))
        ).scalar_one()
        assert "Hello" in msg.content
        assert msg.payload["locale"] == "en"


def test_analyst_prompt_includes_language_lock_directive():
    from app.graphs.nodes import _build_tool_selection_prompt

    state_vi = _state()
    state_vi["locale"] = "vi"
    prompt_vi = _build_tool_selection_prompt(state_vi, [])
    assert "'vi'" in prompt_vi

    state_en = _state()
    state_en["locale"] = "en"
    prompt_en = _build_tool_selection_prompt(state_en, [])
    assert "'en'" in prompt_en
    assert "'vi'" not in prompt_en


def test_graph_routes_task_to_analyze():
    from app.graphs.nodes import route_after_intent

    assert route_after_intent({"intent": "task"}) == "analyze"
    assert route_after_intent({"intent": "unclear"}) == "analyze"
    assert route_after_intent({"intent": "greeting"}) == "greeting"
    assert route_after_intent({"intent": "smalltalk"}) == "greeting"


def _smart_llm():
    """LLM mock that branches on response_format to drive intent_router then analyze→ask_user."""
    from app.graphs.nodes import INTENT_SCHEMA, TOOL_SELECTION_SCHEMA

    intent_calls = []
    llm = AsyncMock()

    async def _generate(*, messages, system, max_tokens, response_format=None):
        if response_format is INTENT_SCHEMA:
            intent_calls.append(1)
            return {"intent": "task", "locale": "vi"}, None
        if response_format is TOOL_SELECTION_SCHEMA:
            return {"tool": "ask_user", "message": "Bạn cần gì thêm?", "confidence": 0.3, "active_mode": "qa"}, None
        return {}, None

    llm.generate = _generate
    return llm, intent_calls


@pytest.mark.asyncio
async def test_resume_from_ask_human_interrupt_with_new_entry_point(client, db_session):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from app.graphs.graph import build_graph

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    llm, intent_calls = _smart_llm()
    saver = MemorySaver()
    graph = build_graph(checkpointer=saver)
    config = _config(str(agent_session.id), str(project_id), llm)
    config["configurable"]["session_factory"] = _session_factory()

    state = _state()
    state["messages"] = [{"role": "user", "content": "tôi cần tạo intent"}]
    await graph.ainvoke(state, config)
    assert len(intent_calls) == 1

    # Resume the ask_human interrupt — must NOT re-enter intent_router.
    await graph.ainvoke(Command(resume={"content": "thêm chi tiết"}), config)
    assert len(intent_calls) == 1


# ---------------------------------------------------------------------------
# Phase 6 — One-question Rhythm (S8)
# ---------------------------------------------------------------------------

def test_analysis_schema_accepts_answer_assessment_and_acknowledgment():
    from app.graphs.nodes import TOOL_SELECTION_SCHEMA

    props = TOOL_SELECTION_SCHEMA["properties"]
    assert "answer_assessment" in props
    assert "acknowledgment" in props
    assert TOOL_SELECTION_SCHEMA["required"] == ["tool"]


# ---------------------------------------------------------------------------
# build_graph tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase A1 (C2): load current draft body just-in-time
# ---------------------------------------------------------------------------

async def _add_artifact_with_version(
    db_session, project_id: uuid.UUID, artifact_type: str, title: str, body: str
):
    """Create an artifact with a current version pointing at `body`.

    Mirrors the flush ordering of the service layer: artifact → version →
    current_version_id, so `current_version` resolves to the version just made.
    """
    from app.models.artifact import Artifact, ArtifactVersion, ChangeSource, VersionStatus

    artifact = Artifact(
        project_id=project_id, type=artifact_type, title=title, extra_metadata={}, status="draft"
    )
    db_session.add(artifact)
    await db_session.flush()
    version = ArtifactVersion(
        artifact_id=artifact.id,
        version_number=1,
        title=title,
        body=body,
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.AI_GENERATION,
        extra_metadata={},
    )
    db_session.add(version)
    await db_session.flush()
    artifact.current_version_id = version.id
    await db_session.commit()
    return artifact


def test_build_prompt_includes_draft_body_block_when_present():
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="problem")
    state["locale"] = "vi"  # populate language_lock so the ordering assert is meaningful
    body = "Người dùng: sinh viên. Trở ngại: trùng lịch học nhóm."
    prompt = _build_tool_selection_prompt(state, [], draft_body=body)

    assert "DRAFT ĐANG CÓ" in prompt
    assert body in prompt
    assert "không hỏi lại" in prompt.lower()
    # draft block must precede the language lock (kept last by contract)
    assert prompt.index("DRAFT ĐANG CÓ") < prompt.index("ngôn ngữ 'vi'")


def test_build_prompt_no_draft_block_when_absent():
    """Regression guard: create-from-scratch prompt unchanged when no draft exists."""
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="problem")
    baseline = _build_tool_selection_prompt(state, [])
    with_none = _build_tool_selection_prompt(state, [], draft_body=None)

    assert "DRAFT ĐANG CÓ" not in with_none
    assert with_none == baseline


@pytest.mark.asyncio
async def test_read_current_body_returns_one_when_multiple(client, db_session):
    """Multiple drafts of the same type: returns exactly one (no crash).

    Picking the *right* target for a deliberate update is the authoritative
    target_artifact_id problem of Phase 4 — A1 only surfaces a single draft as context.
    """
    from app.graphs.tools import read_current_body

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    await _add_artifact_with_version(db_session, project_id, "problem", "Draft A", "body A")
    await _add_artifact_with_version(db_session, project_id, "problem", "Draft B", "body B")

    async with TestSessionFactory() as db:
        result = await read_current_body(db=db, project_id=project_id, artifact_type="problem")

    assert result is not None
    assert result["body"] in {"body A", "body B"}


@pytest.mark.asyncio
async def test_read_current_body_returns_none_without_current_version(client, db_session):
    from app.graphs.tools import read_current_body
    from app.models.artifact import Artifact

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    # An artifact with no current_version_id must not surface as a draft.
    db_session.add(
        Artifact(project_id=project_id, type="problem", title="Trống", extra_metadata={}, status="draft")
    )
    await db_session.commit()

    async with TestSessionFactory() as db:
        result = await read_current_body(db=db, project_id=project_id, artifact_type="problem")

    assert result is None


@pytest.mark.asyncio
async def test_analyze_node_loads_current_draft_body_into_prompt(client, db_session):
    """M7/M8: a new `problem` session must see the existing draft body in its prompt."""
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    draft_body = "Đối tượng: sinh viên năm 2. Trở ngại: lịch học nhóm hay bị trùng giờ làm thêm."
    await _add_artifact_with_version(db_session, project_id, "problem", "Vấn đề lịch nhóm", draft_body)

    session = AgentSession(
        project_id=project_id, artifact_type="problem", workflow_area="analysis", graph_checkpoint={}
    )
    db_session.add(session)
    await db_session.commit()

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(
        return_value=({"next_action": "done", "confidence": 0.5, "gaps": [], "proposals": []}, None)
    )

    state = _state(artifact_type="problem")
    config = _config(str(session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    await analyze_node(state, config)

    prompt = mock_llm.generate.call_args.kwargs["messages"][0]["content"]
    assert draft_body in prompt, "Existing draft body must be injected as context"
    assert "DRAFT ĐANG CÓ" in prompt


# ---------------------------------------------------------------------------
# Phase A2 (C1): incremental running draft (working_draft)
# ---------------------------------------------------------------------------

def test_analysis_schema_accepts_draft_update():
    """`draft_update` is additive and optional — the required set must not change."""
    from app.graphs.nodes import TOOL_SELECTION_SCHEMA

    assert "draft_update" in TOOL_SELECTION_SCHEMA["properties"]
    assert TOOL_SELECTION_SCHEMA["properties"]["draft_update"]["type"] == "string"
    assert TOOL_SELECTION_SCHEMA["required"] == ["tool"]


def test_build_prompt_includes_working_draft_block_when_present():
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="problem")
    state["locale"] = "vi"  # populate language_lock so the ordering assert is meaningful
    state["working_draft"] = "## Vấn đề\n- Sinh viên trùng lịch học nhóm với giờ làm thêm."

    prompt = _build_tool_selection_prompt(state, [])

    assert "DRAFT ĐANG XÂY DỰNG" in prompt
    assert state["working_draft"] in prompt
    # The running draft must precede the language lock (kept last by contract).
    assert prompt.index("DRAFT ĐANG XÂY DỰNG") < prompt.index("ngôn ngữ 'vi'")


def test_build_prompt_no_working_draft_block_when_absent():
    """Regression guard: prompt unchanged when no running draft exists."""
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="problem")
    baseline = _build_tool_selection_prompt(state, [])
    state["working_draft"] = None

    assert "DRAFT ĐANG XÂY DỰNG" not in _build_tool_selection_prompt(state, [])
    assert _build_tool_selection_prompt(state, []) == baseline


@pytest.mark.asyncio
async def test_analyze_node_persists_working_draft_from_draft_update(client, db_session):
    """M10: when the LLM emits draft_update, it becomes the running working_draft."""
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    new_draft = "## Mục tiêu\n- Tăng tỷ lệ giữ chân người dùng lên 30%."
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=({
        "next_action": "ask",
        "confidence": 0.4,
        "gaps": [],
        "message": "Còn ràng buộc nào không?",
        "draft_update": new_draft,
    }, None))

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert result["working_draft"] == new_draft


@pytest.mark.asyncio
async def test_analyze_node_preserves_working_draft_when_no_update(client, db_session):
    """A turn with no draft_update must keep the prior draft, not None it out."""
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    prior = "## Mục tiêu\n- Đã ghi nhận từ lượt trước."
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=({
        "next_action": "ask",
        "confidence": 0.3,
        "gaps": [],
        "message": "Bạn cần gì thêm?",
    }, None))

    state = _state(artifact_type="goal")
    state["working_draft"] = prior
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert result["working_draft"] == prior


# ---------------------------------------------------------------------------
# Multi-angle: active_mode + mode_hint + proactive directive
# ---------------------------------------------------------------------------

def test_active_mode_field_in_analysis_schema():
    """T1: `active_mode` is additive and optional — the required set must not change."""
    from app.graphs.nodes import TOOL_SELECTION_SCHEMA

    assert "active_mode" in TOOL_SELECTION_SCHEMA["properties"]
    assert "active_mode" not in TOOL_SELECTION_SCHEMA.get("required", [])


def test_active_mode_schema_accepts_new_vocabulary():
    """Phase 6: the enum carries the spec §7.1 values alongside the legacy ones."""
    from app.graphs.nodes import TOOL_SELECTION_SCHEMA

    enum = TOOL_SELECTION_SCHEMA["properties"]["active_mode"]["enum"]
    for value in ("discovery", "structuring", "revision", "finalization"):
        assert value in enum


def test_normalize_active_mode_legacy_qa_to_discovery():
    from app.graphs.nodes import _normalize_active_mode

    assert _normalize_active_mode("qa") == "discovery"


def test_normalize_active_mode_legacy_explore_to_structuring():
    """explore -> structuring (NOT discovery) so [qa, explore] keeps variety >= 2."""
    from app.graphs.nodes import _normalize_active_mode

    assert _normalize_active_mode("explore") == "structuring"
    assert _normalize_active_mode("draft") == "structuring"
    assert _normalize_active_mode("critique") == "critique"


def test_variety_preserved_after_normalization():
    from app.graphs.nodes import _normalize_active_mode

    normalized = {_normalize_active_mode(m) for m in ("qa", "explore")}
    assert len(normalized) >= 2


@pytest.mark.asyncio
async def test_active_mode_passes_through_analyze_node(client, db_session):
    """T2: an `active_mode` the LLM emits survives into the persisted analysis_result.

    `active_mode` lives inside analysis_result (no new state channel), so the eval layer
    can mine it from AgentRun.analysis_result — it must not be stripped by analyze_node.
    """
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=({
        "next_action": "ask",
        "confidence": 0.4,
        "gaps": [],
        "message": "Ta thử soi lại nhé?",
        "active_mode": "critique",
    }, None))

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert result["analysis_result"]["active_mode"] == "critique"


def test_mode_hint_injects_directive_into_prompt():
    """T4a: a user-supplied mode_hint must surface the requested mode in the prompt."""
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="goal")
    state["mode_hint"] = "critique"

    prompt = _build_tool_selection_prompt(state, [])

    assert "critique" in prompt


def test_no_mode_hint_injects_proactive_rule():
    """T4b: with no hint, the prompt must carry the proactive mode-switch rule."""
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="goal")
    state["mode_hint"] = None

    prompt = _build_tool_selection_prompt(state, [])

    # The proactive rule now steers the agent to voice critique/explore via `respond` instead of
    # wrapping it in a question — guard that intent, not the old enum-era wording.
    assert "chủ động" in prompt
    assert "respond" in prompt


def test_mode_directive_precedes_language_lock():
    """The mode directive must sit before the language lock (lock stays last by contract)."""
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="goal")
    state["locale"] = "vi"
    state["mode_hint"] = "explore"

    prompt = _build_tool_selection_prompt(state, [])

    assert prompt.index("explore") < prompt.index("ngôn ngữ 'vi'")


@pytest.mark.asyncio
async def test_mode_hint_cleared_after_single_turn(client, db_session):
    """T5: analyze_node consumes mode_hint and clears it within the same turn."""
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=({
        "next_action": "ask",
        "confidence": 0.3,
        "gaps": [],
        "message": "Bạn cần gì thêm?",
    }, None))

    state = _state(artifact_type="goal")
    state["mode_hint"] = "critique"
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert result["mode_hint"] is None


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
