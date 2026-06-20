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
        "slot_coverage": None,
        "coverage_ratio": None,
        "coverage_complete": None,
        "coverage_stall_count": None,
        "last_asked_slot": None,
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
    from app.graphs.nodes import _build_analyst_prompt

    state = _state()
    state["conversation_summary"] = "Ràng buộc — KHÔNG paraphrase\n- Ngân sách tối đa 50 triệu"
    state["messages"] = [{"role": "user", "content": f"Tin nhắn {i}"} for i in range(5)]

    prompt = _build_analyst_prompt(state, [])

    assert "Tóm tắt hội thoại đã tích lũy" in prompt
    assert "Ngân sách tối đa 50 triệu" in prompt
    assert "Tin nhắn 1" not in prompt
    assert "Tin nhắn 2" in prompt


def test_build_prompt_falls_back_to_5_messages_when_empty():
    from app.graphs.nodes import _build_analyst_prompt

    state = _state()
    state["messages"] = [{"role": "user", "content": f"Tin nhắn {i}"} for i in range(6)]

    prompt = _build_analyst_prompt(state, [])

    assert "Tóm tắt hội thoại đã tích lũy" not in prompt
    assert "Tin nhắn 0" not in prompt
    assert "Tin nhắn 1" in prompt


def test_coverage_hint_injected_in_prompt_when_incomplete():
    from app.graphs.nodes import _build_analyst_prompt

    state = _state(artifact_type="problem")
    state["slot_coverage"] = {
        "who": "filled",
        "obstacle": "filled",
        "root_cause": "empty",
        "frequency": "empty",
        "impact": "empty",
    }
    state["coverage_complete"] = False

    prompt = _build_analyst_prompt(state, [])

    assert "Coverage" in prompt
    assert "root_cause" in prompt
    # Gap-inventory marker — distinguishes the coverage hint from the slot directive.
    assert "các khía cạnh còn thiếu" in prompt
    assert "trả lời cụt chỉ bằng câu hỏi" in prompt
    assert "một câu hỏi chính" in prompt


def test_no_coverage_hint_when_complete():
    from app.graphs.nodes import _build_analyst_prompt

    state = _state(artifact_type="problem")
    state["slot_coverage"] = {"root_cause": "filled"}
    state["coverage_complete"] = True

    prompt = _build_analyst_prompt(state, [])

    assert "Coverage" not in prompt


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


def test_slot_gate_blocks_propose_when_incomplete():
    from app.graphs.nodes import route_node

    state = _state(artifact_type="problem", turn_count=2, analysis_result={"next_action": "propose"})
    state["coverage_complete"] = False

    assert route_node(state) == "ask_human"


def test_slot_gate_allows_propose_when_complete():
    from app.graphs.nodes import route_node

    state = _state(artifact_type="problem", turn_count=2, analysis_result={"next_action": "propose"})
    state["coverage_complete"] = True

    assert route_node(state) == "confirm"


def test_slot_gate_blocks_done_when_incomplete():
    from app.graphs.nodes import route_node

    state = _state(artifact_type="problem", turn_count=2, analysis_result={"next_action": "done"})
    state["coverage_complete"] = False

    assert route_node(state) == "ask_human"


def test_slot_gate_never_overrides_turn_limit():
    from app.graphs.nodes import route_node
    from langgraph.graph import END

    state = _state(artifact_type="problem", turn_count=10, analysis_result={"next_action": "propose"})
    state["coverage_complete"] = False

    assert route_node(state) == END


def test_slot_gate_none_coverage_fails_open():
    from app.graphs.nodes import route_node

    state = _state(artifact_type="problem", turn_count=2, analysis_result={"next_action": "propose"})
    state["coverage_complete"] = None

    assert route_node(state) == "confirm"


@pytest.mark.parametrize(
    "brd_key",
    ["intent", "goal", "stakeholder", "capability", "constraint", "assumption", "risk", "open_question"],
)
def test_slot_gate_blocks_propose_for_each_brd_key(brd_key):
    from app.graphs.nodes import route_node

    state = _state(artifact_type=brd_key, turn_count=2, analysis_result={"next_action": "propose"})
    state["coverage_complete"] = False

    assert route_node(state) == "ask_human"


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
    from app.graphs.nodes import _build_analyst_prompt

    state_vi = _state()
    state_vi["locale"] = "vi"
    prompt_vi = _build_analyst_prompt(state_vi, [])
    assert "'vi'" in prompt_vi

    state_en = _state()
    state_en["locale"] = "en"
    prompt_en = _build_analyst_prompt(state_en, [])
    assert "'en'" in prompt_en
    assert "'vi'" not in prompt_en


def test_graph_routes_task_to_analyze():
    from app.graphs.nodes import route_after_intent

    assert route_after_intent({"intent": "task"}) == "analyze"
    assert route_after_intent({"intent": "unclear"}) == "analyze"
    assert route_after_intent({"intent": "greeting"}) == "greeting"
    assert route_after_intent({"intent": "smalltalk"}) == "greeting"


def _smart_llm():
    """LLM mock that branches on response_format to drive intent_router then analyze→ask."""
    from app.graphs.nodes import ANALYSIS_SCHEMA, INTENT_SCHEMA

    intent_calls = []
    llm = AsyncMock()

    async def _generate(*, messages, system, max_tokens, response_format=None):
        if response_format is INTENT_SCHEMA:
            intent_calls.append(1)
            return {"intent": "task", "locale": "vi"}, None
        if response_format is ANALYSIS_SCHEMA:
            return {"next_action": "ask", "confidence": 0.3, "gaps": [], "message": "Bạn cần gì thêm?", "proposals": []}, None
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


@pytest.mark.asyncio
async def test_resume_from_old_topology_checkpoint(client, db_session):
    """A checkpoint saved under the old topology still resumes without intent_router."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
    from langgraph.types import Command

    from app.graphs.graph import build_graph
    from app.graphs.nodes import analyze_node, ask_human_node, route_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    llm, _ = _smart_llm()
    saver = MemorySaver()
    config = _config(str(agent_session.id), str(project_id), llm)
    config["configurable"]["session_factory"] = _session_factory()

    # OLD topology: entry point "analyze", no intent_router node.
    old = StateGraph(WorkflowState)
    old.add_node("analyze", analyze_node)
    old.add_node("ask_human", ask_human_node)
    old.set_entry_point("analyze")
    old.add_conditional_edges("analyze", route_node, {"ask_human": "ask_human", "confirm": "ask_human", END: END})
    old.add_edge("ask_human", END)
    old_graph = old.compile(checkpointer=saver)

    state = _state()
    state["messages"] = [{"role": "user", "content": "tôi cần tạo intent"}]
    await old_graph.ainvoke(state, config)

    # NEW topology resumes the SAME checkpoint — must not crash on the missing intent_router node.
    new_graph = build_graph(checkpointer=saver)
    await new_graph.ainvoke(Command(resume={"content": "thêm chi tiết"}), config)


# ---------------------------------------------------------------------------
# Phase 5 — Payload Envelope (options / blocks / locale) (S6, S7)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_ask_human_node_payload_kind_question(mock_interrupt, client, db_session):
    from app.graphs.nodes import ask_human_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    state = _state(analysis_result={"next_action": "ask", "message": "Mục tiêu chính là gì?"})
    state["locale"] = "vi"
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await ask_human_node(state, config)

    async with TestSessionFactory() as db:
        msg = (
            await db.execute(select(AgentMessage).where(AgentMessage.session_id == agent_session.id))
        ).scalar_one()
        assert msg.payload["kind"] == "question"
        assert msg.payload["locale"] == "vi"
        assert msg.content == "Mục tiêu chính là gì?"


@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_confirm_node_payload_options_two_choices(mock_interrupt, client, db_session):
    from app.graphs.nodes import confirm_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    state["locale"] = "vi"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await confirm_node(state, config)

    async with TestSessionFactory() as db:
        msg = (
            await db.execute(select(AgentMessage).where(AgentMessage.session_id == agent_session.id))
        ).scalar_one()
        assert msg.payload["kind"] == "confirm"
        assert len(msg.payload["options"]) >= 2
        for opt in msg.payload["options"]:
            assert "id" in opt and "label" in opt and "value" in opt
        assert msg.content


@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_propose_artifacts_node_payload_blocks(mock_interrupt, client, db_session):
    from app.graphs.nodes import propose_artifacts_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    proposals = [{"artifact_type": "goal", "title": "Tăng doanh thu", "body": "Mô tả"}]
    state = _state(analysis_result={"next_action": "propose", "confidence": 0.9, "proposals": proposals})
    state["locale"] = "vi"
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await propose_artifacts_node(state, config)

    async with TestSessionFactory() as db:
        msg = (
            await db.execute(
                select(AgentMessage).where(
                    AgentMessage.session_id == agent_session.id,
                    AgentMessage.role == AgentMessageRole.AGENT,
                )
            )
        ).scalar_one()
        assert msg.payload["kind"] == "proposal"
        assert isinstance(msg.payload["blocks"], list)
        assert len(msg.payload["blocks"]) >= 1
        assert msg.content


@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_propose_artifacts_node_idempotent_on_resume(mock_interrupt, client, db_session):
    from app.graphs.nodes import propose_artifacts_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    proposals = [{"artifact_type": "goal", "title": "T", "body": "B"}]
    state = _state(analysis_result={"next_action": "propose", "confidence": 0.9, "proposals": proposals})
    state["locale"] = "vi"
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await propose_artifacts_node(state, config)
    await propose_artifacts_node(state, config)

    async with TestSessionFactory() as db:
        msgs = (
            await db.execute(
                select(AgentMessage).where(
                    AgentMessage.session_id == agent_session.id,
                    AgentMessage.role == AgentMessageRole.AGENT,
                )
            )
        ).scalars().all()
        assert len(msgs) == 1


# ---------------------------------------------------------------------------
# Phase 6 — One-question Rhythm (S8)
# ---------------------------------------------------------------------------

def test_analysis_schema_accepts_answer_assessment_and_acknowledgment():
    from app.graphs.nodes import ANALYSIS_SCHEMA

    props = ANALYSIS_SCHEMA["properties"]
    assert "answer_assessment" in props
    assert "acknowledgment" in props
    # Additive only — required set must not change.
    assert ANALYSIS_SCHEMA["required"] == ["next_action", "confidence"]


@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_ask_human_node_missing_optional_fields_no_error(mock_interrupt, client, db_session):
    from app.graphs.nodes import ask_human_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state(analysis_result={"next_action": "ask", "confidence": 0.7, "message": "Deadline?"})
    state["locale"] = "vi"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await ask_human_node(state, config)

    async with TestSessionFactory() as db:
        msg = (
            await db.execute(select(AgentMessage).where(AgentMessage.session_id == agent_session.id))
        ).scalar_one()
        assert msg.content == "Deadline?"


@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_ask_human_node_prepends_acknowledgment_when_present(mock_interrupt, client, db_session):
    from app.graphs.nodes import ask_human_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state(analysis_result={
        "next_action": "ask",
        "confidence": 0.7,
        "message": "Deadline là khi nào?",
        "answer_assessment": "complete",
        "acknowledgment": "Đã rõ mục tiêu.",
    })
    state["locale"] = "vi"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await ask_human_node(state, config)

    async with TestSessionFactory() as db:
        msg = (
            await db.execute(select(AgentMessage).where(AgentMessage.session_id == agent_session.id))
        ).scalar_one()
        assert "Đã rõ mục tiêu." in msg.content
        assert "Deadline là khi nào?" in msg.content


@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_ask_human_node_no_acknowledgment_first_turn(mock_interrupt, client, db_session):
    from app.graphs.nodes import ask_human_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state(analysis_result={
        "next_action": "ask",
        "confidence": 0.7,
        "message": "Mục tiêu là gì?",
        "acknowledgment": "",
    })
    state["locale"] = "vi"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await ask_human_node(state, config)

    async with TestSessionFactory() as db:
        msg = (
            await db.execute(select(AgentMessage).where(AgentMessage.session_id == agent_session.id))
        ).scalar_one()
        assert msg.content == "Mục tiêu là gì?"


@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_ask_human_node_idempotent_on_resume(mock_interrupt, client, db_session):
    from app.graphs.nodes import ask_human_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    state = _state(analysis_result={
        "next_action": "ask",
        "confidence": 0.7,
        "message": "Deadline?",
        "acknowledgment": "Lần đầu.",
    })
    state["locale"] = "vi"
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await ask_human_node(state, config)
    # Resume: same run_id but a different acknowledgment must NOT create a second message.
    state["analysis_result"]["acknowledgment"] = "Lần hai khác hẳn."
    await ask_human_node(state, config)

    async with TestSessionFactory() as db:
        msgs = (
            await db.execute(select(AgentMessage).where(AgentMessage.session_id == agent_session.id))
        ).scalars().all()
        assert len(msgs) == 1


def test_analyst_prompt_includes_one_question_directive():
    from app.graphs.nodes import _build_analyst_prompt

    prompt = _build_analyst_prompt(_state(), [])
    assert "một câu hỏi chính" in prompt
    assert "trả lời cụt chỉ bằng câu hỏi" in prompt
    assert "KHÔNG hỏi lại cùng nội dung/gap" in prompt
    assert "chuyển progression sang phần khác" in prompt
    assert "gap chưa được khai thác" in prompt


@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_ask_human_node_non_string_acknowledgment_no_crash(mock_interrupt, client, db_session):
    """LLM may return a non-string acknowledgment (schema not enforced at runtime) — must not crash."""
    from app.graphs.nodes import ask_human_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state(analysis_result={
        "next_action": "ask",
        "confidence": 0.7,
        "message": "Deadline?",
        "acknowledgment": 42,
    })
    state["locale"] = "vi"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await ask_human_node(state, config)

    async with TestSessionFactory() as db:
        msg = (
            await db.execute(select(AgentMessage).where(AgentMessage.session_id == agent_session.id))
        ).scalar_one()
        assert "Deadline?" in msg.content


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
