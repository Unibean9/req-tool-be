import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from app.graphs.policy import GovernanceDenied
from app.graphs.state import (
    DEFAULT_ARTIFACT_CHAIN,
    DEFAULT_METHOD_PROFILE,
    DEFAULT_READINESS,
    WorkflowState,
)
from app.models.agent import (
    AgentMessage,
    AgentRun,
    AgentSession,
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
        "last_critiqued_draft_hash": None,
        "locale": None,
        "turn_type": None,
        "triage_reply": None,
        "section_coverage": None,
        "coverage_complete": None,
        "section_coverage_stall_count": None,
        "assumptions": [],
        "risks": [],
        "open_questions": [],
        "focused_artifact_id": None,
        "draft_body": None,
        "method_profile": dict(DEFAULT_METHOD_PROFILE),
        "artifact_chain": dict(DEFAULT_ARTIFACT_CHAIN),
        "readiness": dict(DEFAULT_READINESS),
        "working_draft": None,
        "candidate_readiness": None,
        "tool_errors": [],
        "feedback_summary": None,
        "verification_status": None,
        "latest_checked_revision": None,
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


async def _project(client) -> uuid.UUID:
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return uuid.UUID(project["id"])


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
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content="", tool_calls=[
        {"id": "scripted:0", "name": "ask_user", "args": {"message": "Bạn có thể mô tả thêm về mục tiêu không?"}}
    ]), None))

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert result["analysis_result"]["tools"][0]["name"] == "ask_user"
    assert result["turn_count"] == 1
    assert result["last_agent_run_id"] is not None


@pytest.mark.asyncio
async def test_analyze_node_resets_critique_state_when_db_focused_artifact_changes(client, db_session):
    from app.graphs.nodes import analyze_node
    from app.models.artifact import Artifact

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    agent_session = await _make_agent_session(client, db_session, project_id)
    parent = Artifact(
        project_id=project_id,
        type="brd",
        title="BRD",
        extra_metadata={},
        status="draft",
    )
    db_session.add(parent)
    await db_session.flush()
    child_a = Artifact(
        project_id=project_id,
        parent_id=parent.id,
        type="vision_objectives",
        title="Vision",
        extra_metadata={},
        status="draft",
    )
    child_b = Artifact(
        project_id=project_id,
        parent_id=parent.id,
        type="problem_statement",
        title="Problem",
        extra_metadata={},
        status="draft",
    )
    db_session.add_all([child_a, child_b])
    await db_session.flush()
    agent_session.focused_artifact_id = child_b.id
    await db_session.commit()

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content="", tool_calls=[
        {"id": "scripted:0", "name": "finalize", "args": {"summary": "Hoàn tất"}}
    ]), None))

    state = _state(artifact_type="problem_statement")
    state["focused_artifact_id"] = str(child_a.id)
    state["critique_rounds"] = 1
    state["last_critiqued_draft_hash"] = "stalehash"
    state["quality_report"] = {"quality_gate_result": "pass"}
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert result["focused_artifact_id"] == str(child_b.id)
    assert result["critique_rounds"] == 0
    assert result["last_critiqued_draft_hash"] is None
    assert result["analysis_result"]["tools"][0]["name"] == "ask_user"
    assert result["analysis_result"]["gated_tool"] == "finalize"


@pytest.mark.asyncio
async def test_analyze_node_feeds_predecessor_artifacts_into_prompt(client, db_session):
    """A derived session must see its `brd` predecessor as analyst context.

    Regression: analyze_node previously read only same-type artifacts, so a
    derived type never saw the upstream source it derives from — leaving the
    analyst blind to requirements and hurting traceability.
    """
    from app.graphs.nodes import analyze_node
    from app.models.artifact import Artifact

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    brd_title = "BRD: Điều phối lịch học nhóm cho sinh viên"
    db_session.add(
        Artifact(project_id=project_id, type="brd", title=brd_title, extra_metadata={}, status="draft")
    )
    await db_session.commit()

    session = AgentSession(
        project_id=project_id, artifact_type="functional_requirement", workflow_area="analysis", graph_checkpoint={}
    )
    db_session.add(session)
    await db_session.commit()

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(
        return_value=({"next_action": "done", "confidence": 0.5, "gaps": [], "proposals": []}, None)
    )

    state = _state(artifact_type="functional_requirement")
    config = _config(str(session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    await analyze_node(state, config)

    prompt = mock_llm.generate.call_args.kwargs["messages"][0]["content"]
    assert brd_title in prompt, "Predecessor BRD title must appear in the analyst prompt context"


def test_output_contract_block_requires_candidate_gap_markers():
    from app.graphs.nodes import _build_output_contract_block

    block = _build_output_contract_block(_state(artifact_type="vision_objectives"))

    assert "inferred" in block
    assert "missing" in block
    assert "needs_confirmation" in block
    assert "phần thiếu" in block


@pytest.mark.asyncio
async def test_analyze_node_feeds_transitive_ancestry_into_prompt(client, db_session):
    """Một type sâu phải thấy tiền nhiệm trực tiếp và tổ tiên xa trong prompt."""
    from app.graphs.nodes import analyze_node
    from app.models.artifact import Artifact

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    brd_title = "BRD: Điều phối lịch học nhóm"
    domain_title = "Domain entity: Lịch nhóm"
    component_title = "Component: Bộ điều phối lịch"
    db_session.add(Artifact(project_id=project_id, type="brd", title=brd_title, extra_metadata={}, status="draft"))
    db_session.add(Artifact(project_id=project_id, type="domain_entity", title=domain_title, extra_metadata={}, status="draft"))
    db_session.add(Artifact(project_id=project_id, type="component", title=component_title, extra_metadata={}, status="draft"))
    await db_session.commit()

    session = AgentSession(
        project_id=project_id, artifact_type="interface", workflow_area="analysis", graph_checkpoint={}
    )
    db_session.add(session)
    await db_session.commit()

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(
        return_value=({"next_action": "done", "confidence": 0.5, "gaps": [], "proposals": []}, None)
    )

    state = _state(artifact_type="interface")
    config = _config(str(session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    await analyze_node(state, config)

    prompt = mock_llm.generate.call_args.kwargs["messages"][0]["content"]
    assert component_title in prompt, "Tiền nhiệm trực tiếp (component) phải có trong prompt"
    assert domain_title in prompt, "Tiền nhiệm bắc cầu (domain_entity) phải có trong prompt"
    assert brd_title in prompt, "Tổ tiên xa (brd) phải có qua transitive closure"


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


@pytest.mark.asyncio
async def test_summarize_degrades_when_response_not_schema_conformant(monkeypatch):
    """A non-conforming summary response (generate raises ValueError) must not crash the turn —
    summarize_node keeps the prior summary and lets the loop continue to analyze."""
    from app.graphs.nodes import summarize_node

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 6)
    state = _state()
    state["conversation_summary"] = "Tóm tắt cũ"
    state["messages"] = [{"role": "user", "content": f"Tin nhắn {i}"} for i in range(7)]
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=ValueError("Phản hồi LLM không khớp JSON Schema tại $: thiếu summary"))

    result = await summarize_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert result["conversation_summary"] == "Tóm tắt cũ"


@pytest.mark.asyncio
async def test_triage_degrades_to_work_turn_when_response_not_schema_conformant(monkeypatch):
    """A non-conforming triage response must default to a work turn (falls through to analyze)."""
    from app.graphs.nodes import route_after_triage, triage_node

    state = _state()
    state["messages"] = [{"role": "user", "content": "Xây hệ thống điểm danh"}]
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=ValueError("Phản hồi LLM không khớp JSON Schema tại $: thiếu turn_type"))

    result = await triage_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert result["turn_type"] == "work"
    assert route_after_triage({**state, **result}) == "analyze"


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


def test_build_prompt_excludes_static_policy():
    """Static policy (tool semantics, synthesis depth) lives in the instruction layers now, so the
    per-turn payload must NOT restate it."""
    from app.graphs.nodes import _build_tool_selection_prompt

    prompt = _build_tool_selection_prompt(_state(), [])

    # The old inline synthesis directive and per-tool description block are gone from the payload.
    assert "ĐỘ SÂU NỘI DUNG" not in prompt
    assert "ghi chú phản biện" not in prompt
    # It still names the tools available this turn so the model knows the current menu.
    assert "Công cụ khả dụng" in prompt


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
    # Harness voice: advance the artifact, not "ask one main question".
    assert "advance" in prompt
    assert "một câu hỏi chính" not in prompt


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
# Language lock (S5) — locale drives the analyst's output-language directive
# ---------------------------------------------------------------------------

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


def _smart_llm():
    """LLM mock driving analyze→ask_user each turn. Counts analyze (tool-selection) calls so a test
    can prove the loop advances across a resume. The analyze pass is the one passing tools=; it gets
    an AIMessage with tool_calls. Other passes (triage) read a dict via response_format."""
    from langchain_core.messages import AIMessage

    analyze_calls = []
    llm = AsyncMock()

    async def _generate(*, messages, system, max_tokens, response_format=None, tools=None):
        if tools is not None:
            analyze_calls.append(1)
            return AIMessage(content="", tool_calls=[
                {"id": "scripted:0", "name": "ask_user", "args": {"message": "Bạn cần gì thêm?"}}
            ]), None
        return {}, None

    llm.generate = _generate
    return llm, analyze_calls


@pytest.mark.asyncio
async def test_resume_from_ask_human_interrupt_continues_loop(client, db_session):
    """A work turn triages straight to analyze and interrupts at ask_user; resuming re-enters the
    interrupted tool and loops back to analyze, so the analyze count advances past the resume."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from app.graphs.graph import build_graph

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    llm, analyze_calls = _smart_llm()
    saver = MemorySaver()
    graph = build_graph(checkpointer=saver)
    config = _config(str(agent_session.id), str(project_id), llm)
    config["configurable"]["session_factory"] = _session_factory()

    state = _state()
    state["messages"] = [{"role": "user", "content": "tôi cần tạo intent"}]
    await graph.ainvoke(state, config)
    assert len(analyze_calls) == 1

    # Resume the ask_human interrupt: the loop continues and analyze runs again.
    await graph.ainvoke(Command(resume={"content": "thêm chi tiết"}), config)
    assert len(analyze_calls) == 2


# ---------------------------------------------------------------------------
# Triage + converse (entry routing)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_triage_classifies_converse_and_drafts_reply():
    from app.graphs.nodes import triage_node

    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(
        {"turn_type": "converse", "locale": "vi", "reply": "Xin chào, bạn muốn xây gì?"}, None
    ))
    state = _state()
    state["messages"] = [{"role": "user", "content": "chào bạn"}]

    result = await triage_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert result["turn_type"] == "converse"
    assert result["locale"] == "vi"
    assert result["triage_reply"] == "Xin chào, bạn muốn xây gì?"


@pytest.mark.asyncio
async def test_triage_classifies_work_drops_reply():
    from app.graphs.nodes import triage_node

    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(
        {"turn_type": "work", "locale": "en", "reply": "ignored"}, None
    ))
    state = _state()
    state["messages"] = [{"role": "user", "content": "I need a user story"}]

    result = await triage_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert result["turn_type"] == "work"
    # reply is only carried for a conversational turn.
    assert result["triage_reply"] is None


@pytest.mark.asyncio
async def test_triage_raises_when_llm_client_none():
    from app.graphs.nodes import triage_node

    config = _config(str(uuid.uuid4()), str(uuid.uuid4()), llm_client=None)
    config["configurable"]["llm_client"] = None
    state = _state()
    state["messages"] = [{"role": "user", "content": "chào"}]

    with pytest.raises(ValueError):
        await triage_node(state, config)


def test_route_after_triage_splits_converse_from_work():
    from app.graphs.nodes import route_after_triage

    assert route_after_triage({"turn_type": "converse"}) == "converse"
    assert route_after_triage({"turn_type": "work"}) == "analyze"
    # Missing/unknown defaults to the analyst, never silently skips work.
    assert route_after_triage({}) == "analyze"


@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_converse_node_replies_and_interrupts_without_agent_run(mock_interrupt, client, db_session):
    """A conversational turn replies and pauses for the human — no AgentRun (the full analyst pass
    never ran), and no LLM call inside converse itself (it reuses the triage-drafted reply)."""
    from app.graphs.nodes import converse_node

    mock_interrupt.return_value = {"content": "tôi muốn xây app abc"}
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    state["locale"] = "vi"
    state["triage_reply"] = "Xin chào! Bạn muốn bắt đầu từ đâu?"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    result = await converse_node(state, config)

    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        msg = (
            await db.execute(select(AgentMessage).where(AgentMessage.session_id == agent_session.id))
        ).scalar_one()
        assert msg.payload["kind"] == "greeting"
        assert msg.content == "Xin chào! Bạn muốn bắt đầu từ đâu?"
        runs = (await db.execute(select(AgentRun).where(AgentRun.session_id == agent_session.id))).scalars().all()
        assert runs == []
    # The human's reply is folded in for analyze to pick up next.
    assert result["messages"][-1]["content"] == "tôi muốn xây app abc"


@pytest.mark.asyncio
async def test_greeting_turn_skips_full_analysis(client, db_session):
    """End to end: a greeting triages to converse and interrupts WITHOUT running analyze — proven by
    zero AgentRun rows (analyze is the only node that records one)."""
    from langgraph.checkpoint.memory import MemorySaver

    from app.graphs.graph import build_graph
    from app.graphs.nodes import TRIAGE_SCHEMA

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    llm = AsyncMock()

    async def _generate(*, messages, system, max_tokens, response_format=None):
        if response_format is TRIAGE_SCHEMA:
            return {"turn_type": "converse", "locale": "vi", "reply": "Xin chào!"}, None
        return {}, None

    llm.generate = _generate
    graph = build_graph(checkpointer=MemorySaver())
    config = _config(str(agent_session.id), str(project_id), llm)
    config["configurable"]["session_factory"] = _session_factory()

    state = _state()
    state["messages"] = [{"role": "user", "content": "chào bạn"}]
    out = await graph.ainvoke(state, config)

    assert "__interrupt__" in out
    async with TestSessionFactory() as db:
        runs = (await db.execute(select(AgentRun).where(AgentRun.session_id == agent_session.id))).scalars().all()
        assert runs == [], "greeting must not trigger the analyst pass"


# ---------------------------------------------------------------------------
# Phase 6 — One-question Rhythm (S8)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# build_graph tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase A1 (C2): load current draft body just-in-time
# ---------------------------------------------------------------------------

async def _add_artifact_with_version(
    db_session,
    project_id: uuid.UUID,
    artifact_type: str,
    title: str,
    body: str,
    parent_id: uuid.UUID | None = None,
):
    """Create an artifact with a current version pointing at `body`.

    Mirrors the flush ordering of the service layer: artifact → version →
    current_version_id, so `current_version` resolves to the version just made.
    """
    from app.models.artifact import Artifact, ArtifactVersion, ChangeSource, VersionStatus

    artifact = Artifact(
        project_id=project_id,
        parent_id=parent_id,
        type=artifact_type,
        title=title,
        extra_metadata={},
        status="draft",
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

    await _add_artifact_with_version(db_session, project_id, "functional_requirement", "Draft A", "body A")
    await _add_artifact_with_version(db_session, project_id, "functional_requirement", "Draft B", "body B")

    async with TestSessionFactory() as db:
        result = await read_current_body(db=db, project_id=project_id, artifact_type="functional_requirement")

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
        Artifact(project_id=project_id, type="functional_requirement", title="Trống", extra_metadata={}, status="draft")
    )
    await db_session.commit()

    async with TestSessionFactory() as db:
        result = await read_current_body(db=db, project_id=project_id, artifact_type="functional_requirement")

    assert result is None


@pytest.mark.asyncio
async def test_analyze_node_loads_current_draft_body_into_prompt(client, db_session):
    """A focused document item must expose its current draft in the prompt."""
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    draft_body = "Đối tượng: sinh viên năm 2. Trở ngại: lịch học nhóm hay bị trùng giờ làm thêm."
    parent = await _add_artifact_with_version(
        db_session,
        project_id,
        "brd",
        "BRD",
        "Container",
    )
    child = await _add_artifact_with_version(
        db_session,
        project_id,
        "problem_statement",
        "Problem Statement",
        draft_body,
        parent_id=parent.id,
    )

    session = AgentSession(
        project_id=project_id,
        artifact_type="problem_statement",
        workflow_area="analysis",
        graph_checkpoint={},
        focused_artifact_id=child.id,
    )
    db_session.add(session)
    await db_session.commit()

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(
        return_value=({"next_action": "done", "confidence": 0.5, "gaps": [], "proposals": []}, None)
    )

    state = _state(artifact_type="problem_statement")
    state["focused_artifact_id"] = str(child.id)
    config = _config(str(session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    await analyze_node(state, config)

    prompt = mock_llm.generate.call_args.kwargs["messages"][0]["content"]
    assert draft_body in prompt, "Existing draft body must be injected as context"
    assert "DRAFT ĐANG CÓ" in prompt


# ---------------------------------------------------------------------------
# Phase A2 (C1): incremental running draft (working_draft)
# ---------------------------------------------------------------------------

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
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content=new_draft, tool_calls=[
        {"id": "scripted:0", "name": "ask_user", "args": {"message": "Còn ràng buộc nào không?"}}
    ]), None))

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

@pytest.mark.asyncio
async def test_active_mode_passes_through_analyze_node(client, db_session):
    """T2: `active_mode` is derived from the gated primary tool and persisted into analysis_result.

    `active_mode` lives inside analysis_result (no new state channel), so the eval layer
    can mine it from AgentRun.analysis_result. It is derived (critique_note → critique), not
    self-reported by the LLM.
    """
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content="", tool_calls=[
        {"id": "scripted:0", "name": "critique_note", "args": {"content": "Ta thử soi lại nhé?"}}
    ]), None))

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert result["analysis_result"]["active_mode"] == "critique"


@pytest.mark.asyncio
async def test_respond_colon_terminated_message_uses_fallback(client, db_session):
    from app.graphs.nodes import _RESPOND_FALLBACK, analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content="", tool_calls=[
        {"id": "scripted:0", "name": "respond", "args": {"message": "Dựa trên thông tin hiện có:"}}
    ]), None))

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    tool_call = result["messages"][0].tool_calls[0]
    assert tool_call["name"] == "respond"
    assert tool_call["args"]["message"] == _RESPOND_FALLBACK
    assert tool_call["args"]["mode"] == "critique"


def test_mode_hint_injects_directive_into_prompt():
    """T4a: a user-supplied mode_hint must surface the requested mode in the prompt."""
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="goal")
    state["mode_hint"] = "critique"

    prompt = _build_tool_selection_prompt(state, [])

    assert "critique" in prompt


def test_feedback_control_block_injects_blockers_and_revision_plan():
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="goal")
    state["quality_report"] = {
        "mode": "critique",
        "score": 0.41,
        "findings": ["Metric success chưa đo được"],
        "suggestions": ["Thêm baseline và target định lượng"],
        "blocking_issues": ["Metric success chưa đo được"],
        "non_blocking_warnings": [],
        "revision_plan": ["Thêm baseline và target định lượng"],
        "quality_gate_result": "fail",
        "recommended_next_action": "revise",
    }
    state["candidate_readiness"] = {
        "state": "well_structured_but_incomplete",
        "can_persist": False,
        "missing": ["Success Metrics"],
        "needs_confirmation": ["Target 15%"],
        "inferred": [],
        "blocking_reasons": ["Thiếu metric bắt buộc"],
    }

    prompt = _build_tool_selection_prompt(state, [])

    assert "FEEDBACK CONTROL" in prompt
    assert "Metric success chưa đo được" in prompt
    assert "Thêm baseline và target định lượng" in prompt
    assert "well_structured_but_incomplete" in prompt
    assert "Success Metrics" in prompt
    assert "revise" in prompt


def test_finalize_degrade_reason_uses_feedback_state():
    from app.graphs.nodes import _degrade_reason

    state = _state(artifact_type="goal")
    state["working_draft"] = "draft"
    state["quality_report"] = {
        "quality_gate_result": "fail",
        "blocking_issues": ["Metric chưa kiểm chứng"],
        "recommended_next_action": "revise",
    }
    state["candidate_readiness"] = {
        "state": "well_structured_but_incomplete",
        "can_persist": False,
        "blocking_reasons": ["Thiếu heading bắt buộc"],
    }

    degrade = _degrade_reason(state, "finalize", "ask_user", {"tools": [{"name": "finalize", "args": {"summary": "Xong"}}]})

    assert degrade is not None
    assert "quality_gate=fail" in degrade["gated_reason"]
    assert "candidate_readiness=well_structured_but_incomplete" in degrade["gated_reason"]
    assert "Metric chưa kiểm chứng" in degrade["message"]


def test_proactive_rule_lives_in_instruction_layer_not_payload():
    """T4b: the proactive mode-switch policy is static — it lives in the decision-policy layer (the
    system prompt), not in the per-turn payload. With no mode_hint the payload carries no mode steer."""
    from app.graphs.nodes import _build_tool_selection_prompt
    from app.instructions import get_instruction, load_instructions

    load_instructions()
    contract = get_instruction(artifact_type="goal", workflow_area="analysis", agent_role=None)
    assert "proactive" in contract.lower()

    state = _state(artifact_type="goal")
    state["mode_hint"] = None
    prompt = _build_tool_selection_prompt(state, [])
    assert "YÊU CẦU MODE" not in prompt


def test_mode_hint_precedes_language_lock():
    """A mode_hint directive must sit before the language lock (lock stays last by contract)."""
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


def test_tool_selection_prompt_includes_artifact_output_contract():
    from app.graphs.nodes import _build_tool_selection_prompt

    prompt = _build_tool_selection_prompt(_state(artifact_type="vision_objectives"), [])

    assert "OUTPUT CONTRACT BẮT BUỘC" in prompt
    assert "## Vision" in prompt
    assert "## Objectives" in prompt
    assert "không copy nguyên transcript" in prompt
    assert "(agent suy diễn, cần xác nhận)" in prompt


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
