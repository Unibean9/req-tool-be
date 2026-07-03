import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from app.graphs.decision_graph import create_node
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
        "key_facts": [],
        "focused_artifact_id": None,
        "draft_body": None,
        "method_profile": dict(DEFAULT_METHOD_PROFILE),
        "artifact_chain": dict(DEFAULT_ARTIFACT_CHAIN),
        "readiness": dict(DEFAULT_READINESS),
        "candidate_readiness": None,
        "tool_errors": [],
        "feedback_summary": None,
        "verification_status": None,
        "latest_checked_revision": None,
        "mode_hint": None,
        "session_elicit_count": 0,
        "decision_nodes": {},
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
        {"id": "scripted:0", "name": "ask_user", "args": {"message": "Can you describe the goal more?"}}
    ]), None))

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert result["analysis_result"]["tools"][0]["name"] == "ask_user"
    assert result["analysis_result"]["model_tool_calls"][0]["name"] == "ask_user"
    assert "ask_user" in result["analysis_result"]["available_tools"]
    assert result["analysis_result"]["dispatched_tool_calls"][0]["name"] == "ask_user"
    assert result["turn_count"] == 1
    assert result["last_agent_run_id"] is not None


@pytest.mark.asyncio
async def test_analyze_node_records_raw_model_tool_calls_before_gate(client, db_session):
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content="", tool_calls=[
        {"id": "scripted:0", "name": "ask_user", "args": {"message": "Can you confirm the target?"}},
        {"id": "scripted:1", "name": "create_decision_node", "args": {"kind": "fact", "statement": "Target unclear"}},
    ]), None))

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    analysis = result["analysis_result"]
    assert [tc["name"] for tc in analysis["model_tool_calls"]] == ["ask_user", "create_decision_node"]
    assert [tc["name"] for tc in analysis["dispatched_tool_calls"]] == ["ask_user"]
    assert analysis["dropped_tool_calls"] == ["create_decision_node"]

    async with TestSessionFactory() as db:
        run = await db.get(AgentRun, uuid.UUID(result["last_agent_run_id"]))
        assert [tc["name"] for tc in run.analysis_result["model_tool_calls"]] == [
            "ask_user",
            "create_decision_node",
        ]


@pytest.mark.asyncio
async def test_analyze_node_audit_omits_tool_body_from_agent_run(client, db_session):
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)
    model_body = "## Vision\nBi mat proposal body.\n\n## Objectives\n- Tang toc.\n\n## Success Metrics\n- 99%."

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content="", tool_calls=[
        {"id": "scripted:0", "name": "write_draft", "args": {"title": "Vision", "body": model_body}}
    ]), None))

    state = _state(artifact_type="vision_objectives")
    state["user_confirmed"] = True  # ELICIT phase so write_draft is in phase (test is about body omission)
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    runtime_call = result["messages"][0].tool_calls[0]
    assert runtime_call["args"]["body"] == model_body
    assert result["analysis_result"]["model_tool_calls"][0]["args"]["body"]["omitted"] is True
    assert model_body not in str(result["analysis_result"])

    async with TestSessionFactory() as db:
        run = await db.get(AgentRun, uuid.UUID(result["last_agent_run_id"]))
        assert model_body not in str(run.analysis_result)


@pytest.mark.asyncio
async def test_analyze_node_binds_only_available_tool_schemas(client, db_session, monkeypatch):
    """Tool schema sent to the LLM must match the state menu, not bind the full registry."""
    from app.graphs.nodes import analyze_node

    monkeypatch.setattr("app.graphs.agent_tools.settings.decision_graph_enabled", True)

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content="", tool_calls=[]), None))

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    await analyze_node(state, config)

    tool_names = {tool["name"] for tool in mock_llm.generate.call_args.kwargs["tools"]}
    assert "create_decision_node" in tool_names
    assert "update_decision_node" not in tool_names
    assert "supersede_decision_node" not in tool_names
    assert "run_critique" not in tool_names
    assert "run_readiness_check" not in tool_names
    assert "finalize" not in tool_names


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

    # Phase 2: after the focus reset clears critique state, finalize would be out of phase on the
    # freshly-focused artifact, so the model picks an in-phase tool (write_draft is valid in any
    # post-confirm phase). The point of this test is the critique-state reset, not the tool identity.
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content="", tool_calls=[
        {"id": "scripted:0", "name": "write_draft", "args": {"title": "Problem", "body": "## Problem\nX."}}
    ]), None))

    state = _state(artifact_type="problem_statement")
    state["user_confirmed"] = True
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
    assert result["analysis_result"]["tools"][0]["name"] == "write_draft"
    assert "gated_tool" not in result["analysis_result"]


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

    brd_title = "BRD: Dieu phoi study scheduling cho sinh vien"
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


def test_output_contract_block_lists_sections_for_graph_view():
    from app.graphs.nodes import _build_output_contract_block

    block = _build_output_contract_block(_state(artifact_type="vision_objectives"))

    # Graph-first per-turn block carries only artifact-specific sections + framing; the
    # recording/status/no-fabrication policy lives in the system prompt (layers 05/10), not here.
    assert "decision graph" in block
    assert "do not hand-write the Markdown body" in block
    assert "## Vision" in block


@pytest.mark.asyncio
async def test_analyze_node_feeds_transitive_ancestry_into_prompt(client, db_session):
    """Mot type sau must thay tien nhiem truc tiep va to tien xa trong prompt."""
    from app.graphs.nodes import analyze_node
    from app.models.artifact import Artifact

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    brd_title = "BRD: Dieu phoi study scheduling"
    domain_title = "Domain entity: Lich nhom"
    component_title = "Component: Bo orchestration lich"
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
    assert component_title in prompt, "Tien nhiem truc tiep (component) must co trong prompt"
    assert domain_title in prompt, "Bridge predecessor (domain_entity) must appear in the prompt"
    assert brd_title in prompt, "To tien xa (brd) must co qua transitive closure"


@pytest.mark.asyncio
async def test_analyze_prefers_strong_client_when_present(client, db_session):
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
    state["messages"] = [{"role": "user", "content": f"Tin nhan {i}"} for i in range(7)]
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=("Tom tat new", None))

    result = await summarize_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert route_before_analyze(state) == "summarize"
    assert result["conversation_summary"] == "Tom tat new"
    llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_degrades_when_response_not_schema_conformant(monkeypatch):
    """A non-conforming summary response (generate raises ValueError) must not crash the turn —
    summarize_node keeps the prior summary and lets the loop continue to analyze."""
    from app.graphs.nodes import summarize_node

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 6)
    state = _state()
    state["conversation_summary"] = "Tom tat cu"
    state["messages"] = [{"role": "user", "content": f"Tin nhan {i}"} for i in range(7)]
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=ValueError("LLM response does not match JSON Schema at $: missing summary"))

    result = await summarize_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert result["conversation_summary"] == "Tom tat cu"


@pytest.mark.asyncio
async def test_triage_degrades_to_work_turn_when_response_not_schema_conformant(monkeypatch):
    """A non-conforming triage response must default to a work turn (falls through to analyze)."""
    from app.graphs.nodes import route_after_triage, triage_node

    state = _state()
    state["messages"] = [{"role": "user", "content": "Xay he thong diem danh"}]
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=ValueError("LLM response does not match JSON Schema at $: missing turn_type"))

    result = await triage_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert result["turn_type"] == "work"
    assert route_after_triage({**state, **result}) == "orchestrator"


def test_summarize_triggers_on_real_ask_loop_message_counts(monkeypatch):
    from app.graphs.nodes import route_before_analyze

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 6)

    routes = {}
    for count in [1, 3, 5, 7, 9, 11, 13]:
        state = _state()
        state["messages"] = [{"role": "user", "content": f"Tin nhan {i}"} for i in range(count)]
        routes[count] = route_before_analyze(state)

    assert routes == {
        1: "orchestrator",
        3: "orchestrator",
        5: "orchestrator",
        7: "summarize",
        9: "orchestrator",
        11: "orchestrator",
        13: "summarize",
    }


def test_summarize_skips_tool_only_loop_even_at_human_threshold(monkeypatch):
    from langchain_core.messages import AIMessage, ToolMessage

    from app.graphs.nodes import route_before_analyze

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 2)
    state = _state()
    state["messages"] = [
        {"role": "user", "content": "A"},
        {"role": "user", "content": "B"},
        {"role": "user", "content": "C"},
        AIMessage(content="", tool_calls=[{"id": "call-1", "name": "explore_note", "args": {"content": "fact"}}]),
        ToolMessage(content="Da ghi nhan", tool_call_id="call-1"),
    ]

    assert route_before_analyze(state) == "orchestrator"


@pytest.mark.asyncio
async def test_summarize_skipped_below_threshold(monkeypatch):
    from app.graphs.nodes import route_before_analyze, summarize_node

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 6)
    state = _state()
    state["conversation_summary"] = "Tom tat cu"
    state["messages"] = [{"role": "user", "content": f"Tin nhan {i}"} for i in range(5)]
    llm = AsyncMock()
    llm.generate = AsyncMock()

    result = await summarize_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert route_before_analyze(state) == "orchestrator"
    assert result["conversation_summary"] == "Tom tat cu"
    llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_summary_preserves_constraints_verbatim(monkeypatch):
    from app.graphs.nodes import summarize_node

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 2)
    state = _state()
    state["messages"] = [
        {"role": "user", "content": "Tao MVP"},
        {"role": "user", "content": "Ngan sach toi da 50 trieu"},
        {"role": "user", "content": "Deadline 2 thang"},
    ]
    summary = (
        "Confirmed requirements\n- Lam MVP\n"
        "Rang buoc — KHONG paraphrase\n- Ngan sach toi da 50 trieu\n"
        "Khoang trong chua ro\n- Thoi han\n"
        "Quyet dinh da thong nhat\n- Chua co"
    )
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(summary, None))

    result = await summarize_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert "Rang buoc — KHONG paraphrase" in result["conversation_summary"]
    assert "Ngan sach toi da 50 trieu" in result["conversation_summary"]


@pytest.mark.asyncio
async def test_summarize_node_uses_default_client(monkeypatch):
    from app.graphs.nodes import summarize_node

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 2)
    state = _state()
    state["messages"] = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
        {"role": "user", "content": "D"},
    ]
    default_llm = AsyncMock()
    default_llm.generate = AsyncMock(return_value=("Tom tat", None))
    strong_llm = AsyncMock()
    strong_llm.generate = AsyncMock()
    config = _config(str(uuid.uuid4()), str(uuid.uuid4()), default_llm)
    config["configurable"]["strong_llm_client"] = strong_llm

    await summarize_node(state, config)

    default_llm.generate.assert_called_once()
    strong_llm.generate.assert_not_called()


def test_build_prompt_carries_summary_not_raw_turns():
    """The payload carries only the compacted summary; raw turns live in the message thread, so the
    payload must not restate them (no double representation)."""
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state()
    state["conversation_summary"] = "Rang buoc — KHONG paraphrase\n- Ngan sach toi da 50 trieu"
    state["messages"] = [{"role": "user", "content": f"Tin nhan {i}"} for i in range(5)]

    prompt = _build_tool_selection_prompt(state, [])

    assert "Accumulated conversation summary" in prompt
    assert "Ngan sach toi da 50 trieu" in prompt
    assert not any(f"Tin nhan {i}" in prompt for i in range(5))


def test_build_prompt_omits_conversation_when_no_summary():
    """No summary → no conversation block at all; the thread is the sole carrier of the dialogue."""
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state()
    state["messages"] = [{"role": "user", "content": f"Tin nhan {i}"} for i in range(6)]

    prompt = _build_tool_selection_prompt(state, [])

    assert "Accumulated conversation summary" not in prompt
    assert not any(f"Tin nhan {i}" in prompt for i in range(6))


def test_analyzer_messages_summary_replaces_old_transcript():
    from app.graphs.nodes import _build_analyzer_messages

    state = _state()
    state["conversation_summary"] = "Tom tat: ngan sach toi da 50 trieu"
    state["messages"] = [
        {"role": "user", "content": "LUOT USER CU KHONG DUOC GUI"},
        {"role": "assistant", "content": "LUOT ASSISTANT CU KHONG DUOC GUI"},
        {"role": "user", "content": "Thong tin trung gian"},
        {"role": "assistant", "content": "Ban muon tao artifact nao?"},
        {"role": "user", "content": "Moi nhat: hay tao PRD"},
    ]

    messages = _build_analyzer_messages(state, "WORKSPACE co summary")
    rendered = str(messages)

    assert "LUOT USER CU KHONG DUOC GUI" not in rendered
    assert "LUOT ASSISTANT CU KHONG DUOC GUI" not in rendered
    assert "Ban muon tao artifact nao?" in rendered
    assert "Moi nhat: hay tao PRD" in rendered


def test_analyzer_history_bounded_from_turn_one_without_summary(monkeypatch):
    from app.graphs.nodes import _analyzer_history_messages

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 6)
    state = _state()
    state["conversation_summary"] = ""
    state["messages"] = [
        item
        for i in range(10)
        for item in ({"role": "user", "content": f"Tin nhan cu {i}"}, {"role": "assistant", "content": f"Tra loi {i}"})
    ]

    history = _analyzer_history_messages(state)
    rendered = str(history)

    assert len(history) < len(state["messages"])
    assert "Tin nhan cu 0" not in rendered
    assert "Tin nhan cu 9" in rendered


def test_analyzer_history_unbounded_below_window_without_summary(monkeypatch):
    from app.graphs.nodes import _analyzer_history_messages

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 6)
    state = _state()
    state["conversation_summary"] = ""
    state["messages"] = [{"role": "user", "content": f"Tin nhan {i}"} for i in range(3)]

    history = _analyzer_history_messages(state)

    assert history == state["messages"]


def test_analyzer_summary_compaction_keeps_matching_tool_use():
    from langchain_core.messages import AIMessage, ToolMessage

    from app.graphs.nodes import _build_analyzer_messages

    state = _state()
    state["conversation_summary"] = "Tom tat cu"
    state["messages"] = [
        {"role": "user", "content": "LUOT USER CU KHONG DUOC GUI"},
        AIMessage(content="", tool_calls=[{"id": "call-1", "name": "ask_user", "args": {"message": "Pain chinh?"}}]),
        ToolMessage(content="Hut nguyen lieu", tool_call_id="call-1"),
        {"role": "user", "content": "Hut nguyen lieu"},
    ]

    messages = _build_analyzer_messages(state, "WORKSPACE")

    assert "LUOT USER CU KHONG DUOC GUI" not in str(messages)
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"][0]["type"] == "tool_use"
    assert messages[0]["content"][0]["id"] == "call-1"
    assert messages[1]["content"][0]["type"] == "tool_result"


def test_analyzer_summary_compaction_keeps_immediate_assistant_context():
    from app.graphs.nodes import _build_analyzer_messages

    state = _state()
    state["conversation_summary"] = "Tom tat cu"
    state["messages"] = [
        {"role": "user", "content": "LUOT USER CU KHONG DUOC GUI"},
        {"role": "assistant", "content": "Thi truong muc tieu la gi?"},
        {"role": "user", "content": "Sinh vien nam 2"},
    ]

    messages = _build_analyzer_messages(state, "WORKSPACE")
    rendered = str(messages)

    assert "LUOT USER CU KHONG DUOC GUI" not in rendered
    assert "Thi truong muc tieu la gi?" in rendered
    assert "Sinh vien nam 2" in rendered


def test_build_prompt_excludes_static_policy():
    """Static policy (tool semantics, synthesis depth) lives in the instruction layers now, so the
    per-turn payload must NOT restate it."""
    from app.graphs.nodes import _build_tool_selection_prompt

    prompt = _build_tool_selection_prompt(_state(), [])

    # The old inline synthesis directive and per-tool description block are gone from the payload.
    assert "DO SAU NOI DUNG" not in prompt
    assert "critique note" not in prompt
    # It still names the tools available this turn so the model knows the current menu.
    assert "Tools available this turn" in prompt


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

    assert "Section coverage" in prompt
    # Gap-inventory marker — lists weak sections, not a single pinned question.
    assert "aspects still missing or unclear" in prompt
    # Harness voice: advance the artifact, not "ask one main question".
    assert "advance" in prompt
    assert "one main question" not in prompt


def test_no_coverage_hint_when_complete():
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="problem")
    state["section_coverage"] = {"problem_statement": "filled"}
    state["coverage_complete"] = True

    prompt = _build_tool_selection_prompt(state, [])

    assert "Section coverage" not in prompt


@pytest.mark.asyncio
async def test_messages_not_truncated_by_summarize(monkeypatch):
    from app.graphs.nodes import summarize_node

    monkeypatch.setattr("app.graphs.nodes.settings.summary_trigger_every", 3)
    state = _state()
    state["messages"] = [{"role": "user", "content": f"Tin nhan {i}"} for i in range(3)]
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=("Tom tat", None))

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
        # Pre-existing keys stay present, unchanged in meaning (P10 backward-compat).
        assert run.token_usage["input"] == 5
        assert run.token_usage["output"] == 10
        assert run.token_usage["total"] == 15
        assert isinstance(run.latency_ms, int)
        assert run.latency_ms >= 0


@pytest.mark.asyncio
async def test_analyze_node_token_usage_has_additive_component_breakdown(client, db_session):
    """P10: token_usage gains a `by_component` key alongside pre-existing keys, never replacing them."""
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
        assert run.token_usage["input"] == 5
        assert run.token_usage["output"] == 10
        assert run.token_usage["total"] == 15
        by_component = run.token_usage["by_component"]
        assert set(by_component.keys()) == {"system", "history", "tools", "draft"}
        assert all(isinstance(v, int) for v in by_component.values())


# ---------------------------------------------------------------------------
# route_node tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_node_appends_one_fingerprint_per_turn_not_per_dispatched_call(client, db_session):
    """P9: a single turn dispatching several identical tool calls at once must add exactly one
    fingerprint to recent_tool_calls, not one per call — otherwise a legitimate multi-call turn would
    be mistaken for several consecutive stuck turns and trip the early-exit threshold immediately."""
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    agent_session = await _make_agent_session(client, db_session, project_id)

    same_call = {"id": "1", "name": "explore_note", "args": {"content": "a"}}
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(
        AIMessage(content="", tool_calls=[same_call, same_call, same_call]),
        {"input": 5, "output": 10, "total": 15},
    ))

    state = _state()
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert len(result["recent_tool_calls"]) == 1


def test_route_node_max_turns_routes_to_end():
    from langgraph.graph import END

    from app.config import settings
    from app.graphs.nodes import route_node

    state = _state(turn_count=settings.max_agent_turns, analysis_result={"next_action": "propose"})
    assert route_node(state) == END


def test_route_node_exits_early_on_repeated_identical_tool_calls():
    """P9: N (3) consecutive identical (name+args) tool-call fingerprints exit before max_agent_turns."""
    from langgraph.graph import END

    from app.graphs.nodes import _tool_call_fingerprint, route_node

    fingerprint = _tool_call_fingerprint("write_draft", {"body": "same body"})
    state = _state(turn_count=2)
    state["recent_tool_calls"] = [fingerprint, fingerprint, fingerprint]
    assert route_node(state) == END


def test_route_node_does_not_exit_on_varying_or_below_threshold_repeats():
    """P9 false-positive guard: varying calls, or fewer than N identical repeats, must not exit early."""
    from app.graphs.nodes import _tool_call_fingerprint, route_node

    varying_fp_a = _tool_call_fingerprint("explore_note", {"content": "a"})
    varying_fp_b = _tool_call_fingerprint("explore_note", {"content": "b"})
    state = _state(turn_count=2)
    state["recent_tool_calls"] = [varying_fp_a, varying_fp_b, varying_fp_a]
    state["messages"] = [AIMessage(content="", tool_calls=[{"id": "1", "name": "explore_note", "args": {"content": "a"}}])]
    assert route_node(state) == "tools"

    same_fp = _tool_call_fingerprint("write_draft", {"body": "same body"})
    state_below_threshold = _state(turn_count=2)
    state_below_threshold["recent_tool_calls"] = [same_fp, same_fp]
    state_below_threshold["messages"] = [
        AIMessage(content="", tool_calls=[{"id": "1", "name": "write_draft", "args": {"body": "same body"}}])
    ]
    assert route_node(state_below_threshold) == "tools"


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

    async def _generate(*, messages, system, max_tokens, response_format=None, tools=None, **_kwargs):
        if tools is not None:
            analyze_calls.append(1)
            return AIMessage(content="", tool_calls=[
                {"id": "scripted:0", "name": "ask_user", "args": {"message": "What else do you need?"}}
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
    state["messages"] = [{"role": "user", "content": "I need to create intent"}]
    await graph.ainvoke(state, config)
    assert len(analyze_calls) == 1

    # Resume the ask_human interrupt: the loop continues and analyze runs again.
    await graph.ainvoke(Command(resume={"content": "them detailed"}), config)
    assert len(analyze_calls) == 2


# ---------------------------------------------------------------------------
# Triage + converse (entry routing)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_triage_classifies_converse_and_drafts_reply():
    from app.graphs.nodes import triage_node

    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=(
        {"turn_type": "converse", "locale": "vi", "reply": "Hello, what do you want to build?"}, None
    ))
    state = _state()
    state["messages"] = [{"role": "user", "content": "hello"}]

    result = await triage_node(state, _config(str(uuid.uuid4()), str(uuid.uuid4()), llm))

    assert result["turn_type"] == "converse"
    assert result["locale"] == "vi"
    assert result["triage_reply"] == "Hello, what do you want to build?"


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
async def test_triage_uses_default_client_even_when_strong_present():
    """triage_node stays on the default/cheap tier regardless of a configured strong tier —
    analyze_node is the only node that prefers strong_llm_client."""
    from app.graphs.nodes import triage_node

    default_llm = AsyncMock()
    default_llm.generate = AsyncMock(return_value=(
        {"turn_type": "converse", "locale": "vi", "reply": "Hello, what do you want to build?"}, None
    ))
    strong_llm = AsyncMock()
    strong_llm.generate = AsyncMock()

    state = _state()
    state["messages"] = [{"role": "user", "content": "hello"}]
    config = _config(str(uuid.uuid4()), str(uuid.uuid4()), default_llm)
    config["configurable"]["strong_llm_client"] = strong_llm

    await triage_node(state, config)

    default_llm.generate.assert_called_once()
    strong_llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_triage_raises_when_llm_client_none():
    from app.graphs.nodes import triage_node

    config = _config(str(uuid.uuid4()), str(uuid.uuid4()), llm_client=None)
    config["configurable"]["llm_client"] = None
    state = _state()
    state["messages"] = [{"role": "user", "content": "chao"}]

    with pytest.raises(ValueError):
        await triage_node(state, config)


def test_route_after_triage_splits_converse_from_work():
    from app.graphs.nodes import route_after_triage

    assert route_after_triage({"turn_type": "converse"}) == "converse"
    assert route_after_triage({"turn_type": "work"}) == "orchestrator"
    # Missing/unknown defaults to the analyst, never silently skips work.
    assert route_after_triage({}) == "orchestrator"


@pytest.mark.asyncio
@patch("app.graphs.nodes.interrupt")
async def test_converse_node_replies_and_interrupts_without_agent_run(mock_interrupt, client, db_session):
    """A conversational turn replies and pauses for the human — no AgentRun (the full analyst pass
    never ran), and no LLM call inside converse itself (it reuses the triage-drafted reply)."""
    from app.graphs.nodes import converse_node

    mock_interrupt.return_value = {"content": "toi muon xay app abc"}
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state()
    state["locale"] = "vi"
    state["triage_reply"] = "Hello! Ban muon bat dau tu dau?"
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    result = await converse_node(state, config)

    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        msg = (
            await db.execute(select(AgentMessage).where(AgentMessage.session_id == agent_session.id))
        ).scalar_one()
        assert msg.payload["kind"] == "greeting"
        assert msg.content == "Hello! Ban muon bat dau tu dau?"
        runs = (await db.execute(select(AgentRun).where(AgentRun.session_id == agent_session.id))).scalars().all()
        assert runs == []
    # The human's reply is folded in for analyze to pick up next.
    assert result["messages"][-1]["content"] == "toi muon xay app abc"


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
            return {"turn_type": "converse", "locale": "vi", "reply": "Hello!"}, None
        return {}, None

    llm.generate = _generate
    graph = build_graph(checkpointer=MemorySaver())
    config = _config(str(agent_session.id), str(project_id), llm)
    config["configurable"]["session_factory"] = _session_factory()

    state = _state()
    state["messages"] = [{"role": "user", "content": "hello"}]
    out = await graph.ainvoke(state, config)

    assert "__interrupt__" in out
    async with TestSessionFactory() as db:
        runs = (await db.execute(select(AgentRun).where(AgentRun.session_id == agent_session.id))).scalars().all()
        assert runs == [], "greeting must not trigger the analyst pass"


# ---------------------------------------------------------------------------
# One-question Rhythm (S8)
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
    body = "User: student. Obstacle: study group schedule conflict."
    prompt = _build_tool_selection_prompt(state, [], draft_body=body)

    assert "CURRENT DRAFT" in prompt
    assert body in prompt
    # The anti-re-ask / mine-the-delta policy is static (question-policy layer), so the per-turn
    # payload carries the draft as data without restating the imperative.
    assert "do not ask again" not in prompt.lower()
    # draft block must precede the language lock (kept last by contract)
    assert prompt.index("CURRENT DRAFT") < prompt.index("language 'vi'")


def test_build_prompt_no_draft_block_when_absent():
    """Regression guard: create-from-scratch prompt unchanged when no draft exists."""
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="problem")
    baseline = _build_tool_selection_prompt(state, [])
    with_none = _build_tool_selection_prompt(state, [], draft_body=None)

    assert "CURRENT DRAFT" not in with_none
    assert with_none == baseline


@pytest.mark.asyncio
async def test_read_current_body_returns_one_when_multiple(client, db_session):
    """Multiple drafts of the same type: returns exactly one (no crash).

    Picking the *right* target for a deliberate update is the authoritative
    target_artifact_id problem — A1 only surfaces a single draft as context.
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
        Artifact(project_id=project_id, type="functional_requirement", title="Trong", extra_metadata={}, status="draft")
    )
    await db_session.commit()

    async with TestSessionFactory() as db:
        result = await read_current_body(db=db, project_id=project_id, artifact_type="functional_requirement")

    assert result is None


@pytest.mark.asyncio
async def test_read_artifacts_with_type_list_issues_one_query(client, db_session):
    """A list of artifact types must batch into a single query, not one per type."""
    from app.graphs.tools import read_artifacts

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    await _add_artifact_with_version(db_session, project_id, "brd", "BRD", "body brd")
    await _add_artifact_with_version(db_session, project_id, "epic", "Epic", "body epic")
    await _add_artifact_with_version(db_session, project_id, "story", "Story", "body story")

    async with TestSessionFactory() as db:
        execute_spy = AsyncMock(wraps=db.execute)
        with patch.object(db, "execute", execute_spy):
            result = await read_artifacts(
                db=db,
                project_id=project_id,
                artifact_type=["brd", "epic", "story"],
                context={"workflow_area": "analysis"},
            )

    assert execute_spy.await_count == 1, "One call for the whole ancestor-type chain, not one per type"
    assert {row["type"] for row in result} == {"brd", "epic", "story"}


@pytest.mark.asyncio
async def test_read_artifacts_with_str_type_keeps_existing_single_type_behavior(client, db_session):
    """Widened signature must not change behavior for the existing str/None cases."""
    from app.graphs.tools import read_artifacts

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    await _add_artifact_with_version(db_session, project_id, "brd", "BRD", "body brd")
    await _add_artifact_with_version(db_session, project_id, "epic", "Epic", "body epic")

    async with TestSessionFactory() as db:
        result = await read_artifacts(
            db=db, project_id=project_id, artifact_type="brd", context={"workflow_area": "analysis"}
        )

    assert [row["type"] for row in result] == ["brd"]


@pytest.mark.asyncio
async def test_analyze_node_loads_current_draft_body_into_prompt(client, db_session):
    """A focused document item must expose its current draft in the prompt."""
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    draft_body = "Doi tuong: sinh vien nam 2. Tro ngai: study scheduling hay bi trung gio lam them."
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
    assert "CURRENT DRAFT" in prompt


async def _update_artifact_body(db_session, artifact, body: str) -> None:
    """Add a new current version with `body`, mirroring how write_draft advances a draft."""
    from app.models.artifact import ArtifactVersion, ChangeSource, VersionStatus

    version = ArtifactVersion(
        artifact_id=artifact.id,
        version_number=2,
        title=artifact.title,
        body=body,
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.AI_GENERATION,
        extra_metadata={},
    )
    db_session.add(version)
    await db_session.flush()
    artifact.current_version_id = version.id
    await db_session.commit()


@pytest.mark.asyncio
async def test_analyze_node_sends_diff_then_omits_unchanged_draft_across_turns(client, db_session):
    """Multi-field draft mutation across turns: full body, then diff-only, then no draft block."""
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    turn1_body = "Doi tuong: sinh vien nam 2.\nTro ngai: study scheduling.\nMuc tieu: giam trung lich."
    parent = await _add_artifact_with_version(db_session, project_id, "brd", "BRD", "Container")
    child = await _add_artifact_with_version(
        db_session, project_id, "problem_statement", "Problem Statement", turn1_body, parent_id=parent.id
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
    config = _config(str(session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    # Turn 1: no previous draft_body in state yet -> full body sent.
    state = _state(artifact_type="problem_statement")
    state["focused_artifact_id"] = str(child.id)
    result1 = await analyze_node(state, config)
    prompt1 = mock_llm.generate.call_args.kwargs["messages"][0]["content"]
    assert turn1_body in prompt1
    assert result1["draft_body"] == turn1_body

    # Turn 2: draft mutated across multiple fields; state carries turn 1's draft_body forward.
    turn2_body = "Doi tuong: sinh vien nam 3.\nTro ngai: study scheduling hay bi trung gio lam them.\nMuc tieu: giam trung lich."
    await _update_artifact_body(db_session, child, turn2_body)
    state2 = {**state, "draft_body": result1["draft_body"], "turn_count": result1["turn_count"]}
    result2 = await analyze_node(state2, config)
    prompt2 = mock_llm.generate.call_args.kwargs["messages"][0]["content"]
    assert "unified diff" in prompt2.lower() or "diff" in prompt2.lower()
    assert turn2_body not in prompt2, "Turn 2 must not resend the full (unchanged-shape) body"
    assert "sinh vien nam 3" in prompt2, "Changed field must be present in the diff"
    assert "study scheduling hay bi trung gio lam them" in prompt2, "Second changed field must be present in the diff"
    assert result2["draft_body"] == turn2_body

    # Turn 3: no further change -> draft block omitted entirely.
    state3 = {**state, "draft_body": result2["draft_body"], "turn_count": result2["turn_count"]}
    await analyze_node(state3, config)
    prompt3 = mock_llm.generate.call_args.kwargs["messages"][0]["content"]
    assert "CURRENT DRAFT" not in prompt3
    assert "diff" not in prompt3.lower()


# ---------------------------------------------------------------------------
# Decision graph as live draft source
# ---------------------------------------------------------------------------

def test_build_prompt_includes_decision_view_block_when_nodes_present():
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="problem")
    state["locale"] = "vi"  # populate language_lock so the ordering assert is meaningful
    state["decision_nodes"] = {
        "N1": create_node(
            kind="fact",
            statement="Sinh vien trung study scheduling voi gio lam them.",
            origin={"source": "test"},
            status="confirmed",
        )
    }

    prompt = _build_tool_selection_prompt(state, [])

    assert "DRAFT IN PROGRESS" in prompt
    assert "Sinh vien trung study scheduling voi gio lam them." in prompt
    # The running draft must precede the language lock (kept last by contract).
    assert prompt.index("DRAFT IN PROGRESS") < prompt.index("language 'vi'")


def test_build_prompt_hides_persisted_draft_when_decision_view_covers_contract():
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="vision_objectives")
    state["decision_nodes"] = {
        "N1": create_node(
            kind="objective",
            statement="Sinh vien can xem ton kho realtime.",
            origin={"source": "test"},
            status="confirmed",
            section="## Vision",
        ),
        "N2": create_node(
            kind="objective",
            statement="Giam thoi gian kiem tra hang ton.",
            origin={"source": "test"},
            status="confirmed",
            section="## Objectives",
        ),
        "N3": create_node(
            kind="objective",
            statement="Do thoi gian lay du lieu ton kho.",
            origin={"source": "test"},
            status="confirmed",
            section="## Success Metrics",
        ),
    }

    prompt = _build_tool_selection_prompt(state, [], "NOI DUNG DRAFT CU KHONG DUOC GUI")

    assert "Sinh vien can xem ton kho realtime." in prompt
    assert "NOI DUNG DRAFT CU KHONG DUOC GUI" not in prompt


def test_build_prompt_keeps_persisted_draft_when_decision_view_is_partial():
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="vision_objectives")
    state["decision_nodes"] = {
        "N1": create_node(
            kind="objective",
            statement="Sinh vien can xem ton kho realtime.",
            origin={"source": "test"},
            status="confirmed",
            section="## Vision",
        )
    }
    draft_body = "\n\n".join(
        [
            "## Vision\nVersion 2 vision.",
            "## Objectives\nVersion 2 objectives.",
            "## Success Metrics\nVersion 2 metrics.",
        ]
    )

    prompt = _build_tool_selection_prompt(state, [], draft_body)

    assert "Sinh vien can xem ton kho realtime." in prompt
    assert "CURRENT DRAFT" in prompt
    assert draft_body in prompt


def test_build_prompt_no_decision_view_block_when_nodes_absent():
    """Without graph, prompt does not use stale live draft from checkpoint."""
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="problem")
    baseline = _build_tool_selection_prompt(state, [])

    assert "DRAFT IN PROGRESS" not in _build_tool_selection_prompt(state, [])
    assert _build_tool_selection_prompt(state, []) == baseline


# ---------------------------------------------------------------------------
# P7: cross-turn cache for _build_decision_view_block
# ---------------------------------------------------------------------------

def _decision_state(statement: str) -> WorkflowState:
    state = _state(artifact_type="problem")
    state["decision_nodes"] = {
        "N1": create_node(
            kind="fact",
            statement=statement,
            origin={"source": "test"},
            status="confirmed",
        )
    }
    return state


def test_decision_view_block_cache_hit_skips_render_view():
    from app.graphs import decision_graph
    from app.graphs.nodes import _build_decision_view_block

    session_id = f"cache-test-{uuid.uuid4()}"
    state = _decision_state("Sinh vien can lich hoc linh hoat.")

    with patch.object(decision_graph, "render_view", wraps=decision_graph.render_view) as spy:
        first = _build_decision_view_block(state, session_id)
        assert spy.call_count == 1
        second = _build_decision_view_block(state, session_id)
        assert spy.call_count == 1  # cache hit: render_view not called again
        assert first == second


def test_decision_view_block_mutation_invalidates_cache():
    from app.graphs import decision_graph
    from app.graphs.nodes import _build_decision_view_block

    session_id = f"cache-test-{uuid.uuid4()}"
    state = _decision_state("Sinh vien can lich hoc linh hoat.")

    with patch.object(decision_graph, "render_view", wraps=decision_graph.render_view) as spy:
        first = _build_decision_view_block(state, session_id)
        assert spy.call_count == 1

        mutated_state = _decision_state("Sinh vien can bao cao tien do hang tuan.")
        second = _build_decision_view_block(mutated_state, session_id)
        assert spy.call_count == 2  # content changed -> recompute, not a stale hit
        assert first != second


def test_decision_view_block_cached_output_matches_uncached_output():
    """Parity: cached path renders byte-identical output to the uncached path for the same input."""
    from app.graphs.nodes import _build_decision_view_block

    state = _decision_state("Sinh vien can lich hoc linh hoat.")

    uncached = _build_decision_view_block(state, None)
    session_id = f"cache-test-{uuid.uuid4()}"
    cached_first_call = _build_decision_view_block(state, session_id)
    cached_second_call = _build_decision_view_block(state, session_id)

    assert uncached == cached_first_call == cached_second_call


@pytest.mark.asyncio
async def test_analyze_node_ignores_content_emitted_with_tool_calls(client, db_session):
    """Under forced tool_choice, content emitted alongside tool_calls is reasoning, not a draft.

    Drafts of record flow through decision_nodes/write_draft, never through AIMessage content.
    """
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    thinking_text = "User just said X; I should ask for constraints before writing."
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content=thinking_text, tool_calls=[
        {"id": "scripted:0", "name": "ask_user", "args": {"message": "Any other constraints?"}}
    ]), None))

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert "draft_update" not in result["analysis_result"]
    assert "working_draft" not in result


@pytest.mark.asyncio
async def test_analyze_node_passes_real_tool_thread_to_llm(client, db_session):
    """Analyze loop after tool-result must keep real tool_use/tool_result, not flatten into transcript."""
    from langchain_core.messages import ToolMessage

    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content="Da du du kien.", tool_calls=[]), None))

    state = _state(artifact_type="goal")
    state["messages"] = [
        {"role": "user", "content": "I want to set goals for a study group product."},
        AIMessage(content="Toi will ghi nhan du kien.", tool_calls=[
            {
                "id": "prev:0",
                "name": "explore_note",
                "args": {"content": "Primary users are study group students."},
            }
        ]),
        ToolMessage(content="Da ghi nhan key fact.", tool_call_id="prev:0"),
        {"role": "user", "content": "Da ghi nhan key fact."},
    ]
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    await analyze_node(state, config)

    sent_messages = mock_llm.generate.call_args.kwargs["messages"]
    assert sent_messages[0] == {"role": "user", "content": "I want to set goals for a study group product."}
    assert sent_messages[1]["role"] == "assistant"
    assert sent_messages[1]["content"][1] == {
        "type": "tool_use",
        "id": "prev:0",
        "name": "explore_note",
        "input": {"content": "Primary users are study group students."},
    }
    assert sent_messages[2]["role"] == "user"
    assert sent_messages[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "prev:0",
        "name": "explore_note",
        "content": "Da ghi nhan key fact.",
    }
    assert sent_messages[2]["content"][1]["type"] == "text"
    assert "You are the analyst" in sent_messages[2]["content"][1]["text"]


@pytest.mark.asyncio
async def test_analyze_node_does_not_create_legacy_draft_when_no_tools(client, db_session):
    """Terminal text must not be written to legacy state; graph/write_draft is the draft source."""
    from app.graphs.nodes import analyze_node, route_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(
        return_value=(AIMessage(content="Ban can bo sung thong tin ton kho nao?", tool_calls=[]), None)
    )

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert "working_draft" not in result
    assert "draft_update" not in result["analysis_result"]
    assert result["messages"][0].tool_calls[0]["name"] == "ask_user"
    assert route_node({**state, **result}) == "tools"


# ---------------------------------------------------------------------------
# Multi-angle: no active_mode lock + mode_hint + proactive directive
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analysis_result_has_no_active_mode(client, db_session):
    """The single-mode-per-turn lock is gone: analysis_result carries no active_mode label."""
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content="", tool_calls=[
        {"id": "scripted:0", "name": "critique_note", "args": {"content": "Ta thu soi lai nhe?"}}
    ]), None))

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    assert "active_mode" not in result["analysis_result"]


@pytest.mark.asyncio
async def test_respond_colon_terminated_message_uses_fallback(client, db_session):
    from app.graphs.nodes import _RESPOND_FALLBACK_BY_LOCALE, analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(AIMessage(content="", tool_calls=[
        {"id": "scripted:0", "name": "respond", "args": {"message": "Dua tren thong tin hien co:"}}
    ]), None))

    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)

    tool_call = result["messages"][0].tool_calls[0]
    assert tool_call["name"] == "respond"
    assert tool_call["args"]["message"] == _RESPOND_FALLBACK_BY_LOCALE["vi"]
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
        "findings": ["Success metric is not measurable yet"],
        "suggestions": ["Add baseline and quantified target"],
        "blocking_issues": ["Success metric is not measurable yet"],
        "non_blocking_warnings": [],
        "revision_plan": ["Add baseline and quantified target"],
        "quality_gate_result": "fail",
        "recommended_next_action": "revise",
    }
    state["candidate_readiness"] = {
        "state": "well_structured_but_incomplete",
        "can_persist": False,
        "missing": ["Success Metrics"],
        "needs_confirmation": ["Target 15%"],
        "inferred": [],
        "blocking_reasons": ["Thieu metric bat buoc"],
    }

    prompt = _build_tool_selection_prompt(state, [])

    assert "FEEDBACK CONTROL" in prompt
    assert "Success metric is not measurable yet" in prompt
    assert "Add baseline and quantified target" in prompt
    assert "well_structured_but_incomplete" in prompt
    assert "Success Metrics" in prompt
    assert "revise" in prompt


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
    assert "MODE REQUEST" not in prompt


def test_mode_hint_precedes_language_lock():
    """A mode_hint directive must sit before the language lock (lock stays last by contract)."""
    from app.graphs.nodes import _build_tool_selection_prompt

    state = _state(artifact_type="goal")
    state["locale"] = "vi"
    state["mode_hint"] = "explore"

    prompt = _build_tool_selection_prompt(state, [])

    assert prompt.index("explore") < prompt.index("language 'vi'")


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
        "message": "What else do you need?",
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


def test_artifact_contract_block_carries_sections_and_taxonomy():
    """Artifact shape (section-coverage contract + taxonomy chain) now lives in the SYSTEM prompt
    via _build_artifact_contract_block, not the per-turn payload."""
    from app.graphs.nodes import _build_artifact_contract_block, _build_tool_selection_prompt

    block = _build_artifact_contract_block(_state(artifact_type="vision_objectives"))
    assert "SECTION COVERAGE REQUIRED" in block
    assert "## Vision" in block
    assert "## Objectives" in block
    assert "view rendered from the decision graph" in block
    assert "ARTIFACT TYPE" in block  # taxonomy chain moved here too

    # And it is no longer duplicated in the per-turn payload.
    prompt = _build_tool_selection_prompt(_state(artifact_type="vision_objectives"), [])
    assert "SECTION COVERAGE REQUIRED" not in prompt


def _final_block_text(message: dict) -> str:
    content = message["content"]
    if isinstance(content, list):
        return str(content[-1].get("text", ""))
    return str(content)


def test_analyzer_messages_keep_user_latest_as_final_block():
    """Recency: the human's latest message must be the final text block, not the workspace payload."""
    from langchain_core.messages import AIMessage, ToolMessage

    from app.graphs.nodes import _build_analyzer_messages

    # Plain-user turn (turn 1)
    st = _state(artifact_type="vision_objectives")
    st["messages"] = [{"role": "user", "content": "Toi muon lam app coffee shop"}]
    msgs = _build_analyzer_messages(st, "WORKSPACE-PAYLOAD")
    assert "Toi muon lam app coffee shop" in _final_block_text(msgs[-1])

    # Tool_result-resume turn: the human reply arrives inside a tool_result block
    st2 = _state(artifact_type="vision_objectives")
    st2["messages"] = [
        {"role": "user", "content": "Toi muon lam app coffee shop"},
        AIMessage(content="", tool_calls=[{"id": "r:0", "name": "ask_user", "args": {"message": "Pain chinh?"}}]),
        ToolMessage(content="Hut nguyen lieu", tool_call_id="r:0"),
        {"role": "user", "content": "Hut nguyen lieu"},
    ]
    msgs2 = _build_analyzer_messages(st2, "WORKSPACE-PAYLOAD")
    final2 = _final_block_text(msgs2[-1])
    assert msgs2[-1]["role"] == "user"
    assert "Hut nguyen lieu" in final2
    assert "WORKSPACE-PAYLOAD" not in final2


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
