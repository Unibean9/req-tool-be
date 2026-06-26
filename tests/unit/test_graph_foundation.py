import pytest
from langgraph.graph import add_messages

from app.config import settings
from app.graphs.policy import ARTIFACT_PREDECESSORS, ApprovalRequired, GovernanceDenied, governed
from app.graphs.state import (
    DEFAULT_ARTIFACT_CHAIN,
    DEFAULT_METHOD_PROFILE,
    DEFAULT_READINESS,
    WorkflowState,
)


@pytest.mark.asyncio
async def test_governed_allows_read_tool():
    called = False

    @governed
    async def read_artifacts():
        nonlocal called
        called = True
        return ["artifact"]

    result = await read_artifacts(context={"workflow_area": "analysis"})

    assert called is True
    assert result == ["artifact"]


@pytest.mark.asyncio
async def test_governed_requires_approval_for_write_tool():
    @governed
    async def create_artifact(*, artifact_type: str, title: str):
        return {"artifact_type": artifact_type, "title": title}

    with pytest.raises(ApprovalRequired) as exc_info:
        await create_artifact(
            artifact_type="goal",
            title="Mục tiêu",
            context={"workflow_area": "analysis", "allowed_types": ["goal"]},
        )

    assert exc_info.value.tool_name == "create_artifact"
    assert exc_info.value.args_snapshot == {"artifact_type": "goal", "title": "Mục tiêu"}


@pytest.mark.asyncio
async def test_governed_denies_unknown_tool():
    @governed
    async def unexpected_tool():
        return "không được gọi"

    with pytest.raises(GovernanceDenied) as exc_info:
        await unexpected_tool(context={"workflow_area": "analysis"})

    assert exc_info.value.tool_name == "unexpected_tool"


@pytest.mark.asyncio
async def test_init_workflow_run_denied_outside_orchestrator():
    @governed
    async def init_workflow_run():
        return "không được gọi"

    with pytest.raises(GovernanceDenied) as exc_info:
        await init_workflow_run(context={"workflow_area": "analysis"})

    assert exc_info.value.tool_name == "init_workflow_run"


@pytest.mark.asyncio
async def test_init_workflow_run_requires_approval_for_orchestrator():
    @governed
    async def init_workflow_run(*, project_id: str):
        return {"project_id": project_id}

    with pytest.raises(ApprovalRequired) as exc_info:
        await init_workflow_run(project_id="project-1", context={"workflow_area": "orchestrator"})

    assert exc_info.value.tool_name == "init_workflow_run"
    assert exc_info.value.args_snapshot == {"project_id": "project-1"}


@pytest.mark.asyncio
async def test_create_artifact_denies_type_outside_allowed_context():
    @governed
    async def create_artifact(*, artifact_type: str):
        return {"artifact_type": artifact_type}

    with pytest.raises(GovernanceDenied) as exc_info:
        await create_artifact(artifact_type="risk", context={"workflow_area": "analysis", "allowed_types": ["goal"]})

    assert exc_info.value.tool_name == "create_artifact"


def test_artifact_predecessors_design_types_trace_to_brd():
    assert ARTIFACT_PREDECESSORS["brd"] == []
    assert ARTIFACT_PREDECESSORS["functional_requirement"] == ["brd"]


def test_workflow_state_structure_and_add_messages_reducer():
    state: WorkflowState = {
        "artifact_type": "goal",
        "workflow_area": "analysis",
        "step_key": "intent_vision",
        "messages": [{"role": "user", "content": "Xin chào"}],
        "conversation_summary": "",
        "analysis_result": None,
        "pending_tool_call_ids": [],
        "last_agent_run_id": None,
        "turn_count": 0,
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
        "mode_hint": None,
        "decision_nodes": {},
    }

    assert state["turn_count"] == 0
    assert set(WorkflowState.__annotations__) == {
        "artifact_type",
        "workflow_area",
        "step_key",
        "messages",
        "conversation_summary",
        "analysis_result",
        "pending_tool_call_ids",
        "last_agent_run_id",
        "turn_count",
        "missing_context",
        "user_confirmed",
        "critique_rounds",
        "quality_report",
        "last_critiqued_draft_hash",
        "locale",
        "turn_type",
        "triage_reply",
        "section_coverage",
        "coverage_complete",
        "section_coverage_stall_count",
        "assumptions",
        "risks",
        "open_questions",
        "key_facts",
        "focused_artifact_id",
        "draft_body",
        "method_profile",
        "artifact_chain",
        "readiness",
        "mode_hint",
        "candidate_readiness",
        "verification_status",
        "latest_checked_revision",
        "tool_errors",
        "feedback_summary",
        "session_elicit_count",
        "decision_nodes",
    }
    assert WorkflowState.__annotations__["messages"].__metadata__[0] is add_messages


def test_max_agent_turns_setting_default():
    # High per-request silent-loop backstop (resets each human turn), not a conversation limit.
    assert settings.max_agent_turns == 30
