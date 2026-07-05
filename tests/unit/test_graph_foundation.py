import pytest
from langgraph.graph import add_messages

from app.config import settings
from app.graphs.policy import ARTIFACT_PREDECESSORS, ApprovalRequired, GovernanceDenied, governed
from app.graphs.state import (
    WorkflowState,
    build_initial_workflow_state,
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
            title="Goal",
            context={"workflow_area": "analysis", "allowed_types": ["goal"]},
        )

    assert exc_info.value.tool_name == "create_artifact"
    assert exc_info.value.args_snapshot == {"artifact_type": "goal", "title": "Goal"}


@pytest.mark.asyncio
async def test_governed_denies_unknown_tool():
    @governed
    async def unexpected_tool():
        return "must not be called"

    with pytest.raises(GovernanceDenied) as exc_info:
        await unexpected_tool(context={"workflow_area": "analysis"})

    assert exc_info.value.tool_name == "unexpected_tool"


@pytest.mark.asyncio
async def test_init_workflow_run_denied_outside_orchestrator():
    @governed
    async def init_workflow_run():
        return "must not be called"

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
    assert ARTIFACT_PREDECESSORS["functional_requirement"] == ["use_case", "business_rules"]


def test_workflow_state_structure_and_add_messages_reducer():
    state: WorkflowState = build_initial_workflow_state(
        artifact_type="goal",
        workflow_area="analysis",
        step_key="intent_vision",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert state["turn_count"] == 0
    assert set(state) == set(WorkflowState.__annotations__)
    assert WorkflowState.__annotations__["messages"].__metadata__[0] is add_messages


def test_max_agent_turns_setting_default():
    # High per-request silent-loop backstop (resets each human turn), not a conversation limit.
    assert settings.max_agent_turns == 30
