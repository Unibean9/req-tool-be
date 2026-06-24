"""Tests for workflow_mode / planning_track assignment in analyze_node (addendum §5, §11, §12)."""

import uuid

import pytest

from app.graphs.nodes import (
    TOOL_SELECTION_SCHEMA,
    _infer_workflow_mode,
    analyze_node,
)
from tests.integration.test_graph_nodes import _config, _make_agent_session, _session_factory, _state


class _LLM:
    def __init__(self, payload):
        self._payload = payload

    async def generate(self, **kwargs):
        return dict(self._payload), None


async def _run(client, db_session, payload, artifact_type="intent"):
    from tests.helpers import create_org, create_project, make_auth_headers

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    state = _state(artifact_type=artifact_type)
    config = _config(str(agent_session.id), str(project_id), _LLM(payload))
    config["configurable"]["session_factory"] = _session_factory()
    return await analyze_node(state, config)


@pytest.mark.asyncio
async def test_workflow_mode_assigned_after_analyze_node(client, db_session):
    result = await _run(client, db_session, {"tools": [{"name": "ask_user", "args": {"message": "?"}}],
                                             "workflow_mode": "brief", "planning_track": "quick"})
    assert result["method_profile"]["current_workflow"] == "brief"
    assert result["method_profile"]["planning_track"] == "quick"


@pytest.mark.asyncio
async def test_active_mode_and_workflow_mode_coexist(client, db_session):
    result = await _run(client, db_session, {"tools": [{"name": "ask_user", "args": {"message": "?"}}],
                                             "active_mode": "structuring", "workflow_mode": "prd"})
    assert result["analysis_result"]["active_mode"] == "structuring"
    assert result["method_profile"]["current_workflow"] == "prd"


@pytest.mark.asyncio
async def test_workflow_mode_invalid_value_falls_back_to_brainstorm(client, db_session):
    result = await _run(client, db_session, {"tools": [{"name": "ask_user", "args": {"message": "?"}}],
                                             "workflow_mode": "invalid_value"})
    assert result["method_profile"]["current_workflow"] == "brainstorm"


@pytest.mark.asyncio
async def test_workflow_mode_product_brief_alias_normalizes_to_brief(client, db_session):
    result = await _run(client, db_session, {"tools": [{"name": "ask_user", "args": {"message": "?"}}],
                                             "workflow_mode": "product_brief"})
    assert result["method_profile"]["current_workflow"] == "brief"


def test_tool_selection_schema_contains_workflow_mode_and_planning_track():
    props = TOOL_SELECTION_SCHEMA["properties"]
    assert set(props["workflow_mode"]["enum"]) == {
        "brainstorm", "brief", "prd", "readiness_check",
        "architecture_readiness",
    }
    assert set(props["planning_track"]["enum"]) == {"quick", "standard", "enterprise"}
    # active_mode (Phase 6) still present alongside.
    for value in ("discovery", "structuring", "critique", "revision", "finalization"):
        assert value in props["active_mode"]["enum"]
    # D1: tools is the required field, not tool.
    assert "tools" in props
    assert TOOL_SELECTION_SCHEMA["required"] == ["tools"]


def test_infer_workflow_mode_low_coverage_returns_brainstorm():
    state = _state(artifact_type="intent")
    state["section_coverage"] = {"vision_objectives": "missing", "problem_statement": "missing"}
    assert _infer_workflow_mode(state) == "brainstorm"


def test_infer_workflow_mode_partial_coverage_by_artifact_type():
    partial = {"vision_objectives": "partial", "problem_statement": "partial"}
    brief_state = _state(artifact_type="brainstorming")
    brief_state["section_coverage"] = partial
    assert _infer_workflow_mode(brief_state) == "brief"

    prd_state = _state(artifact_type="product_brief")
    prd_state["section_coverage"] = partial
    assert _infer_workflow_mode(prd_state) == "prd"


def test_planning_track_quick_is_default():
    state = _state(artifact_type="intent")
    assert state["method_profile"]["planning_track"] == "quick"
