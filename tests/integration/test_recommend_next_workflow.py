"""Tests for recommend_next_workflow (addendum §6, §8, §10.1, §19.3)."""

import pytest
from sqlalchemy import select

from app.graphs.agent_tools import (
    _compute_recommendation,
    _recommend_next_workflow_impl,
    get_available_tools,
)
from app.models.agent import AgentToolCall
from tests.conftest import TestSessionFactory
from tests.integration.test_graph_nodes import _config, _make_agent_run, _make_agent_session, _session_factory, _state

_ALL_SECTIONS = [
    "vision_objectives", "problem_statement", "stakeholder_register", "scope_capabilities",
    "business_rules", "constraints_assumptions", "risks_issues",
]


def _coverage(**overrides):
    cov = {s: "missing" for s in _ALL_SECTIONS}
    cov.update(overrides)
    return cov


def test_recommend_returns_brief_when_coverage_sparse():
    result = _compute_recommendation(_coverage(vision_objectives="partial"), "quick")
    assert result["recommended_next_workflow"] == "brief"


def test_recommend_returns_prd_when_brief_complete():
    result = _compute_recommendation(
        _coverage(vision_objectives="filled", problem_statement="filled",
                  stakeholder_register="filled", scope_capabilities="filled"),
        "standard",
    )
    assert result["recommended_next_workflow"] == "prd"


def test_recommend_returns_readiness_check_when_prd_near_full():
    result = _compute_recommendation({s: "filled" for s in _ALL_SECTIONS}, "standard")
    assert result["recommended_next_workflow"] == "readiness_check"


def test_quick_track_does_not_recommend_architecture_readiness():
    result = _compute_recommendation({s: "filled" for s in _ALL_SECTIONS}, "quick")
    assert result["recommended_next_workflow"] != "architecture_readiness"


def test_blocking_gaps_populated_when_sections_empty():
    result = _compute_recommendation(
        _coverage(vision_objectives="filled", problem_statement="filled",
                  stakeholder_register="filled", scope_capabilities="filled"),
        "standard",
    )
    assert "business_rules" in result["blocking_gaps"]


def test_confidence_low_when_many_sections_missing():
    result = _compute_recommendation(_coverage(vision_objectives="filled"), "quick")
    assert result["confidence"] == "low"


def test_recommend_in_available_tools_with_signal():
    state = _state(artifact_type="intent")
    state["section_coverage"] = _coverage(vision_objectives="partial", problem_statement="partial")
    names = {t.name for t in get_available_tools(state)}
    assert "recommend_next_workflow" in names


def test_recommend_not_available_when_no_signal():
    state = _state(artifact_type="intent")
    names = {t.name for t in get_available_tools(state)}
    assert "recommend_next_workflow" not in names


@pytest.mark.asyncio
async def test_result_persisted_to_audit_log(client, db_session):
    project_id = await _project_id(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="intent")
    state["section_coverage"] = _coverage(vision_objectives="filled", problem_statement="filled")
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await _recommend_next_workflow_impl("intent", "quick", state, config, "call_1")

    async with TestSessionFactory() as db:
        rows = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
    audit = [r for r in rows if r.tool_name == "recommend_next_workflow"]
    assert len(audit) == 1
    assert audit[0].input_snapshot["recommended_next_workflow"]


@pytest.mark.asyncio
async def test_recommended_next_workflow_written_to_state(client, db_session):
    project_id = await _project_id(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="intent")
    state["section_coverage"] = {s: "filled" for s in _ALL_SECTIONS}
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    command = await _recommend_next_workflow_impl("intent", "quick", state, config, "call_1")

    assert command.update["method_profile"]["recommended_next_workflow"] == "readiness_check"


@pytest.mark.asyncio
async def test_recommend_reflects_current_coverage_same_turn(client, db_session):
    """The tool derives from the live section_coverage, not a possibly-stale artifact_chain."""
    project_id = await _project_id(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="intent")
    state["artifact_chain"] = {"brainstorming": "missing", "product_brief": "missing", "prd": "missing"}
    # Same-turn coverage update: brief sections now filled.
    state["section_coverage"] = _coverage(
        vision_objectives="filled", problem_statement="filled",
        stakeholder_register="filled", scope_capabilities="filled",
    )
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    command = await _recommend_next_workflow_impl("intent", "standard", state, config, "call_1")

    assert command.update["method_profile"]["recommended_next_workflow"] == "prd"


async def _project_id(client):
    import uuid

    from tests.helpers import create_org, create_project, make_auth_headers

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return uuid.UUID(project["id"])
