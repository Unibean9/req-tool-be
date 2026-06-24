"""Tests for the readiness rubric and run_readiness_check tool (addendum §10.2, §15)."""

import pytest
from sqlalchemy import select

from app.graphs.agent_tools import _run_readiness_check_impl, get_available_tools
from app.graphs.readiness import compute_readiness_score
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


def test_readiness_returns_not_ready_when_critical_sections_empty():
    report = compute_readiness_score(_coverage())
    assert report["ready"] is False
    assert report["readiness_score"] < 0.3
    assert report["blocking_gaps"]


def test_readiness_returns_ready_when_all_sections_sufficient():
    report = compute_readiness_score({s: "filled" for s in _ALL_SECTIONS})
    assert report["ready"] is True
    assert report["readiness_score"] >= 0.7


def test_scope_stability_fails_when_out_of_scope_missing():
    cov = {s: "filled" for s in _ALL_SECTIONS}
    cov["scope_capabilities"] = {"in_scope": "filled"}  # granular, no out_of_scope
    report = compute_readiness_score(cov)
    assert "scope_stability" in report["blocking_gaps"]


def test_readiness_score_weighted_average_of_10_dimensions():
    # Fill the first four sections (5 dimensions pass), leave the rest missing (5 fail).
    cov = _coverage(
        vision_objectives="filled", problem_statement="filled",
        stakeholder_register="filled", scope_capabilities="filled",
    )
    report = compute_readiness_score(cov)
    assert 0.4 <= report["readiness_score"] <= 0.6


def test_recommended_next_step_populated_in_output():
    cov = {s: "filled" for s in _ALL_SECTIONS}
    cov["constraints_assumptions"] = "missing"
    report = compute_readiness_score(cov)
    assert report["recommended_next_step"] in ("architecture_readiness", "complete_constraints")


@pytest.mark.asyncio
async def test_run_readiness_check_persisted_to_audit_log(client, db_session):
    project_id = await _project_id(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="intent")
    state["section_coverage"] = {s: "filled" for s in _ALL_SECTIONS}
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    await _run_readiness_check_impl("draft", state, config, "call_1")

    async with TestSessionFactory() as db:
        rows = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
    assert any(r.tool_name == "run_readiness_check" for r in rows)


@pytest.mark.asyncio
async def test_readiness_state_updated_after_check(client, db_session):
    project_id = await _project_id(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="intent")
    state["section_coverage"] = {s: "filled" for s in _ALL_SECTIONS}
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    command = await _run_readiness_check_impl("draft", state, config, "call_1")

    assert command.update["readiness"]["requirements_ready"] is True


def test_run_readiness_check_in_available_tools_with_draft():
    state = _state(artifact_type="intent")
    state["working_draft"] = "một bản nháp"
    state["critique_rounds"] = 1  # readiness now requires at least one critique round (Phase 3)
    names = {t.name for t in get_available_tools(state)}
    assert "run_readiness_check" in names


async def _project_id(client):
    import uuid

    from tests.helpers import create_org, create_project, make_auth_headers

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return uuid.UUID(project["id"])
