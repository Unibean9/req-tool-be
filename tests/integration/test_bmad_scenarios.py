"""Checkpoint C smoke tests — the 5 addendum §20 scenarios + active_mode/workflow_mode coexistence.

These are behavior smoke tests (not exhaustive units): each confirms the headline behavior of one
scenario through analyze_node or the pure BMAD helpers.
"""

import uuid

import pytest
from langchain_core.messages import AIMessage

from app.graphs.agent_tools import (
    _compute_recommendation,
    _run_readiness_check_impl,
    _write_note_impl,
)
from app.graphs.nodes import analyze_node
from app.graphs.readiness import compute_readiness_score
from tests.integration.test_graph_nodes import _config, _make_agent_run, _make_agent_session, _session_factory, _state

_ALL_SECTIONS = [
    "vision_objectives", "problem_statement", "stakeholder_register", "scope_capabilities",
    "business_rules", "constraints_assumptions", "risks_issues",
]


class _LLM:
    def __init__(self, payload):
        self._payload = payload

    async def generate(self, **kwargs):
        tool_calls = [
            {"id": f"scripted:{i}", "name": item["name"], "args": item.get("args") or {}}
            for i, item in enumerate(self._payload.get("tools") or [])
        ]
        return AIMessage(content=self._payload.get("draft_update", ""), tool_calls=tool_calls), None


async def _project_id(client):
    from tests.helpers import create_org, create_project, make_auth_headers

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return uuid.UUID(project["id"])


async def _analyze(client, db_session, payload, artifact_type="intent", state_mut=None):
    project_id = await _project_id(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    state = _state(artifact_type=artifact_type)
    if state_mut:
        state_mut(state)
    config = _config(str(agent_session.id), str(project_id), _LLM(payload))
    config["configurable"]["session_factory"] = _session_factory()
    return await analyze_node(state, config), project_id, agent_session


@pytest.mark.asyncio
async def test_scenario1_vague_idea(client, db_session):
    """Vague idea: no workflow_mode reported, sparse coverage -> brainstorm/brief on the quick track."""
    result, _, _ = await _analyze(client, db_session, {"tools": [{"name": "ask_user", "args": {"message": "?"}}]})
    assert result["method_profile"]["current_workflow"] in {"brainstorm", "brief"}
    assert result["method_profile"]["planning_track"] == "quick"


@pytest.mark.asyncio
async def test_scenario2_clear_direction_records_assumptions_and_risks(client, db_session):
    """Clear direction: notes feed structured assumptions + risks.

    workflow_mode is no longer LLM-reported (it is inferred from DB coverage), so this scenario no
    longer asserts an echoed brief/prd value; the surviving behavior is the structured note parsing.
    """
    note = await _write_note_impl(
        "ASSUMPTION: users have phones | confidence: high\nRISK: vendor lock-in | likelihood: medium",
        _state(artifact_type="intent"),
        "call_1",
        "explore_note",
    )
    assert note.update["assumptions"]
    assert note.update["risks"]


@pytest.mark.asyncio
async def test_scenario3_build_immediately_flags_blocking_gaps(client, db_session):
    """Wants to build now but PRD is weak -> readiness check returns blocking gaps."""
    project_id = await _project_id(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    run = await _make_agent_run(db_session, agent_session)
    state = _state(artifact_type="intent")
    state["section_coverage"] = {s: "missing" for s in _ALL_SECTIONS}
    state["last_agent_run_id"] = str(run.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    command = await _run_readiness_check_impl("draft", state, config, "call_1")
    assert command.update["readiness"]["requirements_ready"] is False
    assert command.update["readiness"]["blocking_gaps"]


def test_scenario4_prd_near_completion():
    """PRD near complete -> high readiness score and a valid next-workflow recommendation."""
    report = compute_readiness_score({s: "filled" for s in _ALL_SECTIONS})
    assert report["readiness_score"] >= 0.7
    rec = _compute_recommendation({s: "filled" for s in _ALL_SECTIONS}, "standard")
    assert rec["recommended_next_workflow"] in {
        "prd", "readiness_check", "architecture_readiness",
    }


def test_scenario5_small_mvp_quick_track_ceiling():
    """Small MVP on the quick track never escalates past readiness_check."""
    rec = _compute_recommendation({s: "filled" for s in _ALL_SECTIONS}, "quick")
    assert rec["recommended_next_workflow"] != "architecture_readiness"


@pytest.mark.asyncio
async def test_active_mode_and_workflow_mode_coexist_end_to_end(client, db_session):
    """active_mode and method_profile.current_workflow are written to distinct state locations.

    Both are now derived (not LLM-echoed): active_mode from the gated primary tool (ask_user →
    discovery), current_workflow from coverage. They still live in distinct state locations."""
    result, _, _ = await _analyze(
        client, db_session,
        {"tools": [{"name": "ask_user", "args": {"message": "?"}}]},
    )
    assert result["analysis_result"]["active_mode"] == "discovery"
    assert result["method_profile"]["current_workflow"] in {"brainstorm", "brief"}
    assert "current_workflow" not in result["analysis_result"]
