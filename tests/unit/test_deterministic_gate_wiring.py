"""Deterministic proposal gate wired into the write_draft path.

`validate_proposal` violations block a proposal before the PROPOSE_ARTIFACTS interrupt and before
any AgentToolCall row is written; warnings ride the synthesis metadata. The `enforce_deterministic_gate`
config flag restores legacy behavior exactly.
"""

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.graphs import agent_tools
from app.graphs.agent_tools import _write_draft_impl
from app.models.agent import AgentToolCall
from app.models.artifact import ArtifactType
from tests.conftest import TestSessionFactory
from tests.integration.test_graph_nodes import _config, _make_agent_run, _session_factory, _state
from tests.unit.test_tool_parity import _focused_items, _make_agent_session, _project


async def _seed(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(db_session, project_id, ArtifactType.VISION_OBJECTIVES)
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="vision_objectives")
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()
    return state, config, run


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_violation_blocks_before_interrupt_and_persist(mock_interrupt, client, db_session):
    state, config, run = await _seed(client, db_session)

    # Empty title is a `Missing required field` violation in validate_proposal.
    command = await _write_draft_impl("", "## Vision\nA concrete vision statement.", state, config, "call_1")

    errors = command.update.get("tool_errors") or []
    assert any(e["code"] == "deterministic_gate_failed" for e in errors)
    assert "title" in errors[0]["message"]
    mock_interrupt.assert_not_called()
    async with TestSessionFactory() as db:
        rows = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalars().all()
        assert rows == []


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_clean_draft_passes_and_warnings_ride_metadata(mock_interrupt, client, db_session):
    state, config, run = await _seed(client, db_session)

    # "fast" is a weasel word -> warning (non-blocking); the proposal proceeds to the interrupt.
    command = await _write_draft_impl("Vision", "## Vision\nA fast onboarding flow.", state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalar_one()
        warnings = row.input_snapshot["synthesis_metadata"]["deterministic_warnings"]
        assert any("fast" in w for w in warnings)


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_flag_off_restores_pre_phase_behavior(mock_interrupt, client, db_session, monkeypatch):
    monkeypatch.setattr(agent_tools.settings, "enforce_deterministic_gate", False)
    state, config, run = await _seed(client, db_session)

    # Same violation as the blocking test, but the flag bypasses the gate -> proposal proceeds.
    command = await _write_draft_impl("", "## Vision\nA concrete vision statement.", state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        rows = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalars().all()
        assert len(rows) == 1
