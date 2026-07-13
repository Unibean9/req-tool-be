"""write_draft's propose-time readiness gate.

`evaluate_candidate_readiness` is computed on every fresh proposal attempt; a candidate that
would fail the approve-time gate (`can_persist=False`) is blocked here instead, before any
AgentToolCall row is written and before the PROPOSE_ARTIFACTS interrupt fires. After two
consecutive blocked attempts for the same focused artifact within one turn, the recoverable
error instructs the model to call ask_user instead of retrying write_draft again.
"""

from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.graphs.agent_tools import _write_draft_impl
from app.models.agent import AgentToolCall
from app.models.artifact import ArtifactType
from tests.conftest import TestSessionFactory
from tests.factories import (
    _config,
    _focused_items,
    _make_agent_run,
    _make_agent_session,
    _project,
    _session_factory,
    _state,
)

INCOMPLETE_BODY = "## Vision\nA vision statement."  # missing ## Objectives / ## Success Metrics
COMPLETE_BODY = "\n\n".join(
    [
        "## Vision\nA concrete vision statement.",
        "## Objectives\n- Ship the thing.",
        "## Success Metrics\n- Adoption reaches 80%.",
    ]
)


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
    return state, config, run, agent_session


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_missing_headings_blocked_before_interrupt_and_persist(mock_interrupt, client, db_session):
    state, config, run, _session = await _seed(client, db_session)

    command = await _write_draft_impl("Vision", INCOMPLETE_BODY, state, config, "call_1")

    errors = command.update.get("tool_errors") or []
    assert any(e["code"] == "candidate_readiness_not_ready" for e in errors)
    assert "ask_user" not in errors[0].get("recovery", "")
    mock_interrupt.assert_not_called()
    assert command.update["readiness_reject_streak"] == 1
    async with TestSessionFactory() as db:
        rows = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalars().all()
        assert rows == []


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_ready_draft_proceeds_to_normal_propose_interrupt(mock_interrupt, client, db_session):
    state, config, run, _session = await _seed(client, db_session)

    command = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalar_one()
        assert row.input_snapshot["candidate_readiness"]["can_persist"] is True


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_second_consecutive_rejection_forces_ask_user_instead_of_propose(mock_interrupt, client, db_session):
    state, config, run, _session = await _seed(client, db_session)

    first = await _write_draft_impl("Vision", INCOMPLETE_BODY, state, config, "call_1")
    assert first.update["readiness_reject_streak"] == 1

    # Same turn, same artifact: state carries the streak forward like a normal in-turn tool retry.
    state["readiness_reject_streak"] = first.update["readiness_reject_streak"]
    second = await _write_draft_impl("Vision", INCOMPLETE_BODY, state, config, "call_2")

    errors = second.update.get("tool_errors") or []
    assert second.update["readiness_reject_streak"] == 2
    assert "ask_user" in errors[0]["recovery"]
    mock_interrupt.assert_not_called()
    async with TestSessionFactory() as db:
        rows = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalars().all()
        assert rows == []


@pytest.mark.asyncio
async def test_resume_command_resets_readiness_reject_streak(client, db_session):
    from app.services.agent_service import AgentService

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    svc = AgentService(db=db_session, graph=None, session_factory=_session_factory())

    command = svc._resume_command(agent_session, {"content": "hi"})

    assert command.update["readiness_reject_streak"] == 0


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_existing_tool_call_reuse_never_bypasses_gate(mock_interrupt, client, db_session):
    """A repair retry that reuses the (run_id, tool_name) row is only reachable after the gate
    already passed for that run_id — a blocked attempt is never persisted, so it always gets a
    fresh run_id on its next attempt and lands back in the gate-checked branch."""
    state, config, run, _session = await _seed(client, db_session)

    first = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")
    mock_interrupt.assert_called_once()
    assert not (first.update.get("tool_errors") or [])

    # Re-invoking with the same run_id (simulating a resume re-executing this ToolNode body) hits
    # the existing_tool_call reuse branch — it must not re-run the gate or reset/increment the streak.
    state["readiness_reject_streak"] = 0
    second = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")

    assert not (second.update.get("tool_errors") or [])
    assert "readiness_reject_streak" not in second.update
    async with TestSessionFactory() as db:
        rows = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalars().all()
        assert len(rows) == 1
