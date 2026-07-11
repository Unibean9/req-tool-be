"""write_draft's early, non-blocking predecessor-staleness warning.

Approval time (`_guard_lifecycle_predecessors`) remains the sole blocking authority for a stale
`based_on` predecessor. This covers the advisory heads-up surfaced at write_draft time instead: a
`stale_predecessor:<type>:<reason>` string riding both `deterministic_warnings` (read by the human
reviewer via the FE snapshot) and the resume `ToolMessage` content (read by the model at its next
turn) — while the draft itself is still written and the interrupt still fires normally.
"""

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.graphs.agent_tools import _write_draft_impl
from app.models.agent import AgentToolCall
from app.models.artifact import ArtifactStatus, ArtifactType, ArtifactVersion, ChangeSource, VersionStatus
from tests.conftest import TestSessionFactory
from tests.factories import (
    _accept_predecessor,
    _config,
    _focused_items,
    _make_agent_run,
    _make_agent_session,
    _project,
    _session_factory,
    _state,
)

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
    return state, config, run, project_id


async def _add_version(db_session, artifact, body: str) -> ArtifactVersion:
    """Accept a new version on an already-`_accept_predecessor`-seeded artifact, so its
    `current_version_id` moves on while a caller can still reference the earlier one."""
    version = ArtifactVersion(
        artifact_id=artifact.id,
        version_number=2,
        title=artifact.title,
        body=body,
        status=VersionStatus.ACCEPTED,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    db_session.add(version)
    await db_session.flush()
    artifact.current_version_id = version.id
    await db_session.commit()
    return version


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_missing_predecessor_warns_on_both_channels(mock_interrupt, client, db_session):
    state, config, run, _project_id = await _seed(client, db_session)
    missing_id = str(uuid.uuid4())
    state["turn_context_artifacts"] = [
        {"id": missing_id, "type": "problem_statement", "current_version_id": str(uuid.uuid4())}
    ]

    command = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    expected = "stale_predecessor:problem_statement:missing_predecessor"
    assert expected in command.update["messages"][0].content
    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalar_one()
        assert expected in row.input_snapshot["synthesis_metadata"]["deterministic_warnings"]


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_retired_predecessor_warns_on_both_channels(mock_interrupt, client, db_session):
    state, config, run, project_id = await _seed(client, db_session)
    predecessor = await _accept_predecessor(db_session, project_id, "problem_statement")
    predecessor.status = ArtifactStatus.ARCHIVED
    await db_session.commit()
    state["turn_context_artifacts"] = [
        {
            "id": str(predecessor.id),
            "type": "problem_statement",
            "current_version_id": str(predecessor.current_version_id),
        }
    ]

    command = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    expected = "stale_predecessor:problem_statement:retired_predecessor"
    assert expected in command.update["messages"][0].content
    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalar_one()
        assert expected in row.input_snapshot["synthesis_metadata"]["deterministic_warnings"]


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_predecessor_version_changed_warns_on_both_channels(mock_interrupt, client, db_session):
    state, config, run, project_id = await _seed(client, db_session)
    predecessor = await _accept_predecessor(db_session, project_id, "problem_statement")
    recorded_version_id = predecessor.current_version_id
    await _add_version(db_session, predecessor, "updated problem statement")
    state["turn_context_artifacts"] = [
        {"id": str(predecessor.id), "type": "problem_statement", "current_version_id": str(recorded_version_id)}
    ]

    command = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    expected = "stale_predecessor:problem_statement:predecessor_version_changed"
    assert expected in command.update["messages"][0].content
    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalar_one()
        assert expected in row.input_snapshot["synthesis_metadata"]["deterministic_warnings"]


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_current_predecessor_produces_no_warning(mock_interrupt, client, db_session):
    state, config, run, project_id = await _seed(client, db_session)
    predecessor = await _accept_predecessor(db_session, project_id, "problem_statement")
    state["turn_context_artifacts"] = [
        {
            "id": str(predecessor.id),
            "type": "problem_statement",
            "current_version_id": str(predecessor.current_version_id),
        }
    ]

    command = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    assert command.update["messages"][0].content == "Vision"
    async with TestSessionFactory() as db:
        row = (await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))).scalar_one()
        warnings = row.input_snapshot["synthesis_metadata"]["deterministic_warnings"]
        assert not any(str(w).startswith("stale_predecessor:") for w in warnings)
