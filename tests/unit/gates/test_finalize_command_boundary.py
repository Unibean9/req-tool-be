"""Command boundary: finalize's fenced, idempotent ledger effect.

Mirrors test_write_draft_command_boundary.py's structure (same cohort/fence/duplicate/race
mechanics, reused unchanged — see draft_command_service.py). finalize's effect has no
AgentToolCall/artifact row (its mutation is a session/interrupt transition), so success here is
judged purely by the DraftCommandLedger row-count/action_type, not by any created row.
"""

import hashlib
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.graphs.agent_tools import _finalize_impl
from app.graphs.decision_graph import create_node, render_view
from app.models.agent import (
    AgentTurnEnvelope,
    DraftCommandLedger,
    TurnExecutionState,
    TurnExecutionStatus,
)
from app.models.user import User
from app.schemas.artifact_synthesis import ArtifactReadinessState
from tests.conftest import TestSessionFactory
from tests.factories import _config, _make_agent_run, _make_agent_session, _project, _session_factory


def _hash(body: str) -> str:
    return hashlib.md5(body.encode()).hexdigest()[:8]


def _passing_state(statement: str = "A draft") -> dict:
    state = {
        "messages": [],
        "user_confirmed": True,
        "artifact_type": "brd",
        "decision_nodes": {
            "N1": create_node(
                kind="objective",
                statement=statement,
                origin={"source": "test"},
                status="confirmed",
            )
        },
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }
    state["last_critiqued_draft_hash"] = _hash(render_view(state["decision_nodes"], state["artifact_type"]))
    return state


async def _seed_turn(db_session, agent_session, *, command_handlers_enabled: bool, owner_id: str, generation: int):
    user = User(email=f"finalize-cmd-turn-{uuid.uuid4()}@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()
    envelope = AgentTurnEnvelope(
        session_id=agent_session.id,
        session_sequence=1,
        original_trigger_id=uuid.uuid4(),
        actor_id=user.id,
        cohort={"command_handlers_enabled": command_handlers_enabled},
        correlation_id=str(uuid.uuid4()),
    )
    db_session.add(envelope)
    await db_session.flush()
    db_session.add(
        TurnExecutionState(
            turn_id=envelope.id,
            status=TurnExecutionStatus.RUNNING,
            owner_id=owner_id,
            ownership_generation=generation,
        )
    )
    await db_session.commit()
    return envelope


async def _seed(
    client, db_session, *, command_handlers_enabled: bool = True, owner_id: str = "owner-a", generation: int = 1
):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    await _make_agent_run(db_session, agent_session)
    envelope = await _seed_turn(
        db_session,
        agent_session,
        command_handlers_enabled=command_handlers_enabled,
        owner_id=owner_id,
        generation=generation,
    )

    # project_id left empty so finalize skips the predecessor-check/executive-summary branch
    # entirely — this test's only concern is the command boundary, not that separate behavior.
    config = _config(str(agent_session.id), "")
    config["configurable"]["session_factory"] = _session_factory()
    config["configurable"]["turn_id"] = str(envelope.id)
    config["configurable"]["turn_owner_id"] = owner_id
    config["configurable"]["turn_ownership_generation"] = generation
    return _passing_state(), config, envelope


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_command_handler_commits_ledger_row(mock_interrupt, client, db_session):
    state, config, envelope = await _seed(client, db_session)

    await _finalize_impl("Done.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        ledger_rows = (
            await db.execute(
                select(DraftCommandLedger).where(
                    DraftCommandLedger.turn_id == envelope.id, DraftCommandLedger.action_type == "finalize"
                )
            )
        ).scalars().all()
        assert len(ledger_rows) == 1
        assert ledger_rows[0].tool_call_id is None
        assert ledger_rows[0].artifact_id is None


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_duplicate_logical_command_id_does_not_add_second_ledger_row(mock_interrupt, client, db_session):
    """Retrying finalize (e.g. resume re-executes the node) with the same summary must not add a
    second ledger row for the same turn/action_type/canonical intent."""
    state, config, envelope = await _seed(client, db_session)

    await _finalize_impl("Done.", state, config, "call_1")
    await _finalize_impl("Done.", state, config, "call_2")

    mock_interrupt.assert_called_once()

    async with TestSessionFactory() as db:
        ledger_rows = (
            await db.execute(
                select(DraftCommandLedger).where(
                    DraftCommandLedger.turn_id == envelope.id, DraftCommandLedger.action_type == "finalize"
                )
            )
        ).scalars().all()
        assert len(ledger_rows) == 1


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_stale_owner_rejected_before_any_ledger_write(mock_interrupt, client, db_session):
    state, config, envelope = await _seed(client, db_session, owner_id="owner-a", generation=1)

    async with TestSessionFactory() as db:
        turn_state = (
            await db.execute(select(TurnExecutionState).where(TurnExecutionState.turn_id == envelope.id))
        ).scalar_one()
        turn_state.owner_id = "owner-b"
        turn_state.ownership_generation = 2
        await db.commit()

    command = await _finalize_impl("Done.", state, config, "call_1")

    errors = command.update.get("tool_errors") or []
    assert len(errors) == 1
    assert errors[0]["code"] == "turn_fence_stale"
    mock_interrupt.assert_not_called()
    async with TestSessionFactory() as db:
        ledger_rows = (
            await db.execute(select(DraftCommandLedger).where(DraftCommandLedger.turn_id == envelope.id))
        ).scalars().all()
        assert ledger_rows == []


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_cohort_flag_off_never_writes_ledger_row(mock_interrupt, client, db_session):
    state, config, envelope = await _seed(client, db_session, command_handlers_enabled=False)

    await _finalize_impl("Done.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        ledger_rows = (
            await db.execute(select(DraftCommandLedger).where(DraftCommandLedger.turn_id == envelope.id))
        ).scalars().all()
        assert ledger_rows == []


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_no_turn_context_uses_fully_legacy_path(mock_interrupt, client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    state = _passing_state()
    config = _config(str(agent_session.id), "")
    config["configurable"]["session_factory"] = _session_factory()

    async with TestSessionFactory() as db:
        ledger_count_before = len((await db.execute(select(DraftCommandLedger))).scalars().all())

    command = await _finalize_impl("Done.", state, config, "call_1")

    assert "tool_errors" not in command.update
    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        ledger_count_after = len((await db.execute(select(DraftCommandLedger))).scalars().all())
        # No turn context at all — this must never write a ledger row, regardless of anything
        # left over in the table by other tests in this module.
        assert ledger_count_after == ledger_count_before
