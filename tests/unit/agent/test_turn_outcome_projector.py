"""`project_terminal_outcome`: the sole function allowed to write a
terminal `AgentSession.status` value, and the only place a `TurnOutcome` audit row is added.

Cohort-snapshot-at-admission pattern: the `turn_outcomes_enabled` flag is read from the
turn's persisted `AgentTurnEnvelope.cohort`, never live from settings — a turn admitted before an
operator flips the flag keeps its own snapshot's behavior for its entire lifetime.
"""

import uuid

import pytest
from sqlalchemy import select

from app.graphs.analysis.turn_outcome_projector import project_non_terminal_outcome, project_terminal_outcome
from app.models.agent import (
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentTurnEnvelope,
    TurnOutcome,
    TurnOutcomeType,
)
from app.models.user import User
from tests.factories import _make_agent_session, _project


async def _seed_envelope(db_session, agent_session, *, turn_outcomes_enabled: bool) -> AgentTurnEnvelope:
    user = User(email=f"outcome-turn-{uuid.uuid4()}@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()
    envelope = AgentTurnEnvelope(
        session_id=agent_session.id,
        session_sequence=1,
        original_trigger_id=uuid.uuid4(),
        actor_id=user.id,
        cohort={"turn_outcomes_enabled": turn_outcomes_enabled},
        correlation_id=str(uuid.uuid4()),
    )
    db_session.add(envelope)
    await db_session.commit()
    return envelope


@pytest.mark.asyncio
async def test_rejects_a_non_terminal_outcome_type():
    session_row = AgentSession(artifact_type="goal", workflow_area="analysis", graph_checkpoint={})
    with pytest.raises(ValueError):
        await project_terminal_outcome(None, session_row, TurnOutcomeType.CONTINUE, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome_type,expected_status",
    [
        (TurnOutcomeType.COMPLETED, AgentSessionStatus.COMPLETED),
        (TurnOutcomeType.TERMINAL_FAILURE, AgentSessionStatus.FAILED),
        (TurnOutcomeType.RECOVERABLE_FAILURE, AgentSessionStatus.TURN_FAILED),
        (TurnOutcomeType.CANCELLED, AgentSessionStatus.EXPIRED),
    ],
)
async def test_maps_each_terminal_outcome_to_its_status_with_no_turn_context(
    client, db_session, outcome_type, expected_status
):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    await project_terminal_outcome(db_session, agent_session, outcome_type, "some reason")
    await db_session.commit()

    assert agent_session.status == expected_status
    # No turn_id — must never write a TurnOutcome row regardless of any cohort flag.
    outcomes = (
        await db_session.execute(select(TurnOutcome).where(TurnOutcome.session_id == agent_session.id))
    ).scalars().all()
    assert outcomes == []


@pytest.mark.asyncio
async def test_writes_turn_outcome_row_when_cohort_flag_enabled(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_envelope(db_session, agent_session, turn_outcomes_enabled=True)

    await project_terminal_outcome(
        db_session, agent_session, TurnOutcomeType.COMPLETED, "graph_ended", turn_id=envelope.id
    )
    await db_session.commit()

    assert agent_session.status == AgentSessionStatus.COMPLETED
    outcomes = (
        await db_session.execute(select(TurnOutcome).where(TurnOutcome.turn_id == envelope.id))
    ).scalars().all()
    assert len(outcomes) == 1
    assert outcomes[0].outcome_type == TurnOutcomeType.COMPLETED
    assert outcomes[0].reason == "graph_ended"
    assert outcomes[0].session_id == agent_session.id


@pytest.mark.asyncio
async def test_duplicate_projection_writes_exactly_one_outcome(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_envelope(db_session, agent_session, turn_outcomes_enabled=True)

    await project_terminal_outcome(
        db_session, agent_session, TurnOutcomeType.COMPLETED, "all_artifact_proposals_approved", turn_id=envelope.id
    )
    await project_terminal_outcome(
        db_session, agent_session, TurnOutcomeType.COMPLETED, "all_artifact_proposals_approved", turn_id=envelope.id
    )
    await db_session.commit()

    outcomes = (
        await db_session.execute(select(TurnOutcome).where(TurnOutcome.turn_id == envelope.id))
    ).scalars().all()
    assert agent_session.status == AgentSessionStatus.COMPLETED
    assert len(outcomes) == 1
    assert outcomes[0].reason == "all_artifact_proposals_approved"


@pytest.mark.asyncio
async def test_no_turn_outcome_row_when_cohort_flag_disabled(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_envelope(db_session, agent_session, turn_outcomes_enabled=False)

    await project_terminal_outcome(
        db_session, agent_session, TurnOutcomeType.TERMINAL_FAILURE, "turn_limit", turn_id=envelope.id
    )
    await db_session.commit()

    assert agent_session.status == AgentSessionStatus.FAILED
    outcomes = (
        await db_session.execute(select(TurnOutcome).where(TurnOutcome.turn_id == envelope.id))
    ).scalars().all()
    assert outcomes == []


@pytest.mark.asyncio
async def test_missing_envelope_for_turn_id_falls_back_to_status_only(client, db_session):
    """Defensive: a turn_id that doesn't resolve to a persisted envelope must never raise — it just
    behaves as if the cohort flag were off (status write only, no TurnOutcome row)."""
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    await project_terminal_outcome(
        db_session, agent_session, TurnOutcomeType.RECOVERABLE_FAILURE, "graph_exception", turn_id=uuid.uuid4()
    )
    await db_session.commit()

    assert agent_session.status == AgentSessionStatus.TURN_FAILED


@pytest.mark.asyncio
async def test_rejects_a_terminal_outcome_type_for_the_non_terminal_producer():
    session_row = AgentSession(artifact_type="goal", workflow_area="analysis", graph_checkpoint={})
    with pytest.raises(ValueError):
        await project_non_terminal_outcome(None, session_row, TurnOutcomeType.COMPLETED)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome_type,expected_status,expected_interrupt",
    [
        (TurnOutcomeType.CONTINUE, AgentSessionStatus.ACTIVE, None),
        (TurnOutcomeType.WAIT_INPUT, AgentSessionStatus.WAITING_FOR_HUMAN, AgentSessionInterruptType.ASK_HUMAN),
        (
            TurnOutcomeType.WAIT_APPROVAL,
            AgentSessionStatus.WAITING_FOR_HUMAN,
            AgentSessionInterruptType.PROPOSE_ARTIFACTS,
        ),
        (TurnOutcomeType.DIRECT_RESPONSE, AgentSessionStatus.ACTIVE, AgentSessionInterruptType.STREAM_RESPONSE),
    ],
)
async def test_maps_each_non_terminal_outcome_to_its_status_and_interrupt_with_no_turn_context(
    client, db_session, outcome_type, expected_status, expected_interrupt
):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    await project_non_terminal_outcome(db_session, agent_session, outcome_type)
    await db_session.commit()

    assert agent_session.status == expected_status
    assert agent_session.interrupt_type == expected_interrupt
    # No turn_id — must never write a TurnOutcome row regardless of any cohort flag.
    outcomes = (
        await db_session.execute(select(TurnOutcome).where(TurnOutcome.session_id == agent_session.id))
    ).scalars().all()
    assert outcomes == []


@pytest.mark.asyncio
async def test_non_terminal_outcome_writes_turn_outcome_row_when_cohort_flag_enabled(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_envelope(db_session, agent_session, turn_outcomes_enabled=True)

    await project_non_terminal_outcome(
        db_session, agent_session, TurnOutcomeType.WAIT_INPUT, "ask_human", turn_id=envelope.id
    )
    await db_session.commit()

    assert agent_session.status == AgentSessionStatus.WAITING_FOR_HUMAN
    assert agent_session.interrupt_type == AgentSessionInterruptType.ASK_HUMAN
    outcomes = (
        await db_session.execute(select(TurnOutcome).where(TurnOutcome.turn_id == envelope.id))
    ).scalars().all()
    assert len(outcomes) == 1
    assert outcomes[0].outcome_type == TurnOutcomeType.WAIT_INPUT


@pytest.mark.asyncio
async def test_non_terminal_duplicate_projection_writes_exactly_one_outcome(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_envelope(db_session, agent_session, turn_outcomes_enabled=True)

    await project_non_terminal_outcome(
        db_session, agent_session, TurnOutcomeType.DIRECT_RESPONSE, turn_id=envelope.id
    )
    await project_non_terminal_outcome(
        db_session, agent_session, TurnOutcomeType.DIRECT_RESPONSE, turn_id=envelope.id
    )
    await db_session.commit()

    outcomes = (
        await db_session.execute(select(TurnOutcome).where(TurnOutcome.turn_id == envelope.id))
    ).scalars().all()
    assert agent_session.status == AgentSessionStatus.ACTIVE
    assert agent_session.interrupt_type == AgentSessionInterruptType.STREAM_RESPONSE
    assert len(outcomes) == 1
