"""Checkpoint v2 crash-window reconciliation (`decide_reconciliation`, `reconcile_turn_checkpoint`).

The pure `decide_reconciliation` function is unit-tested directly for every crash-window
combination named in the phase brief (Step 2): crash before checkpoint, crash before outcome
projection, and the already-consistent/normal-resume cases. `reconcile_turn_checkpoint` is tested
against real `TurnOutcome`/`AgentCheckpoint` rows to confirm the I/O wrapper gathers the right
inputs for the pure function.
"""

import uuid

import pytest

from app.graphs.analysis.turn_reconciliation import (
    ReconciliationOutcome,
    decide_reconciliation,
    reconcile_turn_checkpoint,
)
from app.models.agent import AgentCheckpoint, AgentTurnEnvelope, TurnOutcome, TurnOutcomeType
from app.models.user import User
from tests.factories import _make_agent_session, _project


@pytest.mark.parametrize(
    "outcome_committed,checkpoint_head_exists,head_has_pending_interrupt,expected",
    [
        # No committed outcome yet: normal, resume as usual — whether or not a checkpoint exists,
        # as long as it still looks mid-flight (a pending interrupt, or no checkpoint at all).
        (False, False, False, ReconciliationOutcome.RESUME),
        (False, True, True, ReconciliationOutcome.RESUME),
        # No committed outcome, but the head checkpoint shows no pending interrupt: the graph may
        # have ended without a terminal projection landing — needs an operator, never auto-resume.
        (False, True, False, ReconciliationOutcome.NEEDS_OPERATOR),
        # Outcome committed and the checkpoint head reflects a terminal (non-pending) state: already
        # consistent, nothing to reconcile.
        (True, True, False, ReconciliationOutcome.ALREADY_CONSISTENT),
        (True, False, False, ReconciliationOutcome.ALREADY_CONSISTENT),
        # Outcome committed but the checkpoint head still shows a pending interrupt: the checkpoint
        # may not reflect the committed terminal transition — needs an operator.
        (True, True, True, ReconciliationOutcome.NEEDS_OPERATOR),
    ],
)
def test_decide_reconciliation_covers_every_crash_window_combination(
    outcome_committed, checkpoint_head_exists, head_has_pending_interrupt, expected
):
    result = decide_reconciliation(
        outcome_committed=outcome_committed,
        checkpoint_head_exists=checkpoint_head_exists,
        head_has_pending_interrupt=head_has_pending_interrupt,
    )
    assert result.outcome == expected
    assert result.detail


async def _seed_turn(db_session, agent_session):
    user = User(email=f"reconcile-{uuid.uuid4()}@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()
    envelope = AgentTurnEnvelope(
        session_id=agent_session.id,
        session_sequence=1,
        original_trigger_id=uuid.uuid4(),
        actor_id=user.id,
        cohort={},
        correlation_id=str(uuid.uuid4()),
    )
    db_session.add(envelope)
    await db_session.commit()
    return envelope


@pytest.mark.asyncio
async def test_reconcile_turn_checkpoint_resumes_with_no_outcome_and_no_checkpoint(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_turn(db_session, session)

    result = await reconcile_turn_checkpoint(db_session, envelope.id)

    assert result.outcome == ReconciliationOutcome.RESUME


@pytest.mark.asyncio
async def test_reconcile_turn_checkpoint_flags_needs_operator_when_outcome_missing_but_head_terminal(
    client, db_session
):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_turn(db_session, session)
    db_session.add(
        AgentCheckpoint(
            session_id=session.id,
            turn_id=envelope.id,
            checkpoint_id="cp-1",
            parent_checkpoint_id=None,
            session_sequence=1,
            ownership_generation=1,
            serde_type="json",
            data=b"{}",
            checkpoint_metadata={},
            new_versions={},
            pending_writes=[],
        )
    )
    await db_session.commit()

    result = await reconcile_turn_checkpoint(db_session, envelope.id)

    assert result.outcome == ReconciliationOutcome.NEEDS_OPERATOR


@pytest.mark.asyncio
async def test_reconcile_turn_checkpoint_already_consistent_when_outcome_and_head_agree(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_turn(db_session, session)
    db_session.add(
        AgentCheckpoint(
            session_id=session.id,
            turn_id=envelope.id,
            checkpoint_id="cp-1",
            parent_checkpoint_id=None,
            session_sequence=1,
            ownership_generation=1,
            serde_type="json",
            data=b"{}",
            checkpoint_metadata={},
            new_versions={},
            pending_writes=[],
        )
    )
    db_session.add(
        TurnOutcome(turn_id=envelope.id, session_id=session.id, outcome_type=TurnOutcomeType.COMPLETED, reason="done")
    )
    await db_session.commit()

    result = await reconcile_turn_checkpoint(db_session, envelope.id)

    assert result.outcome == ReconciliationOutcome.ALREADY_CONSISTENT
