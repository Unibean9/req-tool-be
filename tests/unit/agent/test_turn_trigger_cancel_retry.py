"""Cancel/retry typed-trigger admission, and message-vs-WAIT_* transition behavior.

Covers the remaining gap from the approval/user-message admission tests: `admit_cancel` and
`admit_retry` follow the exact same ownership-fence contract (row lock, idempotency key,
authorization recheck) as `admit_user_message`/`admit_approval`, and `admit_user_message` already
resolves a message arriving while the session is `WAIT_INPUT`/`WAIT_APPROVAL` — this only adds the
missing test coverage for that transition table.
"""

import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.models.agent import (
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentTurnEnvelope,
    AgentTurnTrigger,
    AgentTurnTriggerType,
    TurnExecutionState,
    TurnExecutionStatus,
    TurnOutcome,
    TurnOutcomeType,
)
from app.models.user import User
from app.services.agent_turn_service import AgentTurnService
from tests.factories import _project


async def _user(db_session) -> User:
    user = User(email=f"turn-cancel-retry-{uuid.uuid4()}@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()
    return user


async def _session(db_session, project_id, user, *, status: AgentSessionStatus, interrupt_type=None) -> AgentSession:
    session = AgentSession(
        project_id=project_id,
        artifact_type="problem",
        workflow_area="analysis",
        status=status,
        interrupt_type=interrupt_type,
        created_by_id=user.id,
    )
    db_session.add(session)
    await db_session.commit()
    return session


@pytest.mark.asyncio
async def test_admit_cancel_is_idempotent_and_fenced(client, db_session):
    project_id = await _project(client)
    user = await _user(db_session)
    session = await _session(db_session, project_id, user, status=AgentSessionStatus.ACTIVE)
    service = AgentTurnService(db_session)

    first = await service.admit_cancel(
        project_id=project_id,
        session_id=session.id,
        user_id=user.id,
        idempotency_key="cancel-1",
    )
    duplicate = await service.admit_cancel(
        project_id=project_id,
        session_id=session.id,
        user_id=user.id,
        idempotency_key="cancel-1",
    )

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.turn_id == first.turn_id

    envelope = await db_session.get(AgentTurnEnvelope, first.turn_id)
    assert envelope is not None
    state = (
        await db_session.execute(select(TurnExecutionState).where(TurnExecutionState.turn_id == first.turn_id))
    ).scalar_one()
    assert state.status == TurnExecutionStatus.TERMINAL

    refreshed = await db_session.get(AgentSession, session.id)
    assert refreshed.status == AgentSessionStatus.EXPIRED


@pytest.mark.asyncio
async def test_admit_cancel_rejects_forged_actor(client, db_session):
    project_id = await _project(client)
    owner = await _user(db_session)
    outsider = await _user(db_session)
    session = await _session(db_session, project_id, owner, status=AgentSessionStatus.ACTIVE)
    service = AgentTurnService(db_session)

    with pytest.raises(HTTPException) as exc:
        await service.admit_cancel(
            project_id=project_id, session_id=session.id, user_id=outsider.id, idempotency_key="x"
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_admit_cancel_rejects_an_already_ended_session(client, db_session):
    project_id = await _project(client)
    user = await _user(db_session)
    session = await _session(db_session, project_id, user, status=AgentSessionStatus.COMPLETED)
    service = AgentTurnService(db_session)

    with pytest.raises(HTTPException) as exc:
        await service.admit_cancel(
            project_id=project_id, session_id=session.id, user_id=user.id, idempotency_key="x"
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_admit_cancel_records_turn_outcome_when_cohort_enabled(client, db_session, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "agent_turn_outcomes_enabled", True)
    project_id = await _project(client)
    user = await _user(db_session)
    session = await _session(db_session, project_id, user, status=AgentSessionStatus.WAITING_FOR_HUMAN)
    service = AgentTurnService(db_session)

    admitted = await service.admit_cancel(
        project_id=project_id, session_id=session.id, user_id=user.id, idempotency_key="cancel-audit"
    )

    outcome = (
        await db_session.execute(select(TurnOutcome).where(TurnOutcome.turn_id == admitted.turn_id))
    ).scalar_one()
    assert outcome.outcome_type == TurnOutcomeType.CANCELLED


@pytest.mark.asyncio
async def test_admit_retry_is_idempotent_and_fenced(client, db_session):
    project_id = await _project(client)
    user = await _user(db_session)
    session = await _session(db_session, project_id, user, status=AgentSessionStatus.TURN_FAILED)
    service = AgentTurnService(db_session)

    first = await service.admit_retry(
        project_id=project_id, session_id=session.id, user_id=user.id, idempotency_key="retry-1"
    )
    duplicate = await service.admit_retry(
        project_id=project_id, session_id=session.id, user_id=user.id, idempotency_key="retry-1"
    )

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.turn_id == first.turn_id

    envelopes = (
        await db_session.execute(select(AgentTurnEnvelope).where(AgentTurnEnvelope.session_id == session.id))
    ).scalars().all()
    assert len(envelopes) == 1

    refreshed = await db_session.get(AgentSession, session.id)
    assert refreshed.status == AgentSessionStatus.ACTIVE
    assert refreshed.active_turn_id == first.turn_id
    # Reactivation only reuses the persisted checkpoint (still {} here, untouched) — retry never
    # re-invokes a model inside admission to produce a new one.
    assert refreshed.graph_checkpoint == {}


@pytest.mark.asyncio
async def test_admit_retry_rejects_when_session_is_not_in_a_retryable_state(client, db_session):
    project_id = await _project(client)
    user = await _user(db_session)
    session = await _session(db_session, project_id, user, status=AgentSessionStatus.ACTIVE)
    service = AgentTurnService(db_session)

    with pytest.raises(HTTPException) as exc:
        await service.admit_retry(
            project_id=project_id, session_id=session.id, user_id=user.id, idempotency_key="x"
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_admit_retry_rejects_forged_actor(client, db_session):
    project_id = await _project(client)
    owner = await _user(db_session)
    outsider = await _user(db_session)
    session = await _session(db_session, project_id, owner, status=AgentSessionStatus.TURN_FAILED)
    service = AgentTurnService(db_session)

    with pytest.raises(HTTPException) as exc:
        await service.admit_retry(
            project_id=project_id, session_id=session.id, user_id=outsider.id, idempotency_key="x"
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_message_arriving_during_wait_input_resumes_immediately(client, db_session):
    """ASK_HUMAN is a direct question: the reply resumes the turn inline, not queued."""
    project_id = await _project(client)
    user = await _user(db_session)
    session = await _session(
        db_session,
        project_id,
        user,
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
    )
    service = AgentTurnService(db_session)

    admitted = await service.admit_user_message(
        project_id=project_id,
        session_id=session.id,
        user_id=user.id,
        content="Câu trả lời",
        idempotency_key="answer-1",
    )

    assert admitted.queued is False
    refreshed = await db_session.get(AgentSession, session.id)
    assert refreshed.status == AgentSessionStatus.ACTIVE
    assert refreshed.interrupt_type is None
    trigger = (
        await db_session.execute(select(AgentTurnTrigger).where(AgentTurnTrigger.message_id == admitted.message.id))
    ).scalar_one()
    assert trigger.trigger_type == AgentTurnTriggerType.USER_MESSAGE


@pytest.mark.asyncio
async def test_message_arriving_during_wait_approval_is_queued_not_dropped(client, db_session):
    """PROPOSE_ARTIFACTS is an approval gate: an incoming message must not jump the fence and
    resume the graph ahead of the pending approval decision — it coalesces into a queued message."""
    project_id = await _project(client)
    user = await _user(db_session)
    session = await _session(
        db_session,
        project_id,
        user,
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
    )
    service = AgentTurnService(db_session)

    admitted = await service.admit_user_message(
        project_id=project_id,
        session_id=session.id,
        user_id=user.id,
        content="Trong khi chờ duyệt",
        idempotency_key="queued-while-waiting-approval",
    )

    assert admitted.queued is True
    assert admitted.message.payload == {"queued": True}
    refreshed = await db_session.get(AgentSession, session.id)
    # Status/interrupt untouched — the pending approval remains the only fence owner.
    assert refreshed.status == AgentSessionStatus.WAITING_FOR_HUMAN
    assert refreshed.interrupt_type == AgentSessionInterruptType.PROPOSE_ARTIFACTS
