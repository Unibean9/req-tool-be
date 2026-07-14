import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.models.agent import AgentSession, AgentSessionStatus, AgentTurnEnvelope, AgentTurnTrigger, TurnExecutionState
from app.models.user import User
from app.services.agent_turn_service import AgentTurnService
from tests.factories import _project


async def _user(db_session) -> User:
    user = User(email=f"turn-{uuid.uuid4()}@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()
    return user


@pytest.mark.asyncio
async def test_admission_is_idempotent_and_keeps_envelope_immutable(client, db_session):
    project_id = await _project(client)
    user = await _user(db_session)
    session = AgentSession(
        project_id=project_id,
        artifact_type="problem",
        workflow_area="analysis",
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        created_by_id=user.id,
    )
    db_session.add(session)
    await db_session.commit()

    service = AgentTurnService(db_session)
    first = await service.admit_user_message(
        project_id=project_id,
        session_id=session.id,
        user_id=user.id,
        content="Mô tả vấn đề",
        idempotency_key="client-turn-1",
    )
    duplicate = await service.admit_user_message(
        project_id=project_id,
        session_id=session.id,
        user_id=user.id,
        content="Nội dung phải bị bỏ qua",
        idempotency_key="client-turn-1",
    )

    envelope = await db_session.get(AgentTurnEnvelope, first.turn_id)
    assert duplicate.duplicate is True
    assert duplicate.turn_id == first.turn_id
    assert duplicate.message.id == first.message.id
    assert envelope is not None
    assert envelope.session_sequence == 1
    assert envelope.cohort["turn_admission"] == "v1"
    assert (await db_session.execute(select(TurnExecutionState).where(TurnExecutionState.turn_id == first.turn_id))).scalar_one()


@pytest.mark.asyncio
async def test_stale_owner_cannot_release_new_generation(client, db_session):
    project_id = await _project(client)
    user = await _user(db_session)
    session = AgentSession(
        project_id=project_id,
        artifact_type="problem",
        workflow_area="analysis",
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        created_by_id=user.id,
    )
    db_session.add(session)
    await db_session.commit()
    service = AgentTurnService(db_session)
    admitted = await service.admit_user_message(
        project_id=project_id,
        session_id=session.id,
        user_id=user.id,
        content="Mô tả vấn đề",
        idempotency_key=str(uuid.uuid4()),
    )

    generation = await service.claim_inline(turn_id=admitted.turn_id, owner_id="inline-a")
    assert generation == 1
    assert await service.release_inline(turn_id=admitted.turn_id, owner_id="inline-b", generation=generation) is False
    state = (await db_session.execute(select(TurnExecutionState).where(TurnExecutionState.turn_id == admitted.turn_id))).scalar_one()
    assert state.owner_id == "inline-a"
    assert state.ownership_generation == generation


@pytest.mark.asyncio
async def test_queued_admission_preserves_turn_reference_for_drain(client, db_session):
    project_id = await _project(client)
    user = await _user(db_session)
    session = AgentSession(
        project_id=project_id,
        artifact_type="problem",
        workflow_area="analysis",
        status=AgentSessionStatus.ACTIVE,
        created_by_id=user.id,
    )
    db_session.add(session)
    await db_session.commit()

    admitted = await AgentTurnService(db_session).admit_user_message(
        project_id=project_id,
        session_id=session.id,
        user_id=user.id,
        content="Đợi lượt trước hoàn tất",
        idempotency_key="queued-turn",
        mode_hint="critique",
    )

    assert admitted.message.payload == {"queued": True, "mode_hint": "critique"}
    assert (await db_session.get(AgentSession, session.id)).status == AgentSessionStatus.ACTIVE
    trigger = (
        await db_session.execute(select(AgentTurnTrigger).where(AgentTurnTrigger.message_id == admitted.message.id))
    ).scalar_one()
    assert trigger.turn_id == admitted.turn_id


@pytest.mark.asyncio
async def test_admission_fails_closed_when_actor_missing_or_forged(client, db_session):
    project_id = await _project(client)
    owner = await _user(db_session)
    outsider = await _user(db_session)
    session = AgentSession(
        project_id=project_id,
        artifact_type="problem",
        workflow_area="analysis",
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        created_by_id=owner.id,
    )
    db_session.add(session)
    await db_session.commit()
    service = AgentTurnService(db_session)

    with pytest.raises(HTTPException) as missing:
        await service.admit_user_message(
            project_id=project_id, session_id=session.id, user_id=None, content="x", idempotency_key="missing"
        )
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as forged:
        await service.admit_user_message(
            project_id=project_id, session_id=session.id, user_id=outsider.id, content="x", idempotency_key="forged"
        )
    assert forged.value.status_code == 404
