"""Proof of concurrency behavior; only runs when a real PostgreSQL instance is configured."""

import asyncio
import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.agent import AgentSession, AgentSessionStatus, AgentTurnEnvelope, TurnExecutionState
from app.models.organization import Organization
from app.models.project import Project
from app.models.user import User
from app.services.agent_turn_service import AgentTurnService
from tests.integration.conftest import assert_postgres_schema_contract

POSTGRES_URL = os.getenv("AGENT_TURN_POSTGRES_URL")
pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def postgres_session_factory():
    if not POSTGRES_URL:
        pytest.skip("AGENT_TURN_POSTGRES_URL is not configured")
    engine = create_async_engine(POSTGRES_URL, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            await assert_postgres_schema_contract(
                connection, table_name="projects", column_name="executive_summary"
            )
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_postgres_admission_and_fence_are_serialized(postgres_session_factory):
    async with postgres_session_factory() as db:
        user = User(email=f"turn-{uuid.uuid4()}@example.com", hashed_password="hash")
        db.add(user)
        await db.flush()
        org = Organization(name="Turn test", slug=f"turn-{uuid.uuid4().hex}", owner_id=user.id)
        db.add(org)
        await db.flush()
        project = Project(org_id=org.id, name="Turn test", slug=f"turn-{uuid.uuid4().hex}")
        db.add(project)
        await db.flush()
        session = AgentSession(
            project_id=project.id,
            artifact_type="problem",
            workflow_area="analysis",
            status=AgentSessionStatus.WAITING_FOR_HUMAN,
            created_by_id=user.id,
        )
        db.add(session)
        await db.commit()
        project_id, session_id, user_id = project.id, session.id, user.id

    async def admit_same_key():
        async with postgres_session_factory() as db:
            return await AgentTurnService(db).admit_user_message(
                project_id=project_id,
                session_id=session_id,
                user_id=user_id,
                content="Một input duy nhất",
                idempotency_key="same-request",
            )

    first, second = await asyncio.gather(admit_same_key(), admit_same_key())
    assert first.turn_id == second.turn_id

    async with postgres_session_factory() as db:
        envelopes = (
            await db.execute(select(AgentTurnEnvelope).where(AgentTurnEnvelope.session_id == session_id))
        ).scalars().all()
        assert len(envelopes) == 1
        assert envelopes[0].session_sequence == 1

    async def claim(owner_id: str):
        async with postgres_session_factory() as db:
            return await AgentTurnService(db).claim_inline(turn_id=first.turn_id, owner_id=owner_id)

    claim_a, claim_b = await asyncio.gather(claim("postgres-a"), claim("postgres-b"))
    assert sorted([claim_a is None, claim_b is None]) == [False, True]

    async with postgres_session_factory() as db:
        state = (
            await db.execute(select(TurnExecutionState).where(TurnExecutionState.turn_id == first.turn_id))
        ).scalar_one()
        assert state.ownership_generation == 1
        assert state.owner_id in {"postgres-a", "postgres-b"}


@pytest.mark.asyncio
async def test_postgres_distinct_keys_queue_behind_one_session_owner(postgres_session_factory):
    async with postgres_session_factory() as db:
        owner = User(email=f"turn-owner-{uuid.uuid4()}@example.com", hashed_password="hash")
        outsider = User(email=f"turn-outsider-{uuid.uuid4()}@example.com", hashed_password="hash")
        db.add_all([owner, outsider])
        await db.flush()
        org = Organization(name="Turn queue test", slug=f"turn-queue-{uuid.uuid4().hex}", owner_id=owner.id)
        db.add(org)
        await db.flush()
        project = Project(org_id=org.id, name="Turn queue test", slug=f"turn-queue-{uuid.uuid4().hex}")
        db.add(project)
        await db.flush()
        session = AgentSession(
            project_id=project.id,
            artifact_type="problem",
            workflow_area="analysis",
            status=AgentSessionStatus.WAITING_FOR_HUMAN,
            created_by_id=owner.id,
        )
        db.add(session)
        await db.commit()
        project_id, session_id, owner_id, outsider_id = project.id, session.id, owner.id, outsider.id

    async def admit(key: str):
        async with postgres_session_factory() as db:
            return await AgentTurnService(db).admit_user_message(
                project_id=project_id,
                session_id=session_id,
                user_id=owner_id,
                content=key,
                idempotency_key=key,
            )

    first, second = await asyncio.gather(admit("distinct-a"), admit("distinct-b"))
    assert {first.queued, second.queued} == {False, True}
    inline = first if not first.queued else second
    queued = second if first is inline else first

    async def claim(turn_id, owner_name):
        async with postgres_session_factory() as db:
            return await AgentTurnService(db).claim_inline(turn_id=turn_id, owner_id=owner_name)

    inline_claim, queued_claim = await asyncio.gather(
        claim(inline.turn_id, "postgres-inline"), claim(queued.turn_id, "postgres-queued")
    )
    assert inline_claim == 1
    assert queued_claim is None

    async with postgres_session_factory() as db:
        persisted_session = await db.get(AgentSession, session_id)
        assert persisted_session is not None
        assert persisted_session.active_turn_id == inline.turn_id

        with pytest.raises(Exception) as missing_actor:
            await AgentTurnService(db).admit_user_message(
                project_id=project_id, session_id=session_id, user_id=None, content="x", idempotency_key="missing-actor"
            )
        assert getattr(missing_actor.value, "status_code", None) == 401

        with pytest.raises(Exception) as forged_actor:
            await AgentTurnService(db).admit_user_message(
                project_id=project_id,
                session_id=session_id,
                user_id=outsider_id,
                content="x",
                idempotency_key="forged-actor",
            )
        assert getattr(forged_actor.value, "status_code", None) == 404

        with pytest.raises(Exception) as wrong_project:
            await AgentTurnService(db).admit_user_message(
                project_id=uuid.uuid4(),
                session_id=session_id,
                user_id=owner_id,
                content="x",
                idempotency_key="wrong-project",
            )
        assert getattr(wrong_project.value, "status_code", None) == 404


@pytest.mark.asyncio
async def test_postgres_concurrent_cancel_with_same_key_is_fenced_to_one_turn(postgres_session_factory):
    async with postgres_session_factory() as db:
        user = User(email=f"turn-cancel-{uuid.uuid4()}@example.com", hashed_password="hash")
        db.add(user)
        await db.flush()
        org = Organization(name="Turn cancel test", slug=f"turn-cancel-{uuid.uuid4().hex}", owner_id=user.id)
        db.add(org)
        await db.flush()
        project = Project(org_id=org.id, name="Turn cancel test", slug=f"turn-cancel-{uuid.uuid4().hex}")
        db.add(project)
        await db.flush()
        session = AgentSession(
            project_id=project.id,
            artifact_type="problem",
            workflow_area="analysis",
            status=AgentSessionStatus.ACTIVE,
            created_by_id=user.id,
        )
        db.add(session)
        await db.commit()
        project_id, session_id, user_id = project.id, session.id, user.id

    async def cancel_same_key():
        async with postgres_session_factory() as db:
            return await AgentTurnService(db).admit_cancel(
                project_id=project_id,
                session_id=session_id,
                user_id=user_id,
                idempotency_key="same-cancel",
            )

    first, second = await asyncio.gather(cancel_same_key(), cancel_same_key())
    assert first.turn_id == second.turn_id

    async with postgres_session_factory() as db:
        envelopes = (
            await db.execute(select(AgentTurnEnvelope).where(AgentTurnEnvelope.session_id == session_id))
        ).scalars().all()
        assert len(envelopes) == 1
        persisted_session = await db.get(AgentSession, session_id)
        assert persisted_session.status == AgentSessionStatus.EXPIRED


@pytest.mark.asyncio
async def test_postgres_concurrent_retry_with_same_key_is_fenced_to_one_turn(postgres_session_factory):
    async with postgres_session_factory() as db:
        user = User(email=f"turn-retry-{uuid.uuid4()}@example.com", hashed_password="hash")
        db.add(user)
        await db.flush()
        org = Organization(name="Turn retry test", slug=f"turn-retry-{uuid.uuid4().hex}", owner_id=user.id)
        db.add(org)
        await db.flush()
        project = Project(org_id=org.id, name="Turn retry test", slug=f"turn-retry-{uuid.uuid4().hex}")
        db.add(project)
        await db.flush()
        session = AgentSession(
            project_id=project.id,
            artifact_type="problem",
            workflow_area="analysis",
            status=AgentSessionStatus.TURN_FAILED,
            created_by_id=user.id,
        )
        db.add(session)
        await db.commit()
        project_id, session_id, user_id = project.id, session.id, user.id

    async def retry_same_key():
        async with postgres_session_factory() as db:
            return await AgentTurnService(db).admit_retry(
                project_id=project_id,
                session_id=session_id,
                user_id=user_id,
                idempotency_key="same-retry",
            )

    first, second = await asyncio.gather(retry_same_key(), retry_same_key())
    assert first.turn_id == second.turn_id

    async with postgres_session_factory() as db:
        envelopes = (
            await db.execute(select(AgentTurnEnvelope).where(AgentTurnEnvelope.session_id == session_id))
        ).scalars().all()
        assert len(envelopes) == 1
        persisted_session = await db.get(AgentSession, session_id)
        assert persisted_session.status == AgentSessionStatus.ACTIVE
        assert persisted_session.active_turn_id == first.turn_id
