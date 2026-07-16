"""Real-concurrency proof for checkpoint v2 CAS append; only runs when PostgreSQL is available."""

import os
import uuid

import pytest
import pytest_asyncio
from langgraph.checkpoint.base import empty_checkpoint
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.graphs.analysis.turn_outcome_projector import StaleTurnOwnershipError
from app.graphs.checkpointer import AgentCheckpointHistorySaver, StaleCheckpointAppendError
from app.models.agent import (
    AgentCheckpoint,
    AgentSession,
    AgentSessionStatus,
    AgentTurnEnvelope,
    AgentTurnEvent,
    AgentTurnEventType,
    TurnExecutionState,
    TurnExecutionStatus,
)
from app.models.organization import Organization
from app.models.project import Project
from app.models.user import User
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
            await assert_postgres_schema_contract(connection)
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()


def _checkpoint(checkpoint_id: str | None = None) -> dict:
    checkpoint = empty_checkpoint()
    if checkpoint_id is not None:
        checkpoint["id"] = checkpoint_id
    checkpoint["channel_values"] = {"messages": [{"role": "user", "content": "hello"}]}
    checkpoint["channel_versions"] = {"messages": "1"}
    return checkpoint


def _config(
    session_id: uuid.UUID,
    *,
    turn_id: uuid.UUID,
    owner_id: str,
    ownership_generation: int,
    parent_checkpoint_id: str | None,
) -> dict:
    return {
        "configurable": {
            "thread_id": str(session_id),
            "turn_id": str(turn_id),
            "turn_owner_id": owner_id,
            "turn_ownership_generation": ownership_generation,
            "checkpoint_id": parent_checkpoint_id,
        }
    }


async def _seed_v2_session_and_turn(postgres_session_factory, *, owner_id: str = "postgres-owner", generation: int = 1):
    async with postgres_session_factory() as db:
        user = User(email=f"checkpoint-v2-{uuid.uuid4()}@example.com", hashed_password="hash")
        db.add(user)
        await db.flush()
        org = Organization(name="Checkpoint v2 test", slug=f"cpv2-{uuid.uuid4().hex}", owner_id=user.id)
        db.add(org)
        await db.flush()
        project = Project(org_id=org.id, name="Checkpoint v2 test", slug=f"cpv2-{uuid.uuid4().hex}")
        db.add(project)
        await db.flush()
        session = AgentSession(
            project_id=project.id,
            artifact_type="problem",
            workflow_area="analysis",
            status=AgentSessionStatus.ACTIVE,
            created_by_id=user.id,
            checkpoint_version="v2",
        )
        db.add(session)
        await db.flush()
        envelope = AgentTurnEnvelope(
            session_id=session.id,
            session_sequence=1,
            original_trigger_id=uuid.uuid4(),
            actor_id=user.id,
            cohort={},
            correlation_id=str(uuid.uuid4()),
        )
        db.add(envelope)
        await db.flush()
        state = TurnExecutionState(
            turn_id=envelope.id,
            status=TurnExecutionStatus.RUNNING,
            owner_id=owner_id,
            ownership_generation=generation,
        )
        db.add(state)
        await db.commit()
        return session.id, envelope.id


@pytest.mark.asyncio
async def test_postgres_concurrent_checkpoint_append_with_same_parent_is_fenced_to_one_writer(
    postgres_session_factory,
):
    session_id, turn_id = await _seed_v2_session_and_turn(postgres_session_factory)
    saver = AgentCheckpointHistorySaver(session_id=str(session_id), session_factory=postgres_session_factory)

    # First append: no parent, establishes the head.
    first_checkpoint = _checkpoint("checkpoint-1")
    await saver.aput(
        _config(session_id, turn_id=turn_id, owner_id="postgres-owner", ownership_generation=1, parent_checkpoint_id=None),
        first_checkpoint,
        {},
        {},
    )

    # Two racing writers both think "checkpoint-1" is still the head and both try to append a
    # *different* next checkpoint on top of it concurrently — real Postgres connections, real
    # `SELECT ... FOR UPDATE` on the session row. Exactly one must win; the loser must see a stale
    # parent (whichever one commits first) and raise, never silently fork history.
    import asyncio

    async def append(checkpoint_id: str):
        try:
            await saver.aput(
                _config(
                    session_id,
                    turn_id=turn_id,
                    owner_id="postgres-owner",
                    ownership_generation=1,
                    parent_checkpoint_id="checkpoint-1",
                ),
                _checkpoint(checkpoint_id),
                {},
                {},
            )
            return "ok"
        except StaleCheckpointAppendError:
            return "stale"

    results = await asyncio.gather(append("checkpoint-2a"), append("checkpoint-2b"))
    assert sorted(results) == ["ok", "stale"]

    async with postgres_session_factory() as db:
        rows = (
            await db.execute(select(AgentCheckpoint).where(AgentCheckpoint.session_id == session_id))
        ).scalars().all()
        # The seed row plus exactly one winner — the loser never persisted a row.
        assert len(rows) == 2
        children_of_first = [row for row in rows if row.parent_checkpoint_id == "checkpoint-1"]
        assert len(children_of_first) == 1

        events = (
            await db.execute(select(AgentTurnEvent).where(AgentTurnEvent.session_id == session_id))
        ).scalars().all()
        checkpoint_events = [e for e in events if e.event_type == AgentTurnEventType.CHECKPOINT_APPENDED]
        # One event per successfully committed checkpoint append (seed + winner), none for the loser.
        assert len(checkpoint_events) == 2
        assert len({e.session_sequence for e in checkpoint_events}) == 2  # monotonic cursor, no dup


@pytest.mark.asyncio
async def test_postgres_checkpoint_append_after_reclaim_fences_stale_owner(postgres_session_factory):
    session_id, turn_id = await _seed_v2_session_and_turn(postgres_session_factory, owner_id="owner-a", generation=1)
    saver = AgentCheckpointHistorySaver(session_id=str(session_id), session_factory=postgres_session_factory)

    await saver.aput(
        _config(session_id, turn_id=turn_id, owner_id="owner-a", ownership_generation=1, parent_checkpoint_id=None),
        _checkpoint("checkpoint-1"),
        {},
        {},
    )

    # Simulate a lease reclaim bumping the fence generation to a new owner.
    async with postgres_session_factory() as db:
        state = (
            await db.execute(select(TurnExecutionState).where(TurnExecutionState.turn_id == turn_id))
        ).scalar_one()
        state.owner_id = "owner-b"
        state.ownership_generation = 2
        await db.commit()

    # The stale owner (generation 1) tries to append on top of the real head — must be fenced
    # before the CAS parent check even runs.
    with pytest.raises(StaleTurnOwnershipError):
        await saver.aput(
            _config(
                session_id, turn_id=turn_id, owner_id="owner-a", ownership_generation=1, parent_checkpoint_id="checkpoint-1"
            ),
            _checkpoint("checkpoint-2-stale"),
            {},
            {},
        )

    async with postgres_session_factory() as db:
        rows = (
            await db.execute(select(AgentCheckpoint).where(AgentCheckpoint.session_id == session_id))
        ).scalars().all()
        assert len(rows) == 1  # the stale owner's write never landed
