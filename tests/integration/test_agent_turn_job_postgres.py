"""Postgres-backed concurrency proof for `AgentTurnJobService`'s claim/reclaim/fence.

Only runs when `AGENT_TURN_POSTGRES_URL` is configured (mirrors `test_agent_turn_postgres.py`).
SQLite (the unit-test backend) does not enforce real `SELECT ... FOR UPDATE`/`SKIP LOCKED`
semantics, so the claim-once and post-reclaim fence invariants can only be proven against real
concurrent Postgres transactions.

This is a new file rather than an extension of `test_agent_turn_postgres.py`: that file's fixture
and helpers are scoped to `AgentTurnService`'s session/trigger-admission concurrency, a different
service with a different seed shape (no `AgentTurnJob` row at all). Keeping this service's
job-claim proof in its own file avoids coupling unrelated fixtures, following the same precedent
`test_draft_command_postgres.py` set for `DraftCommandService`.
"""

import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.graphs.analysis.turn_outcome_projector import StaleTurnOwnershipError, project_terminal_outcome
from app.models.agent import (
    AgentSession,
    AgentSessionStatus,
    AgentTurnEnvelope,
    AgentTurnJob,
    AgentTurnJobStatus,
    TurnExecutionState,
    TurnOutcomeType,
)
from app.models.organization import Organization
from app.models.project import Project
from app.models.user import User
from app.services.agent_turn_job_service import MAX_ATTEMPTS_BEFORE_DEAD_LETTER, AgentTurnJobService

POSTGRES_URL = os.getenv("AGENT_TURN_POSTGRES_URL")
EXPECTED_ALEMBIC_REVISION = "d2e5c8ecc7e0"
pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def postgres_session_factory():
    if not POSTGRES_URL:
        pytest.skip("AGENT_TURN_POSTGRES_URL is not configured")
    engine = create_async_engine(POSTGRES_URL, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            assert connection.dialect.name == "postgresql"
            # No Base.metadata.create_all() here: schema must come from `alembic upgrade head`,
            # same as CI, so a mis-stamped local database fails loudly instead of being masked.
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
            assert revision == EXPECTED_ALEMBIC_REVISION
            jobs_table = await connection.scalar(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = current_schema() AND table_name = 'agent_turn_jobs'"
                )
            )
            assert jobs_table == 1
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def _reset_agent_turn_jobs(postgres_session_factory):
    """`claim()`/`reclaim_expired()` scan the whole `agent_turn_jobs` table by design, and every
    commit here is a real commit against a persistent database (unlike the SQLite unit fixture,
    which rolls back per test) — so a row left over from an earlier run of this file would
    otherwise still be visible to a later test's unscoped scan. Clear it before each test."""
    async with postgres_session_factory() as db:
        await db.execute(delete(AgentTurnJob))
        await db.commit()
    yield


async def _seed_session(session_factory) -> uuid.UUID:
    async with session_factory() as db:
        user = User(email=f"turn-job-{uuid.uuid4()}@example.com", hashed_password="hash")
        db.add(user)
        await db.flush()
        org = Organization(name="Turn job test", slug=f"turn-job-{uuid.uuid4().hex}", owner_id=user.id)
        db.add(org)
        await db.flush()
        project = Project(org_id=org.id, name="Turn job test", slug=f"turn-job-{uuid.uuid4().hex}")
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
        await db.flush()
        await db.commit()
        return session.id


async def _seed_turn(session_factory, *, session_id: uuid.UUID, session_sequence: int) -> uuid.UUID:
    async with session_factory() as db:
        session = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
        envelope = AgentTurnEnvelope(
            session_id=session.id,
            session_sequence=session_sequence,
            original_trigger_id=uuid.uuid4(),
            actor_id=session.created_by_id,
            cohort={"turn_admission": "v1"},
            correlation_id=str(uuid.uuid4()),
        )
        db.add(envelope)
        await db.flush()
        db.add(TurnExecutionState(turn_id=envelope.id))
        await db.commit()
        turn_id = envelope.id

    async with session_factory() as db:
        job = await AgentTurnJobService(db).enqueue(turn_id=turn_id, expected_transition_version=0, cohort={})
        return job.turn_id


async def _seed_job(session_factory) -> uuid.UUID:
    session_id = await _seed_session(session_factory)
    return await _seed_turn(session_factory, session_id=session_id, session_sequence=1)


@pytest.mark.asyncio
async def test_postgres_two_workers_claim_the_same_job_only_one_wins(postgres_session_factory):
    turn_id = await _seed_job(postgres_session_factory)

    async def claim(worker_id: str):
        async with postgres_session_factory() as db:
            return await AgentTurnJobService(db).claim(worker_id=worker_id)

    claim_a, claim_b = await asyncio.gather(claim("postgres-a"), claim("postgres-b"))
    winners = [claimed for claimed in (claim_a, claim_b) if claimed is not None]
    assert len(winners) == 1
    assert sorted([claim_a is None, claim_b is None]) == [False, True]

    async with postgres_session_factory() as db:
        job = (await db.execute(select(AgentTurnJob).where(AgentTurnJob.turn_id == turn_id))).scalar_one()
        assert job.status == AgentTurnJobStatus.CLAIMED
        assert job.lease_owner in {"postgres-a", "postgres-b"}
        assert job.lease_generation == 1
        state = (
            await db.execute(select(TurnExecutionState).where(TurnExecutionState.turn_id == turn_id))
        ).scalar_one()
        assert state.ownership_generation == 1


@pytest.mark.asyncio
async def test_postgres_lease_expiry_reclaim_lets_another_worker_claim_and_fences_stale_owner(
    postgres_session_factory,
):
    await _seed_job(postgres_session_factory)

    async with postgres_session_factory() as db:
        first_claim = await AgentTurnJobService(db).claim(worker_id="postgres-a")
        assert first_claim is not None
        stale_job_id = first_claim.id
        stale_generation = first_claim.lease_generation

    # Force the lease to look expired, as if postgres-a's heartbeat had stopped.
    async with postgres_session_factory() as db:
        job = (await db.execute(select(AgentTurnJob).where(AgentTurnJob.id == stale_job_id))).scalar_one()
        job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await db.commit()

    async with postgres_session_factory() as db:
        reclaimed = await AgentTurnJobService(db).reclaim_expired()
        assert len(reclaimed) == 1
        assert reclaimed[0].id == stale_job_id
        assert reclaimed[0].status == AgentTurnJobStatus.QUEUED
        assert reclaimed[0].attempt == 1

    async def claim(worker_id: str):
        async with postgres_session_factory() as db:
            return await AgentTurnJobService(db).claim(worker_id=worker_id)

    reclaim_a, reclaim_b = await asyncio.gather(claim("postgres-c"), claim("postgres-d"))
    winners = [claimed for claimed in (reclaim_a, reclaim_b) if claimed is not None]
    assert len(winners) == 1
    new_owner_generation = winners[0].lease_generation
    assert new_owner_generation != stale_generation

    async with postgres_session_factory() as db:
        service = AgentTurnJobService(db)
        assert (
            await service.renew_heartbeat(
                job_id=stale_job_id, worker_id="postgres-a", lease_generation=stale_generation
            )
            is False
        )
        assert (
            await service.complete(
                job_id=stale_job_id,
                worker_id="postgres-a",
                lease_generation=stale_generation,
                status=AgentTurnJobStatus.SUCCEEDED,
            )
            is False
        )

    async with postgres_session_factory() as db:
        job = (await db.execute(select(AgentTurnJob).where(AgentTurnJob.id == stale_job_id))).scalar_one()
        assert job.status == AgentTurnJobStatus.CLAIMED
        assert job.lease_generation == new_owner_generation


@pytest.mark.asyncio
async def test_postgres_concurrent_claim_respects_session_head_of_line_ordering(postgres_session_factory):
    """Two queued jobs in the same session, sequence 1 and 2: a claim() race must never let the
    sequence-2 job be claimed while sequence-1's job is still non-terminal (QUEUED here), even
    when both jobs' rows are simultaneously lockable by SKIP LOCKED."""
    session_id = await _seed_session(postgres_session_factory)
    turn_1 = await _seed_turn(postgres_session_factory, session_id=session_id, session_sequence=1)
    turn_2 = await _seed_turn(postgres_session_factory, session_id=session_id, session_sequence=2)

    async def claim(worker_id: str):
        async with postgres_session_factory() as db:
            return await AgentTurnJobService(db).claim(worker_id=worker_id)

    claim_a, claim_b = await asyncio.gather(claim("postgres-a"), claim("postgres-b"))
    winners = [claimed for claimed in (claim_a, claim_b) if claimed is not None]
    assert len(winners) == 1
    assert winners[0].turn_id == turn_1

    async with postgres_session_factory() as db:
        job_2 = (await db.execute(select(AgentTurnJob).where(AgentTurnJob.turn_id == turn_2))).scalar_one()
        assert job_2.status == AgentTurnJobStatus.QUEUED
        assert job_2.lease_owner is None


@pytest.mark.asyncio
async def test_postgres_claim_self_heal_dead_letters_after_max_attempts(postgres_session_factory):
    """A worker that repeatedly crashes right after claiming (lease always expires before it
    renews) must eventually dead-letter via claim()'s own self-heal path, not just via the
    separately-scheduled reclaim_expired() scanner — otherwise a crash-looping worker can starve
    the dead-letter cap forever since claim() runs far more often than the recovery scan."""
    turn_id = await _seed_job(postgres_session_factory)

    async def claim_then_expire_lease(worker_id: str):
        async with postgres_session_factory() as db:
            claimed = await AgentTurnJobService(db).claim(worker_id=worker_id)
        if claimed is None:
            return None
        async with postgres_session_factory() as db:
            job = (await db.execute(select(AgentTurnJob).where(AgentTurnJob.id == claimed.id))).scalar_one()
            job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
            await db.commit()
        return claimed

    # The first claim() (job still QUEUED) does not count as a retry attempt; each subsequent
    # self-heal claim on an expired lease increments `attempt` by one, so reaching
    # MAX_ATTEMPTS_BEFORE_DEAD_LETTER self-heals takes one initial claim plus that many more calls.
    # The last call is the one that crosses the cap: claim() dead-letters the job instead of
    # handing it back, so that final call returns None.
    total_calls = MAX_ATTEMPTS_BEFORE_DEAD_LETTER + 1
    for attempt in range(total_calls):
        result = await claim_then_expire_lease(f"crash-loop-{attempt}")
        is_last = attempt == total_calls - 1
        assert (result is None) == is_last

    async with postgres_session_factory() as db:
        job = (await db.execute(select(AgentTurnJob).where(AgentTurnJob.turn_id == turn_id))).scalar_one()
        assert job.status == AgentTurnJobStatus.DEAD_LETTER
        assert job.attempt >= MAX_ATTEMPTS_BEFORE_DEAD_LETTER

    async with postgres_session_factory() as db:
        no_more_claims = await AgentTurnJobService(db).claim(worker_id="postgres-late")
        assert no_more_claims is None


@pytest.mark.asyncio
async def test_postgres_dead_letter_does_not_block_next_turn_head_of_line(postgres_session_factory):
    """Policy decision: a dead-lettered job does not hold up the next turn in the same session.
    `claim()`'s head-of-line candidate set is built from `non_terminal` statuses only, and
    `DEAD_LETTER` is not one of them — so once turn 1 exhausts its retries and self-heals into
    `DEAD_LETTER`, turn 2 (the next `session_sequence` in the same session) becomes head-of-line
    and is immediately claimable, with no operator intervention required. This mirrors
    `test_postgres_claim_self_heal_dead_letters_after_max_attempts`'s crash-loop setup for turn 1,
    then adds a second, ordinary queued turn in the same session to prove it is not blocked."""
    session_id = await _seed_session(postgres_session_factory)
    turn_1 = await _seed_turn(postgres_session_factory, session_id=session_id, session_sequence=1)
    turn_2 = await _seed_turn(postgres_session_factory, session_id=session_id, session_sequence=2)

    async def claim_then_expire_lease(worker_id: str):
        async with postgres_session_factory() as db:
            claimed = await AgentTurnJobService(db).claim(worker_id=worker_id)
        if claimed is None or claimed.turn_id != turn_1:
            return claimed
        async with postgres_session_factory() as db:
            job = (await db.execute(select(AgentTurnJob).where(AgentTurnJob.id == claimed.id))).scalar_one()
            job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
            await db.commit()
        return claimed

    # Crash-loop turn 1's job past the dead-letter cap via claim()'s own self-heal path, exactly
    # like test_postgres_claim_self_heal_dead_letters_after_max_attempts, while turn 2 stays
    # QUEUED and untouched behind it.
    total_calls = MAX_ATTEMPTS_BEFORE_DEAD_LETTER + 1
    for attempt in range(total_calls):
        await claim_then_expire_lease(f"crash-loop-{attempt}")

    async with postgres_session_factory() as db:
        job_1 = (await db.execute(select(AgentTurnJob).where(AgentTurnJob.turn_id == turn_1))).scalar_one()
        assert job_1.status == AgentTurnJobStatus.DEAD_LETTER

    # Turn 1 is now DEAD_LETTER (terminal, not in claim()'s non_terminal set) — turn 2 must be
    # claimable immediately, with no dependency on any manual dead-letter cleanup.
    async with postgres_session_factory() as db:
        turn_2_claim = await AgentTurnJobService(db).claim(worker_id="postgres-after-dead-letter")
        assert turn_2_claim is not None
        assert turn_2_claim.turn_id == turn_2
        assert turn_2_claim.status == AgentTurnJobStatus.CLAIMED


@pytest.mark.asyncio
async def test_postgres_stale_owner_outcome_projection_is_fenced_after_reclaim(postgres_session_factory):
    """Cross-worker reclaim-then-stale-write proof for the outcome projector's fence: worker A
    claims a job, its lease is treated as expired and reclaimed by worker B (bumping
    `TurnExecutionState.ownership_generation` for real, under Postgres row locking), and worker A
    then still tries to project a terminal outcome using the generation it originally observed.
    That write must be rejected before it touches `AgentSession.status`, not silently applied or
    silently skipped — the same "outcome" leg of the two-workers-claim-once invariant the other
    tests in this file already prove for `claim`/`renew_heartbeat`/`complete`.
    """
    session_id = await _seed_session(postgres_session_factory)
    turn_id = await _seed_turn(postgres_session_factory, session_id=session_id, session_sequence=1)

    async with postgres_session_factory() as db:
        first_claim = await AgentTurnJobService(db).claim(worker_id="postgres-a")
        assert first_claim is not None
        stale_generation = first_claim.lease_generation
        stale_job_id = first_claim.id

    # Force the lease to look expired, as if postgres-a's heartbeat had stopped mid-execution.
    async with postgres_session_factory() as db:
        job = (await db.execute(select(AgentTurnJob).where(AgentTurnJob.id == stale_job_id))).scalar_one()
        job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await db.commit()

    async with postgres_session_factory() as db:
        reclaimed = await AgentTurnJobService(db).reclaim_expired()
        assert len(reclaimed) == 1

    async with postgres_session_factory() as db:
        new_claim = await AgentTurnJobService(db).claim(worker_id="postgres-b")
        assert new_claim is not None
        assert new_claim.lease_generation != stale_generation

    # Worker A, unaware it was reclaimed, still tries to project a terminal outcome for the turn
    # at the generation it originally observed.
    async with postgres_session_factory() as db:
        session_row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
        pre_status = session_row.status
        with pytest.raises(StaleTurnOwnershipError):
            await project_terminal_outcome(
                db,
                session_row,
                TurnOutcomeType.COMPLETED,
                "graph_ended",
                turn_id=turn_id,
                owner_id="postgres-a",
                expected_ownership_generation=stale_generation,
            )
        await db.rollback()

    # The stale write must never have landed: status is unchanged, and the current generation is
    # still the new owner's, not worker A's.
    async with postgres_session_factory() as db:
        session_row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
        assert session_row.status == pre_status
        state = (
            await db.execute(select(TurnExecutionState).where(TurnExecutionState.turn_id == turn_id))
        ).scalar_one()
        assert state.ownership_generation == new_claim.lease_generation
        assert state.ownership_generation != stale_generation
