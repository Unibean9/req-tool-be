"""Unit coverage for `execute_claimed_job`'s completion contract: it always reports an outcome to
the job ledger in a `finally` block, maps a graph exception to a `FAILED` completion, and treats an
already-reclaimed `complete()` as a normal non-error path rather than something to raise.

These tests do not exercise the heartbeat renewal loop's real timing - SQLite (this fixture's
backend) gives no real row locking either, so any timing/concurrency proof belongs in a Postgres
integration test instead. `agent_turn_job_heartbeat_interval_seconds` is left at its production
default here, comfortably longer than these tests take to run, so the loop never ticks before the
mocked graph call returns and the heartbeat task is cancelled.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from app.models.agent import (
    AgentTurnEnvelope,
    AgentTurnJob,
    AgentTurnJobStatus,
    TurnExecutionState,
)
from app.models.user import User
from app.services.agent_turn_worker import _run_one_scan, execute_claimed_job
from tests.conftest import TestSessionFactory


@pytest_asyncio.fixture(autouse=True)
async def _reset_agent_turn_jobs(db_session):
    """`reclaim_expired()` (called through `_run_one_scan`) scans the whole `agent_turn_jobs`
    table by design, and this module's tests commit for real — reset before each test so a job
    left over from an earlier test in this file is not still visible to a later scan."""
    await db_session.execute(delete(AgentTurnJob))
    await db_session.commit()
    yield


async def _seed_claimed_job(db_session, *, owner_id: str = "worker-a", generation: int = 1) -> AgentTurnJob:
    user = User(email=f"turn-worker-{uuid.uuid4()}@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()
    envelope = AgentTurnEnvelope(
        session_id=uuid.uuid4(),
        session_sequence=1,
        original_trigger_id=uuid.uuid4(),
        actor_id=user.id,
        cohort={},
        correlation_id=str(uuid.uuid4()),
    )
    db_session.add(envelope)
    await db_session.flush()
    job = AgentTurnJob(
        turn_id=envelope.id,
        status=AgentTurnJobStatus.CLAIMED,
        lease_owner=owner_id,
        lease_generation=generation,
        cohort={},
    )
    db_session.add(job)
    await db_session.commit()
    return job


class _StubAgentService:
    """A minimal double: only `_run_graph` is exercised by `execute_claimed_job`."""

    def __init__(self, run_graph):
        self._run_graph = run_graph


@pytest.mark.asyncio
async def test_execute_claimed_job_completes_succeeded_on_normal_graph_return(db_session):
    job = await _seed_claimed_job(db_session)
    calls: list[dict] = []

    async def _run_graph(**kwargs):
        calls.append(kwargs)

    agent_service = _StubAgentService(_run_graph)

    await execute_claimed_job(
        agent_service=agent_service,
        session_factory=TestSessionFactory,
        job=job,
        worker_id="worker-a",
        run_graph_kwargs={"session_id": uuid.uuid4(), "turn_id": job.turn_id},
    )

    # owner_id/ownership_generation must always come from the job, never from run_graph_kwargs.
    assert calls[0]["owner_id"] == "worker-a"
    assert calls[0]["ownership_generation"] == 1

    # `execute_claimed_job` completes the job through its own short-lived session, so this
    # session's identity map still holds the pre-completion object - refresh to observe the write.
    await db_session.refresh(job)
    assert job.status == AgentTurnJobStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_execute_claimed_job_completes_failed_when_graph_raises(db_session):
    job = await _seed_claimed_job(db_session)

    async def _run_graph(**kwargs):
        raise RuntimeError("graph blew up")

    agent_service = _StubAgentService(_run_graph)

    # The exception must not propagate - the wrapper reports it as a FAILED completion instead.
    await execute_claimed_job(
        agent_service=agent_service,
        session_factory=TestSessionFactory,
        job=job,
        worker_id="worker-a",
        run_graph_kwargs={"session_id": uuid.uuid4(), "turn_id": job.turn_id},
    )

    await db_session.refresh(job)
    assert job.status == AgentTurnJobStatus.FAILED
    assert "graph blew up" in (job.last_error or "")


@pytest.mark.asyncio
async def test_execute_claimed_job_does_not_raise_when_already_reclaimed(db_session):
    """Simulates a worker whose lease was reclaimed by another worker while its graph was still
    running: by the time this worker reaches `complete()`, the job's lease_owner/generation have
    already moved on. `complete()` returning False for that CAS mismatch must not raise."""
    job = await _seed_claimed_job(db_session, owner_id="worker-a", generation=1)

    async def _run_graph(**kwargs):
        # Simulate a concurrent reclaim landing while this worker's graph call is in flight.
        async with TestSessionFactory() as other_db:
            other_job = (
                await other_db.execute(select(AgentTurnJob).where(AgentTurnJob.id == job.id))
            ).scalar_one()
            other_job.lease_owner = "worker-b"
            other_job.lease_generation = 2
            other_job.status = AgentTurnJobStatus.CLAIMED
            await other_db.commit()

    agent_service = _StubAgentService(_run_graph)

    await execute_claimed_job(
        agent_service=agent_service,
        session_factory=TestSessionFactory,
        job=job,
        worker_id="worker-a",
        run_graph_kwargs={"session_id": uuid.uuid4(), "turn_id": job.turn_id},
    )

    await db_session.refresh(job)
    # Worker A's stale completion must not overwrite worker B's ownership.
    assert job.lease_owner == "worker-b"
    assert job.lease_generation == 2


@pytest.mark.asyncio
async def test_run_one_scan_requeues_expired_lease_and_returns_it(db_session, caplog):
    """`_run_one_scan` is the recovery scanner loop's single-iteration body: it must call
    `reclaim_expired()` through its own short-lived session, return whatever that call reclaimed,
    and log a structured line per reclaimed job so an operator can see it happen without a
    dashboard. This does not need Postgres: no concurrent claim is exercised here, only that the
    wrapper calls through and logs correctly."""
    job = await _seed_claimed_job(db_session)
    envelope_id = job.turn_id
    db_session.add(TurnExecutionState(turn_id=envelope_id, ownership_generation=job.lease_generation))
    job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await db_session.commit()

    with caplog.at_level("INFO", logger="app.services.agent_turn_worker"):
        reclaimed = await _run_one_scan(session_factory=TestSessionFactory)

    assert len(reclaimed) == 1
    assert reclaimed[0].id == job.id
    assert reclaimed[0].status == AgentTurnJobStatus.QUEUED
    assert reclaimed[0].attempt == 1
    assert any("agent_turn_job_reclaimed" in record.message for record in caplog.records)
    assert any(str(job.id) in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_run_one_scan_returns_empty_when_nothing_expired(db_session):
    """No expired lease anywhere means no reclaim and no log noise."""
    reclaimed = await _run_one_scan(session_factory=TestSessionFactory)
    assert reclaimed == []
