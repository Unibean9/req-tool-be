"""Unit coverage for `AgentTurnJobService`'s enqueue idempotency, status enum values, and the
generation-mismatch rejection path of `renew_heartbeat`/`complete`.

SQLite (this fixture's backend) does not enforce real `SELECT ... FOR UPDATE`/`SKIP LOCKED`
semantics, so it cannot prove claim-once/reclaim fencing under real concurrency — that invariant
is proven separately by the Postgres integration test. These tests only exercise the pure
application-level CAS logic (compare-then-write against a single in-process session), which
behaves identically regardless of backend.
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
from app.services.agent_turn_job_service import AgentTurnJobService
from tests.factories import _make_agent_session, _project


@pytest_asyncio.fixture(autouse=True)
async def _reset_agent_turn_jobs(db_session):
    """`claim()`/`reclaim_expired()` scan the whole `agent_turn_jobs` table by design (a durable
    queue serves every session), and this module's methods commit for real rather than only
    flushing — so, unlike most other tests here, a job committed by an earlier test in this file
    would otherwise still be visible to a later test's unscoped scan. Reset before each test."""
    await db_session.execute(delete(AgentTurnJob))
    await db_session.commit()
    yield


async def _seed_envelope(db_session, agent_session, *, session_sequence: int = 1) -> AgentTurnEnvelope:
    user = User(email=f"turn-job-{uuid.uuid4()}@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()
    envelope = AgentTurnEnvelope(
        session_id=agent_session.id,
        session_sequence=session_sequence,
        original_trigger_id=uuid.uuid4(),
        actor_id=user.id,
        cohort={"turn_admission": "v1"},
        correlation_id=str(uuid.uuid4()),
    )
    db_session.add(envelope)
    await db_session.flush()
    db_session.add(TurnExecutionState(turn_id=envelope.id))
    await db_session.commit()
    return envelope


def test_agent_turn_job_status_enum_values():
    assert {status.value for status in AgentTurnJobStatus} == {
        "queued",
        "claimed",
        "running",
        "succeeded",
        "failed",
        "dead_letter",
    }


@pytest.mark.asyncio
async def test_enqueue_is_idempotent_per_turn(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_envelope(db_session, agent_session)

    service = AgentTurnJobService(db_session)
    first = await service.enqueue(turn_id=envelope.id, expected_transition_version=0, cohort={"a": 1})
    second = await service.enqueue(turn_id=envelope.id, expected_transition_version=0, cohort={"a": 1})

    assert first.id == second.id
    rows = (await db_session.execute(select(AgentTurnJob).where(AgentTurnJob.turn_id == envelope.id))).scalars().all()
    assert len(rows) == 1
    assert rows[0].status == AgentTurnJobStatus.QUEUED


@pytest.mark.asyncio
async def test_renew_heartbeat_rejects_generation_mismatch(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_envelope(db_session, agent_session)
    service = AgentTurnJobService(db_session)
    job = await service.enqueue(turn_id=envelope.id, expected_transition_version=0, cohort={})
    claimed = await service.claim(worker_id="worker-a")
    assert claimed is not None
    assert claimed.id == job.id
    generation = claimed.lease_generation

    assert await service.renew_heartbeat(job_id=job.id, worker_id="worker-a", lease_generation=generation + 1) is False
    assert await service.renew_heartbeat(job_id=job.id, worker_id="worker-b", lease_generation=generation) is False
    assert await service.renew_heartbeat(job_id=job.id, worker_id="worker-a", lease_generation=generation) is True

    refreshed = await db_session.get(AgentTurnJob, job.id)
    assert refreshed.heartbeat_at is not None


@pytest.mark.asyncio
async def test_complete_rejects_generation_mismatch_and_does_not_overwrite_new_owner(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_envelope(db_session, agent_session)
    service = AgentTurnJobService(db_session)
    job = await service.enqueue(turn_id=envelope.id, expected_transition_version=0, cohort={})
    claimed = await service.claim(worker_id="worker-a")
    stale_generation = claimed.lease_generation

    # Simulate a reclaim moving the fence forward from under the stale owner.
    job.status = AgentTurnJobStatus.QUEUED
    job.lease_owner = None
    await db_session.commit()
    reclaimed = await service.claim(worker_id="worker-b")
    assert reclaimed is not None
    assert reclaimed.lease_generation != stale_generation

    assert (
        await service.complete(
            job_id=job.id,
            worker_id="worker-a",
            lease_generation=stale_generation,
            status=AgentTurnJobStatus.SUCCEEDED,
        )
        is False
    )
    still_owned_by_new_owner = await db_session.get(AgentTurnJob, job.id)
    assert still_owned_by_new_owner.lease_owner == "worker-b"
    assert still_owned_by_new_owner.status == AgentTurnJobStatus.CLAIMED

    assert (
        await service.complete(
            job_id=job.id,
            worker_id="worker-b",
            lease_generation=reclaimed.lease_generation,
            status=AgentTurnJobStatus.SUCCEEDED,
        )
        is True
    )
    finished = await db_session.get(AgentTurnJob, job.id)
    assert finished.status == AgentTurnJobStatus.SUCCEEDED


def test_complete_rejects_non_terminal_status():
    with pytest.raises(ValueError):
        import asyncio

        async def _call():
            await AgentTurnJobService(None).complete(
                job_id=uuid.uuid4(), worker_id="w", lease_generation=1, status=AgentTurnJobStatus.QUEUED
            )

        asyncio.run(_call())


@pytest.mark.asyncio
async def test_claim_returns_none_when_no_claimable_jobs(client, db_session):
    service = AgentTurnJobService(db_session)
    assert await service.claim(worker_id="worker-a") is None


@pytest.mark.asyncio
async def test_claim_respects_head_of_line_ordering_within_session(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    earlier = await _seed_envelope(db_session, agent_session, session_sequence=1)
    later = await _seed_envelope(db_session, agent_session, session_sequence=2)
    service = AgentTurnJobService(db_session)
    earlier_job = await service.enqueue(turn_id=earlier.id, expected_transition_version=0, cohort={})
    later_job = await service.enqueue(turn_id=later.id, expected_transition_version=0, cohort={})

    claimed_earlier = await service.claim(worker_id="worker-a")
    assert claimed_earlier is not None
    assert claimed_earlier.id == earlier_job.id

    # Earlier job is still claimed/running with a live lease: the later job in the same
    # session must not be claimable yet, even though it is queued.
    assert await service.claim(worker_id="worker-b") is None

    # Once the earlier job's lease has expired, claim() still self-heals it first (still the
    # lower session_sequence) rather than skipping straight to the later job.
    earlier_job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await db_session.commit()
    reclaimed_earlier = await service.claim(worker_id="worker-c")
    assert reclaimed_earlier is not None
    assert reclaimed_earlier.id == earlier_job.id

    # Only once the earlier job reaches a terminal status does the later job become claimable.
    assert await service.claim(worker_id="worker-d") is None
    assert await service.complete(
        job_id=earlier_job.id,
        worker_id="worker-c",
        lease_generation=reclaimed_earlier.lease_generation,
        status=AgentTurnJobStatus.SUCCEEDED,
    )
    claimed_later = await service.claim(worker_id="worker-e")
    assert claimed_later is not None
    assert claimed_later.id == later_job.id


@pytest.mark.asyncio
async def test_reclaim_expired_requeues_and_dead_letters_by_attempt_cap(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    requeue_envelope = await _seed_envelope(db_session, agent_session, session_sequence=1)
    dead_letter_envelope = await _seed_envelope(db_session, agent_session, session_sequence=2)
    service = AgentTurnJobService(db_session)

    requeue_job = await service.enqueue(turn_id=requeue_envelope.id, expected_transition_version=0, cohort={})
    dead_letter_job = await service.enqueue(
        turn_id=dead_letter_envelope.id, expected_transition_version=0, cohort={}
    )
    dead_letter_job.attempt = 4
    await db_session.commit()

    claimed_requeue = await service.claim(worker_id="worker-a")
    assert claimed_requeue.id == requeue_job.id
    # dead_letter_job is still blocked by head-of-line while requeue_job is active; force it
    # into a claimed state directly to exercise reclaim_expired's own status/lease handling.
    # `lease_generation` must mirror this turn's own `TurnExecutionState.ownership_generation`
    # (bumped here the same way claim() would have), not the unrelated requeue turn's value.
    dead_letter_state = (
        await db_session.execute(
            select(TurnExecutionState).where(TurnExecutionState.turn_id == dead_letter_envelope.id)
        )
    ).scalar_one()
    dead_letter_state.ownership_generation += 1
    dead_letter_job.status = AgentTurnJobStatus.CLAIMED
    dead_letter_job.lease_owner = "worker-x"
    dead_letter_job.lease_generation = dead_letter_state.ownership_generation
    await db_session.commit()

    requeue_job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    dead_letter_job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await db_session.commit()

    reclaimed = await service.reclaim_expired()
    reclaimed_ids = {job.id: job for job in reclaimed}
    assert requeue_job.id in reclaimed_ids
    assert reclaimed_ids[requeue_job.id].status == AgentTurnJobStatus.QUEUED
    assert reclaimed_ids[requeue_job.id].attempt == 1
    assert dead_letter_job.id in reclaimed_ids
    assert reclaimed_ids[dead_letter_job.id].status == AgentTurnJobStatus.DEAD_LETTER
    assert reclaimed_ids[dead_letter_job.id].attempt == 5
