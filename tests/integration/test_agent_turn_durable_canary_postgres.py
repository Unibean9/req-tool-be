"""Postgres-backed end-to-end proof for the `execution_mode="durable"` canary dispatch path.

Exercises the full lifecycle a real durable turn goes through: `AgentService.handle_user_message`
admits and enqueues the turn (never spawning an inline `asyncio.create_task`), worker A claims the
job, worker A's lease is simulated as expired (as if the process died), the recovery scanner
reclaims the job, worker B claims the now-requeued job and runs it to completion via
`execute_claimed_job` + `AgentService.build_run_graph_kwargs_for_turn`, and finally worker A's own
`complete()` call is confirmed to lose the CAS race (`False`) since it is no longer the lease owner.

This is a new file rather than an extension of `test_agent_turn_job_postgres.py`: that file's
scope is `AgentTurnJobService` in isolation (seeded envelopes/jobs directly, no `AgentService`, no
real admission or graph execution). This file instead proves the seam between admission
(`AgentTurnService` via `AgentService.handle_user_message`), the job ledger
(`AgentTurnJobService`), and the worker (`agent_turn_worker.execute_claimed_job` +
`AgentService.build_run_graph_kwargs_for_turn`) all compose correctly end-to-end, following the
same one-file-per-seam precedent `test_artifact_proposal_completion_postgres.py` set for the
approval-completion seam.
"""

import os
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.agent import AgentSession, AgentSessionStatus, AgentTurnJob, AgentTurnJobStatus
from app.models.organization import Organization
from app.models.project import Project
from app.models.user import User
from app.services.agent_service import AgentService
from app.services.agent_turn_job_service import AgentTurnJobService
from app.services.agent_turn_worker import execute_claimed_job

POSTGRES_URL = os.getenv("AGENT_TURN_POSTGRES_URL")
EXPECTED_ALEMBIC_REVISION = "d38efc70b4f0"
pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def postgres_session_factory():
    if not POSTGRES_URL:
        pytest.skip("AGENT_TURN_POSTGRES_URL is not configured")
    engine = create_async_engine(POSTGRES_URL, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            assert connection.dialect.name == "postgresql"
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
            assert revision == EXPECTED_ALEMBIC_REVISION
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def _reset_agent_turn_jobs(postgres_session_factory):
    async with postgres_session_factory() as db:
        await db.execute(delete(AgentTurnJob))
        await db.commit()
    yield


def _mock_graph():
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={})
    return graph


async def _seed_session(session_factory) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    async with session_factory() as db:
        user = User(email=f"durable-canary-{uuid.uuid4()}@example.com", hashed_password="hash")
        db.add(user)
        await db.flush()
        org = Organization(name="Durable canary", slug=f"durable-canary-{uuid.uuid4().hex}", owner_id=user.id)
        db.add(org)
        await db.flush()
        project = Project(org_id=org.id, name="Durable canary", slug=f"durable-canary-{uuid.uuid4().hex}")
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
        return project.id, session.id, user.id


@pytest.mark.asyncio
async def test_postgres_durable_canary_full_lifecycle_survives_worker_death_and_reclaim(
    postgres_session_factory, monkeypatch
):
    monkeypatch.setattr("app.services.agent_turn_service.settings.agent_turn_admission_enabled", True)
    monkeypatch.setattr("app.services.agent_turn_service.settings.agent_execution_mode", "durable")

    project_id, session_id, user_id = await _seed_session(postgres_session_factory)

    async with postgres_session_factory() as db:
        service = AgentService(db=db, graph=_mock_graph(), session_factory=postgres_session_factory)
        await service.handle_user_message(
            project_id=project_id,
            session_id=session_id,
            content="Durable canary turn",
            user_id=user_id,
            idempotency_key="durable-canary-1",
        )

    # handle_user_message must not have spawned an inline asyncio task: the job row is the only
    # thing that exists for this turn right after admission.
    async with postgres_session_factory() as db:
        job = (await db.execute(select(AgentTurnJob))).scalar_one()
        assert job.status == AgentTurnJobStatus.QUEUED
        turn_id = job.turn_id
        session_row = await db.get(AgentSession, session_id)
        # Admission's own mutation (ACTIVE, interrupt cleared) has landed, but the turn has not
        # executed yet — no worker has claimed it, so status must not have advanced past that.
        assert session_row.status == AgentSessionStatus.ACTIVE

    # Worker A claims the job.
    async with postgres_session_factory() as db:
        job_a = await AgentTurnJobService(db).claim(worker_id="worker-a")
        assert job_a is not None
        assert job_a.turn_id == turn_id
        stale_generation = job_a.lease_generation
        stale_job_id = job_a.id

    # Simulate worker A's death: force its lease to look expired without it ever renewing/completing.
    async with postgres_session_factory() as db:
        stale_job = (await db.execute(select(AgentTurnJob).where(AgentTurnJob.id == stale_job_id))).scalar_one()
        stale_job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await db.commit()

    # Recovery scanner reclaims the expired lease, requeueing the job.
    async with postgres_session_factory() as db:
        reclaimed = await AgentTurnJobService(db).reclaim_expired()
        assert len(reclaimed) == 1
        assert reclaimed[0].id == stale_job_id
        assert reclaimed[0].status == AgentTurnJobStatus.QUEUED

    # Worker B claims the requeued job and runs it to completion.
    async with postgres_session_factory() as db:
        job_b = await AgentTurnJobService(db).claim(worker_id="worker-b")
        assert job_b is not None
        assert job_b.turn_id == turn_id
        assert job_b.lease_generation != stale_generation

    async with postgres_session_factory() as db:
        agent_service = AgentService(db=db, graph=_mock_graph(), session_factory=postgres_session_factory)
        run_graph_kwargs = await agent_service.build_run_graph_kwargs_for_turn(turn_id=turn_id)

    await execute_claimed_job(
        agent_service=agent_service,
        session_factory=postgres_session_factory,
        job=job_b,
        worker_id="worker-b",
        run_graph_kwargs=run_graph_kwargs,
    )

    async with postgres_session_factory() as db:
        session_row = await db.get(AgentSession, session_id)
        assert session_row.status == AgentSessionStatus.COMPLETED
        completed_job = (await db.execute(select(AgentTurnJob).where(AgentTurnJob.turn_id == turn_id))).scalar_one()
        assert completed_job.status == AgentTurnJobStatus.SUCCEEDED

    # Worker A, unaware it was ever reclaimed, tries to report completion at the generation it
    # originally observed — that must lose the CAS race, not silently overwrite worker B's outcome.
    async with postgres_session_factory() as db:
        stale_complete = await AgentTurnJobService(db).complete(
            job_id=stale_job_id,
            worker_id="worker-a",
            lease_generation=stale_generation,
            status=AgentTurnJobStatus.SUCCEEDED,
        )
        assert stale_complete is False
