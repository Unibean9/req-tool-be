"""Postgres-backed race proof for final approval's ordered graph-resume transition.

This test exists to provide evidence, not inference from reading code, that the existing
`.with_for_update()` row lock on the session row genuinely prevents two concurrent approvals from
racing the pending-proposal count into a double (or premature) resume. SQLite (used by the
rest of the unit suite) cannot exercise real concurrent-transaction row locking, so this proof runs
only against a real Postgres instance, mirroring test_draft_command_postgres.py / test_
agent_turn_postgres.py's `AGENT_TURN_POSTGRES_URL`-gated pattern.
"""

import asyncio
import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.agent import (
    AgentRun,
    AgentSession,
    AgentSessionStatus,
    AgentToolCall,
    AgentToolCallStatus,
    AgentTurnEnvelope,
    TurnOutcome,
)
from app.models.organization import Organization
from app.models.project import Project
from app.models.user import User
from app.services.agent_service import AgentService

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
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
            assert revision == EXPECTED_ALEMBIC_REVISION
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _seed_two_proposals(session_factory):
    async with session_factory() as db:
        user = User(email=f"completion-race-{uuid.uuid4()}@example.com", hashed_password="hash")
        db.add(user)
        await db.flush()
        org = Organization(name="Completion race", slug=f"completion-race-{uuid.uuid4().hex}", owner_id=user.id)
        db.add(org)
        await db.flush()
        project = Project(org_id=org.id, name="Completion race", slug=f"completion-race-{uuid.uuid4().hex}")
        db.add(project)
        await db.flush()
        session = AgentSession(
            project_id=project.id,
            artifact_type="brd",
            workflow_area="analysis",
            status=AgentSessionStatus.WAITING_FOR_HUMAN,
            created_by_id=user.id,
        )
        db.add(session)
        await db.flush()
        run = AgentRun(session_id=session.id, analysis_result={})
        db.add(run)
        await db.flush()
        tool_call_1 = AgentToolCall(
            run_id=run.id, tool_name="create_artifact", status=AgentToolCallStatus.PROPOSED, input_snapshot={}
        )
        tool_call_2 = AgentToolCall(
            run_id=run.id, tool_name="create_artifact", status=AgentToolCallStatus.PROPOSED, input_snapshot={}
        )
        db.add_all([tool_call_1, tool_call_2])
        approval_turn_1 = AgentTurnEnvelope(
            session_id=session.id,
            session_sequence=1,
            original_trigger_id=uuid.uuid4(),
            actor_id=user.id,
            cohort={"turn_outcomes_enabled": True},
            correlation_id=str(uuid.uuid4()),
        )
        approval_turn_2 = AgentTurnEnvelope(
            session_id=session.id,
            session_sequence=2,
            original_trigger_id=uuid.uuid4(),
            actor_id=user.id,
            cohort={"turn_outcomes_enabled": True},
            correlation_id=str(uuid.uuid4()),
        )
        db.add_all([approval_turn_1, approval_turn_2])
        await db.commit()
        return session.id, tool_call_1.id, tool_call_2.id, approval_turn_1.id, approval_turn_2.id


async def _approve_then_check_completion(session_factory, session_id, tool_call_id, turn_id):
    async with session_factory() as db:
        tool_call = await db.get(AgentToolCall, tool_call_id)
        tool_call.status = AgentToolCallStatus.EXECUTED
        await db.commit()
    async with session_factory() as db:
        svc = AgentService(db, graph=None, session_factory=session_factory)
        result = await svc._prepare_resume_when_all_artifact_proposals_approved(
            session_id=session_id, llm_client=object()
        )
        # `approve_tool_call()` releases its lease immediately after this helper. The losing approval
        # race phải đóng transaction `FOR UPDATE`, nếu không release_inline() sẽ ném lỗi vì
        # transaction còn mở.
        assert not db.in_transaction()
        return result


@pytest.mark.asyncio
async def test_postgres_concurrent_last_two_approvals_schedule_one_resume_without_terminal_outcome(postgres_session_factory):
    """Two concurrent final approval transitions must commit one ordered resume, never complete
    the session themselves (each caller opens its own transaction/session, exactly the shape
    `approve_tool_call` uses in production), and must not resume before both proposals are
    actually EXECUTED."""
    session_id, tc1_id, tc2_id, approval_turn_1_id, approval_turn_2_id = await _seed_two_proposals(
        postgres_session_factory
    )

    results = await asyncio.gather(
        _approve_then_check_completion(postgres_session_factory, session_id, tc1_id, approval_turn_1_id),
        _approve_then_check_completion(postgres_session_factory, session_id, tc2_id, approval_turn_2_id),
    )

    async with postgres_session_factory() as db:
        session = await db.get(AgentSession, session_id)
        assert session.status == AgentSessionStatus.ACTIVE
        assert session.interrupt_type is None
        outcomes = (
            await db.execute(select(TurnOutcome).where(TurnOutcome.session_id == session_id))
        ).scalars().all()
        assert outcomes == []
        assert sum(result is not None for result in results) == 1
        tool_calls = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id.in_(
                select(AgentRun.id).where(AgentRun.session_id == session_id)
            )))
        ).scalars().all()
        assert {tc.status for tc in tool_calls} == {AgentToolCallStatus.EXECUTED}


@pytest.mark.asyncio
async def test_postgres_repeated_concurrent_final_approval_never_races(postgres_session_factory):
    """Repeats the race several times (fresh session/proposals each iteration) to raise confidence
    that the row lock's correctness is not merely an artifact of one lucky interleaving."""
    for _ in range(5):
        session_id, tc1_id, tc2_id, approval_turn_1_id, approval_turn_2_id = await _seed_two_proposals(
            postgres_session_factory
        )
        results = await asyncio.gather(
            _approve_then_check_completion(postgres_session_factory, session_id, tc1_id, approval_turn_1_id),
            _approve_then_check_completion(postgres_session_factory, session_id, tc2_id, approval_turn_2_id),
        )
        async with postgres_session_factory() as db:
            session = await db.get(AgentSession, session_id)
            assert session.status == AgentSessionStatus.ACTIVE
            outcomes = (
                await db.execute(select(TurnOutcome).where(TurnOutcome.session_id == session_id))
            ).scalars().all()
            assert outcomes == []
            assert sum(result is not None for result in results) == 1
