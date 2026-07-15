"""Postgres-backed fault-injection tests for write_draft's command boundary.

Only runs when AGENT_TURN_POSTGRES_URL is configured (mirrors test_agent_turn_postgres.py). These
tests exercise DraftCommandService directly against real Postgres row-locking/unique-constraint
semantics, since SQLite unit tests cannot prove concurrent-transaction fencing behavior.
"""

import asyncio
import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.agent import (
    AgentRun,
    AgentSession,
    AgentSessionStatus,
    AgentToolCall,
    AgentToolCallStatus,
    AgentTurnEnvelope,
    DraftCommandLedger,
    TurnExecutionState,
    TurnExecutionStatus,
)
from app.models.organization import Organization
from app.models.project import Project
from app.models.user import User
from app.services.draft_command_service import (
    DraftCommandService,
    canonical_write_draft_intent,
    write_draft_logical_command_id,
)

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
            ledger_table = await connection.scalar(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = current_schema() AND table_name = 'agent_draft_commands'"
                )
            )
            assert ledger_table == 1
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _seed_turn(session_factory, *, owner_id: str, generation: int):
    async with session_factory() as db:
        user = User(email=f"draft-cmd-{uuid.uuid4()}@example.com", hashed_password="hash")
        db.add(user)
        await db.flush()
        org = Organization(name="Draft cmd test", slug=f"draft-cmd-{uuid.uuid4().hex}", owner_id=user.id)
        db.add(org)
        await db.flush()
        project = Project(org_id=org.id, name="Draft cmd test", slug=f"draft-cmd-{uuid.uuid4().hex}")
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
        await db.flush()
        envelope = AgentTurnEnvelope(
            session_id=session.id,
            session_sequence=1,
            original_trigger_id=uuid.uuid4(),
            actor_id=user.id,
            cohort={"command_handlers_enabled": True},
            correlation_id=str(uuid.uuid4()),
        )
        db.add(envelope)
        await db.flush()
        db.add(
            TurnExecutionState(
                turn_id=envelope.id,
                status=TurnExecutionStatus.RUNNING,
                owner_id=owner_id,
                ownership_generation=generation,
            )
        )
        run = AgentRun(session_id=session.id, analysis_result={})
        db.add(run)
        await db.commit()
        return envelope.id, run.id


@pytest.mark.asyncio
async def test_postgres_stale_owner_rejected_after_reclaim(postgres_session_factory):
    """Owner A's snapshot is captured before owner B reclaims (bumps ownership_generation); A's
    late fence check must be rejected under real Postgres row-locking, not just SQLite."""
    turn_id, _run_id = await _seed_turn(postgres_session_factory, owner_id="owner-a", generation=1)

    async with postgres_session_factory() as db:
        state = (
            await db.execute(select(TurnExecutionState).where(TurnExecutionState.turn_id == turn_id))
        ).scalar_one()
        state.owner_id = "owner-b"
        state.ownership_generation = 2
        await db.commit()

    async with postgres_session_factory() as db:
        service = DraftCommandService(db)
        fenced = await service.fence_or_none(turn_id=turn_id, owner_id="owner-a", expected_generation=1)
        assert fenced is None

    async with postgres_session_factory() as db:
        service = DraftCommandService(db)
        current = await service.fence_or_none(turn_id=turn_id, owner_id="owner-b", expected_generation=2)
        assert current is not None


@pytest.mark.asyncio
async def test_postgres_duplicate_logical_command_concurrent_writers(postgres_session_factory):
    """Two concurrent attempts race to insert the same logical_command_id; the unique constraint
    must let exactly one commit succeed, and the loser's check_duplicate must then observe the
    winner's committed row rather than double-writing the effect."""
    turn_id, run_id = await _seed_turn(postgres_session_factory, owner_id="owner-a", generation=1)
    canonical_intent = canonical_write_draft_intent("Vision", "same content for both writers")
    logical_command_id = write_draft_logical_command_id(turn_id, canonical_intent, None)

    async def attempt(call_suffix: str):
        async with postgres_session_factory() as db:
            service = DraftCommandService(db)
            existing = await service.check_duplicate(logical_command_id)
            if existing is not None:
                return "duplicate", existing.id
            tool_call = AgentToolCall(
                run_id=run_id,
                tool_name="write_draft",
                input_snapshot={"call": call_suffix},
                status=AgentToolCallStatus.PROPOSED,
            )
            db.add(tool_call)
            await db.flush()
            service.record_effect(
                turn_id=turn_id,
                logical_command_id=logical_command_id,
                action_type="write_draft",
                tool_call=tool_call,
                artifact_id=None,
                attempt=0,
            )
            try:
                await db.commit()
                return "committed", tool_call.id
            except IntegrityError:
                await db.rollback()
                return "raced_out", None

    results = await asyncio.gather(attempt("a"), attempt("b"))
    outcomes = [status for status, _ in results]
    # Exactly one writer must have won (committed); the other must either have observed the
    # duplicate up front or raced into the unique constraint and rolled back cleanly.
    assert outcomes.count("committed") == 1
    assert set(outcomes) <= {"committed", "duplicate", "raced_out"}

    async with postgres_session_factory() as db:
        ledger_rows = (
            await db.execute(select(DraftCommandLedger).where(DraftCommandLedger.turn_id == turn_id))
        ).scalars().all()
        assert len(ledger_rows) == 1
        tool_calls = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run_id))
        ).scalars().all()
        assert len(tool_calls) == 1


@pytest.mark.asyncio
async def test_postgres_reconciliation_reads_committed_effect_without_replay(postgres_session_factory):
    """Simulates crash-after-commit-before-observation: a fresh session/connection reads back the
    already-committed ledger + tool call and must reuse it, never re-running the artifact write."""
    turn_id, run_id = await _seed_turn(postgres_session_factory, owner_id="owner-a", generation=1)
    canonical_intent = canonical_write_draft_intent("Vision", "reconciliation content")
    logical_command_id = write_draft_logical_command_id(turn_id, canonical_intent, None)

    async with postgres_session_factory() as db:
        service = DraftCommandService(db)
        tool_call = AgentToolCall(
            run_id=run_id,
            tool_name="write_draft",
            input_snapshot={"call": "original"},
            status=AgentToolCallStatus.PROPOSED,
        )
        db.add(tool_call)
        await db.flush()
        service.record_effect(
            turn_id=turn_id,
            logical_command_id=logical_command_id,
            action_type="write_draft",
            tool_call=tool_call,
            artifact_id=None,
            attempt=0,
        )
        await db.commit()
        committed_tool_call_id = tool_call.id
    # Session above is fully closed/disposed of its connection here — a brand new session below
    # stands in for a reconciling process reading state after a crash.
    async with postgres_session_factory() as db:
        service = DraftCommandService(db)
        reconciled = await service.check_duplicate(logical_command_id)
        assert reconciled is not None
        assert reconciled.id == committed_tool_call_id

        tool_calls = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run_id))
        ).scalars().all()
        assert len(tool_calls) == 1
