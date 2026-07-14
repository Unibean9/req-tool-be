"""Phase 4 command boundary: write_draft's fenced, idempotent effect path.

Only reachable when the admitting turn's cohort snapshot has `command_handlers_enabled=True` and
the config carries `turn_id`/`turn_owner_id`/`turn_ownership_generation` — the plumbing gap the
phase-04 brief calls out (RunnableConfig/state only, never the public tool JSON schema). Any turn
without that context, or with the cohort flag off, must behave byte-identically to the pre-Phase-4
legacy path — see test_write_draft_*.py for that regression coverage.
"""

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.graphs.agent_tools import _write_draft_impl
from app.models.agent import (
    AgentToolCall,
    AgentToolCallStatus,
    AgentTurnEnvelope,
    DraftCommandEffectState,
    DraftCommandLedger,
    TurnExecutionState,
    TurnExecutionStatus,
)
from app.models.artifact import ArtifactType
from app.models.user import User
from tests.conftest import TestSessionFactory
from tests.factories import (
    _config,
    _focused_items,
    _make_agent_run,
    _make_agent_session,
    _project,
    _session_factory,
    _state,
)

COMPLETE_BODY = "\n\n".join(
    [
        "## Vision\nA concrete vision statement.",
        "## Objectives\n- Ship the thing.",
        "## Success Metrics\n- Adoption reaches 80%.",
    ]
)
OTHER_COMPLETE_BODY = "\n\n".join(
    [
        "## Vision\nA different vision statement entirely.",
        "## Objectives\n- Ship the other thing.",
        "## Success Metrics\n- Adoption reaches 90%.",
    ]
)


async def _seed_turn(db_session, agent_session, *, command_handlers_enabled: bool, owner_id: str, generation: int):
    """Admit a turn directly (bypassing AgentTurnService admission flow, which is Phase 2/3
    territory) so this test controls the cohort snapshot and claimed owner/generation exactly."""
    user = User(email=f"cmd-turn-{uuid.uuid4()}@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()
    envelope = AgentTurnEnvelope(
        session_id=agent_session.id,
        session_sequence=1,
        original_trigger_id=uuid.uuid4(),
        actor_id=user.id,
        cohort={"command_handlers_enabled": command_handlers_enabled},
        correlation_id=str(uuid.uuid4()),
    )
    db_session.add(envelope)
    await db_session.flush()
    db_session.add(
        TurnExecutionState(
            turn_id=envelope.id,
            status=TurnExecutionStatus.RUNNING,
            owner_id=owner_id,
            ownership_generation=generation,
        )
    )
    await db_session.commit()
    return envelope


async def _seed(client, db_session, *, command_handlers_enabled: bool = True, owner_id: str = "owner-a", generation: int = 1):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(db_session, project_id, ArtifactType.VISION_OBJECTIVES)
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)
    envelope = await _seed_turn(
        db_session,
        agent_session,
        command_handlers_enabled=command_handlers_enabled,
        owner_id=owner_id,
        generation=generation,
    )

    state = _state(artifact_type="vision_objectives")
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()
    config["configurable"]["turn_id"] = str(envelope.id)
    config["configurable"]["turn_owner_id"] = owner_id
    config["configurable"]["turn_ownership_generation"] = generation
    return state, config, run, envelope


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_command_handler_commits_effect_and_ledger_row(mock_interrupt, client, db_session):
    state, config, run, envelope = await _seed(client, db_session)

    command = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        tool_call = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalar_one()
        ledger_rows = (
            await db.execute(select(DraftCommandLedger).where(DraftCommandLedger.turn_id == envelope.id))
        ).scalars().all()
        assert len(ledger_rows) == 1
        assert ledger_rows[0].tool_call_id == tool_call.id
        assert ledger_rows[0].action_type == "write_draft"


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_duplicate_logical_command_id_reuses_outcome_without_new_row(mock_interrupt, client, db_session):
    """Two execution attempts (different tool-call IDs) with the same turn/content/base-version
    are the same logical command: the second call must not create a second AgentToolCall or a
    second ledger row, even though its provider tool-call ID differs from the first."""
    state, config, run, envelope = await _seed(client, db_session)

    first = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")
    second = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_2")

    assert not (first.update.get("tool_errors") or [])
    assert not (second.update.get("tool_errors") or [])
    async with TestSessionFactory() as db:
        tool_calls = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
        ledger_rows = (
            await db.execute(select(DraftCommandLedger).where(DraftCommandLedger.turn_id == envelope.id))
        ).scalars().all()
        assert len(tool_calls) == 1
        assert len(ledger_rows) == 1


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_distinct_intent_same_turn_gets_distinct_logical_command(mock_interrupt, client, db_session):
    """Different draft content in the same turn is a distinct logical command, not a duplicate."""
    state, config, run, envelope = await _seed(client, db_session)

    await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")
    await _write_draft_impl("Vision", OTHER_COMPLETE_BODY, state, config, "call_2")

    async with TestSessionFactory() as db:
        ledger_rows = (
            await db.execute(select(DraftCommandLedger).where(DraftCommandLedger.turn_id == envelope.id))
        ).scalars().all()
        assert len(ledger_rows) == 2
        assert ledger_rows[0].logical_command_id != ledger_rows[1].logical_command_id


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_stale_owner_rejected_before_any_mutation(mock_interrupt, client, db_session):
    """Owner A's ownership_generation snapshot (captured at claim time) no longer matches the
    live TurnExecutionState (owner B reclaimed) — A's late commit must be rejected before writing
    any AgentToolCall/ledger row, with a typed recoverable observation."""
    state, config, run, envelope = await _seed(client, db_session, owner_id="owner-a", generation=1)

    # Owner B reclaims the turn (bumps ownership_generation, changes owner_id) — simulates
    # AgentTurnService.claim_inline being called again after A's lease expired.
    async with TestSessionFactory() as db:
        turn_state = (
            await db.execute(select(TurnExecutionState).where(TurnExecutionState.turn_id == envelope.id))
        ).scalar_one()
        turn_state.owner_id = "owner-b"
        turn_state.ownership_generation = 2
        await db.commit()

    command = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")

    errors = command.update.get("tool_errors") or []
    assert len(errors) == 1
    assert errors[0]["code"] == "turn_fence_stale"
    mock_interrupt.assert_not_called()
    async with TestSessionFactory() as db:
        tool_calls = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
        ledger_rows = (
            await db.execute(select(DraftCommandLedger).where(DraftCommandLedger.turn_id == envelope.id))
        ).scalars().all()
        assert tool_calls == []
        assert ledger_rows == []


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_expired_lease_rejected_before_any_mutation(mock_interrupt, client, db_session):
    from datetime import UTC, datetime, timedelta

    state, config, run, envelope = await _seed(client, db_session, owner_id="owner-a", generation=1)
    async with TestSessionFactory() as db:
        turn_state = (
            await db.execute(select(TurnExecutionState).where(TurnExecutionState.turn_id == envelope.id))
        ).scalar_one()
        turn_state.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await db.commit()

    command = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")

    errors = command.update.get("tool_errors") or []
    assert errors and errors[0]["code"] == "turn_fence_stale"
    mock_interrupt.assert_not_called()
    async with TestSessionFactory() as db:
        assert (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalar_one_or_none() is None


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_race_loser_reuses_winners_committed_outcome(mock_interrupt, client, db_session):
    """Simulates two writers racing the same logical_command_id: a competing row is inserted
    (standing in for a concurrent writer) between this call's duplicate pre-check and its commit,
    so the pre-check sees no duplicate but the commit hits the unique constraint. The call must
    reuse the winner's already-committed outcome instead of propagating an unhandled IntegrityError."""
    state, config, run, envelope = await _seed(client, db_session)

    from app.services import draft_command_service as dcs_module

    original_check_duplicate = dcs_module.DraftCommandService.check_duplicate
    call_count = {"n": 0}

    async def racy_check_duplicate(self, logical_command_id):
        call_count["n"] += 1
        if call_count["n"] == 1:
            async with TestSessionFactory() as winner_db:
                winner_tool_call = AgentToolCall(
                    run_id=run.id,
                    tool_name="write_draft",
                    input_snapshot={
                        "candidate_readiness": {"state": "sufficient"},
                        "synthesis_metadata": {"deterministic_warnings": []},
                    },
                    status=AgentToolCallStatus.PROPOSED,
                )
                winner_db.add(winner_tool_call)
                await winner_db.flush()
                winner_db.add(
                    DraftCommandLedger(
                        turn_id=envelope.id,
                        logical_command_id=logical_command_id,
                        action_type="write_draft",
                        tool_call_id=winner_tool_call.id,
                        artifact_id=None,
                        effect_state=DraftCommandEffectState.COMMITTED,
                        attempt=0,
                    )
                )
                await winner_db.commit()
            return None
        return await original_check_duplicate(self, logical_command_id)

    with patch.object(dcs_module.DraftCommandService, "check_duplicate", racy_check_duplicate):
        command = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        tool_calls = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
        ledger_rows = (
            await db.execute(select(DraftCommandLedger).where(DraftCommandLedger.turn_id == envelope.id))
        ).scalars().all()
        # Only the winner's row survives — this call's own insert never committed.
        assert len(tool_calls) == 1
        assert len(ledger_rows) == 1


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_cohort_flag_off_uses_fully_legacy_idempotency(mock_interrupt, client, db_session):
    """Turn present but cohort snapshot has command_handlers_enabled=False (recorded at admission
    time, e.g. before an operator flipped the global flag) — legacy (run_id, tool_name) idempotency
    applies and no ledger row is ever written, even though a turn_id is present in config."""
    state, config, run, envelope = await _seed(client, db_session, command_handlers_enabled=False)

    command = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        tool_calls = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
        ledger_rows = (
            await db.execute(select(DraftCommandLedger).where(DraftCommandLedger.turn_id == envelope.id))
        ).scalars().all()
        assert len(tool_calls) == 1
        assert ledger_rows == []


@pytest.mark.asyncio
@patch("app.graphs.agent_tools.interrupt")
async def test_no_turn_context_uses_fully_legacy_path(mock_interrupt, client, db_session):
    """No turn_id in config at all (default flag-off cohort in this codebase today) — write_draft
    must run its 100% legacy branch regardless of any TurnExecutionState existing elsewhere."""
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    [focused] = await _focused_items(db_session, project_id, ArtifactType.VISION_OBJECTIVES)
    agent_session.focused_artifact_id = focused.id
    await db_session.commit()
    run = await _make_agent_run(db_session, agent_session)

    state = _state(artifact_type="vision_objectives")
    state["user_confirmed"] = True
    state["last_agent_run_id"] = str(run.id)
    state["focused_artifact_id"] = str(focused.id)
    config = _config(str(agent_session.id), str(project_id))
    config["configurable"]["session_factory"] = _session_factory()

    command = await _write_draft_impl("Vision", COMPLETE_BODY, state, config, "call_1")

    assert not (command.update.get("tool_errors") or [])
    mock_interrupt.assert_called_once()
    async with TestSessionFactory() as db:
        tool_calls = (
            await db.execute(select(AgentToolCall).where(AgentToolCall.run_id == run.id))
        ).scalars().all()
        assert len(tool_calls) == 1
        no_turn_ledger_rows = (
            await db.execute(select(DraftCommandLedger).where(DraftCommandLedger.tool_call_id == tool_calls[0].id))
        ).scalars().all()
        assert no_turn_ledger_rows == []
