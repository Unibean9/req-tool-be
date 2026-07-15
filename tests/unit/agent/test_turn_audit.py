"""`record_run_and_dispatch` attribution and redaction contract (Phase 8 Step 2 + 3).

Two invariants under test:

1. Correlation join: given an `AgentTurnEnvelope.correlation_id`, the new `AgentRun.turn_id`
   attribution column lets on-call reach the run, and the turn also reaches its `AgentTurnEvent`,
   `TurnOutcome`, and `DraftCommandLedger` rows — i.e. trigger -> turn -> attempt -> command ->
   outcome -> event is a real, queryable chain, not aspirational.
2. Redaction: no raw value for any `_AUDIT_TEXT_ARG_KEYS` key survives into the persisted
   `AgentRun.analysis_result` — `record_run_and_dispatch` already redacts via `_audit_arg_value`
   (this test proves it holds end-to-end, not just on the helper in isolation).
"""

import uuid

import pytest
from sqlalchemy import select

from app.graphs.analysis.turn_audit import _AUDIT_TEXT_ARG_KEYS, record_run_and_dispatch
from app.models.agent import (
    AgentRun,
    AgentTurnEnvelope,
    AgentTurnEvent,
    AgentTurnEventType,
    DraftCommandLedger,
    TurnOutcome,
    TurnOutcomeType,
)
from app.models.user import User
from tests.factories import _make_agent_session, _project, _session_factory


async def _seed_envelope(db_session, agent_session, *, correlation_id: str) -> AgentTurnEnvelope:
    user = User(email=f"turn-audit-{uuid.uuid4()}@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()
    envelope = AgentTurnEnvelope(
        session_id=agent_session.id,
        session_sequence=1,
        original_trigger_id=uuid.uuid4(),
        actor_id=user.id,
        cohort={},
        correlation_id=correlation_id,
    )
    db_session.add(envelope)
    await db_session.commit()
    return envelope


@pytest.mark.asyncio
async def test_correlation_id_reaches_run_event_outcome_and_command_via_turn_id(client, db_session):
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    correlation_id = f"corr-{uuid.uuid4()}"
    envelope = await _seed_envelope(db_session, agent_session, correlation_id=correlation_id)

    run = AgentRun(session_id=agent_session.id, turn_id=envelope.id, analysis_result={})
    event = AgentTurnEvent(
        session_id=agent_session.id,
        turn_id=envelope.id,
        session_sequence=1,
        event_type=AgentTurnEventType.CHECKPOINT_APPENDED,
        payload={},
    )
    outcome = TurnOutcome(turn_id=envelope.id, session_id=agent_session.id, outcome_type=TurnOutcomeType.COMPLETED)
    command = DraftCommandLedger(
        turn_id=envelope.id,
        logical_command_id=f"cmd-{uuid.uuid4()}",
        action_type="write_draft",
    )
    db_session.add_all([run, event, outcome, command])
    await db_session.commit()

    # Start the join from correlation_id only, as on-call would.
    joined_envelope = (
        await db_session.execute(select(AgentTurnEnvelope).where(AgentTurnEnvelope.correlation_id == correlation_id))
    ).scalar_one()
    turn_id = joined_envelope.id

    joined_run = (await db_session.execute(select(AgentRun).where(AgentRun.turn_id == turn_id))).scalar_one()
    joined_event = (await db_session.execute(select(AgentTurnEvent).where(AgentTurnEvent.turn_id == turn_id))).scalar_one()
    joined_outcome = (await db_session.execute(select(TurnOutcome).where(TurnOutcome.turn_id == turn_id))).scalar_one()
    joined_command = (
        await db_session.execute(select(DraftCommandLedger).where(DraftCommandLedger.turn_id == turn_id))
    ).scalar_one()

    assert joined_run.id == run.id
    assert joined_event.id == event.id
    assert joined_outcome.id == outcome.id
    assert joined_command.id == command.id


@pytest.mark.asyncio
async def test_record_run_and_dispatch_redacts_configured_text_arg_keys(client, db_session):
    assert "body" in _AUDIT_TEXT_ARG_KEYS
    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)
    secret = "SECRET-do-not-persist-verbatim-9f3a"

    run_id, analysis_result, dispatched_tools, _dispatched_tool_calls = await record_run_and_dispatch(
        session_factory=_session_factory(),
        session_id=agent_session.id,
        analysis_result_base={
            "tools": [],
            "model_tool_calls": [],
            "raw_model_tool_calls": [],
            "dropped_tool_calls": [],
            "available_tools": [],
            "locale": "en",
            "coverage_complete": True,
            "session_phase": None,
        },
        token_usage={"input": 10, "output": 5, "total": 15},
        latency_ms=1,
        gated_tools=[{"name": "custom_tool", "args": {"body": secret, "other": "kept"}}],
        direct_response="",
        locale="en",
    )

    # `dispatched_tools`/`dispatched_tool_calls` legitimately carry the raw args — they are what
    # actually gets executed. Only `analysis_result` (the audit/persistence copy) must be redacted.
    assert dispatched_tools[0]["args"]["body"] == secret

    assert secret not in str(analysis_result)

    persisted = (
        await db_session.execute(select(AgentRun).where(AgentRun.id == uuid.UUID(run_id)))
    ).scalar_one()
    assert secret not in str(persisted.analysis_result)
    # The non-secret sibling key must survive untouched — proves this is redaction, not blanket
    # scrubbing of every arg.
    assert "kept" in str(persisted.analysis_result)
