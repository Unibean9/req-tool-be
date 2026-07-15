"""Outbox/event log (`emit_turn_event`, `list_turn_events`, `redact_event_payload`):
v1 no-op, v2 cursor/ordering, redaction before persistence, and authorized replay.
"""

import uuid

import pytest
from sqlalchemy import select

from app.graphs.analysis.turn_event_log import (
    TurnEventAuthorizationError,
    emit_turn_event,
    latest_checkpoint_id_for_session,
    list_turn_events,
    redact_event_payload,
)
from app.models.agent import AgentTurnEvent, AgentTurnEventType
from tests.factories import _make_agent_session, _project


def test_redact_event_payload_omits_denylisted_text_keys():
    redacted = redact_event_payload({"content": "some raw prompt text", "outcome_type": "completed"})

    assert redacted["outcome_type"] == "completed"
    assert redacted["content"]["omitted"] is True
    assert "some raw prompt text" not in str(redacted)


@pytest.mark.asyncio
async def test_emit_turn_event_is_a_noop_for_v1_sessions(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    assert session.checkpoint_version == "v1"

    await emit_turn_event(
        db_session,
        session_row=session,
        turn_id=None,
        event_type=AgentTurnEventType.CHECKPOINT_APPENDED,
        parent_checkpoint_id=None,
        payload={"x": 1},
    )
    await db_session.commit()

    rows = (
        await db_session.execute(select(AgentTurnEvent).where(AgentTurnEvent.session_id == session.id))
    ).scalars().all()
    assert rows == []
    assert session.event_cursor == 0


@pytest.mark.asyncio
async def test_emit_turn_event_increments_cursor_and_redacts_for_v2_sessions(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    session.checkpoint_version = "v2"
    await db_session.commit()

    await emit_turn_event(
        db_session,
        session_row=session,
        turn_id=None,
        event_type=AgentTurnEventType.OUTCOME_COMMITTED,
        parent_checkpoint_id="cp-1",
        payload={"content": "raw text", "outcome_type": "completed"},
    )
    await db_session.commit()

    assert session.event_cursor == 1
    rows = (
        await db_session.execute(select(AgentTurnEvent).where(AgentTurnEvent.session_id == session.id))
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].session_sequence == 1
    assert rows[0].payload["content"]["omitted"] is True
    assert rows[0].payload["outcome_type"] == "completed"


@pytest.mark.asyncio
async def test_emit_turn_event_dedups_a_race_on_the_same_sequence(client, db_session):
    """Two callers computing the same next-cursor value must not duplicate/reorder — the unique
    constraint on (session_id, session_sequence) rejects the loser, swallowed as a no-op."""
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    session.checkpoint_version = "v2"
    await db_session.commit()

    db_session.add(
        AgentTurnEvent(
            session_id=session.id,
            turn_id=None,
            session_sequence=1,
            event_type=AgentTurnEventType.CHECKPOINT_APPENDED,
            parent_checkpoint_id=None,
            payload={},
        )
    )
    await db_session.commit()
    # session.event_cursor is still 0 in this row's own in-memory state — simulating a racing
    # caller that read the pre-increment cursor value concurrently with the row above.

    await emit_turn_event(
        db_session,
        session_row=session,
        turn_id=None,
        event_type=AgentTurnEventType.OUTCOME_COMMITTED,
        parent_checkpoint_id=None,
        payload={},
    )
    await db_session.commit()

    rows = (
        await db_session.execute(
            select(AgentTurnEvent).where(AgentTurnEvent.session_id == session.id, AgentTurnEvent.session_sequence == 1)
        )
    ).scalars().all()
    assert len(rows) == 1
    # The race loser never advanced the cursor past the value it collided on.
    assert session.event_cursor == 0


@pytest.mark.asyncio
async def test_latest_checkpoint_id_for_session_is_none_with_no_history(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)

    assert await latest_checkpoint_id_for_session(db_session, session.id) is None


@pytest.mark.asyncio
async def test_list_turn_events_orders_by_sequence_and_respects_after_cursor(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    session.checkpoint_version = "v2"
    await db_session.commit()

    for _ in range(3):
        await emit_turn_event(
            db_session,
            session_row=session,
            turn_id=None,
            event_type=AgentTurnEventType.CHECKPOINT_APPENDED,
            parent_checkpoint_id=None,
            payload={},
        )
    await db_session.commit()

    all_events = await list_turn_events(db_session, session_id=session.id, project_id=project_id)
    assert [e.session_sequence for e in all_events] == [1, 2, 3]

    tail = await list_turn_events(db_session, session_id=session.id, project_id=project_id, after_cursor=1)
    assert [e.session_sequence for e in tail] == [2, 3]


@pytest.mark.asyncio
async def test_list_turn_events_rejects_cross_project_read(client, db_session):
    project_id = await _project(client)
    other_project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)

    with pytest.raises(TurnEventAuthorizationError):
        await list_turn_events(db_session, session_id=session.id, project_id=other_project_id)


@pytest.mark.asyncio
async def test_list_turn_events_rejects_wrong_owner_when_user_id_given(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    session.created_by_id = uuid.uuid4()
    await db_session.commit()

    with pytest.raises(TurnEventAuthorizationError):
        await list_turn_events(
            db_session, session_id=session.id, project_id=project_id, user_id=uuid.uuid4()
        )
