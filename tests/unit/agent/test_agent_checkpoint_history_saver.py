"""Checkpoint v2 (`AgentCheckpointHistorySaver`): CAS-append history, real multi-checkpoint
`alist()`, and `DelegatingCheckpointer`'s cohort dispatch.

`AgentSessionCheckpointer` (v1) is completely untouched by this module — see
`test_agent_checkpointer.py` for v1's own tests and `test_agent_checkpointer_v1_compat.py` for the
fail-safe fixtures required by the phase 7 brief's v1 compatibility section.
"""

import uuid

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from sqlalchemy import select

from app.graphs.analysis.turn_outcome_projector import StaleTurnOwnershipError
from app.graphs.checkpointer import (
    AgentCheckpointHistorySaver,
    DelegatingCheckpointer,
    MissingCheckpointHistoryTurnContextError,
    StaleCheckpointAppendError,
)
from app.models.agent import (
    AgentCheckpoint,
    AgentTurnEnvelope,
    AgentTurnEvent,
    AgentTurnEventType,
    TurnExecutionState,
    TurnExecutionStatus,
)
from app.models.user import User
from tests.factories import _make_agent_session, _project, _session_factory


async def _seed_v2_turn(db_session, agent_session, *, owner_id: str = "owner-a", generation: int = 1):
    agent_session.checkpoint_version = "v2"
    await db_session.commit()

    user = User(email=f"checkpoint-v2-{uuid.uuid4()}@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()
    envelope = AgentTurnEnvelope(
        session_id=agent_session.id,
        session_sequence=1,
        original_trigger_id=uuid.uuid4(),
        actor_id=user.id,
        cohort={},
        correlation_id=str(uuid.uuid4()),
    )
    db_session.add(envelope)
    await db_session.flush()
    state = TurnExecutionState(
        turn_id=envelope.id,
        status=TurnExecutionStatus.RUNNING,
        owner_id=owner_id,
        ownership_generation=generation,
    )
    db_session.add(state)
    await db_session.commit()
    return envelope


def _checkpoint() -> dict:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"messages": [{"role": "user", "content": "Hello"}]}
    checkpoint["channel_versions"] = {"messages": "1"}
    checkpoint["versions_seen"] = {"analyze": {"messages": "1"}}
    return checkpoint


def _config(
    session_id: uuid.UUID,
    *,
    turn_id: uuid.UUID,
    owner_id: str = "owner-a",
    ownership_generation: int = 1,
    parent_checkpoint_id: str | None = None,
) -> dict:
    configurable = {
        "thread_id": str(session_id),
        "turn_id": str(turn_id),
        "turn_owner_id": owner_id,
        "turn_ownership_generation": ownership_generation,
        "checkpoint_version": "v2",
    }
    if parent_checkpoint_id is not None:
        configurable["checkpoint_id"] = parent_checkpoint_id
    return {"configurable": configurable}


@pytest.mark.asyncio
async def test_aput_appends_first_checkpoint_with_null_parent(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_v2_turn(db_session, session)
    saver = AgentCheckpointHistorySaver(session_id=str(session.id), session_factory=_session_factory())
    checkpoint = _checkpoint()

    result_config = await saver.aput(
        _config(session.id, turn_id=envelope.id), checkpoint, {"step": 1}, {"messages": "1"}
    )

    assert result_config["configurable"]["checkpoint_id"] == checkpoint["id"]
    rows = (
        await db_session.execute(select(AgentCheckpoint).where(AgentCheckpoint.session_id == session.id))
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].parent_checkpoint_id is None
    assert rows[0].turn_id == envelope.id
    assert rows[0].ownership_generation == 1


@pytest.mark.asyncio
async def test_aput_chains_second_checkpoint_onto_first_as_parent(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_v2_turn(db_session, session)
    saver = AgentCheckpointHistorySaver(session_id=str(session.id), session_factory=_session_factory())

    first = _checkpoint()
    config1 = await saver.aput(_config(session.id, turn_id=envelope.id), first, {}, {})

    second = _checkpoint()
    second["id"] = str(uuid.uuid4())
    config2 = await saver.aput(
        _config(
            session.id,
            turn_id=envelope.id,
            parent_checkpoint_id=config1["configurable"]["checkpoint_id"],
        ),
        second,
        {},
        {},
    )

    assert config2["configurable"]["checkpoint_id"] == second["id"]
    rows = (
        await db_session.execute(
            select(AgentCheckpoint).where(AgentCheckpoint.session_id == session.id).order_by(AgentCheckpoint.created_at)
        )
    ).scalars().all()
    assert len(rows) == 2
    assert rows[1].parent_checkpoint_id == rows[0].checkpoint_id


@pytest.mark.asyncio
async def test_aput_rejects_stale_parent_checkpoint_id(client, db_session):
    """A caller replaying an outdated `checkpoint_id` (a stale owner, or a fork attempt) must never
    silently overwrite or fork the linear history chain."""
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_v2_turn(db_session, session)
    saver = AgentCheckpointHistorySaver(session_id=str(session.id), session_factory=_session_factory())

    await saver.aput(_config(session.id, turn_id=envelope.id), _checkpoint(), {}, {})

    forked = _checkpoint()
    forked["id"] = str(uuid.uuid4())
    with pytest.raises(StaleCheckpointAppendError):
        await saver.aput(
            _config(session.id, turn_id=envelope.id, parent_checkpoint_id="not-the-real-head"),
            forked,
            {},
            {},
        )

    rows = (
        await db_session.execute(select(AgentCheckpoint).where(AgentCheckpoint.session_id == session.id))
    ).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_aput_rejects_stale_ownership_generation(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_v2_turn(db_session, session, generation=5)
    saver = AgentCheckpointHistorySaver(session_id=str(session.id), session_factory=_session_factory())

    with pytest.raises(StaleTurnOwnershipError):
        await saver.aput(
            _config(session.id, turn_id=envelope.id, ownership_generation=1),
            _checkpoint(),
            {},
            {},
        )

    rows = (
        await db_session.execute(select(AgentCheckpoint).where(AgentCheckpoint.session_id == session.id))
    ).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_aput_without_turn_context_raises_missing_context_error(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    session.checkpoint_version = "v2"
    await db_session.commit()
    saver = AgentCheckpointHistorySaver(session_id=str(session.id), session_factory=_session_factory())

    with pytest.raises(MissingCheckpointHistoryTurnContextError):
        await saver.aput({"configurable": {"thread_id": str(session.id)}}, _checkpoint(), {}, {})


@pytest.mark.asyncio
async def test_aput_emits_checkpoint_appended_event(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_v2_turn(db_session, session)
    saver = AgentCheckpointHistorySaver(session_id=str(session.id), session_factory=_session_factory())

    await saver.aput(_config(session.id, turn_id=envelope.id), _checkpoint(), {}, {})

    events = (
        await db_session.execute(select(AgentTurnEvent).where(AgentTurnEvent.session_id == session.id))
    ).scalars().all()
    assert len(events) == 1
    assert events[0].event_type == AgentTurnEventType.CHECKPOINT_APPENDED
    assert events[0].session_sequence == 1
    # The saver commits through its own session_factory-opened session, so db_session's identity
    # map holds a stale copy of this row until explicitly refreshed.
    await db_session.refresh(session)
    assert session.event_cursor == 1


@pytest.mark.asyncio
async def test_aput_writes_scopes_pending_writes_to_the_head_row(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_v2_turn(db_session, session)
    saver = AgentCheckpointHistorySaver(session_id=str(session.id), session_factory=_session_factory())
    config = await saver.aput(_config(session.id, turn_id=envelope.id), _checkpoint(), {}, {})

    await saver.aput_writes(config, [("messages", [{"role": "user", "content": "ok"}])], task_id="task-1")

    tuples = [item async for item in saver.alist(config)]
    assert len(tuples) == 1
    assert tuples[0].pending_writes == [("task-1", "messages", [{"role": "user", "content": "ok"}])]


@pytest.mark.asyncio
async def test_alist_returns_full_history_newest_first(client, db_session):
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_v2_turn(db_session, session)
    saver = AgentCheckpointHistorySaver(session_id=str(session.id), session_factory=_session_factory())

    first = _checkpoint()
    config1 = await saver.aput(_config(session.id, turn_id=envelope.id), first, {}, {})
    second = _checkpoint()
    second["id"] = str(uuid.uuid4())
    await saver.aput(
        _config(session.id, turn_id=envelope.id, parent_checkpoint_id=config1["configurable"]["checkpoint_id"]),
        second,
        {},
        {},
    )

    tuples = [item async for item in saver.alist(None)]
    assert [t.checkpoint["id"] for t in tuples] == [second["id"], first["id"]]
    assert tuples[0].parent_config["configurable"]["checkpoint_id"] == first["id"]


@pytest.mark.asyncio
async def test_delegating_checkpointer_dispatches_on_checkpoint_version(client, db_session):
    project_id = await _project(client)
    v1_session = await _make_agent_session(client, db_session, project_id)
    v2_session = await _make_agent_session(client, db_session, project_id)
    envelope = await _seed_v2_turn(db_session, v2_session)
    delegator = DelegatingCheckpointer(_session_factory())

    v1_for = delegator._for({"configurable": {"thread_id": str(v1_session.id)}})
    v2_for = delegator._for(
        {
            "configurable": {
                "thread_id": str(v2_session.id),
                "checkpoint_version": "v2",
                "turn_id": str(envelope.id),
                "turn_owner_id": "owner-a",
                "turn_ownership_generation": 1,
            }
        }
    )

    assert type(v1_for).__name__ == "AgentSessionCheckpointer"
    assert isinstance(v2_for, AgentCheckpointHistorySaver)
