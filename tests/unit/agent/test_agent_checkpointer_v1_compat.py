"""v1 compatibility fixtures (phase 7 brief, section 6): a paused v1 session must keep resuming
through the legacy `AgentSessionCheckpointer` reader — byte-identical — now that `AgentCheckpointHistorySaver`
(v2) and the `agent_checkpoints`/`agent_turn_events` tables exist in the same schema. Malformed
payload, a checkpoint_id/pending-writes mismatch, and an undecodable serde blob must all fail safe
(raise or exclude the stale data) rather than silently completing or resuming into a wrong state.
"""

import base64
import uuid

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from sqlalchemy import select

from app.graphs.checkpointer import AgentSessionCheckpointer, DelegatingCheckpointer
from app.models.agent import AgentSession
from tests.factories import _make_agent_session, _project, _session_factory


def _checkpoint() -> dict:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"messages": [{"role": "user", "content": "Hello"}]}
    checkpoint["channel_versions"] = {"messages": "1"}
    checkpoint["versions_seen"] = {"analyze": {"messages": "1"}}
    return checkpoint


def _config(session_id: uuid.UUID) -> dict:
    return {"configurable": {"thread_id": str(session_id)}}


@pytest.mark.asyncio
async def test_v1_session_paused_before_v2_release_still_resumes_through_legacy_reader(client, db_session):
    """A session admitted before checkpoint v2 existed keeps `checkpoint_version == "v1"` (the
    column's `server_default`) and must be routed to `AgentSessionCheckpointer` by
    `DelegatingCheckpointer`, exactly as if `AgentCheckpointHistorySaver`/v2 tables did not exist —
    the v2 cohort must never be inferred from checkpoint shape."""
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    assert session.checkpoint_version == "v1"

    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    checkpoint = _checkpoint()
    await checkpointer.aput(_config(session.id), checkpoint, {"source": "paused"}, {})

    delegator = DelegatingCheckpointer(_session_factory())
    resolved = delegator._for({"configurable": {"thread_id": str(session.id)}})
    assert type(resolved).__name__ == "AgentSessionCheckpointer"

    loaded = await resolved.aget_tuple(_config(session.id))
    assert loaded is not None
    assert loaded.checkpoint == checkpoint


@pytest.mark.asyncio
async def test_malformed_serialized_payload_raises_instead_of_silently_resuming(client, db_session):
    """A corrupted `data`/`serde_type` pair (e.g. truncated write, bit rot) must raise on decode —
    never silently return an empty/partial checkpoint that a caller could mistake for a fresh
    session or a valid resume point."""
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    session.graph_checkpoint = {
        "data": base64.b64encode(b"not a valid serialized checkpoint").decode("ascii"),
        "serde_type": "json",
        "metadata": {},
        "new_versions": {},
        "checkpoint_id": "cp-corrupt",
        "pending_writes": [],
    }
    await db_session.commit()

    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())

    with pytest.raises(Exception):
        await checkpointer.aget_tuple(_config(session.id))


@pytest.mark.asyncio
async def test_pending_writes_from_a_superseded_checkpoint_id_are_excluded_not_applied(client, db_session):
    """Pending writes stamped with a `checkpoint_id` that no longer matches the session's current
    head must never leak into the next resume — `aget_tuple` filters them out rather than
    surfacing stale interrupt/tool-write state as if it belonged to the live checkpoint."""
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    checkpoint = _checkpoint()
    config = await checkpointer.aput(_config(session.id), checkpoint, {}, {})

    # A pending write tagged with a checkpoint_id from a prior (superseded) checkpoint.
    async with _session_factory()() as db:
        stmt = select(AgentSession).where(AgentSession.id == session.id)
        row = (await db.execute(stmt)).scalar_one()
        payload = dict(row.graph_checkpoint)
        payload["pending_writes"] = [
            {
                "task_id": "stale-task",
                "channel": "__interrupt__",
                "serde_type": "json",
                "data": base64.b64encode(b'"stale"').decode("ascii"),
                "checkpoint_id": "some-superseded-checkpoint-id",
            }
        ]
        row.graph_checkpoint = payload
        await db.commit()

    loaded = await checkpointer.aget_tuple(config)

    assert loaded is not None
    assert loaded.pending_writes == []


@pytest.mark.asyncio
async def test_duplicate_pending_interrupt_writes_for_the_same_checkpoint_are_all_visible(client, db_session):
    """Two pending writes tagged with the SAME (current) checkpoint_id — e.g. a duplicate
    interrupt delivery — must both surface through `aget_tuple`'s `pending_writes`; the reader does
    not itself dedupe (that responsibility sits with `AgentService._resume_command`'s interrupt-id
    walk), so this fixture only asserts the reader fails safe by preserving both rather than
    silently dropping one."""
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())
    checkpoint = _checkpoint()
    config = await checkpointer.aput(_config(session.id), checkpoint, {}, {})

    await checkpointer.aput_writes(
        config, [("messages", [{"role": "user", "content": "first"}])], task_id="task-1"
    )
    await checkpointer.aput_writes(
        config, [("messages", [{"role": "user", "content": "duplicate"}])], task_id="task-1"
    )

    loaded = await checkpointer.aget_tuple(config)
    assert len(loaded.pending_writes) == 2


@pytest.mark.asyncio
async def test_unknown_serde_type_raises_instead_of_guessing_a_schema(client, db_session):
    """An unrecognized `serde_type` (e.g. from a future/rolled-back schema version) must raise
    through the checkpoint serde, not be guessed at or defaulted into an empty checkpoint."""
    project_id = await _project(client)
    session = await _make_agent_session(client, db_session, project_id)
    session.graph_checkpoint = {
        "data": base64.b64encode(b"{}").decode("ascii"),
        "serde_type": "unknown-future-schema-v99",
        "metadata": {},
        "new_versions": {},
        "checkpoint_id": "cp-unknown",
        "pending_writes": [],
    }
    await db_session.commit()

    checkpointer = AgentSessionCheckpointer(session_id=str(session.id), session_factory=_session_factory())

    with pytest.raises(Exception):
        await checkpointer.aget_tuple(_config(session.id))
