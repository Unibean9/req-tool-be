"""Append-only outbox for committed logical-turn transitions (checkpoint v2 cohort only).

`AgentTurnEvent` rows are a side effect of an already-committed transition — never a separate
best-effort write and never a second source of truth. `emit_turn_event` must be called inside the
same transaction as the transition it records, before the caller's own `db.commit()`; it never
commits or flushes past a savepoint itself, mirroring `turn_outcome_projector`'s "caller owns the
transaction" contract.

Only sessions on the `checkpoint_version == "v2"` cohort ever get an event row appended, matching
the cohort-snapshot discipline used across every other new control-plane surface in this codebase —
a v1 session calling either projector function here still behaves byte-identically to before this
module existed. Retention/expiry policy and a general read-model projector are intentionally out of
scope for this pass: this module stores redacted history for authorized replay, nothing more, until
a separate rollout defines retention.
"""

import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import AgentCheckpoint, AgentSession, AgentTurnEvent, AgentTurnEventType

logger = logging.getLogger(__name__)


class TurnEventAuthorizationError(Exception):
    """Raised when a caller requests turn events for a session outside its own project/ownership.

    A plain exception (not `HTTPException`): this reader has no single HTTP call site yet, so the
    caller decides how to surface it, same reasoning as `StaleTurnOwnershipError`.
    """

    def __init__(self, *, session_id: uuid.UUID, project_id: uuid.UUID) -> None:
        self.session_id = session_id
        self.project_id = project_id
        super().__init__(f"session_id={session_id} is not visible to project_id={project_id}")


def redact_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Redact nested text/tool-argument content before an event row is ever persisted.

    Reuses the same allowlist-by-key redaction `turn_audit.py` already applies to `AgentRun.
    analysis_result` — a payload here must never carry a raw prompt, secret or unredacted tool
    argument any more than that existing audited surface does. Imported lazily (not at module
    level) because `turn_audit` transitively imports `agent_tools`, which imports back into
    `turn_outcome_projector` (the module that imports this one) — a module-level import here would
    be a circular import.
    """
    from app.graphs.analysis.turn_audit import _audit_arg_value

    return {key: _audit_arg_value(str(key), value) for key, value in payload.items()}


async def latest_checkpoint_id_for_session(db: AsyncSession, session_id: uuid.UUID) -> str | None:
    """The v2 checkpoint history head's `checkpoint_id` for a session, or `None` if it has none yet.

    Used to stamp `parent_checkpoint_id` on an outcome-committed event; a plain read, not part of any
    CAS decision (the saver's own CAS append owns that).
    """
    return (
        await db.execute(
            select(AgentCheckpoint.checkpoint_id)
            .where(AgentCheckpoint.session_id == session_id)
            .order_by(AgentCheckpoint.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def emit_turn_event(
    db: AsyncSession,
    *,
    session_row: AgentSession,
    turn_id: uuid.UUID | None,
    event_type: AgentTurnEventType,
    parent_checkpoint_id: str | None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Append one ordered, redacted event row, gated by `session_row.checkpoint_version == "v2"`.

    No-ops (silently) for any v1-cohort session — v1 never gets an event log. `session_row.
    event_cursor` is the single ordering/dedup source: this function increments it and uses the new
    value as `session_sequence`, so a duplicate/racing append that collides on the same cursor value
    hits the table's unique constraint and is swallowed as a no-op rather than corrupting order.
    """
    if session_row.checkpoint_version != "v2":
        return
    next_sequence = session_row.event_cursor + 1
    try:
        async with db.begin_nested():
            db.add(
                AgentTurnEvent(
                    session_id=session_row.id,
                    turn_id=turn_id,
                    session_sequence=next_sequence,
                    event_type=event_type,
                    parent_checkpoint_id=parent_checkpoint_id,
                    payload=redact_event_payload(payload or {}),
                )
            )
            await db.flush()
    except IntegrityError:
        logger.info(
            "turn_event_race_duplicate_noop session_id=%s session_sequence=%s", session_row.id, next_sequence
        )
        return
    session_row.event_cursor = next_sequence


async def list_turn_events(
    db: AsyncSession,
    *,
    session_id: uuid.UUID,
    project_id: uuid.UUID,
    user_id: uuid.UUID | None = None,
    after_cursor: int | None = None,
    limit: int | None = None,
) -> list[AgentTurnEvent]:
    """Read events ordered by `session_sequence`, rechecking project/owner membership first.

    Never trusts a bare `session_id`: same authorization shape as `AgentService.get_session` — a
    session outside the caller's project (or, when `user_id` is given, not owned by that user) raises
    `TurnEventAuthorizationError` instead of returning any row. `after_cursor` makes delivery
    idempotent for a client that already consumed up to some cursor; a stale/expired cursor value
    just yields the same never-consumed tail deterministically, since a cursor is only an ordering
    filter here, not a token with its own separate expiry state to validate. This is additive read
    API surface, not wired into any existing SSE stream in this pass.
    """
    owner_query = select(AgentSession.id).where(AgentSession.id == session_id, AgentSession.project_id == project_id)
    if user_id is not None:
        owner_query = owner_query.where(AgentSession.created_by_id == user_id)
    owned = (await db.execute(owner_query)).scalar_one_or_none()
    if owned is None:
        raise TurnEventAuthorizationError(session_id=session_id, project_id=project_id)

    stmt = select(AgentTurnEvent).where(AgentTurnEvent.session_id == session_id)
    if after_cursor is not None:
        stmt = stmt.where(AgentTurnEvent.session_sequence > after_cursor)
    stmt = stmt.order_by(AgentTurnEvent.session_sequence.asc())
    if limit is not None:
        stmt = stmt.limit(limit)
    return list((await db.execute(stmt)).scalars().all())
