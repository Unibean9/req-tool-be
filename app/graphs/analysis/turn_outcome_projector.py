"""Terminal and non-terminal `AgentSession.status` projector.

ADR 0001: `TurnOutcome` is the sole owner of the turn decision; `AgentSession.status` and SSE
are compatibility projections of it. `project_terminal_outcome` is the one function allowed to set
`AgentSession.status` to a terminal value (`COMPLETED`/`FAILED`/`TURN_FAILED`/`EXPIRED`), and
`project_non_terminal_outcome` is the one function allowed to set it to a non-terminal
waiting/continuing value (`ACTIVE`/`WAITING_FOR_HUMAN` with the matching `interrupt_type`) — every
call site that used to assign one of those directly now routes through the matching function
instead.

This does not change *when* a branch is terminal vs. non-terminal (the existing no-outcome vs.
forced-stop distinction is untouched) — only *how* the status write happens, and only additively records
a `TurnOutcome` row when the admitting turn's cohort has `turn_outcomes_enabled=True` (snapshotted
once at admission, never read live from settings). No turn context, or the flag off for that
turn's cohort, means only the compatibility status write happens — byte-identical to the direct
write it replaces.

The caller still owns the transaction: these functions only mutate `session_row` and add the
optional `TurnOutcome` row to `db`; they never flush or commit.
"""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.graphs.analysis.turn_event_log import emit_turn_event, latest_checkpoint_id_for_session
from app.models.agent import (
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentTurnEnvelope,
    AgentTurnEventType,
    TurnExecutionState,
    TurnOutcome,
    TurnOutcomeType,
)

logger = logging.getLogger(__name__)


class StaleTurnOwnershipError(Exception):
    """Raised when an outcome projection is attempted by an owner/generation pair that no longer
    matches `TurnExecutionState.ownership_generation` for the turn.

    This is a plain exception, not an `HTTPException`: callers of the outcome projector are not
    always inside an HTTP request context (a durable worker loop is the reason this fence exists),
    so the caller is responsible for catching this and deciding how to log/audit it.
    """

    def __init__(
        self,
        *,
        turn_id: uuid.UUID | None,
        owner_id: str,
        expected_generation: int,
        actual_generation: int | None,
    ) -> None:
        self.turn_id = turn_id
        self.owner_id = owner_id
        self.expected_generation = expected_generation
        self.actual_generation = actual_generation
        super().__init__(
            f"stale turn ownership: turn_id={turn_id} owner_id={owner_id} "
            f"expected_generation={expected_generation} actual_generation={actual_generation}"
        )

# Compatibility mapping for every outcome type this codebase currently commits terminal status
# for. `cancelled` maps to `EXPIRED` because that is the one remaining
# direct-terminal-write call site (lazy session abandonment) with no better existing status value —
# an abandoned session was never explicitly failed or completed, it was given up on, which is closer
# to "cancelled" than to a failure or a completion.
_STATUS_BY_OUTCOME: dict[TurnOutcomeType, AgentSessionStatus] = {
    TurnOutcomeType.COMPLETED: AgentSessionStatus.COMPLETED,
    TurnOutcomeType.TERMINAL_FAILURE: AgentSessionStatus.FAILED,
    TurnOutcomeType.RECOVERABLE_FAILURE: AgentSessionStatus.TURN_FAILED,
    TurnOutcomeType.CANCELLED: AgentSessionStatus.EXPIRED,
}

# The subset of TurnOutcomeType that represents a committed terminal transition. Callers that need
# to distinguish "a terminal outcome landed" from "any outcome row exists" (e.g. crash-window
# reconciliation, which must not treat a non-terminal WAIT_APPROVAL/WAIT_INPUT/CONTINUE row as proof
# the turn concluded) filter on this set rather than re-deriving it.
TERMINAL_OUTCOME_TYPES = frozenset(_STATUS_BY_OUTCOME.keys())

# Compatibility mapping for the non-terminal outcomes: each pair is the exact
# (status, interrupt_type) shape the two direct-write call sites this replaces already produced,
# so routing through this table changes nothing observable in REST/SSE payloads.
_STATUS_AND_INTERRUPT_BY_OUTCOME: dict[TurnOutcomeType, tuple[AgentSessionStatus, AgentSessionInterruptType | None]] = {
    TurnOutcomeType.CONTINUE: (AgentSessionStatus.ACTIVE, None),
    TurnOutcomeType.WAIT_INPUT: (AgentSessionStatus.WAITING_FOR_HUMAN, AgentSessionInterruptType.ASK_HUMAN),
    TurnOutcomeType.WAIT_APPROVAL: (
        AgentSessionStatus.WAITING_FOR_HUMAN,
        AgentSessionInterruptType.PROPOSE_ARTIFACTS,
    ),
    TurnOutcomeType.DIRECT_RESPONSE: (AgentSessionStatus.ACTIVE, AgentSessionInterruptType.STREAM_RESPONSE),
}


async def _record_outcome_if_enabled(
    db: AsyncSession,
    session_row: AgentSession,
    turn_id: uuid.UUID | None,
    outcome_type: TurnOutcomeType,
    reason: str | None,
) -> None:
    """Additively record a `TurnOutcome` audit row, gated by the turn's cohort snapshot.

    No-ops (with no side effect other than a log line) whenever there is no turn context, the
    cohort has the flag off, or a row for this turn already exists — including the race where a
    concurrent writer wins the unique-turn constraint first. Takes the whole session row (not just
    its id) so the legacy no-turn-context path never touches `session_row.id` — some call sites
    exercise that path with a bare status/interrupt_type double that has no `id` attribute.
    """
    if turn_id is None:
        return
    session_id = session_row.id
    envelope = await db.get(AgentTurnEnvelope, turn_id)
    outcomes_enabled = bool((envelope.cohort or {}).get("turn_outcomes_enabled")) if envelope else False
    if not outcomes_enabled:
        return

    existing = (
        await db.execute(select(TurnOutcome).where(TurnOutcome.turn_id == turn_id))
    ).scalar_one_or_none()
    if existing is not None:
        logger.info("turn_outcome_duplicate_noop turn_id=%s session_id=%s", turn_id, session_id)
        return
    try:
        # Flush inside a savepoint so a concurrent winner of the unique turn fence does
        # not poison the caller's transaction; the winner remains the authoritative row.
        async with db.begin_nested():
            db.add(
                TurnOutcome(
                    turn_id=turn_id,
                    session_id=session_id,
                    outcome_type=outcome_type,
                    reason=reason,
                )
            )
            await db.flush()
    except IntegrityError:
        logger.info("turn_outcome_race_duplicate_noop turn_id=%s session_id=%s", turn_id, session_id)
        return
    logger.info(
        "turn_outcome_committed outcome=%s reason=%s turn_id=%s session_id=%s",
        outcome_type.value,
        reason,
        turn_id,
        session_id,
    )


async def _emit_outcome_event_if_v2(
    db: AsyncSession,
    session_row: AgentSession,
    turn_id: uuid.UUID | None,
    outcome_type: TurnOutcomeType,
    reason: str | None,
) -> None:
    """Append an `OUTCOME_COMMITTED` outbox row alongside the status write, for v2-cohort sessions
    only (`emit_turn_event` no-ops for v1). Skips the extra checkpoint-head lookup entirely for the
    common v1 case.

    Uses `getattr` (not direct attribute access) because several existing call sites pass a
    lightweight duck-typed `session_row` stand-in (not a real `AgentSession`) that predates this
    cohort field — those callers have no turn-v2 concept at all and must keep behaving exactly like
    v1 (no-op) rather than raising `AttributeError`.
    """
    if getattr(session_row, "checkpoint_version", "v1") != "v2":
        return
    parent_checkpoint_id = await latest_checkpoint_id_for_session(db, session_row.id)
    await emit_turn_event(
        db,
        session_row=session_row,
        turn_id=turn_id,
        event_type=AgentTurnEventType.OUTCOME_COMMITTED,
        parent_checkpoint_id=parent_checkpoint_id,
        payload={"outcome_type": outcome_type.value, "reason": reason},
    )


async def _check_ownership_fence(
    db: AsyncSession,
    turn_id: uuid.UUID | None,
    owner_id: str | None,
    expected_ownership_generation: int | None,
) -> None:
    """Raise `StaleTurnOwnershipError` if the caller's `owner_id`/`expected_ownership_generation`
    no longer match `TurnExecutionState.ownership_generation` for `turn_id`.

    A no-op whenever either optional parameter is `None`: the caller is not running under any
    execution fence (e.g. the current inline execution path with no claimed job), so behavior stays
    exactly as it was before this fence existed — this is the byte-identical-for-`None` guarantee
    the fence must not break.
    """
    if owner_id is None or expected_ownership_generation is None:
        return
    state = (
        await db.execute(
            select(TurnExecutionState).where(TurnExecutionState.turn_id == turn_id).with_for_update()
        )
    ).scalar_one_or_none()
    actual_generation = state.ownership_generation if state is not None else None
    if state is None or state.ownership_generation != expected_ownership_generation:
        raise StaleTurnOwnershipError(
            turn_id=turn_id,
            owner_id=owner_id,
            expected_generation=expected_ownership_generation,
            actual_generation=actual_generation,
        )


async def check_ownership_fence(
    db: AsyncSession,
    turn_id: uuid.UUID | None,
    owner_id: str | None,
    expected_ownership_generation: int | None,
) -> None:
    """Public fence check for call sites that mutate `session_row.status`/`interrupt_type` directly
    without going through `project_terminal_outcome`/`project_non_terminal_outcome` — e.g. a
    non-terminal pair with no matching `TurnOutcomeType` entry in `_STATUS_AND_INTERRUPT_BY_OUTCOME`.

    Same `StaleTurnOwnershipError`/no-op-when-`None` contract as the fence embedded in those two
    functions. A pure pass-through to `_check_ownership_fence`, with no `TurnOutcome` audit row —
    unlike the other two, there is no outcome type here to record.
    """
    await _check_ownership_fence(db, turn_id, owner_id, expected_ownership_generation)


async def project_terminal_outcome(
    db: AsyncSession,
    session_row: AgentSession,
    outcome_type: TurnOutcomeType,
    reason: str | None,
    *,
    turn_id: uuid.UUID | None = None,
    owner_id: str | None = None,
    expected_ownership_generation: int | None = None,
) -> None:
    """Set `session_row.status` to the terminal value `outcome_type` maps to.

    Does not touch `session_row.interrupt_type` — callers keep deciding that themselves, exactly as
    before this refactor, since it is not uniform across the call sites this replaces.

    `owner_id`/`expected_ownership_generation` are optional: when both are given, the ownership
    fence is checked (in the same transaction as this status write, via the caller's `db`) before
    anything is mutated — see `_check_ownership_fence`. Leaving either as `None` (the default)
    skips the fence entirely.
    """
    status = _STATUS_BY_OUTCOME.get(outcome_type)
    if status is None:
        raise ValueError(f"{outcome_type!r} is not a terminal outcome type")

    await _check_ownership_fence(db, turn_id, owner_id, expected_ownership_generation)
    await _record_outcome_if_enabled(db, session_row, turn_id, outcome_type, reason)
    session_row.status = status
    await _emit_outcome_event_if_v2(db, session_row, turn_id, outcome_type, reason)


async def project_non_terminal_outcome(
    db: AsyncSession,
    session_row: AgentSession,
    outcome_type: TurnOutcomeType,
    reason: str | None = None,
    *,
    turn_id: uuid.UUID | None = None,
    owner_id: str | None = None,
    expected_ownership_generation: int | None = None,
) -> None:
    """Set `session_row.status`/`interrupt_type` to the non-terminal pair `outcome_type` maps to.

    Unlike `project_terminal_outcome`, this also sets `interrupt_type` because the two call sites
    it replaces always assign both fields together for a given outcome.

    See `project_terminal_outcome` for the optional `owner_id`/`expected_ownership_generation`
    fence contract — identical here.
    """
    mapped = _STATUS_AND_INTERRUPT_BY_OUTCOME.get(outcome_type)
    if mapped is None:
        raise ValueError(f"{outcome_type!r} is not a non-terminal outcome type")
    status, interrupt_type = mapped

    await _check_ownership_fence(db, turn_id, owner_id, expected_ownership_generation)
    await _record_outcome_if_enabled(db, session_row, turn_id, outcome_type, reason)
    session_row.status = status
    session_row.interrupt_type = interrupt_type
    await _emit_outcome_event_if_v2(db, session_row, turn_id, outcome_type, reason)
