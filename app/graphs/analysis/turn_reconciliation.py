"""Crash-window reconciliation for checkpoint v2 resume.

Compares a turn's committed `TurnOutcome` against its v2 checkpoint history head to detect the
crash windows ADR 0001's replay/effect/fencing section names: a business effect/outbox committed
before the checkpoint landed, a checkpoint committed before outcome projection ran, or a stale fence
generation appended to an active session. This is a single targeted check for the v2 resume path
only — v1 sessions never call it and keep their existing resume behavior unchanged.

Not a general saga/reconciliation framework: it only distinguishes "safe to resume", "already
consistent" and "needs an operator". The caller must never silently resume or re-invoke the model on
a `NEEDS_OPERATOR` result.
"""

import enum
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.graphs.analysis.turn_outcome_projector import TERMINAL_OUTCOME_TYPES
from app.models.agent import AgentCheckpoint, TurnOutcome


class ReconciliationOutcome(enum.StrEnum):
    RESUME = "resume"
    ALREADY_CONSISTENT = "already_consistent"
    NEEDS_OPERATOR = "needs_operator"


@dataclass(frozen=True)
class ReconciliationResult:
    outcome: ReconciliationOutcome
    detail: str


def decide_reconciliation(
    *,
    outcome_committed: bool,
    checkpoint_head_exists: bool,
    head_has_pending_interrupt: bool,
) -> ReconciliationResult:
    """Pure decision function — no I/O, so every crash-window combination is unit-testable without
    seeding database rows.

    `head_has_pending_interrupt` is the same "does the head checkpoint carry an unresolved
    `__interrupt__` write" signal `AgentService._pending_interrupt_ids` already uses for v1 — used
    here only as a mid-flight-vs-ended proxy, not as a resume mechanism itself.
    """
    if not outcome_committed:
        if checkpoint_head_exists and not head_has_pending_interrupt:
            return ReconciliationResult(
                ReconciliationOutcome.NEEDS_OPERATOR,
                "checkpoint history head shows no pending interrupt but no TurnOutcome is committed "
                "for this turn — the graph may have ended without a terminal projection landing",
            )
        return ReconciliationResult(
            ReconciliationOutcome.RESUME,
            "no committed outcome yet; checkpoint head is mid-flight or absent",
        )
    if checkpoint_head_exists and head_has_pending_interrupt:
        return ReconciliationResult(
            ReconciliationOutcome.NEEDS_OPERATOR,
            "TurnOutcome is committed but the checkpoint history head still shows a pending "
            "interrupt — the checkpoint may not reflect the committed terminal transition",
        )
    return ReconciliationResult(
        ReconciliationOutcome.ALREADY_CONSISTENT,
        "TurnOutcome is committed and the checkpoint history head reflects a terminal state",
    )


async def reconcile_turn_checkpoint(db: AsyncSession, turn_id: uuid.UUID) -> ReconciliationResult:
    """I/O wrapper: gathers the committed `TurnOutcome` and v2 checkpoint history head for `turn_id`,
    then defers the actual decision to `decide_reconciliation`."""
    outcome = (
        await db.execute(
            select(TurnOutcome).where(
                TurnOutcome.turn_id == turn_id, TurnOutcome.outcome_type.in_(TERMINAL_OUTCOME_TYPES)
            )
        )
    ).scalar_one_or_none()
    head = (
        await db.execute(
            select(AgentCheckpoint)
            .where(AgentCheckpoint.turn_id == turn_id)
            .order_by(AgentCheckpoint.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    head_has_pending_interrupt = False
    if head is not None:
        head_has_pending_interrupt = any(
            item.get("channel") == "__interrupt__" for item in (head.pending_writes or [])
        )
    return decide_reconciliation(
        outcome_committed=outcome is not None,
        checkpoint_head_exists=head is not None,
        head_has_pending_interrupt=head_has_pending_interrupt,
    )
