"""Command boundary for write_draft's mutating effect (Phase 4).

Only reached when the admitting turn's cohort snapshot has `command_handlers_enabled=True`
(recorded once at admission by `AgentTurnService.admit_user_message`, never read live from
settings at effect time). A turn cohort with the flag off, or a call with no turn context at all,
never touches this module — `draft_lifecycle._write_draft_impl` keeps running the fully legacy
path in that case.

This module owns exactly three responsibilities, all against the caller's already-open, short-lived
DB session (no LLM/external call happens inside that session, matching the existing write_draft
transaction boundary):
  1. business-identity duplicate check (`logical_command_id` unique ledger lookup);
  2. fence validation (`TurnExecutionState` row-locked and checked for live owner/generation/lease);
  3. recording the committed effect ledger row alongside the artifact effect the caller persists.

No LLM/external call and no `interrupt()` happens here — those stay exactly where they are today,
after the caller's transaction commits.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import (
    AgentToolCall,
    DraftCommandEffectState,
    DraftCommandLedger,
    TurnExecutionState,
    TurnExecutionStatus,
)


def canonical_write_draft_intent(title: str, body: str) -> str:
    """Normalize the model-facing intent so retries with the same content hash identically.

    Deliberately excludes anything nondeterministic (timestamps, tool-call id, run id): those are
    correlation only, never business identity (see module docstring / phase-04 brief).
    """
    return json.dumps(
        {"title": str(title or "").strip(), "body": str(body or "").strip()},
        sort_keys=True,
        ensure_ascii=False,
    )


def write_draft_logical_command_id(
    turn_id: uuid.UUID, canonical_intent: str, expected_base_version_id: uuid.UUID | None
) -> str:
    """Stable business identity: turn + action type + canonical intent + expected base version.

    Two write_draft calls in the same turn with the same content and the same expected base version
    are the same logical command (retry) regardless of tool-call ID/run ID; a different base version
    or different content is a distinct logical command.
    """
    payload = "|".join(
        ["write_draft", str(turn_id), str(expected_base_version_id or ""), canonical_intent]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WriteDraftCommandOutcome:
    status: str  # "committed" | "duplicate" | "fenced"
    logical_command_id: str
    tool_call: AgentToolCall | None = None


class DraftCommandService:
    """Fenced, idempotent effect boundary for write_draft's business mutation.

    Operates on the caller's own session/transaction — it does not open its own transaction — so the
    fence check, duplicate check, and effect/ledger commit are part of the same short DB transaction
    the caller already holds, per the phase-04 constraint that fencing must be validated inside the
    same transaction as the mutation (a pre-check-only fence would leave a race window).
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def check_duplicate(self, logical_command_id: str) -> AgentToolCall | None:
        ledger = (
            await self.db.execute(
                select(DraftCommandLedger).where(DraftCommandLedger.logical_command_id == logical_command_id)
            )
        ).scalar_one_or_none()
        if ledger is None or ledger.tool_call_id is None:
            return None
        return await self.db.get(AgentToolCall, ledger.tool_call_id)

    async def fence_or_none(
        self, *, turn_id: uuid.UUID, owner_id: str, expected_generation: int
    ) -> TurnExecutionState | None:
        """Row-lock the turn's execution state; return it only if `owner_id`/`expected_generation`
        are still current and the lease has not expired. Returns None (fenced — caller must reject
        before any mutation) otherwise."""
        state = (
            await self.db.execute(
                select(TurnExecutionState).where(TurnExecutionState.turn_id == turn_id).with_for_update()
            )
        ).scalar_one_or_none()
        if state is None:
            return None
        if state.status == TurnExecutionStatus.TERMINAL:
            return None
        if state.owner_id != owner_id or state.ownership_generation != expected_generation:
            return None
        lease_expires_at = state.lease_expires_at
        if lease_expires_at is not None:
            # SQLite (unit tests) round-trips DateTime(timezone=True) as naive; Postgres keeps it
            # aware. Normalize to UTC-aware before comparing so both backends behave identically.
            if lease_expires_at.tzinfo is None:
                lease_expires_at = lease_expires_at.replace(tzinfo=UTC)
            if lease_expires_at <= datetime.now(UTC):
                return None
        return state

    def record_effect(
        self,
        *,
        turn_id: uuid.UUID,
        logical_command_id: str,
        tool_call: AgentToolCall,
        artifact_id: uuid.UUID | None,
        attempt: int,
    ) -> None:
        """Add the ledger row in the same (still-open) transaction as the effect the caller just
        persisted — the caller commits both atomically."""
        self.db.add(
            DraftCommandLedger(
                turn_id=turn_id,
                logical_command_id=logical_command_id,
                action_type="write_draft",
                tool_call_id=tool_call.id,
                artifact_id=artifact_id,
                effect_state=DraftCommandEffectState.COMMITTED,
                attempt=attempt,
            )
        )
