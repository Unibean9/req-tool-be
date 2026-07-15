"""Command boundary for mutating effects of write_draft, finalize, create_artifact_link
and propose_retirement.

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


def _logical_command_id(
    action_type: str,
    turn_id: uuid.UUID,
    canonical_intent: str,
    expected_base_version_id: uuid.UUID | None = None,
) -> str:
    """Stable business identity shared by every action_type: action + turn + canonical intent +
    expected base version (when the action has one). Two calls in the same turn with the same
    action_type/content/base-version are the same logical command (retry) regardless of provider
    tool-call ID/run ID; anything different is a distinct logical command.

    Each action_type gets its own thin wrapper below (not reused across action_types directly) so a
    canonical-intent format change for one action can never accidentally collide with another's
    namespace — the action_type is folded into the hash input either way, but the wrappers keep each
    call site's intent shape self-documenting.
    """
    payload = "|".join([action_type, str(turn_id), str(expected_base_version_id or ""), canonical_intent])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_write_draft_intent(title: str, body: str) -> str:
    """Normalize the model-facing intent so retries with the same content hash identically.

    Deliberately excludes anything nondeterministic (timestamps, tool-call id, run id): those are
    correlation only, never business identity (see module docstring).
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
    return _logical_command_id("write_draft", turn_id, canonical_intent, expected_base_version_id)


def canonical_finalize_intent(summary: str) -> str:
    """Normalize finalize's model-facing intent (its closing summary) for retry hashing."""
    return json.dumps({"summary": str(summary or "").strip()}, sort_keys=True, ensure_ascii=False)


def finalize_logical_command_id(turn_id: uuid.UUID, canonical_intent: str) -> str:
    """finalize has no artifact base version (its effect is a session/interrupt transition, not an
    artifact mutation), so the identity is just turn + canonical intent."""
    return _logical_command_id("finalize", turn_id, canonical_intent)


def canonical_artifact_link_intent(
    source_artifact_id: uuid.UUID | str, target_artifact_id: uuid.UUID | str, relation_type: str
) -> str:
    """Normalize create_artifact_link's execution-time intent for retry hashing."""
    return json.dumps(
        {
            "source_artifact_id": str(source_artifact_id),
            "target_artifact_id": str(target_artifact_id),
            "relation_type": str(relation_type),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def create_artifact_link_logical_command_id(turn_id: uuid.UUID, canonical_intent: str) -> str:
    return _logical_command_id("create_artifact_link", turn_id, canonical_intent)


def canonical_retirement_intent(artifact_id: uuid.UUID | str, superseded_by_artifact_id: uuid.UUID | str | None) -> str:
    """Normalize propose_retirement's execution-time intent for retry hashing."""
    return json.dumps(
        {
            "artifact_id": str(artifact_id),
            "superseded_by_artifact_id": str(superseded_by_artifact_id) if superseded_by_artifact_id else None,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def propose_retirement_logical_command_id(turn_id: uuid.UUID, canonical_intent: str) -> str:
    return _logical_command_id("propose_retirement", turn_id, canonical_intent)


@dataclass(frozen=True)
class WriteDraftCommandOutcome:
    status: str  # "committed" | "duplicate" | "fenced"
    logical_command_id: str
    tool_call: AgentToolCall | None = None


class DraftCommandService:
    """Fenced, idempotent effect boundary for write_draft's business mutation.

    Operates on the caller's own session/transaction — it does not open its own transaction — so the
    fence check, duplicate check, and effect/ledger commit are part of the same short DB transaction
    the caller already holds, since fencing must be validated inside the same transaction as the
    mutation (a pre-check-only fence would leave a race window).
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def find_ledger(self, logical_command_id: str) -> DraftCommandLedger | None:
        """Look up the ledger row itself, for action_types whose effect is not an AgentToolCall
        (e.g. finalize/artifact-link execution) — `check_duplicate` below stays the write_draft-
        shaped convenience wrapper so its existing callers/tests are unaffected."""
        return (
            await self.db.execute(
                select(DraftCommandLedger).where(DraftCommandLedger.logical_command_id == logical_command_id)
            )
        ).scalar_one_or_none()

    async def check_duplicate(self, logical_command_id: str) -> AgentToolCall | None:
        ledger = await self.find_ledger(logical_command_id)
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
        action_type: str,
        attempt: int,
        tool_call: AgentToolCall | None = None,
        artifact_id: uuid.UUID | None = None,
    ) -> None:
        """Add the ledger row in the same (still-open) transaction as the effect the caller just
        persisted — the caller commits both atomically.

        `tool_call`/`artifact_id` are optional: finalize's effect is a session/interrupt
        transition, not an AgentToolCall or artifact row, so it records a ledger row with both
        left `None` — the unique `logical_command_id` constraint is still the exactly-once
        invariant either way.
        """
        self.db.add(
            DraftCommandLedger(
                turn_id=turn_id,
                logical_command_id=logical_command_id,
                action_type=action_type,
                tool_call_id=tool_call.id if tool_call is not None else None,
                artifact_id=artifact_id,
                effect_state=DraftCommandEffectState.COMMITTED,
                attempt=attempt,
            )
        )
