"""Admission and ownership fence shared by the inline runner and the future worker."""

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.graphs.analysis.turn_outcome_projector import project_terminal_outcome
from app.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCall,
    AgentToolCallStatus,
    AgentTurnEnvelope,
    AgentTurnTrigger,
    AgentTurnTriggerType,
    TurnExecutionState,
    TurnExecutionStatus,
    TurnOutcomeType,
)


@dataclass(frozen=True)
class AdmittedTurn:
    turn_id: uuid.UUID
    message: AgentMessage
    duplicate: bool
    queued: bool
    prior_status: AgentSessionStatus
    prior_interrupt_type: AgentSessionInterruptType | None
    # Whether the session's latest message, as observed under the same row lock and before this
    # turn's own message was inserted, was an agent direct-response. Must be captured before the
    # insert: once this turn's user message is the latest row, the check can no longer see it.
    prior_latest_message_is_direct_response: bool
    # The envelope's cohort snapshot (execution_mode, etc.), so a caller deciding inline-vs-durable
    # dispatch can read it without a second round trip. `None` on the duplicate-idempotency-replay
    # return path, where the caller never dispatches a second time and therefore never reads it.
    cohort: dict[str, Any] | None = None


@dataclass(frozen=True)
class AdmittedApprovalTurn:
    turn_id: uuid.UUID
    duplicate: bool


@dataclass(frozen=True)
class AdmittedCancelTurn:
    turn_id: uuid.UUID
    duplicate: bool


@dataclass(frozen=True)
class AdmittedRetryTurn:
    turn_id: uuid.UUID
    duplicate: bool


def idempotency_key_hash(value: str | None) -> str | None:
    if value is None:
        return None
    if not value or len(value) > 255:
        raise HTTPException(422, detail="Idempotency-Key must contain 1 to 255 characters")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AgentTurnService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _latest_message_is_direct_response(self, session_id: uuid.UUID) -> bool:
        """Mirrors `AgentService._latest_message_is_direct_response`. Must be called before this
        turn's own message is inserted, otherwise it always sees the just-inserted USER message
        instead of the prior AGENT response it is meant to detect."""
        latest = (
            await self.db.execute(
                select(AgentMessage)
                .where(AgentMessage.session_id == session_id)
                .order_by(AgentMessage.created_at.desc(), AgentMessage.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        return bool(
            latest
            and latest.role == AgentMessageRole.AGENT
            and isinstance(latest.payload, dict)
            and latest.payload.get("kind") == "response"
        )

    async def admit_user_message(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        user_id: uuid.UUID | None,
        content: str,
        idempotency_key: str | None,
        mode_hint: str | None = None,
    ) -> AdmittedTurn:
        """Write the trigger and decide inline/queue under the same session row lock."""
        if user_id is None:
            raise HTTPException(401, detail="Authenticated actor is required for agent turn admission")
        key_hash = idempotency_key_hash(idempotency_key)
        # The legacy ingress just read the session using this same request's session. End that
        # read transaction before opening the row-locked admission boundary; never hold a
        # transaction across an LLM call.
        if self.db.in_transaction():
            await self.db.commit()
        async with self.db.begin():
            session = (
                await self.db.execute(
                    select(AgentSession)
                    .where(AgentSession.id == session_id, AgentSession.project_id == project_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if session is None or session.created_by_id is None or session.created_by_id != user_id:
                raise HTTPException(404, detail="Agent session not found")

            prior_status = session.status
            prior_interrupt_type = session.interrupt_type
            # Must be captured now, before any message insert below (including the duplicate-replay
            # branch's read, which inserts nothing but still must see the state as of admission).
            prior_latest_message_is_direct_response = await self._latest_message_is_direct_response(session_id)

            if key_hash is not None:
                existing = (
                    await self.db.execute(
                        select(AgentTurnTrigger)
                        .where(
                            AgentTurnTrigger.session_id == session_id,
                            AgentTurnTrigger.trigger_type == AgentTurnTriggerType.USER_MESSAGE,
                            AgentTurnTrigger.idempotency_key_hash == key_hash,
                        )
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if existing is not None and existing.turn_id is not None and existing.message_id is not None:
                    message = await self.db.get(AgentMessage, existing.message_id)
                    if message is None:
                        raise HTTPException(409, detail="Idempotent turn is incomplete")
                    return AdmittedTurn(
                        turn_id=existing.turn_id,
                        message=message,
                        duplicate=True,
                        queued=bool(isinstance(message.payload, dict) and message.payload.get("queued")),
                        prior_status=prior_status,
                        prior_interrupt_type=prior_interrupt_type,
                        prior_latest_message_is_direct_response=prior_latest_message_is_direct_response,
                    )

            if session.status in (AgentSessionStatus.COMPLETED, AgentSessionStatus.FAILED, AgentSessionStatus.EXPIRED):
                raise HTTPException(400, detail="Session has ended and cannot accept more messages")
            queued = (
                (session.status == AgentSessionStatus.ACTIVE
                 and session.interrupt_type != AgentSessionInterruptType.STREAM_RESPONSE)
                or session.interrupt_type == AgentSessionInterruptType.PROPOSE_ARTIFACTS
            )
            if not queued and session.interrupt_type not in (
                AgentSessionInterruptType.ASK_HUMAN,
                AgentSessionInterruptType.STREAM_RESPONSE,
                None,
            ):
                raise HTTPException(400, detail="Session is not waiting for a user message")

            payload: dict[str, str | bool] | None = None
            if queued:
                payload = {"queued": True}
            # mode_hint is persisted on the message even for a turn that runs immediately: the
            # inline dispatch path still has it from its own request closure, but a durable worker
            # reconstructing this turn later from `turn_id` alone has no closure to read it from.
            if mode_hint:
                payload = {**(payload or {}), "mode_hint": mode_hint}
            message = AgentMessage(session_id=session.id, role=AgentMessageRole.USER, content=content, payload=payload)
            self.db.add(message)
            await self.db.flush()
            trigger = AgentTurnTrigger(
                session_id=session.id,
                trigger_type=AgentTurnTriggerType.USER_MESSAGE,
                idempotency_key_hash=key_hash,
                actor_id=user_id,
                message_id=message.id,
            )
            self.db.add(trigger)
            await self.db.flush()
            session.turn_sequence += 1
            if not queued:
                # This is only a legacy projection; the envelope/trigger/state remains the control plane.
                session.status = AgentSessionStatus.ACTIVE
                session.interrupt_type = None
            cohort = {
                "turn_admission": "v1",
                "policy_resolver_mode": settings.agent_policy_resolver_mode,
                "execution_mode": settings.agent_execution_mode,
                # Snapshotted once at admission: write_draft's command handler reads this
                # per-turn value, never settings.agent_command_handlers_enabled live, so a flag flip
                # mid-flight cannot change behavior for an already-admitted turn.
                "command_handlers_enabled": settings.agent_command_handlers_enabled,
                # Same snapshot-at-admission contract, gating whether the
                # terminal projector also writes a TurnOutcome audit row for this turn.
                "turn_outcomes_enabled": settings.agent_turn_outcomes_enabled,
                # The session's status/interrupt_type as observed under this same row lock, before
                # the mutation below flips it to ACTIVE/None for an immediately-runnable turn. A
                # durable worker reconstructing this turn's initial_state/resume_command later (once
                # the live session row has already moved on) reads these back instead of the inline
                # dispatch path's in-memory prior_status/prior_interrupt_type, which it never sees.
                "prior_status": prior_status.value,
                "prior_interrupt_type": prior_interrupt_type.value if prior_interrupt_type is not None else None,
                "prior_latest_message_is_direct_response": prior_latest_message_is_direct_response,
            }
            envelope = AgentTurnEnvelope(
                session_id=session.id,
                session_sequence=session.turn_sequence,
                original_trigger_id=trigger.id,
                actor_id=user_id,
                message_id=message.id,
                cohort=cohort,
                correlation_id=str(uuid.uuid4()),
            )
            self.db.add(envelope)
            await self.db.flush()
            trigger.turn_id = envelope.id
            if not queued:
                # The session lock is the shared ownership boundary: two different idempotency
                # keys can never both run the graph against the same checkpoint/session.
                session.active_turn_id = envelope.id
            self.db.add(TurnExecutionState(turn_id=envelope.id, status=TurnExecutionStatus.PENDING))
        return AdmittedTurn(
            turn_id=envelope.id,
            message=message,
            duplicate=False,
            queued=queued,
            prior_status=prior_status,
            prior_interrupt_type=prior_interrupt_type,
            prior_latest_message_is_direct_response=prior_latest_message_is_direct_response,
            cohort=cohort,
        )

    async def admit_approval(
        self,
        *,
        project_id: uuid.UUID,
        tool_call_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> AdmittedApprovalTurn:
        """Admit an approval turn from an already-authorized tool call; never trusts turn context from the client."""
        if self.db.in_transaction():
            await self.db.commit()
        async with self.db.begin():
            tool_call = (
                await self.db.execute(select(AgentToolCall).where(AgentToolCall.id == tool_call_id).with_for_update())
            ).scalar_one_or_none()
            if tool_call is None:
                raise HTTPException(404, detail="Tool call not found")
            session = (
                await self.db.execute(
                    select(AgentSession)
                    .where(AgentSession.id == tool_call.run.session_id, AgentSession.project_id == project_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if session is None or session.created_by_id != user_id:
                raise HTTPException(404, detail="Tool call not found")
            existing = (
                await self.db.execute(
                    select(AgentTurnTrigger)
                    .where(AgentTurnTrigger.tool_call_id == tool_call_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if existing is not None and existing.turn_id is not None:
                return AdmittedApprovalTurn(turn_id=existing.turn_id, duplicate=True)
            if tool_call.status != AgentToolCallStatus.PROPOSED:
                raise HTTPException(400, detail="Tool call is not in proposed status")
            if session.status != AgentSessionStatus.WAITING_FOR_HUMAN:
                raise HTTPException(400, detail="Session is not waiting for approval")
            trigger = AgentTurnTrigger(
                session_id=session.id,
                trigger_type=AgentTurnTriggerType.APPROVAL,
                actor_id=user_id,
                tool_call_id=tool_call.id,
            )
            self.db.add(trigger)
            await self.db.flush()
            session.turn_sequence += 1
            cohort = {
                "turn_admission": "v1",
                "policy_resolver_mode": settings.agent_policy_resolver_mode,
                "execution_mode": settings.agent_execution_mode,
                "command_handlers_enabled": settings.agent_command_handlers_enabled,
                "turn_outcomes_enabled": settings.agent_turn_outcomes_enabled,
            }
            envelope = AgentTurnEnvelope(
                session_id=session.id,
                session_sequence=session.turn_sequence,
                original_trigger_id=trigger.id,
                actor_id=user_id,
                cohort=cohort,
                correlation_id=str(uuid.uuid4()),
            )
            self.db.add(envelope)
            await self.db.flush()
            trigger.turn_id = envelope.id
            session.active_turn_id = envelope.id
            self.db.add(TurnExecutionState(turn_id=envelope.id, status=TurnExecutionStatus.PENDING))
        return AdmittedApprovalTurn(turn_id=envelope.id, duplicate=False)

    async def admit_cancel(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        user_id: uuid.UUID | None,
        idempotency_key: str | None,
        reason: str | None = None,
    ) -> AdmittedCancelTurn:
        """Admit a typed cancel trigger and project the CANCELLED terminal outcome for it.

        Cancel has no resume step waiting on a graph run the way approval does, so this admission
        boundary is also the terminal owner for the turn it opens — there is no separate executor
        to defer the outcome to.
        """
        if user_id is None:
            raise HTTPException(401, detail="Authenticated actor is required for agent turn admission")
        key_hash = idempotency_key_hash(idempotency_key)
        if self.db.in_transaction():
            await self.db.commit()
        async with self.db.begin():
            session = (
                await self.db.execute(
                    select(AgentSession)
                    .where(AgentSession.id == session_id, AgentSession.project_id == project_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if session is None or session.created_by_id is None or session.created_by_id != user_id:
                raise HTTPException(404, detail="Agent session not found")

            if key_hash is not None:
                existing = (
                    await self.db.execute(
                        select(AgentTurnTrigger)
                        .where(
                            AgentTurnTrigger.session_id == session_id,
                            AgentTurnTrigger.trigger_type == AgentTurnTriggerType.CANCEL,
                            AgentTurnTrigger.idempotency_key_hash == key_hash,
                        )
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if existing is not None and existing.turn_id is not None:
                    return AdmittedCancelTurn(turn_id=existing.turn_id, duplicate=True)

            if session.status in (AgentSessionStatus.COMPLETED, AgentSessionStatus.FAILED, AgentSessionStatus.EXPIRED):
                raise HTTPException(400, detail="Session has already ended and cannot be cancelled")

            trigger = AgentTurnTrigger(
                session_id=session.id,
                trigger_type=AgentTurnTriggerType.CANCEL,
                idempotency_key_hash=key_hash,
                actor_id=user_id,
            )
            self.db.add(trigger)
            await self.db.flush()
            session.turn_sequence += 1
            cohort = {
                "turn_admission": "v1",
                "policy_resolver_mode": settings.agent_policy_resolver_mode,
                "execution_mode": settings.agent_execution_mode,
                "command_handlers_enabled": settings.agent_command_handlers_enabled,
                "turn_outcomes_enabled": settings.agent_turn_outcomes_enabled,
            }
            envelope = AgentTurnEnvelope(
                session_id=session.id,
                session_sequence=session.turn_sequence,
                original_trigger_id=trigger.id,
                actor_id=user_id,
                cohort=cohort,
                correlation_id=str(uuid.uuid4()),
            )
            self.db.add(envelope)
            await self.db.flush()
            trigger.turn_id = envelope.id
            session.active_turn_id = envelope.id
            self.db.add(
                TurnExecutionState(
                    turn_id=envelope.id,
                    status=TurnExecutionStatus.TERMINAL,
                    attempt=1,
                    transition_version=1,
                )
            )
            await project_terminal_outcome(
                self.db, session, TurnOutcomeType.CANCELLED, reason or "user_cancelled", turn_id=envelope.id
            )
        return AdmittedCancelTurn(turn_id=envelope.id, duplicate=False)

    async def admit_retry(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        user_id: uuid.UUID | None,
        idempotency_key: str | None,
    ) -> AdmittedRetryTurn:
        """Admit a typed retry trigger that resumes the session's persisted checkpoint.

        Per ADR 0001, retry never re-invokes the model to re-decide: this only opens a new turn
        envelope over the existing, unchanged `graph_checkpoint` and reactivates the session so
        the next graph run resumes it — exactly the same reactivation this admission boundary
        already does for a plain user message, and no model call happens inside it either way.
        """
        if user_id is None:
            raise HTTPException(401, detail="Authenticated actor is required for agent turn admission")
        key_hash = idempotency_key_hash(idempotency_key)
        if self.db.in_transaction():
            await self.db.commit()
        async with self.db.begin():
            session = (
                await self.db.execute(
                    select(AgentSession)
                    .where(AgentSession.id == session_id, AgentSession.project_id == project_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if session is None or session.created_by_id is None or session.created_by_id != user_id:
                raise HTTPException(404, detail="Agent session not found")

            if key_hash is not None:
                existing = (
                    await self.db.execute(
                        select(AgentTurnTrigger)
                        .where(
                            AgentTurnTrigger.session_id == session_id,
                            AgentTurnTrigger.trigger_type == AgentTurnTriggerType.RETRY,
                            AgentTurnTrigger.idempotency_key_hash == key_hash,
                        )
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if existing is not None and existing.turn_id is not None:
                    return AdmittedRetryTurn(turn_id=existing.turn_id, duplicate=True)

            if session.status != AgentSessionStatus.TURN_FAILED:
                raise HTTPException(400, detail="Session is not in a retryable state")

            trigger = AgentTurnTrigger(
                session_id=session.id,
                trigger_type=AgentTurnTriggerType.RETRY,
                idempotency_key_hash=key_hash,
                actor_id=user_id,
            )
            self.db.add(trigger)
            await self.db.flush()
            session.turn_sequence += 1
            cohort = {
                "turn_admission": "v1",
                "policy_resolver_mode": settings.agent_policy_resolver_mode,
                "execution_mode": settings.agent_execution_mode,
                "command_handlers_enabled": settings.agent_command_handlers_enabled,
                "turn_outcomes_enabled": settings.agent_turn_outcomes_enabled,
            }
            envelope = AgentTurnEnvelope(
                session_id=session.id,
                session_sequence=session.turn_sequence,
                original_trigger_id=trigger.id,
                actor_id=user_id,
                cohort=cohort,
                correlation_id=str(uuid.uuid4()),
            )
            self.db.add(envelope)
            await self.db.flush()
            trigger.turn_id = envelope.id
            session.active_turn_id = envelope.id
            session.status = AgentSessionStatus.ACTIVE
            self.db.add(TurnExecutionState(turn_id=envelope.id, status=TurnExecutionStatus.PENDING))
        return AdmittedRetryTurn(turn_id=envelope.id, duplicate=False)

    async def claim_inline(self, *, turn_id: uuid.UUID, owner_id: str, lease_seconds: int = 120) -> int | None:
        """Claim via row lock; never hold the transaction while the caller calls the LLM."""
        now = datetime.now(UTC)
        async with self.db.begin():
            envelope = await self.db.get(AgentTurnEnvelope, turn_id)
            if envelope is None:
                return None
            session = (
                await self.db.execute(
                    select(AgentSession).where(AgentSession.id == envelope.session_id).with_for_update()
                )
            ).scalar_one_or_none()
            if session is None or session.active_turn_id != turn_id:
                return None
            state = (
                await self.db.execute(
                    select(TurnExecutionState).where(TurnExecutionState.turn_id == turn_id).with_for_update()
                )
            ).scalar_one_or_none()
            if state is None:
                return None
            if state.status == TurnExecutionStatus.TERMINAL:
                return None
            if (
                state.owner_id
                and state.owner_id != owner_id
                and state.lease_expires_at
                and state.lease_expires_at > now
            ):
                return None
            state.owner_id = owner_id
            state.ownership_generation += 1
            state.transition_version += 1
            state.attempt += 1
            state.status = TurnExecutionStatus.RUNNING
            state.lease_expires_at = now + timedelta(seconds=lease_seconds)
            return state.ownership_generation

    async def release_inline(self, *, turn_id: uuid.UUID, owner_id: str, generation: int) -> bool:
        """Prevent a stale executor's old lease from clearing a newer executor's ownership."""
        async with self.db.begin():
            state = (
                await self.db.execute(
                    select(TurnExecutionState).where(TurnExecutionState.turn_id == turn_id).with_for_update()
                )
            ).scalar_one_or_none()
            if state is None or state.owner_id != owner_id or state.ownership_generation != generation:
                return False
            state.owner_id = None
            state.lease_expires_at = None
            state.transition_version += 1
            if state.status == TurnExecutionStatus.RUNNING:
                state.status = TurnExecutionStatus.WAITING
            return True
