"""Admission và ownership fence dùng chung cho inline runner và worker tương lai."""

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentTurnEnvelope,
    AgentTurnTrigger,
    AgentTurnTriggerType,
    TurnExecutionState,
    TurnExecutionStatus,
)


@dataclass(frozen=True)
class AdmittedTurn:
    turn_id: uuid.UUID
    message: AgentMessage
    duplicate: bool
    queued: bool
    prior_status: AgentSessionStatus
    prior_interrupt_type: AgentSessionInterruptType | None


def idempotency_key_hash(value: str | None) -> str | None:
    if value is None:
        return None
    if not value or len(value) > 255:
        raise HTTPException(422, detail="Idempotency-Key must contain 1 to 255 characters")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AgentTurnService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

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
        """Ghi trigger và quyết định inline/queue dưới cùng session row lock."""
        if user_id is None:
            raise HTTPException(401, detail="Authenticated actor is required for agent turn admission")
        key_hash = idempotency_key_hash(idempotency_key)
        # Ingress legacy vừa đọc session bằng cùng request session. Kết thúc read transaction
        # trước khi mở boundary admission có row lock; không giữ transaction qua LLM.
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
                if mode_hint:
                    payload["mode_hint"] = mode_hint
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
                # Đây chỉ là projection legacy; envelope/trigger/state vẫn là control plane.
                session.status = AgentSessionStatus.ACTIVE
                session.interrupt_type = None
            cohort = {
                "turn_admission": "v1",
                "policy_resolver_mode": settings.agent_policy_resolver_mode,
                "execution_mode": settings.agent_execution_mode,
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
                # Khóa session là boundary ownership chung: hai idempotency key khác không thể
                # cùng được chạy graph trên một checkpoint/session.
                session.active_turn_id = envelope.id
            self.db.add(TurnExecutionState(turn_id=envelope.id, status=TurnExecutionStatus.PENDING))
        return AdmittedTurn(
            turn_id=envelope.id,
            message=message,
            duplicate=False,
            queued=queued,
            prior_status=prior_status,
            prior_interrupt_type=prior_interrupt_type,
        )

    async def claim_inline(self, *, turn_id: uuid.UUID, owner_id: str, lease_seconds: int = 120) -> int | None:
        """Claim bằng row lock; không giữ transaction khi caller gọi LLM."""
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
        """Không cho executor lease cũ xoá ownership của executor mới."""
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
