import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from typing import Any

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from app.documents.registry import container_for
from app.models.agent import (
    AgentMessage,
    AgentRun,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCall,
)
from app.models.artifact import Artifact
from app.schemas.agent import public_tool_call_input_snapshot
from app.services.agent_tool_visibility import public_tool_call_filter
from app.services.document_service import DocumentService


class AgentEventService:
    def __init__(self, db: AsyncSession, session_factory: Any = None):
        self.db = db
        self.session_factory = session_factory

    async def stream_session_events(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        request: Request,
        interval_seconds: float = 0.5,
    ) -> AsyncIterator[str]:
        snapshot = await self.build_snapshot(project_id=project_id, session_id=session_id, user_id=user_id)
        yield _sse(event="snapshot", data=snapshot, event_id=_event_id(session_id, 0))

        sequence = 1
        fingerprint = _snapshot_fingerprint(snapshot)
        while not await request.is_disconnected():
            status = snapshot["session"]["status"]
            if status in {AgentSessionStatus.COMPLETED.value, AgentSessionStatus.FAILED.value}:
                yield _sse(
                    event="stream_closed",
                    data={"type": "stream_closed", "status": status},
                    event_id=_event_id(session_id, sequence),
                )
                return

            await asyncio.sleep(interval_seconds)
            next_snapshot = await self.build_snapshot(
                project_id=project_id,
                session_id=session_id,
                user_id=user_id,
            )
            next_fingerprint = _snapshot_fingerprint(next_snapshot)
            if next_fingerprint != fingerprint:
                sequence += 1
                yield _sse(event="snapshot", data=next_snapshot, event_id=_event_id(session_id, sequence))
                snapshot = next_snapshot
                fingerprint = next_fingerprint

    async def build_snapshot(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> dict[str, Any]:
        session = (
            await self.db.execute(
                select(AgentSession)
                .where(
                    AgentSession.id == session_id,
                    AgentSession.project_id == project_id,
                    AgentSession.created_by_id == user_id,
                )
                .execution_options(populate_existing=True)
            )
        ).scalar_one_or_none()
        if not session:
            raise HTTPException(404, detail="Agent session not found")

        messages = (
            (
                await self.db.execute(
                    select(AgentMessage)
                    .where(AgentMessage.session_id == session_id)
                    .order_by(AgentMessage.created_at)
                    .execution_options(populate_existing=True)
                )
            )
            .scalars()
            .all()
        )
        tool_calls = (
            (
                await self.db.execute(
                    select(AgentToolCall)
                    .join(AgentRun, AgentToolCall.run_id == AgentRun.id)
                    .where(AgentRun.session_id == session_id)
                    .where(public_tool_call_filter())
                    .order_by(AgentToolCall.created_at)
                    .execution_options(populate_existing=True)
                )
            )
            .scalars()
            .all()
        )
        document = await self._document_for_session(session, project_id)

        return {
            "type": "snapshot",
            "session": {
                "id": session.id,
                "project_id": session.project_id,
                "created_by_id": session.created_by_id,
                "artifact_type": session.artifact_type,
                "workflow_area": session.workflow_area,
                "focused_artifact_id": session.focused_artifact_id,
                "status": session.status,
                "ui_status": _ui_status(session.status, session.interrupt_type),
                "interrupt_type": session.interrupt_type,
                "missing_context": session.missing_context,
                "document": document,
                "updated_at": session.updated_at,
            },
            "messages": [
                {
                    "id": message.id,
                    "session_id": message.session_id,
                    "role": message.role,
                    "content": message.content,
                    "payload": message.payload,
                    "created_at": message.created_at,
                    "updated_at": message.updated_at,
                }
                for message in messages
            ],
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "run_id": tool_call.run_id,
                    "tool_name": tool_call.tool_name,
                    "input_snapshot": public_tool_call_input_snapshot(tool_call.input_snapshot),
                    "status": tool_call.status,
                    "created_artifact_id": tool_call.created_artifact_id,
                    "created_version_id": tool_call.created_version_id,
                    "resolved_at": tool_call.resolved_at,
                    "created_at": tool_call.created_at,
                    "updated_at": tool_call.updated_at,
                }
                for tool_call in tool_calls
            ],
        }

    async def _document_for_session(
        self,
        session: AgentSession,
        project_id: uuid.UUID,
    ):
        document_type = container_for(session.artifact_type)
        if session.focused_artifact_id is not None:
            focused = await self.db.get(Artifact, session.focused_artifact_id)
            if focused is not None and focused.project_id == project_id:
                if focused.parent_id is not None:
                    parent = await self.db.get(Artifact, focused.parent_id)
                    document_type = parent.type.value if parent is not None else document_type
                elif focused.type.value in {"brd", "prd", "sad"}:
                    document_type = focused.type.value
        if document_type is None and session.artifact_type in {"brd", "prd", "sad"}:
            document_type = session.artifact_type
        if document_type is None:
            return None
        return await DocumentService(self.db).get_document(
            project_id=project_id,
            document_type=document_type,
        )


def _ui_status(status: Any, interrupt_type: Any) -> str:
    """Derive a coarse UI-facing status from session.status + interrupt_type.

    Pure mapping, no DB. Accepts enum members or their string values so it works both
    inside build_snapshot (enums) and against serialized rows. See spec section 4.2.
    """
    status_val = getattr(status, "value", status)
    interrupt_val = getattr(interrupt_type, "value", interrupt_type)
    if status_val == AgentSessionStatus.ACTIVE.value:
        if interrupt_val == AgentSessionInterruptType.STREAM_RESPONSE.value:
            return "waiting_input"
        return "processing"
    if status_val == AgentSessionStatus.WAITING_FOR_HUMAN.value:
        if interrupt_val == AgentSessionInterruptType.PROPOSE_ARTIFACTS.value:
            return "waiting_approval"
        return "waiting_input"
    if status_val == AgentSessionStatus.FAILED.value:
        return "error"
    return "idle"


def _sse(*, event: str, data: dict[str, Any], event_id: str) -> str:
    payload = json.dumps(jsonable_encoder(data), ensure_ascii=False, separators=(",", ":"))
    return f"id: {event_id}\nevent: {event}\ndata: {payload}\n\n"


def _event_id(session_id: uuid.UUID, sequence: int) -> str:
    return f"{datetime.now(UTC).isoformat()}:{session_id}:{sequence}"


def _snapshot_fingerprint(snapshot: dict[str, Any]) -> str:
    payload = jsonable_encoder(snapshot)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime, uuid.UUID)):
        return str(value)
    return str(value)
