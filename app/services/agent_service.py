import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from langgraph.types import Command
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.documents.registry import container_for
from app.graphs.checkpointer import AgentSessionCheckpointer
from app.graphs.policy import ARTIFACT_PREDECESSORS
from app.graphs.state import (
    DEFAULT_ARTIFACT_CHAIN,
    DEFAULT_METHOD_PROFILE,
    DEFAULT_READINESS,
)
from app.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentRun,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCall,
    AgentToolCallStatus,
)
from app.models.artifact import (
    Artifact,
    ArtifactStatus,
    ArtifactVersion,
    ChangeSource,
)
from app.schemas.agent import AgentSessionResponse
from app.schemas.artifact_synthesis import synthesis_metadata_dict, synthesis_metadata_from_snapshot
from app.services.agent_tool_visibility import public_tool_call_filter
from app.services.document_service import DocumentService


class AgentService:
    def __init__(self, db: AsyncSession, graph: Any, session_factory: Any):
        self.db = db
        self.graph = graph
        self.session_factory = session_factory

    async def create_session(
        self,
        *,
        project_id: uuid.UUID,
        artifact_type: str,
        step_key: str | None = None,
        workflow_area: str = "analysis",
        agent_role: str | None = None,
        provider_config_id: uuid.UUID | None = None,
        created_by_id: uuid.UUID | None = None,
        focused_artifact_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        missing = await self._check_predecessors(project_id, artifact_type)
        if missing:
            raise HTTPException(
                409,
                detail={
                    "detail": "Predecessor artifact chưa được accepted",
                    "missing_context": missing,
                },
            )
        if focused_artifact_id is not None:
            focused = await self.db.get(Artifact, focused_artifact_id)
            if focused is None or focused.project_id != project_id:
                raise HTTPException(422, detail="focused_artifact_id không thuộc dự án")
            if focused.parent_id is None:
                raise HTTPException(422, detail="Agent phải focus vào document item, không phải container")
            if focused.type.value != artifact_type:
                raise HTTPException(
                    422,
                    detail="artifact_type phải khớp với document item được focus",
                )

        try:
            session = AgentSession(
                project_id=project_id,
                artifact_type=artifact_type,
                step_key=step_key,
                workflow_area=workflow_area,
                agent_role=agent_role,
                status=AgentSessionStatus.WAITING_FOR_HUMAN,
                graph_checkpoint={},
                missing_context=missing or None,
                focused_artifact_id=focused_artifact_id,
                provider_config_id=provider_config_id,
                created_by_id=created_by_id,
            )
            self.db.add(session)
            await self.db.flush()
            await self.db.commit()
        except IntegrityError:
            await self.db.rollback()
            existing_query = select(AgentSession).where(
                AgentSession.project_id == project_id,
                AgentSession.artifact_type == artifact_type,
                AgentSession.status.in_(["active", "waiting_for_human"]),
            )
            if created_by_id is not None:
                existing_query = existing_query.where(AgentSession.created_by_id == created_by_id)
            existing = (await self.db.execute(existing_query)).scalar_one_or_none()
            raise HTTPException(
                409,
                detail={
                    "detail": "Active session already exists",
                    "session_id": str(existing.id) if existing else None,
                },
            ) from None

        return await self.create_session_response(session, missing)

    async def create_session_response(self, session: AgentSession, missing: list[str]) -> dict[str, Any]:
        document_type = await self._document_type_for_session(session)
        return {
            "session_id": str(session.id),
            "missing_context": missing,
            "artifact_type": session.artifact_type,
            "focused_artifact_id": session.focused_artifact_id,
            "document_type": document_type,
        }

    async def get_session_response(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> AgentSessionResponse:
        session = await self.get_session(project_id=project_id, session_id=session_id, user_id=user_id)
        await self._load_graph_state_values(session_id)
        document_type = await self._document_type_for_session(session)
        document = (
            await DocumentService(self.db).get_document(
                project_id=project_id,
                document_type=document_type,
            )
            if document_type
            else None
        )
        return AgentSessionResponse.model_validate(session).model_copy(
            update={
                "ui_status": _session_ui_status(session.status, session.interrupt_type),
                "document": document,
            }
        )

    async def _document_type_for_session(self, session: AgentSession) -> str | None:
        if session.focused_artifact_id is not None:
            focused = await self.db.get(Artifact, session.focused_artifact_id)
            if focused is not None:
                if focused.parent_id is not None:
                    parent = await self.db.get(Artifact, focused.parent_id)
                    if parent is not None:
                        return parent.type.value
                if focused.type.value in {"brd", "prd", "sad"}:
                    return focused.type.value
        if session.artifact_type in {"brd", "prd", "sad"}:
            return session.artifact_type
        return container_for(session.artifact_type)

    async def get_session(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> AgentSession:
        query = select(AgentSession).where(
            AgentSession.id == session_id,
            AgentSession.project_id == project_id,
        )
        if user_id is not None:
            query = query.where(AgentSession.created_by_id == user_id)
        session = (await self.db.execute(query)).scalar_one_or_none()
        if not session:
            raise HTTPException(404, detail="Agent session không tồn tại")
        return session

    async def _load_graph_state_values(self, session_id: uuid.UUID) -> dict[str, Any] | None:
        if self.graph is None:
            return None
        try:
            snapshot = await self.graph.aget_state({"configurable": {"thread_id": str(session_id)}})
        except Exception as exc:
            raise HTTPException(500, detail="Không thể đọc checkpoint workspace") from exc
        values = getattr(snapshot, "values", None)
        return values if isinstance(values, dict) else None

    async def delete_session(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> None:
        session = await self.get_session(project_id=project_id, session_id=session_id, user_id=user_id)
        await self.db.delete(session)
        await self.db.commit()

    async def handle_user_message(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        content: str,
        user_id: uuid.UUID | None = None,
        llm_client: Any = None,
        mode_hint: str | None = None,
    ) -> AgentMessage:
        session = await self.get_session(project_id=project_id, session_id=session_id, user_id=user_id)

        # S2 — never silently drop a valid message while the agent is busy. Queue it and return 200.
        # A queued message carries only its content; a mode_hint on it is intentionally not
        # replayed (the queue stores no steer) — acceptable for MVP, revisit post-MVP if needed.
        if session.status == AgentSessionStatus.ACTIVE:
            return await self._queue_message(session.id, content)
        if session.status in (AgentSessionStatus.COMPLETED, AgentSessionStatus.FAILED):
            raise HTTPException(400, detail="Session đã kết thúc, không thể nhận thêm message")
        # status == WAITING_FOR_HUMAN below.
        # PROPOSE_ARTIFACTS waits for an approval decision, not free-text — queue the text (no carve-out).
        if session.interrupt_type == AgentSessionInterruptType.PROPOSE_ARTIFACTS:
            return await self._queue_message(session.id, content)
        if session.interrupt_type not in (AgentSessionInterruptType.ASK_HUMAN, None):
            raise HTTPException(400, detail="Session không ở trạng thái chờ user message")

        is_first_message = session.interrupt_type is None

        msg = AgentMessage(session_id=session.id, role=AgentMessageRole.USER, content=content)
        self.db.add(msg)
        session.status = AgentSessionStatus.ACTIVE
        session.interrupt_type = None
        await self.db.commit()

        strong_llm_client = None
        if llm_client is None:
            llm_client, strong_llm_client = await self._resolve_llm_client(session.provider_config_id)

        if is_first_message:
            initial_state = {
                "artifact_type": session.artifact_type,
                "workflow_area": session.workflow_area,
                "step_key": session.step_key,
                "messages": [{"role": "user", "content": content}],
                "conversation_summary": "",
                "analysis_result": None,
                "pending_tool_call_ids": [],
                "last_agent_run_id": None,
                "turn_count": 0,
                "missing_context": session.missing_context or [],
                "user_confirmed": None,
                "locale": None,
                "turn_type": None,
                "triage_reply": None,
                "section_coverage": None,
                "coverage_complete": None,
                "section_coverage_stall_count": None,
                "assumptions": [],
                "risks": [],
                "open_questions": [],
                "focused_artifact_id": (
                    str(session.focused_artifact_id)
                    if session.focused_artifact_id is not None
                    else None
                ),
                "draft_body": None,
                "method_profile": dict(DEFAULT_METHOD_PROFILE),
                "artifact_chain": dict(DEFAULT_ARTIFACT_CHAIN),
                "readiness": dict(DEFAULT_READINESS),
                "working_draft": None,
                "mode_hint": mode_hint,
            }
            resume_command = None
        else:
            initial_state = None
            resume_command = self._resume_command(
                session, {"content": content}, state_update={"mode_hint": mode_hint} if mode_hint else None
            )

        asyncio.create_task(
            self._run_graph(
                session_id=session.id,
                project_id=project_id,
                artifact_type=session.artifact_type,
                step_key=session.step_key,
                workflow_area=session.workflow_area,
                agent_role=session.agent_role,
                focused_artifact_id=session.focused_artifact_id,
                missing_context=session.missing_context or [],
                llm_client=llm_client,
                strong_llm_client=strong_llm_client,
                initial_state=initial_state,
                resume_command=resume_command,
            )
        )

        return msg

    async def _queue_message(self, session_id: uuid.UUID, content: str) -> AgentMessage:
        """Persist a user message as queued (payload.queued=True) without starting a graph turn.

        Drained later by _drain_queue once the current turn ends COMPLETED/FAILED.
        """
        msg = AgentMessage(
            session_id=session_id,
            role=AgentMessageRole.USER,
            content=content,
            payload={"queued": True},
        )
        self.db.add(msg)
        await self.db.commit()
        await self.db.refresh(msg)
        return msg

    async def list_messages(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> list[AgentMessage]:
        await self.get_session(project_id=project_id, session_id=session_id, user_id=user_id)
        rows = (
            await self.db.execute(
                select(AgentMessage)
                .where(AgentMessage.session_id == session_id)
                .order_by(AgentMessage.created_at)
            )
        ).scalars().all()
        return list(rows)

    async def list_tool_calls(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> list[AgentToolCall]:
        await self.get_session(project_id=project_id, session_id=session_id, user_id=user_id)
        rows = (
            await self.db.execute(
                select(AgentToolCall)
                .join(AgentRun, AgentToolCall.run_id == AgentRun.id)
                .where(AgentRun.session_id == session_id)
                .where(public_tool_call_filter())
                .order_by(AgentToolCall.created_at)
            )
        ).scalars().all()
        return list(rows)

    async def approve_tool_call(
        self,
        *,
        project_id: uuid.UUID,
        tool_call_id: uuid.UUID,
        created_by_id: uuid.UUID | None,
        user_id: uuid.UUID | None = None,
        _llm_client: Any = None,
    ) -> AgentToolCall:
        tool_call, session_id = await self._get_tool_call_with_idor(tool_call_id, project_id, user_id=user_id)
        if tool_call.status == AgentToolCallStatus.EXECUTED:
            return tool_call
        if tool_call.status == AgentToolCallStatus.REJECTED:
            raise HTTPException(400, detail="Tool call đã bị reject")
        if tool_call.status != AgentToolCallStatus.PROPOSED:
            raise HTTPException(400, detail="Tool call không ở trạng thái proposed")

        artifact, version = await self._execute_create_artifact(
            project_id=project_id,
            snapshot=tool_call.input_snapshot or {},
            run_id=tool_call.run_id,
            tool_call_id=tool_call.id,
            created_by_id=created_by_id,
        )

        tool_call.status = AgentToolCallStatus.EXECUTED
        tool_call.created_artifact_id = artifact.id
        tool_call.created_version_id = version.id
        tool_call.resolved_at = datetime.now(UTC)
        await self.db.commit()

        await self._complete_when_all_artifact_proposals_approved(session_id=session_id)
        await self.db.refresh(tool_call)
        return tool_call

    async def reject_tool_call(
        self,
        *,
        project_id: uuid.UUID,
        tool_call_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
        llm_client: Any = None,
    ) -> AgentToolCall:
        tool_call, session_id = await self._get_tool_call_with_idor(tool_call_id, project_id, user_id=user_id)
        if tool_call.status == AgentToolCallStatus.REJECTED:
            return tool_call
        if tool_call.status == AgentToolCallStatus.EXECUTED:
            raise HTTPException(400, detail="Tool call đã được approve")
        if tool_call.status != AgentToolCallStatus.PROPOSED:
            raise HTTPException(400, detail="Tool call không ở trạng thái proposed")

        tool_call.status = AgentToolCallStatus.REJECTED
        tool_call.resolved_at = datetime.now(UTC)
        await self.db.commit()

        await self._check_and_resume(project_id=project_id, session_id=session_id, llm_client=llm_client)
        await self.db.refresh(tool_call)
        return tool_call

    async def request_edit(
        self,
        *,
        project_id: uuid.UUID,
        tool_call_id: uuid.UUID,
        note: str,
        base_version_id: uuid.UUID | None = None,
        user_id: uuid.UUID | None = None,
        llm_client: Any = None,
    ) -> AgentToolCall:
        tool_call, session_id = await self._get_tool_call_with_idor(tool_call_id, project_id, user_id=user_id)
        if tool_call.status != AgentToolCallStatus.PROPOSED:
            raise HTTPException(400, detail="Tool call không ở trạng thái proposed")
        await self._guard_current_base_version(tool_call.input_snapshot or {}, base_version_id)

        tool_call.status = AgentToolCallStatus.SUPERSEDED
        tool_call.resolved_at = datetime.now(UTC)

        msg = AgentMessage(session_id=session_id, role=AgentMessageRole.USER, content=note)
        self.db.add(msg)
        await self.db.commit()

        await self._check_and_resume(project_id=project_id, session_id=session_id, llm_client=llm_client)
        await self.db.refresh(tool_call)
        return tool_call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _check_predecessors(self, project_id: uuid.UUID, artifact_type: str) -> list[str]:
        predecessors = ARTIFACT_PREDECESSORS.get(artifact_type, [])
        missing: list[str] = []
        for pred in predecessors:
            count = (
                await self.db.execute(
                    select(func.count(Artifact.id)).where(
                        Artifact.project_id == project_id,
                        Artifact.type == pred,
                        Artifact.status == ArtifactStatus.ACCEPTED,
                    )
                )
            ).scalar() or 0
            if count == 0:
                missing.append(pred)
        return missing

    async def _get_tool_call_with_idor(
        self,
        tool_call_id: uuid.UUID,
        project_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> tuple[AgentToolCall, uuid.UUID]:
        query = (
            select(AgentToolCall, AgentRun.session_id.label("session_id"))
            .join(AgentRun, AgentToolCall.run_id == AgentRun.id)
            .join(AgentSession, AgentRun.session_id == AgentSession.id)
            .where(AgentToolCall.id == tool_call_id)
            .where(AgentSession.project_id == project_id)
            .where(public_tool_call_filter())
            .with_for_update()
        )
        if user_id is not None:
            query = query.where(AgentSession.created_by_id == user_id)
        row = (await self.db.execute(query)).one_or_none()
        if not row:
            raise HTTPException(404, detail="Tool call không tồn tại")
        tool_call, session_id = row
        return tool_call, session_id

    async def _execute_create_artifact(
        self,
        *,
        project_id: uuid.UUID,
        snapshot: dict[str, Any],
        run_id: uuid.UUID,
        tool_call_id: uuid.UUID,
        created_by_id: uuid.UUID | None,
    ) -> tuple[Artifact, ArtifactVersion]:
        focused_artifact_id = snapshot.get("focused_artifact_id")
        if not focused_artifact_id:
            raise HTTPException(
                422,
                detail="Tool call thiếu focused_artifact_id; vui lòng chọn document item hiện tại",
            )
        title = snapshot.get("title", "Untitled")
        body = snapshot.get("body", "")
        try:
            synthesis_metadata = synthesis_metadata_dict(snapshot)
        except ValueError as exc:
            raise HTTPException(422, detail="Tool call metadata synthesis không hợp lệ") from exc

        try:
            artifact, version = await DocumentService(self.db).create_item_version(
                artifact_id=uuid.UUID(str(focused_artifact_id)),
                project_id=project_id,
                title=title,
                body=body,
                created_by_id=created_by_id,
                change_source=ChangeSource.AI_GENERATION,
                agent_run_id=run_id,
                tool_call_id=tool_call_id,
                metadata=synthesis_metadata,
                mark_accepted=True,
            )
        except ValueError as exc:
            raise HTTPException(404, detail="Document item được focus không tồn tại") from exc

        return artifact, version

    async def _guard_current_base_version(
        self,
        snapshot: dict[str, Any],
        requested_base_version_id: uuid.UUID | None,
    ) -> None:
        focused_artifact_id = snapshot.get("focused_artifact_id")
        if not focused_artifact_id:
            raise HTTPException(422, detail="Tool call thiếu focused_artifact_id")
        focused = await self.db.get(Artifact, uuid.UUID(str(focused_artifact_id)))
        if focused is None:
            raise HTTPException(404, detail="Document item được focus không tồn tại")

        snapshot_base = None
        try:
            snapshot_base = synthesis_metadata_from_snapshot(snapshot).base_version_id
        except ValueError:
            raw_base = snapshot.get("base_version_id")
            snapshot_base = uuid.UUID(str(raw_base)) if raw_base else None
        base_version_id = requested_base_version_id if requested_base_version_id is not None else snapshot_base
        if base_version_id != focused.current_version_id:
            raise HTTPException(
                409,
                detail={
                    "detail": "Bản nháp sửa đang dựa trên version cũ",
                    "base_version_id": str(base_version_id) if base_version_id else None,
                    "current_version_id": str(focused.current_version_id) if focused.current_version_id else None,
                },
            )

    async def _check_and_resume(
        self, *, project_id: uuid.UUID, session_id: uuid.UUID, llm_client: Any = None
    ) -> None:
        session_row = (
            await self.db.execute(
                select(AgentSession).where(AgentSession.id == session_id).with_for_update()
            )
        ).scalar_one()

        # Guard: concurrent approve/reject may have already resumed this session.
        if session_row.status != AgentSessionStatus.WAITING_FOR_HUMAN:
            return

        pending_count = (
            await self.db.execute(
                select(func.count(AgentToolCall.id))
                .join(AgentRun, AgentToolCall.run_id == AgentRun.id)
                .where(AgentRun.session_id == session_id)
                .where(AgentToolCall.status == AgentToolCallStatus.PROPOSED)
                .where(public_tool_call_filter())
            )
        ).scalar() or 0

        if pending_count > 0:
            return

        strong_llm_client = None
        if llm_client is None:
            llm_client, strong_llm_client = await self._resolve_llm_client(session_row.provider_config_id)

        resume_command = self._resume_command(session_row, {"all_resolved": True})
        session_row.status = AgentSessionStatus.ACTIVE
        session_row.interrupt_type = None
        await self.db.commit()

        asyncio.create_task(
            self._run_graph(
                session_id=session_id,
                project_id=project_id,
                artifact_type=session_row.artifact_type,
                step_key=session_row.step_key,
                workflow_area=session_row.workflow_area,
                agent_role=session_row.agent_role,
                focused_artifact_id=session_row.focused_artifact_id,
                missing_context=session_row.missing_context or [],
                llm_client=llm_client,
                strong_llm_client=strong_llm_client,
                initial_state=None,
                resume_command=resume_command,
            )
        )

    async def _complete_when_all_artifact_proposals_approved(self, *, session_id: uuid.UUID) -> None:
        session_row = (
            await self.db.execute(
                select(AgentSession).where(AgentSession.id == session_id).with_for_update()
            )
        ).scalar_one()
        if session_row.status != AgentSessionStatus.WAITING_FOR_HUMAN:
            return
        pending_count = (
            await self.db.execute(
                select(func.count(AgentToolCall.id))
                .join(AgentRun, AgentToolCall.run_id == AgentRun.id)
                .where(AgentRun.session_id == session_id)
                .where(AgentToolCall.status == AgentToolCallStatus.PROPOSED)
                .where(public_tool_call_filter())
            )
        ).scalar() or 0
        if pending_count > 0:
            return
        session_row.status = AgentSessionStatus.COMPLETED
        session_row.interrupt_type = None
        await self.db.commit()

    async def _run_graph(
        self,
        *,
        session_id: uuid.UUID,
        project_id: uuid.UUID,
        artifact_type: str,
        step_key: str | None,
        workflow_area: str,
        agent_role: str | None,
        missing_context: list[str],
        llm_client: Any,
        strong_llm_client: Any = None,
        focused_artifact_id: uuid.UUID | None = None,
        initial_state: dict[str, Any] | None,
        resume_command: Any,
    ) -> None:
        config = self._make_config(session_id, project_id, llm_client, agent_role, strong_llm_client=strong_llm_client)
        timeout = settings.agent_turn_timeout_seconds
        if focused_artifact_id is None:
            async with self.session_factory() as db:
                focused_artifact_id = (
                    await db.execute(
                        select(AgentSession.focused_artifact_id).where(AgentSession.id == session_id)
                    )
                ).scalar_one_or_none()
        try:
            if resume_command is not None:
                # wait_for MUST wrap ainvoke INSIDE this coroutine. Wrapping from outside would raise
                # CancelledError (a BaseException) which `except Exception` cannot catch → session stuck ACTIVE.
                result = await asyncio.wait_for(self.graph.ainvoke(resume_command, config), timeout=timeout)
            else:
                state = initial_state or {
                    "artifact_type": artifact_type,
                    "workflow_area": workflow_area,
                    "step_key": step_key,
                    "messages": [],
                    "conversation_summary": "",
                    "analysis_result": None,
                    "pending_tool_call_ids": [],
                    "last_agent_run_id": None,
                    "turn_count": 0,
                    "missing_context": missing_context,
                    "user_confirmed": None,
                    "locale": None,
                    "turn_type": None,
                    "triage_reply": None,
                    "section_coverage": None,
                    "coverage_complete": None,
                    "section_coverage_stall_count": None,
                    "assumptions": [],
                    "risks": [],
                    "open_questions": [],
                    "focused_artifact_id": (
                        str(focused_artifact_id)
                        if focused_artifact_id is not None
                        else None
                    ),
                    "draft_body": None,
                    "method_profile": dict(DEFAULT_METHOD_PROFILE),
                    "artifact_chain": dict(DEFAULT_ARTIFACT_CHAIN),
                    "readiness": dict(DEFAULT_READINESS),
                    "working_draft": None,
                    "mode_hint": None,
                }
                result = await asyncio.wait_for(self.graph.ainvoke(state, config), timeout=timeout)

            # The graph paused iff its final state carries an __interrupt__. When it instead reached
            # END, a WAITING_FOR_HUMAN left behind by a tool re-running on resume (tool-loop: the tool
            # re-sets WAITING before its interrupt() returns the resume value) is stale → COMPLETED.
            graph_ended = not (isinstance(result, dict) and "__interrupt__" in result)
            async with self.session_factory() as db:
                row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
                if row.status == AgentSessionStatus.ACTIVE or (
                    graph_ended and row.status == AgentSessionStatus.WAITING_FOR_HUMAN
                ):
                    row.status = AgentSessionStatus.COMPLETED
                await db.commit()
        except TimeoutError:
            async with self.session_factory() as db:
                row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
                if row.status not in (AgentSessionStatus.WAITING_FOR_HUMAN, AgentSessionStatus.COMPLETED):
                    row.status = AgentSessionStatus.FAILED
                    db.add(
                        AgentMessage(
                            session_id=session_id,
                            role=AgentMessageRole.AGENT,
                            content=_agent_timeout_message(timeout),
                        )
                    )
                await db.commit()
        except Exception as exc:
            async with self.session_factory() as db:
                row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
                if row.status not in (AgentSessionStatus.WAITING_FOR_HUMAN, AgentSessionStatus.COMPLETED):
                    row.status = AgentSessionStatus.FAILED
                    db.add(
                        AgentMessage(
                            session_id=session_id,
                            role=AgentMessageRole.AGENT,
                            content=_agent_failure_message(exc),
                        )
                    )
                await db.commit()

        # Drain queued messages at the END of every turn — covers both handle_user_message and the
        # approve/reject path via _check_and_resume. Only fires after COMPLETED/FAILED (see _drain_queue);
        # a turn that paused at WAITING_FOR_HUMAN is awaiting a specific input and must not be fed a queued one.
        await self._drain_queue(
            session_id=session_id,
            project_id=project_id,
            artifact_type=artifact_type,
            step_key=step_key,
            workflow_area=workflow_area,
            agent_role=agent_role,
            missing_context=missing_context,
            llm_client=llm_client,
            strong_llm_client=strong_llm_client,
        )

    async def _drain_queue(
        self,
        *,
        session_id: uuid.UUID,
        project_id: uuid.UUID,
        artifact_type: str,
        step_key: str | None,
        workflow_area: str,
        agent_role: str | None,
        missing_context: list[str],
        llm_client: Any,
        strong_llm_client: Any = None,
    ) -> None:
        async with self.session_factory() as db:
            session_row = (
                await db.execute(select(AgentSession).where(AgentSession.id == session_id))
            ).scalar_one()
            # Only drain after a turn truly ended. WAITING_FOR_HUMAN means the graph paused on a
            # specific question/approval — feeding a queued message here would be the wrong input.
            if session_row.status not in (AgentSessionStatus.COMPLETED, AgentSessionStatus.FAILED):
                return

            queued = (
                await db.execute(
                    select(AgentMessage)
                    .where(
                        AgentMessage.session_id == session_id,
                        AgentMessage.role == AgentMessageRole.USER,
                        AgentMessage.payload["queued"].as_boolean().is_(True),
                    )
                    .order_by(AgentMessage.created_at.asc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if queued is None:
                return

            # Mark dequeued (reassign dict so SQLAlchemy detects the JSON change).
            new_payload = dict(queued.payload or {})
            new_payload["queued"] = False
            queued.payload = new_payload
            content = queued.content

            session_row.status = AgentSessionStatus.ACTIVE
            session_row.interrupt_type = None
            await db.commit()

        initial_state = {
            "artifact_type": artifact_type,
            "workflow_area": workflow_area,
            "step_key": step_key,
            "messages": [{"role": "user", "content": content}],
            "conversation_summary": "",
            "analysis_result": None,
            "pending_tool_call_ids": [],
            "last_agent_run_id": None,
            "turn_count": 0,
            "missing_context": missing_context,
            "user_confirmed": None,
            "locale": None,
            "turn_type": None,
            "triage_reply": None,
            "section_coverage": None,
            "coverage_complete": None,
            "section_coverage_stall_count": None,
            "assumptions": [],
            "risks": [],
            "open_questions": [],
            "focused_artifact_id": (
                str(session_row.focused_artifact_id)
                if session_row.focused_artifact_id is not None
                else None
            ),
            "draft_body": None,
            "method_profile": dict(DEFAULT_METHOD_PROFILE),
            "artifact_chain": dict(DEFAULT_ARTIFACT_CHAIN),
            "readiness": dict(DEFAULT_READINESS),
            "working_draft": None,
            "mode_hint": None,
        }
        # Max 1 graph task per session: this runs only after the prior turn finished.
        asyncio.create_task(
            self._run_graph(
                session_id=session_id,
                project_id=project_id,
                artifact_type=artifact_type,
                step_key=step_key,
                workflow_area=workflow_area,
                agent_role=agent_role,
                focused_artifact_id=session_row.focused_artifact_id,
                missing_context=missing_context,
                llm_client=llm_client,
                strong_llm_client=strong_llm_client,
                initial_state=initial_state,
                resume_command=None,
            )
        )

    def _make_config(
        self,
        session_id: uuid.UUID,
        project_id: uuid.UUID,
        llm_client: Any,
        agent_role: str | None = None,
        *,
        strong_llm_client: Any = None,
    ) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": str(session_id),
                "project_id": str(project_id),
                "session_factory": self.session_factory,
                "llm_client": llm_client,
                "strong_llm_client": strong_llm_client,
                "agent_role": agent_role,
            }
        }

    def _resume_command(
        self, session: AgentSession, value: dict[str, Any], state_update: dict[str, Any] | None = None
    ) -> Command:
        interrupt_ids = self._pending_interrupt_ids(session)
        resume = {iid: value for iid in interrupt_ids} if interrupt_ids else value
        # state_update lets a resuming turn seed state (e.g. a one-shot mode_hint) before the
        # interrupted node re-runs — applied by LangGraph as a normal channel update.
        if state_update:
            return Command(resume=resume, update=state_update)
        return Command(resume=resume)

    def _pending_interrupt_ids(self, session: AgentSession) -> list[str]:
        payload = session.graph_checkpoint or {}
        pending_writes = payload.get("pending_writes") or []
        if not pending_writes:
            return []

        current_id = payload.get("checkpoint_id")
        checker = AgentSessionCheckpointer(session_id=str(session.id), session_factory=self.session_factory)
        interrupt_ids: list[str] = []
        seen: set[str] = set()
        for item in reversed(pending_writes):
            if current_id is not None and item.get("checkpoint_id") != current_id:
                continue
            try:
                _, channel, value = checker._load_pending_write(item)
            except Exception:
                continue
            if channel != "__interrupt__":
                continue
            interrupts = value if isinstance(value, list) else [value]
            for interrupt in reversed(interrupts):
                interrupt_id = getattr(interrupt, "id", None)
                if not interrupt_id:
                    continue
                interrupt_id = str(interrupt_id)
                if interrupt_id in seen:
                    continue
                seen.add(interrupt_id)
                interrupt_ids.append(interrupt_id)
        return list(reversed(interrupt_ids))

    async def _resolve_llm_client(self, provider_config_id: uuid.UUID | None) -> tuple[Any, Any | None]:
        if not provider_config_id:
            return None, None
        from app.core.crypto import decrypt_token
        from app.models.llm_provider import LLMProviderConfig
        from app.services.llm_clients import LLMClientFactory

        config_row = (
            await self.db.execute(select(LLMProviderConfig).where(LLMProviderConfig.id == provider_config_id))
        ).scalar_one_or_none()
        if not config_row or not config_row.encrypted_api_key:
            return None, None
        api_key = decrypt_token(config_row.encrypted_api_key)
        if not api_key:
            raise ValueError("API key không thể giải mã — có thể lệch key rotation")
        secret_key = None
        if config_row.encrypted_secret_key:
            secret_key = decrypt_token(config_row.encrypted_secret_key)
            if not secret_key:
                raise ValueError("secret_key không thể giải mã — có thể lệch key rotation")
        default_client = LLMClientFactory.create(
            provider_type=config_row.provider_type,
            api_key=api_key,
            secret_key=secret_key,
            model=config_row.model_name,
            region=config_row.region,
        )
        strong_client = None
        if config_row.strong_model_name:
            strong_client = LLMClientFactory.create(
                provider_type=config_row.provider_type,
                api_key=api_key,
                secret_key=secret_key,
                model=config_row.strong_model_name,
                region=config_row.region,
            )
        return default_client, strong_client


def _agent_timeout_message(timeout: float) -> str:
    return (
        f"Agent mất quá nhiều thời gian để phản hồi (quá {int(timeout)}s) nên lượt này đã được dừng. "
        "Bạn vui lòng thử gửi lại yêu cầu."
    )


def _agent_failure_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return f"Agent không thể hoàn tất lượt phân tích hiện tại. Lý do kỹ thuật: {message[:500]}"


def _session_ui_status(status: Any, interrupt_type: Any) -> str:
    status_val = getattr(status, "value", status)
    interrupt_val = getattr(interrupt_type, "value", interrupt_type)
    if status_val == AgentSessionStatus.ACTIVE.value:
        return "processing"
    if status_val == AgentSessionStatus.WAITING_FOR_HUMAN.value:
        if interrupt_val == AgentSessionInterruptType.PROPOSE_ARTIFACTS.value:
            return "waiting_approval"
        return "waiting_input"
    if status_val == AgentSessionStatus.FAILED.value:
        return "error"
    return "idle"
