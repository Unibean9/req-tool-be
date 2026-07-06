import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from langgraph.types import Command
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.documents.registry import all_container_types, container_for
from app.graphs.checkpointer import AgentSessionCheckpointer
from app.graphs.gate_logging import log_gate_decision
from app.graphs.lifecycle_context import has_stale_curation
from app.graphs.policy import ARTIFACT_PREDECESSORS
from app.graphs.state import build_initial_workflow_state
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
    ArtifactLink,
    ArtifactStatus,
    ArtifactVersion,
    ChangeSource,
    RelationType,
)
from app.schemas.agent import AgentSessionResponse
from app.schemas.artifact import ArtifactLinkCreateRequest
from app.schemas.artifact_synthesis import (
    evaluate_candidate_readiness,
    synthesis_metadata_dict,
    synthesis_metadata_from_snapshot,
)
from app.services.agent_tool_visibility import public_tool_call_filter
from app.services.artifact_service import ArtifactInUseError, ArtifactLinkService, ArtifactService
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)


def _snapshot_base_version_id(snapshot: dict[str, Any]) -> uuid.UUID | None:
    try:
        return synthesis_metadata_from_snapshot(snapshot).base_version_id
    except ValueError:
        raw_base = snapshot.get("base_version_id")
        return uuid.UUID(str(raw_base)) if raw_base else None


def _stale_base_version_detail(
    *,
    snapshot: dict[str, Any],
    requested_base_version_id: uuid.UUID | None,
    current_version_id: uuid.UUID | None,
) -> dict[str, Any] | None:
    base_version_id = (
        requested_base_version_id if requested_base_version_id is not None else _snapshot_base_version_id(snapshot)
    )
    if base_version_id == current_version_id:
        return None
    return {
        "detail": "Revision draft is based on an old version",
        "base_version_id": str(base_version_id) if base_version_id else None,
        "current_version_id": str(current_version_id) if current_version_id else None,
    }


_CANDIDATE_READINESS_REJECTION_DETAIL = "Candidate is not ready enough to persist as an official version"


def _candidate_readiness_rejection_feedback(
    snapshot: dict[str, Any], exc: HTTPException
) -> dict[str, Any] | None:
    if exc.status_code != 422 or not isinstance(exc.detail, dict):
        return None
    if exc.detail.get("detail") != _CANDIDATE_READINESS_REJECTION_DETAIL:
        return None
    return {
        "candidate_readiness_rejection": {
            **exc.detail,
            "focused_artifact_id": snapshot.get("focused_artifact_id"),
        }
    }


def _feedback_summary_recovery_reason(feedback_summary: dict[str, Any] | None, user_message: str | None) -> str:
    if not feedback_summary:
        return "user_requested_edit" if user_message is not None else "proposal_superseded"
    for key in ("stale_base_version", "candidate_readiness_rejection", "lifecycle_persist_rejection"):
        if key in feedback_summary:
            return key
    return next(iter(feedback_summary), "feedback_summary")


def _is_artifact_body_proposal(tool_name: str) -> bool:
    return tool_name == "create_artifact" or tool_name.startswith("write_draft:")


def _approval_tool_kind(tool_name: str) -> str:
    if tool_name.startswith("create_artifact_link:"):
        return "create_artifact_link"
    if tool_name.startswith("propose_retirement:"):
        return "propose_retirement"
    return tool_name


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
        # Soft gate: missing predecessors no longer block session creation — every artifact stays
        # navigable, and `missing_context` (already returned in the response) carries the warning.
        missing = await self._check_predecessors(project_id, artifact_type)
        if focused_artifact_id is not None:
            focused = await self.db.get(Artifact, focused_artifact_id)
            if focused is None or focused.project_id != project_id:
                raise HTTPException(422, detail="focused_artifact_id does not belong to the project")
            if focused.parent_id is None:
                raise HTTPException(422, detail="Agent must focus on a document item, not a container")
            if focused.type.value != artifact_type:
                raise HTTPException(
                    422,
                    detail="artifact_type must match the focused document item",
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
                if focused.type.value in all_container_types():
                    return focused.type.value
        if session.artifact_type in all_container_types():
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
            raise HTTPException(404, detail="Agent session not found")
        return session

    async def _load_graph_state_values(self, session_id: uuid.UUID) -> dict[str, Any] | None:
        if self.graph is None:
            return None
        try:
            snapshot = await self.graph.aget_state({"configurable": {"thread_id": str(session_id)}})
        except Exception as exc:
            raise HTTPException(500, detail="Cannot read checkpoint workspace") from exc
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
        # The queue row carries the message content AND its mode_hint so the drained turn replays the
        # user's requested mode exactly as a fresh message would.
        #
        # Exception: ACTIVE + STREAM_RESPONSE means the graph halted via interrupt() while keeping
        # status=ACTIVE (conversational Q&A). This is not a "busy" session — it is waiting for a
        # reply. Fall through to the resume path below rather than queuing.
        if session.status == AgentSessionStatus.ACTIVE:
            if session.interrupt_type != AgentSessionInterruptType.STREAM_RESPONSE:
                return await self._queue_message(session.id, content, mode_hint)
        if session.status in (AgentSessionStatus.COMPLETED, AgentSessionStatus.FAILED):
            raise HTTPException(400, detail="Session has ended and cannot accept more messages")
        # status == WAITING_FOR_HUMAN or ACTIVE+STREAM_RESPONSE below.
        # PROPOSE_ARTIFACTS waits for an approval decision, not free-text — queue the text (no carve-out).
        if session.interrupt_type == AgentSessionInterruptType.PROPOSE_ARTIFACTS:
            return await self._queue_message(session.id, content, mode_hint)
        if session.interrupt_type not in (
            AgentSessionInterruptType.ASK_HUMAN,
            AgentSessionInterruptType.STREAM_RESPONSE,
            None,
        ):
            raise HTTPException(400, detail="Session is not waiting for a user message")

        is_first_message = session.interrupt_type is None

        strong_llm_client = None
        if llm_client is None:
            llm_client, strong_llm_client = await self._resolve_llm_client(session.provider_config_id)

        msg = AgentMessage(session_id=session.id, role=AgentMessageRole.USER, content=content)
        self.db.add(msg)
        session.status = AgentSessionStatus.ACTIVE
        session.interrupt_type = None
        await self.db.commit()

        if is_first_message:
            initial_state = build_initial_workflow_state(
                artifact_type=session.artifact_type,
                workflow_area=session.workflow_area,
                step_key=session.step_key,
                messages=[{"role": "user", "content": content}],
                missing_context=session.missing_context or [],
                focused_artifact_id=session.focused_artifact_id,
                mode_hint=mode_hint,
            )
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

    async def _queue_message(
        self, session_id: uuid.UUID, content: str, mode_hint: str | None = None
    ) -> AgentMessage:
        """Persist a user message as queued (payload.queued=True) without starting a graph turn.

        The mode_hint (if any) rides the payload so _drain_queue can replay it. Drained later by
        _drain_queue once the current turn ends COMPLETED/FAILED.
        """
        payload: dict[str, Any] = {"queued": True}
        if mode_hint:
            payload["mode_hint"] = mode_hint
        msg = AgentMessage(
            session_id=session_id,
            role=AgentMessageRole.USER,
            content=content,
            payload=payload,
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
            (
                await self.db.execute(
                    select(AgentMessage).where(AgentMessage.session_id == session_id).order_by(AgentMessage.created_at)
                )
            )
            .scalars()
            .all()
        )
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
            (
                await self.db.execute(
                    select(AgentToolCall)
                    .join(AgentRun, AgentToolCall.run_id == AgentRun.id)
                    .where(AgentRun.session_id == session_id)
                    .where(public_tool_call_filter())
                    .order_by(AgentToolCall.created_at)
                )
            )
            .scalars()
            .all()
        )
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
            raise HTTPException(400, detail="Tool call has been rejected")
        if tool_call.status != AgentToolCallStatus.PROPOSED:
            raise HTTPException(400, detail="Tool call is not in proposed status")

        snapshot = tool_call.input_snapshot or {}
        artifact: Artifact | None = None
        version: ArtifactVersion | None = None
        tool_kind = _approval_tool_kind(tool_call.tool_name)
        if _is_artifact_body_proposal(tool_call.tool_name):
            stale_detail = await self._guard_current_base_version(project_id, snapshot, None, raise_on_stale=False)
            if stale_detail is not None:
                await self._supersede_tool_call_for_in_loop_recovery(
                    project_id=project_id,
                    session_id=session_id,
                    tool_call=tool_call,
                    feedback_summary={
                        "stale_base_version": {**stale_detail, "artifact_id": snapshot.get("focused_artifact_id")}
                    },
                    llm_client=_llm_client,
                )
                raise HTTPException(409, detail=stale_detail)
            lifecycle_rejection = await self._guard_lifecycle_predecessors(project_id, snapshot)
            if lifecycle_rejection is not None:
                await self._supersede_tool_call_for_in_loop_recovery(
                    project_id=project_id,
                    session_id=session_id,
                    tool_call=tool_call,
                    feedback_summary={"lifecycle_persist_rejection": lifecycle_rejection},
                    llm_client=_llm_client,
                )
                raise HTTPException(409, detail=lifecycle_rejection)
            try:
                artifact, version = await self._execute_create_artifact(
                    project_id=project_id,
                    snapshot=snapshot,
                    run_id=tool_call.run_id,
                    tool_call_id=tool_call.id,
                    created_by_id=created_by_id,
                )
            except HTTPException as exc:
                feedback_summary = _candidate_readiness_rejection_feedback(snapshot, exc)
                if feedback_summary is None:
                    raise
                await self._supersede_tool_call_for_in_loop_recovery(
                    project_id=project_id,
                    session_id=session_id,
                    tool_call=tool_call,
                    feedback_summary=feedback_summary,
                    llm_client=_llm_client,
                )
                raise
        elif tool_kind == "create_artifact_link":
            link = await self._execute_create_artifact_link(
                project_id=project_id,
                session_id=session_id,
                snapshot=snapshot,
                created_by_id=created_by_id,
            )
            snapshot = {**snapshot, "created_link_id": str(link.id)}
            tool_call.input_snapshot = snapshot
        elif tool_kind == "propose_retirement":
            artifact = await self._execute_retirement(
                project_id=project_id,
                session_id=session_id,
                snapshot=snapshot,
                created_by_id=created_by_id,
            )
        else:
            raise HTTPException(400, detail=f"Unsupported approval tool: {tool_call.tool_name}")

        tool_call.status = AgentToolCallStatus.EXECUTED
        if artifact is not None:
            tool_call.created_artifact_id = artifact.id
        if version is not None:
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
            raise HTTPException(400, detail="Tool call has been approved")
        if tool_call.status != AgentToolCallStatus.PROPOSED:
            raise HTTPException(400, detail="Tool call is not in proposed status")

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
            raise HTTPException(400, detail="Tool call is not in proposed status")
        # request_edit resumes the graph, so a stale base is recovered in-loop rather than 409'd: the
        # condition is seeded into feedback_summary so the resumed turn re-reads and rebases.
        snapshot = tool_call.input_snapshot or {}
        feedback_summary: dict[str, Any] | None = None
        if _is_artifact_body_proposal(tool_call.tool_name):
            stale_detail = await self._guard_current_base_version(
                project_id, snapshot, base_version_id, raise_on_stale=False
            )
        else:
            stale_detail = None
        if stale_detail is not None:
            # Seeded as a single-key feedback_summary on the resume boundary. This replaces (not merges)
            # the channel, but that is safe here: orchestrator_node rebuilds the transient signals
            # (resurfaced_questions, sweep gaps, dropped/out-of-phase tools) every turn, and resetting
            # the ignored-signal counter at a human edit boundary is intentional — the same rationale as
            # the turn_count reset in _resume_command (a human just intervened, so "ignored for N turns"
            # restarts). orchestrator_node preserves stale_base_version (it never pops the key).
            feedback_summary = {
                "stale_base_version": {**stale_detail, "artifact_id": snapshot.get("focused_artifact_id")}
            }

        await self._supersede_tool_call_for_in_loop_recovery(
            project_id=project_id,
            session_id=session_id,
            tool_call=tool_call,
            feedback_summary=feedback_summary,
            user_message=note,
            llm_client=llm_client,
        )
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
            raise HTTPException(404, detail="Tool call does not exist")
        tool_call, session_id = row
        return tool_call, session_id

    async def _supersede_tool_call_for_in_loop_recovery(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        tool_call: AgentToolCall,
        feedback_summary: dict[str, Any] | None,
        user_message: str | None = None,
        llm_client: Any = None,
    ) -> None:
        tool_call.status = AgentToolCallStatus.SUPERSEDED
        tool_call.resolved_at = datetime.now(UTC)
        if user_message is not None:
            self.db.add(AgentMessage(session_id=session_id, role=AgentMessageRole.USER, content=user_message))
        await self.db.commit()

        log_gate_decision(
            "in_loop_feedback_recovery",
            "seeded",
            reason=_feedback_summary_recovery_reason(feedback_summary, user_message),
            extra={
                "session_id": str(session_id),
                "tool_call_id": str(tool_call.id),
                "tool_name": tool_call.tool_name,
            },
        )
        state_update = {"feedback_summary": feedback_summary} if feedback_summary is not None else None
        # The tool_call is already committed SUPERSEDED and the feedback is seeded; the resume is a
        # best-effort continuation. Never let a resume failure propagate — callers on the approve
        # path raise the client-facing 409/422 immediately after this and must not have it masked
        # by a 500 from the resume.
        try:
            await self._check_and_resume(
                project_id=project_id,
                session_id=session_id,
                llm_client=llm_client,
                state_update=state_update,
            )
        except Exception:
            logger.exception(
                "in-loop recovery resume failed for session %s tool_call %s", session_id, tool_call.id
            )

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
                detail="Tool call is missing focused_artifact_id; choose the current document item",
            )
        title = snapshot.get("title", "Untitled")
        body = snapshot.get("body", "")
        try:
            synthesis_metadata = synthesis_metadata_dict(snapshot)
        except ValueError as exc:
            raise HTTPException(422, detail="Tool call metadata synthesis is invalid") from exc
        lifecycle_metadata = snapshot.get("lifecycle_metadata")
        if not isinstance(lifecycle_metadata, dict):
            lifecycle_metadata = {}
        source_evidence = snapshot.get("source_evidence")
        if not isinstance(source_evidence, list):
            source_evidence = None
        body = str(body or "").strip()
        snapshot["body"] = body
        self._validate_candidate_readiness_for_persist(snapshot, synthesis_metadata)

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
                metadata={**synthesis_metadata, **lifecycle_metadata},
                auto_evidence=source_evidence,
                mark_accepted=True,
            )
        except ValueError as exc:
            raise HTTPException(404, detail="Focused document item does not exist") from exc

        return artifact, version

    async def _approval_actor_id(
        self,
        *,
        session_id: uuid.UUID,
        created_by_id: uuid.UUID | None,
    ) -> uuid.UUID:
        if created_by_id is not None:
            return created_by_id
        actor_id = await self.db.scalar(select(AgentSession.created_by_id).where(AgentSession.id == session_id))
        if actor_id is None:
            raise HTTPException(422, detail="Approval requires a project member user")
        return actor_id

    async def _execute_create_artifact_link(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        snapshot: dict[str, Any],
        created_by_id: uuid.UUID | None,
    ) -> ArtifactLink:
        try:
            body = ArtifactLinkCreateRequest(
                source_artifact_id=uuid.UUID(str(snapshot.get("source_artifact_id"))),
                target_artifact_id=uuid.UUID(str(snapshot.get("target_artifact_id"))),
                relation_type=RelationType(str(snapshot.get("relation_type"))),
                metadata=snapshot.get("metadata") if isinstance(snapshot.get("metadata"), dict) else {},
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, detail="Tool call link input is invalid") from exc
        actor_id = await self._approval_actor_id(session_id=session_id, created_by_id=created_by_id)
        try:
            response = await ArtifactLinkService(self.db).create(
                project_id=project_id,
                body=body,
                created_by_id=actor_id,
            )
        except PermissionError as exc:
            raise HTTPException(403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        link = await self.db.get(ArtifactLink, response.id)
        if link is None:
            raise HTTPException(500, detail="Artifact link was not created")
        return link

    async def _execute_retirement(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        snapshot: dict[str, Any],
        created_by_id: uuid.UUID | None,
    ) -> Artifact:
        try:
            artifact_id = uuid.UUID(str(snapshot.get("artifact_id")))
            superseded_by_raw = snapshot.get("superseded_by_artifact_id")
            superseded_by_id = uuid.UUID(str(superseded_by_raw)) if superseded_by_raw else None
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, detail="Tool call retirement input is invalid") from exc
        reason = str(snapshot.get("reason") or "").strip()
        if not reason:
            raise HTTPException(422, detail="Tool call retirement input is missing reason")
        actor_id = await self._approval_actor_id(session_id=session_id, created_by_id=created_by_id)
        try:
            return await ArtifactService(self.db).archive_artifact(
                project_id=project_id,
                artifact_id=artifact_id,
                user_id=actor_id,
                reason=reason,
                superseded_by_id=superseded_by_id,
                source="agent_retirement",
            )
        except ArtifactInUseError as exc:
            raise HTTPException(
                409,
                detail={
                    "message": "Artifact has live downstream dependents",
                    "artifact_ids": [str(artifact_id) for artifact_id in exc.artifact_ids],
                },
            ) from exc
        except PermissionError as exc:
            raise HTTPException(403, detail=str(exc)) from exc
        except ValueError as exc:
            status_code = 404 if "not found" in str(exc).lower() else 400
            raise HTTPException(status_code, detail=str(exc)) from exc

    def _validate_candidate_readiness_for_persist(
        self,
        snapshot: dict[str, Any],
        synthesis_metadata: dict[str, Any],
    ) -> None:
        readiness = evaluate_candidate_readiness(
            artifact_type=str(snapshot.get("artifact_type") or synthesis_metadata.get("artifact_type") or ""),
            body=str(snapshot.get("body") or ""),
            synthesis_metadata=synthesis_metadata,
        )
        if readiness.can_persist:
            return
        log_gate_decision(
            "candidate_readiness_persist",
            "rejected_422",
            reason=readiness.state.value,
            extra={"focused_artifact_id": snapshot.get("focused_artifact_id")},
        )
        raise HTTPException(
            422,
            detail={
                "detail": "Candidate is not ready enough to persist as an official version",
                **readiness.model_dump(mode="json"),
            },
        )

    async def _guard_lifecycle_predecessors(
        self,
        project_id: uuid.UUID,
        snapshot: dict[str, Any],
    ) -> dict[str, Any] | None:
        lifecycle_metadata = snapshot.get("lifecycle_metadata")
        if not isinstance(lifecycle_metadata, dict):
            return None
        based_on = lifecycle_metadata.get("based_on")
        if not isinstance(based_on, dict) or not based_on:
            return None

        stale_predecessors: list[dict[str, Any]] = []
        # Deadlock-avoidance invariant: predecessor rows are always locked FOR UPDATE in a canonical
        # order (sorted by artifact_id) so two concurrent approves with overlapping predecessor sets
        # can never acquire the same two rows in opposite order. The focused artifact is locked first
        # by _guard_current_base_version; the artifact DAG is acyclic, so a focused artifact is never
        # also a predecessor in a conflicting approve, keeping the global order consistent. Any future
        # caller that locks predecessor rows MUST preserve this sort.
        for predecessor_id, based_on_version_id in sorted((str(k), str(v)) for k, v in based_on.items()):
            try:
                artifact_id = uuid.UUID(predecessor_id)
            except (TypeError, ValueError):
                stale_predecessors.append(
                    {
                        "artifact_id": predecessor_id,
                        "based_on_version_id": based_on_version_id,
                        "current_version_id": None,
                        "reason": "invalid_artifact_id",
                    }
                )
                continue
            predecessor = (
                await self.db.execute(
                    select(Artifact)
                    .where(Artifact.project_id == project_id)
                    .where(Artifact.id == artifact_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if predecessor is None:
                stale_predecessors.append(
                    {
                        "artifact_id": predecessor_id,
                        "based_on_version_id": based_on_version_id,
                        "current_version_id": None,
                        "reason": "missing_predecessor",
                    }
                )
                continue
            current_version_id = str(predecessor.current_version_id) if predecessor.current_version_id else None
            if predecessor.status == ArtifactStatus.ARCHIVED:
                reason = "retired_predecessor"
            elif current_version_id != based_on_version_id:
                reason = "predecessor_version_changed"
            else:
                continue
            stale_predecessors.append(
                {
                    "artifact_id": predecessor_id,
                    "based_on_version_id": based_on_version_id,
                    "current_version_id": current_version_id,
                    "reason": reason,
                }
            )

        if not stale_predecessors:
            return None

        curation_decision = lifecycle_metadata.get("curation_decision")
        curation_ok = isinstance(curation_decision, dict) and has_stale_curation(
            {
                "curation_action": curation_decision.get("action"),
                "curation_justification": curation_decision.get("justification"),
            }
        )
        log_gate_decision(
            "lifecycle_persist_guard",
            "rejected_409",
            reason="predecessor_diverged",
            extra={
                "focused_artifact_id": snapshot.get("focused_artifact_id"),
                "stale_predecessor_count": len(stale_predecessors),
                "curation_declared": curation_ok,
            },
        )
        return {
            "detail": "Proposal is based on predecessor versions that are no longer current",
            "focused_artifact_id": snapshot.get("focused_artifact_id"),
            "stale_predecessors": stale_predecessors,
            "curation_declared": curation_ok,
        }

    async def _guard_current_base_version(
        self,
        project_id: uuid.UUID,
        snapshot: dict[str, Any],
        requested_base_version_id: uuid.UUID | None,
        *,
        raise_on_stale: bool = True,
    ) -> dict[str, Any] | None:
        """Detect a stale base version.

        With ``raise_on_stale=False`` it returns the stale detail so callers can seed in-loop feedback
        before preserving their HTTP-level error contract.
        """
        focused_artifact_id = snapshot.get("focused_artifact_id")
        if not focused_artifact_id:
            raise HTTPException(422, detail="Tool call missing focused_artifact_id")
        try:
            focused = await DocumentService(self.db).get_document_item_artifact(
                artifact_id=uuid.UUID(str(focused_artifact_id)),
                project_id=project_id,
                for_update=True,
            )
        except ValueError as exc:
            raise HTTPException(404, detail="Focused document item does not exist") from exc

        detail = _stale_base_version_detail(
            snapshot=snapshot,
            requested_base_version_id=requested_base_version_id,
            current_version_id=focused.current_version_id,
        )
        if detail is not None and raise_on_stale:
            raise HTTPException(409, detail=detail)
        return detail

    async def _check_and_resume(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        llm_client: Any = None,
        state_update: dict[str, Any] | None = None,
    ) -> None:
        session_row = (
            await self.db.execute(select(AgentSession).where(AgentSession.id == session_id).with_for_update())
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

        resume_command = self._resume_command(session_row, {"all_resolved": True}, state_update=state_update)
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
            await self.db.execute(select(AgentSession).where(AgentSession.id == session_id).with_for_update())
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
                    await db.execute(select(AgentSession.focused_artifact_id).where(AgentSession.id == session_id))
                ).scalar_one_or_none()
        try:
            if resume_command is not None:
                # wait_for MUST wrap ainvoke INSIDE this coroutine. Wrapping from outside would raise
                # CancelledError (a BaseException) which `except Exception` cannot catch → session stuck ACTIVE.
                result = await asyncio.wait_for(self.graph.ainvoke(resume_command, config), timeout=timeout)
            else:
                state = initial_state or build_initial_workflow_state(
                    artifact_type=artifact_type,
                    workflow_area=workflow_area,
                    step_key=step_key,
                    messages=[],
                    missing_context=missing_context,
                    focused_artifact_id=focused_artifact_id,
                    mode_hint=None,
                )
                result = await asyncio.wait_for(self.graph.ainvoke(state, config), timeout=timeout)

            # Only END (no __interrupt__) means the turn truly finished. A status left behind by a
            # paused tool is stale and must not be promoted: WAITING_FOR_HUMAN re-set by a tool
            # re-running on resume, or ACTIVE+STREAM_RESPONSE while halted on a conversational ask (D4).
            graph_ended = not (isinstance(result, dict) and "__interrupt__" in result)
            # A graph that ENDs while its last message still carries tool_calls did not finish: it hit
            # the per-request turn cap in route_node before the pending interrupt-bearing tool (e.g.
            # ask_user) could run, so no __interrupt__ surfaced. Since turn_count resets each human turn,
            # tripping the cap means the model looped silently within one request without ever
            # interacting — a genuine runaway, not a long conversation. Fail it loudly (mirroring the
            # timeout branch) rather than mislabel it COMPLETED.
            turn_limit_hit = graph_ended and _result_has_pending_tool_calls(result)
            async with self.session_factory() as db:
                row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
                if turn_limit_hit and row.status != AgentSessionStatus.COMPLETED:
                    row.status = AgentSessionStatus.FAILED
                    db.add(
                        AgentMessage(
                            session_id=session_id,
                            role=AgentMessageRole.AGENT,
                            content=_agent_turn_limit_message(),
                        )
                    )
                elif graph_ended and row.status in (AgentSessionStatus.ACTIVE, AgentSessionStatus.WAITING_FOR_HUMAN):
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
            session_row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
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
            queued_mode_hint = new_payload.get("mode_hint")

            session_row.status = AgentSessionStatus.ACTIVE
            session_row.interrupt_type = None
            await db.commit()

        initial_state = build_initial_workflow_state(
            artifact_type=artifact_type,
            workflow_area=workflow_area,
            step_key=step_key,
            messages=[{"role": "user", "content": content}],
            missing_context=missing_context,
            focused_artifact_id=session_row.focused_artifact_id,
            mode_hint=queued_mode_hint,
        )
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
        # A resume is a human-in-the-loop boundary (reply or approval), so the silent-loop circuit
        # breaker resets here: turn_count counts internal steps per request, not across the session.
        # The diagnosis judge budget is per-turn (not cumulative), so it resets on the same boundary —
        # otherwise only the first high-risk section in a session could ever escalate.
        # state_update lets a resuming turn also seed state (e.g. a one-shot mode_hint) before the
        # interrupted node re-runs — applied by LangGraph as a normal channel update.
        return Command(resume=resume, update={"turn_count": 0, "diagnosis_judge_calls_used": 0, **(state_update or {})})

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
        from app.models.llm_provider import LLMProviderConfig, LLMProviderStatus
        from app.services.llm_clients import LLMClientFactory

        config_row = (
            await self.db.execute(select(LLMProviderConfig).where(LLMProviderConfig.id == provider_config_id))
        ).scalar_one_or_none()
        if not config_row or not config_row.encrypted_api_key:
            return None, None
        if config_row.status != LLMProviderStatus.ACTIVE:
            raise HTTPException(422, detail="LLM provider config must pass health check before use")
        api_key = decrypt_token(config_row.encrypted_api_key)
        if not api_key:
            raise ValueError("API key cannot be decrypted - key rotation may be out of sync")
        secret_key = None
        if config_row.encrypted_secret_key:
            secret_key = decrypt_token(config_row.encrypted_secret_key)
            if not secret_key:
                raise ValueError("secret_key cannot be decrypted - key rotation may be out of sync")
        default_client = LLMClientFactory.create(
            provider_type=config_row.provider_type,
            api_key=api_key,
            secret_key=secret_key,
            model=config_row.model_name,
            region=config_row.region,
            base_url=config_row.base_url,
        )
        strong_client = None
        if config_row.strong_model_name:
            strong_client = LLMClientFactory.create(
                provider_type=config_row.provider_type,
                api_key=api_key,
                secret_key=secret_key,
                model=config_row.strong_model_name,
                region=config_row.region,
                base_url=config_row.base_url,
            )
        return default_client, strong_client


def _result_has_pending_tool_calls(result: Any) -> bool:
    """Whether the graph's final state ends on a message that still carries undispatched tool_calls.

    True only when route_node returned END (turn cap) while analyze_node had emitted an
    interrupt-bearing tool that never ran — the signature of a force-terminated, non-resumable turn.
    """
    if not isinstance(result, dict):
        return False
    messages = result.get("messages") or []
    return bool(messages) and bool(getattr(messages[-1], "tool_calls", None))


def _agent_turn_limit_message() -> str:
    return (
        "The session reached the analysis turn limit and was stopped before completion. "
        "Please create a new session to continue."
    )


def _agent_timeout_message(timeout: float) -> str:
    return (
        f"Agent took too long to respond (over {int(timeout)}s) so this turn was stopped. "
        "Please send the request again."
    )


def _agent_failure_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return f"Agent could not complete the current analysis turn. Technical reason: {message[:500]}"


def _session_ui_status(status: Any, interrupt_type: Any) -> str:
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
