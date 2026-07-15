import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException
from langgraph.types import Command
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.documents.registry import all_container_types, container_for
from app.graphs.analysis.turn_outcome_projector import check_ownership_fence, project_terminal_outcome
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
    AgentTurnEnvelope,
    AgentTurnTrigger,
    AgentTurnTriggerType,
    TurnOutcomeType,
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
from app.services.agent_turn_job_service import AgentTurnJobService
from app.services.agent_turn_service import AgentTurnService
from app.services.artifact_service import ArtifactInUseError, ArtifactLinkService, ArtifactService
from app.services.document_service import DocumentService
from app.services.draft_command_service import (
    DraftCommandService,
    canonical_artifact_link_intent,
    canonical_retirement_intent,
    create_artifact_link_logical_command_id,
    propose_retirement_logical_command_id,
)

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

    async def _latest_message_is_direct_response(self, session_id: uuid.UUID) -> bool:
        """Distinguish a direct-response resting state from a new session with no chat turn."""
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

        def _new_session() -> AgentSession:
            return AgentSession(
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

        try:
            session = _new_session()
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

            if existing is not None:
                locked_existing = (
                    await self.db.execute(existing_query.with_for_update())
                ).scalar_one_or_none()
                if locked_existing is not None and await expire_abandoned_session(self.db, locked_existing):
                    await self.db.commit()
                    try:
                        session = _new_session()
                        self.db.add(session)
                        await self.db.flush()
                        await self.db.commit()
                    except IntegrityError:
                        await self.db.rollback()
                        raise HTTPException(
                            409,
                            detail={"detail": "Active session already exists", "session_id": None},
                        ) from None
                    return await self.create_session_response(session, missing)

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
        idempotency_key: str | None = None,
    ) -> AgentMessage:
        admitted = None
        if settings.agent_turn_admission_enabled:
            # Không đọc status rồi quyết định ở đây: admission giữ session row lock để một
            # checkpoint chỉ có một turn inline; turn đến sau được persist dạng queue.
            admitted = await AgentTurnService(self.db).admit_user_message(
                project_id=project_id,
                session_id=session_id,
                user_id=user_id,
                content=content,
                idempotency_key=idempotency_key,
                mode_hint=mode_hint,
            )
            if admitted.duplicate or admitted.queued:
                return admitted.message
            session = await self.get_session(project_id=project_id, session_id=session_id, user_id=user_id)
        else:
            session = await self.get_session(project_id=project_id, session_id=session_id, user_id=user_id)

        # S2 — never silently drop a valid message while the agent is busy. Queue it and return 200.
        # The queue row carries the message content AND its mode_hint so the drained turn replays the
        # user's requested mode exactly as a fresh message would.
        #
        # Exception: ACTIVE + STREAM_RESPONSE means the graph halted via interrupt() while keeping
        # status=ACTIVE (conversational Q&A). This is not a "busy" session — it is waiting for a
        # reply. Fall through to the resume path below rather than queuing.
        if not settings.agent_turn_admission_enabled and session.status == AgentSessionStatus.ACTIVE:
            if session.interrupt_type != AgentSessionInterruptType.STREAM_RESPONSE:
                return await self._queue_message(
                    session.id,
                    content,
                    mode_hint,
                    project_id=project_id,
                    user_id=user_id,
                    idempotency_key=idempotency_key,
                )
        if not settings.agent_turn_admission_enabled and session.status in (
            AgentSessionStatus.COMPLETED,
            AgentSessionStatus.FAILED,
            AgentSessionStatus.EXPIRED,
        ):
            raise HTTPException(400, detail="Session has ended and cannot accept more messages")
        # status == WAITING_FOR_HUMAN, TURN_FAILED, or ACTIVE+STREAM_RESPONSE below.
        # PROPOSE_ARTIFACTS waits for an approval decision, not free-text — queue the text (no carve-out).
        if (
            not settings.agent_turn_admission_enabled
            and session.interrupt_type == AgentSessionInterruptType.PROPOSE_ARTIFACTS
        ):
            return await self._queue_message(
                session.id, content, mode_hint, project_id=project_id, user_id=user_id, idempotency_key=idempotency_key
            )
        if not settings.agent_turn_admission_enabled and session.interrupt_type not in (
            AgentSessionInterruptType.ASK_HUMAN,
            AgentSessionInterruptType.STREAM_RESPONSE,
            None,
        ):
            raise HTTPException(400, detail="Session is not waiting for a user message")

        # TURN_FAILED and direct responses both have no interrupt. They must continue from the
        # checkpoint with partial state; treating them as a first message would erase prior context.
        decision_status = admitted.prior_status if admitted is not None else session.status
        decision_interrupt_type = admitted.prior_interrupt_type if admitted is not None else session.interrupt_type
        # Must be read here, before this turn's own message is inserted below (admission already
        # inserted its message earlier, under the same row lock, before computing this snapshot).
        decision_latest_message_is_direct_response = (
            admitted.prior_latest_message_is_direct_response
            if admitted is not None
            else await self._latest_message_is_direct_response(session.id)
        )

        strong_llm_client = None
        if llm_client is None:
            llm_client, strong_llm_client = await self._resolve_llm_client(session.provider_config_id)

        admitted_turn_id: uuid.UUID | None = None
        if admitted is not None:
            msg = admitted.message
            admitted_turn_id = admitted.turn_id
        else:
            msg = AgentMessage(session_id=session.id, role=AgentMessageRole.USER, content=content)
            self.db.add(msg)
            session.status = AgentSessionStatus.ACTIVE
            session.interrupt_type = None
            await self.db.commit()

        initial_state, resume_command = await self._build_user_message_turn_state(
            session=session,
            content=content,
            mode_hint=mode_hint,
            decision_status=decision_status,
            decision_interrupt_type=decision_interrupt_type,
            decision_latest_message_is_direct_response=decision_latest_message_is_direct_response,
        )

        def _runner_factory(
            *,
            turn_id: uuid.UUID | None = None,
            owner_id: str | None = None,
            ownership_generation: int | None = None,
        ):
            return self._run_graph(
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
                turn_id=turn_id,
                owner_id=owner_id,
                ownership_generation=ownership_generation,
            )

        if admitted_turn_id is not None:
            # execution_mode is read from the cohort snapshotted at admission time, never from
            # live settings here — a process config change must never change an already-admitted
            # turn's dispatch route (Compatibility contract). Durable dispatch is exercised only in
            # test/CI at this point; no default/env config enables it.
            admitted_cohort = admitted.cohort or {} if admitted is not None else {}
            execution_mode = admitted_cohort.get("execution_mode", "inline")
            if execution_mode == "durable":
                async with self.session_factory() as db:
                    await AgentTurnJobService(db).enqueue(
                        turn_id=admitted_turn_id,
                        expected_transition_version=0,
                        cohort=admitted.cohort or {},
                    )
            else:
                asyncio.create_task(
                    self._run_admitted_graph(turn_id=admitted_turn_id, runner_factory=_runner_factory)
                )
        else:
            asyncio.create_task(_runner_factory())

        return msg

    async def _run_admitted_graph(self, *, turn_id: uuid.UUID, runner_factory: Any) -> None:
        """Adapter inline dùng cùng ownership contract với durable worker tương lai.

        `runner_factory` builds the graph-invocation coroutine lazily, called only after the claim
        below succeeds — this is how owner_id/ownership_generation (unknown until claimed) reach the
        tool handler's config (command boundary) without executing any graph code early.
        """
        owner_id = f"inline:{uuid.uuid4()}"
        async with self.session_factory() as db:
            generation = await AgentTurnService(db).claim_inline(turn_id=turn_id, owner_id=owner_id)
        if generation is None:
            logger.warning("agent_turn_claim_conflict turn_id=%s", turn_id)
            return
        runner = runner_factory(turn_id=turn_id, owner_id=owner_id, ownership_generation=generation)
        try:
            await runner
        finally:
            async with self.session_factory() as db:
                released = await AgentTurnService(db).release_inline(
                    turn_id=turn_id, owner_id=owner_id, generation=generation
                )
            if not released:
                logger.warning("agent_turn_release_stale turn_id=%s generation=%s", turn_id, generation)

    async def _build_user_message_turn_state(
        self,
        *,
        session: AgentSession,
        content: str,
        mode_hint: str | None,
        decision_status: AgentSessionStatus,
        decision_interrupt_type: AgentSessionInterruptType | None,
        decision_latest_message_is_direct_response: bool,
    ) -> tuple[dict[str, Any] | None, Command | None]:
        """Build the (initial_state, resume_command) pair for one USER_MESSAGE turn.

        Shared by the inline dispatch path in `handle_user_message` and the durable worker's
        `build_run_graph_kwargs_for_turn`, so the checkpoint-continuation / first-message / resume
        decision never forks into two copies that could drift apart. `decision_status` /
        `decision_interrupt_type` / `decision_latest_message_is_direct_response` are all snapshots
        of session/message state as observed at the point this turn was admitted, BEFORE this
        turn's own user message was inserted (either the in-memory `AdmittedTurn` fields the inline
        caller already had, or the same values read back from the turn's persisted cohort for a
        durable worker running long after the live session row has moved on) — none of them may be
        re-derived from the live session/message rows here, since by the time this method runs the
        new user message is already the latest row for the session.
        """
        is_turn_failed = decision_status == AgentSessionStatus.TURN_FAILED
        is_direct_response_wait = (
            decision_status == AgentSessionStatus.WAITING_FOR_HUMAN
            and decision_interrupt_type is None
            and decision_latest_message_is_direct_response
        )
        is_checkpoint_continuation = is_turn_failed or is_direct_response_wait
        is_first_message = not is_checkpoint_continuation and decision_interrupt_type is None

        if is_checkpoint_continuation:
            # Minimal partial-state update against the existing checkpoint: only the new message and
            # the per-turn counters (turn_count, readiness_reject_streak, diagnosis_judge_calls_used)
            # reset, matching _resume_command's reset convention for the same channels, so prior
            # WorkflowState channels (decision_nodes, draft_body, etc.) carry forward untouched
            # instead of being reset by build_initial_workflow_state.
            initial_state = {
                "messages": [{"role": "user", "content": content}],
                "turn_count": 0,
                "readiness_reject_streak": 0,
                "diagnosis_judge_calls_used": 0,
            }
            resume_command = None
        elif is_first_message:
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
        return initial_state, resume_command

    async def build_run_graph_kwargs_for_turn(self, *, turn_id: uuid.UUID) -> dict[str, Any]:
        """Reconstruct `_run_graph`'s keyword arguments for an already-admitted turn purely from
        persisted state, with no HTTP request closure available. Used by the durable worker
        entrypoint once it has claimed a job for `turn_id`.

        Only `AgentTurnTriggerType.USER_MESSAGE` is supported so far; other trigger types raise
        `NotImplementedError` until a later increment builds their kwargs the same way.
        """
        envelope = await self.db.get(AgentTurnEnvelope, turn_id)
        if envelope is None:
            raise ValueError(f"agent turn envelope not found for turn_id={turn_id}")
        trigger = await self.db.get(AgentTurnTrigger, envelope.original_trigger_id)
        if trigger is None:
            raise ValueError(f"agent turn trigger not found for turn_id={turn_id}")
        if trigger.trigger_type != AgentTurnTriggerType.USER_MESSAGE:
            raise NotImplementedError(
                f"run_graph_kwargs construction for trigger_type={trigger.trigger_type.value} is not "
                "implemented yet; only USER_MESSAGE is supported by the durable worker so far"
            )
        if trigger.message_id is None:
            raise ValueError(f"user_message trigger for turn_id={turn_id} has no message_id")

        session = await self.db.get(AgentSession, envelope.session_id)
        if session is None:
            raise ValueError(f"agent session not found for turn_id={turn_id}")
        message = await self.db.get(AgentMessage, trigger.message_id)
        if message is None:
            raise ValueError(f"agent message not found for turn_id={turn_id}")

        cohort = envelope.cohort or {}
        prior_status_raw = cohort.get("prior_status")
        if prior_status_raw is None:
            raise ValueError(f"turn cohort for turn_id={turn_id} is missing the prior_status snapshot")
        decision_status = AgentSessionStatus(prior_status_raw)
        prior_interrupt_type_raw = cohort.get("prior_interrupt_type")
        decision_interrupt_type = (
            AgentSessionInterruptType(prior_interrupt_type_raw) if prior_interrupt_type_raw is not None else None
        )
        if "prior_latest_message_is_direct_response" not in cohort:
            raise ValueError(
                f"turn cohort for turn_id={turn_id} is missing the prior_latest_message_is_direct_response snapshot"
            )
        decision_latest_message_is_direct_response = bool(cohort["prior_latest_message_is_direct_response"])
        mode_hint = message.payload.get("mode_hint") if isinstance(message.payload, dict) else None

        llm_client, strong_llm_client = await self._resolve_llm_client(session.provider_config_id)
        initial_state, resume_command = await self._build_user_message_turn_state(
            session=session,
            content=message.content,
            mode_hint=mode_hint,
            decision_status=decision_status,
            decision_interrupt_type=decision_interrupt_type,
            decision_latest_message_is_direct_response=decision_latest_message_is_direct_response,
        )

        return {
            "session_id": session.id,
            "project_id": session.project_id,
            "artifact_type": session.artifact_type,
            "step_key": session.step_key,
            "workflow_area": session.workflow_area,
            "agent_role": session.agent_role,
            "focused_artifact_id": session.focused_artifact_id,
            "missing_context": session.missing_context or [],
            "llm_client": llm_client,
            "strong_llm_client": strong_llm_client,
            "initial_state": initial_state,
            "resume_command": resume_command,
            "turn_id": turn_id,
        }

    async def _queue_message(
        self,
        session_id: uuid.UUID,
        content: str,
        mode_hint: str | None = None,
        *,
        project_id: uuid.UUID | None = None,
        user_id: uuid.UUID | None = None,
        idempotency_key: str | None = None,
    ) -> AgentMessage:
        """Persist a user message as queued (payload.queued=True) without starting a graph turn.

        The mode_hint (if any) rides the payload so _drain_queue can replay it. Drained later by
        _drain_queue once the current turn ends COMPLETED/FAILED.
        """
        if settings.agent_turn_admission_enabled:
            if project_id is None:
                raise RuntimeError("project_id is required for admitted queued messages")
            admitted = await AgentTurnService(self.db).admit_user_message(
                project_id=project_id,
                session_id=session_id,
                user_id=user_id,
                content=content,
                idempotency_key=idempotency_key,
                mode_hint=mode_hint,
            )
            return admitted.message
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
        await self._guard_session_not_ended(session_id)
        if tool_call.status == AgentToolCallStatus.REJECTED:
            raise HTTPException(400, detail="Tool call has been rejected")
        if tool_call.status != AgentToolCallStatus.PROPOSED:
            raise HTTPException(400, detail="Tool call is not in proposed status")

        # Approval creates its own server-side logical turn. Never reuse the proposal turn and
        # never accept owner/generation from the HTTP request.
        actor_id = user_id or created_by_id
        admitted = None
        owner_id = None
        generation = None
        # Direct legacy service calls without an authenticated actor stay on their compatibility
        # path. The REST path always supplies user_id and therefore cannot bypass admission.
        if actor_id is not None:
            admitted = await AgentTurnService(self.db).admit_approval(
                project_id=project_id, tool_call_id=tool_call_id, user_id=actor_id
            )
            owner_id = f"approval:{uuid.uuid4()}"
            generation = await AgentTurnService(self.db).claim_inline(turn_id=admitted.turn_id, owner_id=owner_id)
            if generation is None:
                # A concurrent request owns the same admitted approval; do not execute a second effect.
                await self.db.refresh(tool_call)
                if tool_call.status == AgentToolCallStatus.EXECUTED:
                    return tool_call
                raise HTTPException(409, detail="Approval turn is already being processed")

        approval_error = False
        approved_tool_call: AgentToolCall | None = None
        resume_context: dict[str, Any] | None = None
        try:
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
                    turn_id=admitted.turn_id if admitted is not None else None,
                    turn_owner_id=owner_id,
                    turn_ownership_generation=generation,
                )
                snapshot = {**snapshot, "created_link_id": str(link.id)}
                tool_call.input_snapshot = snapshot
            elif tool_kind == "propose_retirement":
                artifact = await self._execute_retirement(
                    project_id=project_id,
                    session_id=session_id,
                    snapshot=snapshot,
                    created_by_id=created_by_id,
                    turn_id=admitted.turn_id if admitted is not None else None,
                    turn_owner_id=owner_id,
                    turn_ownership_generation=generation,
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
            # Approval is never a terminal owner.  The final resolved proposal makes the session
            # runnable again; the resumed graph alone may emit a terminal TurnOutcome.
            resume_context = await self._prepare_resume_when_all_artifact_proposals_approved(
                session_id=session_id, llm_client=_llm_client
            )
            approved_tool_call = tool_call
        except BaseException:
            approval_error = True
            # Approval owns its own in-flight transaction. Only roll back on the error branch;
            # an unconditional rollback after commit would expire the `tool_call` about to be
            # returned and could force a lazy-load outside the router's greenlet.
            if self.db.in_transaction():
                await self.db.rollback()
            raise
        finally:
            # Every path after claim must release the lease. Success paths already committed at
            # the effect/completion boundary; the error path already rolled back at its origin.
            if admitted is not None and owner_id is not None and generation is not None:
                try:
                    await AgentTurnService(self.db).release_inline(
                        turn_id=admitted.turn_id, owner_id=owner_id, generation=generation
                    )
                except Exception:
                    if not approval_error:
                        raise
                    logger.exception("approval lease release failed after approval error")
        if approved_tool_call is None:
            raise RuntimeError("Approval completed without a tool call result")
        if resume_context is not None:
            def _runner_factory(
                *,
                turn_id: uuid.UUID | None = None,
                owner_id: str | None = None,
                ownership_generation: int | None = None,
            ):
                return self._run_graph(
                    session_id=session_id,
                    project_id=project_id,
                    artifact_type=resume_context["artifact_type"],
                    step_key=resume_context["step_key"],
                    workflow_area=resume_context["workflow_area"],
                    agent_role=resume_context["agent_role"],
                    focused_artifact_id=resume_context["focused_artifact_id"],
                    missing_context=resume_context["missing_context"],
                    llm_client=resume_context["llm_client"],
                    strong_llm_client=resume_context["strong_llm_client"],
                    initial_state=None,
                    resume_command=resume_context["resume_command"],
                    turn_id=turn_id,
                    owner_id=owner_id,
                    ownership_generation=ownership_generation,
                )

            # The effect executor's lease is released in `finally` above before this task is
            # created.  Scheduling earlier lets the resumed runner race its own approval fence
            # and lose the claim.
            if admitted is not None:
                asyncio.create_task(self._run_admitted_graph(turn_id=admitted.turn_id, runner_factory=_runner_factory))
            else:
                # Compatibility path for old direct service callers that have no authenticated
                # actor and therefore no admissible approval turn.  It still resumes the graph;
                # it must never directly project COMPLETED.
                asyncio.create_task(_runner_factory())
        await self.db.refresh(approved_tool_call)
        return approved_tool_call

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
        await self._guard_session_not_ended(session_id)
        if tool_call.status == AgentToolCallStatus.EXECUTED:
            raise HTTPException(400, detail="Tool call has been approved")
        if tool_call.status != AgentToolCallStatus.PROPOSED:
            raise HTTPException(400, detail="Tool call is not in proposed status")

        tool_call.status = AgentToolCallStatus.REJECTED
        tool_call.resolved_at = datetime.now(UTC)
        await self.db.commit()

        # Người dùng đã quyết định không persist proposal; đây là một terminal outcome hợp lệ dù graph
        # resume không tạo thêm tool call hoặc direct response.
        await self._check_and_resume(
            project_id=project_id,
            session_id=session_id,
            llm_client=llm_client,
            allow_empty_completion=True,
        )
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

    async def _guard_session_not_ended(self, session_id: uuid.UUID) -> None:
        # The approve/reject path has no built-in session-status check (unlike
        # handle_user_message), so a session that ended after a tool call was proposed would
        # otherwise still accept an approval/rejection decision on it.
        session_status = (
            await self.db.execute(select(AgentSession.status).where(AgentSession.id == session_id))
        ).scalar_one()
        if session_status in (AgentSessionStatus.COMPLETED, AgentSessionStatus.FAILED, AgentSessionStatus.EXPIRED):
            raise HTTPException(400, detail="Session has ended and cannot process tool call decisions")

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
        turn_id: uuid.UUID | None = None,
        turn_owner_id: str | None = None,
        turn_ownership_generation: int | None = None,
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
        # The command boundary only runs when approval carried the full server-side logical turn context.
        command_service: DraftCommandService | None = None
        logical_command_id: str | None = None
        turn_state = None
        if turn_id is not None and turn_owner_id is not None and turn_ownership_generation is not None:
            command_service = DraftCommandService(self.db)
            canonical_intent = canonical_artifact_link_intent(
                body.source_artifact_id, body.target_artifact_id, body.relation_type.value
            )
            logical_command_id = create_artifact_link_logical_command_id(turn_id, canonical_intent)
            existing_ledger = await command_service.find_ledger(logical_command_id)
            if existing_ledger is not None and existing_ledger.artifact_id is not None:
                existing_link = await self.db.get(ArtifactLink, existing_ledger.artifact_id)
                if existing_link is not None:
                    return existing_link
            turn_state = await command_service.fence_or_none(
                turn_id=turn_id, owner_id=turn_owner_id, expected_generation=turn_ownership_generation
            )
            if turn_state is None:
                raise HTTPException(409, detail="Turn fence is stale for create_artifact_link")
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
        if command_service is not None and logical_command_id is not None:
            command_service.record_effect(
                turn_id=turn_id,
                logical_command_id=logical_command_id,
                action_type="create_artifact_link",
                artifact_id=link.id,
                attempt=turn_state.attempt if turn_state is not None else 0,
            )
        return link

    async def _execute_retirement(
        self,
        *,
        project_id: uuid.UUID,
        session_id: uuid.UUID,
        snapshot: dict[str, Any],
        created_by_id: uuid.UUID | None,
        turn_id: uuid.UUID | None = None,
        turn_owner_id: str | None = None,
        turn_ownership_generation: int | None = None,
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
        # The command boundary only runs when approval carried the full server-side logical turn context.
        command_service: DraftCommandService | None = None
        logical_command_id: str | None = None
        turn_state = None
        if turn_id is not None and turn_owner_id is not None and turn_ownership_generation is not None:
            command_service = DraftCommandService(self.db)
            canonical_intent = canonical_retirement_intent(artifact_id, superseded_by_id)
            logical_command_id = propose_retirement_logical_command_id(turn_id, canonical_intent)
            existing_ledger = await command_service.find_ledger(logical_command_id)
            if existing_ledger is not None and existing_ledger.artifact_id is not None:
                existing_artifact = await self.db.get(Artifact, existing_ledger.artifact_id)
                if existing_artifact is not None:
                    return existing_artifact
            turn_state = await command_service.fence_or_none(
                turn_id=turn_id, owner_id=turn_owner_id, expected_generation=turn_ownership_generation
            )
            if turn_state is None:
                raise HTTPException(409, detail="Turn fence is stale for propose_retirement")
        try:
            archived = await ArtifactService(self.db).archive_artifact(
                project_id=project_id,
                artifact_id=artifact_id,
                user_id=actor_id,
                reason=reason,
                superseded_by_id=superseded_by_id,
                source="agent_retirement",
            )
            if command_service is not None and logical_command_id is not None:
                command_service.record_effect(
                    turn_id=turn_id,
                    logical_command_id=logical_command_id,
                    action_type="propose_retirement",
                    artifact_id=archived.id,
                    attempt=turn_state.attempt if turn_state is not None else 0,
                )
            return archived
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
        allow_empty_completion: bool = False,
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
                allow_empty_completion=allow_empty_completion,
            )
        )

    async def _prepare_resume_when_all_artifact_proposals_approved(
        self, *, session_id: uuid.UUID, llm_client: Any = None
    ) -> dict[str, Any] | None:
        """Commit the ordered approval-resume transition without deciding terminal state.

        The caller schedules the returned graph runner only after it has released the approval
        effect fence.  Returning a data-only context also keeps this transaction free of graph
        execution and makes the ordering explicit for a future durable worker.
        """
        session_row = (
            await self.db.execute(select(AgentSession).where(AgentSession.id == session_id).with_for_update())
        ).scalar_one()
        if session_row.status != AgentSessionStatus.WAITING_FOR_HUMAN:
            # A concurrent decision already transitioned this session.  Close the lock transaction
            # before the approval caller attempts to release its own fence.
            await self.db.commit()
            return None
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
            # This approval is not the last one in the batch.
            await self.db.commit()
            return None
        strong_llm_client = None
        if llm_client is None:
            llm_client, strong_llm_client = await self._resolve_llm_client(session_row.provider_config_id)
        resume_command = self._resume_command(session_row, {"all_resolved": True})
        resume_context = {
            "artifact_type": session_row.artifact_type,
            "step_key": session_row.step_key,
            "workflow_area": session_row.workflow_area,
            "agent_role": session_row.agent_role,
            "focused_artifact_id": session_row.focused_artifact_id,
            "missing_context": session_row.missing_context or [],
            "llm_client": llm_client,
            "strong_llm_client": strong_llm_client,
            "resume_command": resume_command,
        }
        session_row.status = AgentSessionStatus.ACTIVE
        session_row.interrupt_type = None
        await self.db.commit()
        return resume_context

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
        allow_empty_completion: bool = False,
        turn_id: uuid.UUID | None = None,
        owner_id: str | None = None,
        ownership_generation: int | None = None,
    ) -> None:
        config = self._make_config(
            session_id,
            project_id,
            llm_client,
            agent_role,
            strong_llm_client=strong_llm_client,
            turn_id=turn_id,
            turn_owner_id=owner_id,
            turn_ownership_generation=ownership_generation,
        )
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
            # Graph END cùng tool call chưa dispatch hoặc cờ circuit-breaker nghĩa là chưa hoàn tất.
            # Synthetic ToolMessage đóng tool-use block của provider, nên chỉ kiểm tra pending call
            # không thể nhận ra forced stop sau ToolMessage cuối cùng.
            forced_stop_reason = _result_forced_stop_reason(result)
            turn_limit_hit = graph_ended and (
                _result_has_pending_tool_calls(result) or forced_stop_reason is not None
            )
            async with self.session_factory() as db:
                row = (
                    await db.execute(select(AgentSession).where(AgentSession.id == session_id))
                ).scalar_one_or_none()
                if row is None:
                    return  # session was deleted while this turn was running
                if turn_limit_hit and row.status != AgentSessionStatus.COMPLETED:
                    reason_code = forced_stop_reason or "turn_limit"
                    logger.error("turn failed: reason_code=%s session_id=%s", reason_code, session_id)
                    log_gate_decision("turn_failure", reason_code, session_id=str(session_id))
                    await project_terminal_outcome(
                        db,
                        row,
                        TurnOutcomeType.TERMINAL_FAILURE,
                        reason_code,
                        turn_id=turn_id,
                        owner_id=owner_id,
                        expected_ownership_generation=ownership_generation,
                    )
                    db.add(
                        AgentMessage(
                            session_id=session_id,
                            role=AgentMessageRole.AGENT,
                            content=_agent_loop_stop_message(reason_code),
                        )
                    )
                elif graph_ended and row.status in (AgentSessionStatus.ACTIVE, AgentSessionStatus.WAITING_FOR_HUMAN):
                    if _result_is_direct_response(result):
                        # A direct response ends the current turn, not the conversation. No
                        # TurnOutcomeType maps to (WAITING_FOR_HUMAN, None), so this stays a direct
                        # field write instead of routing through project_non_terminal_outcome — but
                        # it still needs the same ownership fence as every other completion branch
                        # here, so a stale/reclaimed durable worker can't silently commit it.
                        await check_ownership_fence(db, turn_id, owner_id, ownership_generation)
                        row.status = AgentSessionStatus.WAITING_FOR_HUMAN
                        row.interrupt_type = None
                    elif _result_has_no_outcome(result) and not allow_empty_completion:
                        # Empty response hoặc toàn bộ tool bị gate loại không tạo proposal đã persist
                        # hay phản hồi cho người dùng. Giữ checkpoint có thể resume thay vì báo artifact
                        # chưa tồn tại là hoàn tất.
                        logger.error("turn failed: reason_code=no_terminal_outcome session_id=%s", session_id)
                        log_gate_decision("turn_failure", "no_terminal_outcome", session_id=str(session_id))
                        await project_terminal_outcome(
                            db,
                            row,
                            TurnOutcomeType.RECOVERABLE_FAILURE,
                            "no_terminal_outcome",
                            turn_id=turn_id,
                            owner_id=owner_id,
                            expected_ownership_generation=ownership_generation,
                        )
                        row.interrupt_type = None
                        db.add(
                            AgentMessage(
                                session_id=session_id,
                                role=AgentMessageRole.AGENT,
                                content=_agent_no_outcome_message(),
                            )
                        )
                    else:
                        await project_terminal_outcome(
                            db,
                            row,
                            TurnOutcomeType.COMPLETED,
                            "graph_ended",
                            turn_id=turn_id,
                            owner_id=owner_id,
                            expected_ownership_generation=ownership_generation,
                        )
                await db.commit()
        except TimeoutError:
            async with self.session_factory() as db:
                row = (
                    await db.execute(select(AgentSession).where(AgentSession.id == session_id))
                ).scalar_one_or_none()
                if row is None:
                    return  # session was deleted while this turn was running
                if row.status not in (AgentSessionStatus.WAITING_FOR_HUMAN, AgentSessionStatus.COMPLETED):
                    logger.exception("turn failed: reason_code=turn_timeout session_id=%s", session_id)
                    log_gate_decision("turn_failure", "turn_timeout", session_id=str(session_id))
                    # TURN_FAILED, not FAILED: the checkpoint's prior WorkflowState survives a timeout,
                    # so the session is resumable.
                    await project_terminal_outcome(
                        db,
                        row,
                        TurnOutcomeType.RECOVERABLE_FAILURE,
                        "turn_timeout",
                        turn_id=turn_id,
                        owner_id=owner_id,
                        expected_ownership_generation=ownership_generation,
                    )
                    row.interrupt_type = None
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
                row = (
                    await db.execute(select(AgentSession).where(AgentSession.id == session_id))
                ).scalar_one_or_none()
                if row is None:
                    return  # session was deleted while this turn was running
                if row.status not in (AgentSessionStatus.WAITING_FOR_HUMAN, AgentSessionStatus.COMPLETED):
                    logger.exception("turn failed: reason_code=graph_exception session_id=%s", session_id)
                    log_gate_decision("turn_failure", "graph_exception", session_id=str(session_id))
                    # TURN_FAILED, not FAILED: same resumability reasoning as the timeout branch above.
                    await project_terminal_outcome(
                        db,
                        row,
                        TurnOutcomeType.RECOVERABLE_FAILURE,
                        "graph_exception",
                        turn_id=turn_id,
                        owner_id=owner_id,
                        expected_ownership_generation=ownership_generation,
                    )
                    row.interrupt_type = None
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
                await db.execute(select(AgentSession).where(AgentSession.id == session_id).with_for_update())
            ).scalar_one()
            # Only drain after a turn truly ended. WAITING_FOR_HUMAN means the graph paused on a
            # specific question/approval — feeding a queued message here would be the wrong input.
            # EXPIRED is terminal-and-inert, not terminal-and-drainable: reviving it to ACTIVE would
            # re-acquire the unique active-session slot the expiry exists to free.
            if session_row.status == AgentSessionStatus.EXPIRED:
                return
            if session_row.status not in (
                AgentSessionStatus.COMPLETED,
                AgentSessionStatus.FAILED,
                AgentSessionStatus.TURN_FAILED,
            ):
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
            queued_turn_id: uuid.UUID | None = None

            if settings.agent_turn_admission_enabled:
                queued_turn_id = (
                    await db.execute(
                        select(AgentTurnTrigger.turn_id).where(AgentTurnTrigger.message_id == queued.id)
                    )
                ).scalar_one_or_none()
                if queued_turn_id is None:
                    logger.error("agent_turn_queue_missing_turn_id session_id=%s message_id=%s", session_id, queued.id)
                    return
                session_row.active_turn_id = queued_turn_id

            was_turn_failed = session_row.status == AgentSessionStatus.TURN_FAILED
            session_row.status = AgentSessionStatus.ACTIVE
            session_row.interrupt_type = None
            await db.commit()

        if was_turn_failed:
            # Same minimal partial-state mechanism as handle_user_message's TURN_FAILED branch: the
            # queued message drains without wiping the failing turn's prior WorkflowState progress.
            initial_state = {
                "messages": [{"role": "user", "content": content}],
                "turn_count": 0,
                "readiness_reject_streak": 0,
                "diagnosis_judge_calls_used": 0,
            }
        else:
            # COMPLETED (and legacy FAILED) sessions keep the full-reset behavior — that boundary is
            # intentional for a genuinely finished session's next queued message.
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
        def _runner_factory(
            *,
            turn_id: uuid.UUID | None = None,
            owner_id: str | None = None,
            ownership_generation: int | None = None,
        ):
            return self._run_graph(
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
                turn_id=turn_id,
                owner_id=owner_id,
                ownership_generation=ownership_generation,
            )

        if queued_turn_id is not None:
            asyncio.create_task(
                self._run_admitted_graph(turn_id=queued_turn_id, runner_factory=_runner_factory)
            )
        else:
            asyncio.create_task(_runner_factory())

    def _make_config(
        self,
        session_id: uuid.UUID,
        project_id: uuid.UUID,
        llm_client: Any,
        agent_role: str | None = None,
        *,
        strong_llm_client: Any = None,
        turn_id: uuid.UUID | None = None,
        turn_owner_id: str | None = None,
        turn_ownership_generation: int | None = None,
    ) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": str(session_id),
                "project_id": str(project_id),
                "session_factory": self.session_factory,
                "llm_client": llm_client,
                "strong_llm_client": strong_llm_client,
                "agent_role": agent_role,
                # Turn context for the command boundary (write_draft). None for any turn
                # not admitted through AgentTurnService — the tool handler falls back to its fully
                # legacy path when this is absent, regardless of the global flag.
                "turn_id": str(turn_id) if turn_id else None,
                "turn_owner_id": turn_owner_id,
                "turn_ownership_generation": turn_ownership_generation,
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
        return Command(
            resume=resume,
            update={
                "turn_count": 0,
                "diagnosis_judge_calls_used": 0,
                "readiness_reject_streak": 0,
                **(state_update or {}),
            },
        )

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


def _result_is_direct_response(result: Any) -> bool:
    """Return whether graph END came from an agent direct response, not workflow completion."""
    if not isinstance(result, dict):
        return False
    analysis_result = result.get("analysis_result")
    return isinstance(analysis_result, dict) and analysis_result.get("response_mode") == "direct"


def _result_has_no_outcome(result: Any) -> bool:
    """Kiểm tra analyze kết thúc mà không dispatch action hoặc tạo direct response."""
    if not isinstance(result, dict):
        return False
    analysis_result = result.get("analysis_result")
    return isinstance(analysis_result, dict) and analysis_result.get("response_mode") == "none"


def _result_forced_stop_reason(result: Any) -> str | None:
    """Đọc lý do circuit-breaker có cấu trúc từ synthetic tool result."""
    if not isinstance(result, dict):
        return None
    messages = result.get("messages") or []
    if not messages:
        return None
    # WorkflowState.messages là lịch sử cộng dồn. Chỉ ToolMessage cuối của lượt hiện tại mới là
    # synthetic result có quyền quyết định terminal state; marker cũ không được làm hỏng lượt mới.
    metadata = getattr(messages[-1], "additional_kwargs", None)
    reason = metadata.get("agent_stop_reason") if isinstance(metadata, dict) else None
    if reason in {"max_agent_turns", "repeated_tool_calls"}:
        return reason
    return None


def _agent_turn_limit_message() -> str:
    return (
        "The session reached the analysis turn limit and was stopped before completion. "
        "Please create a new session to continue."
    )


def _agent_loop_stop_message(reason_code: str) -> str:
    if reason_code == "repeated_tool_calls":
        return (
            "Phiên đã dừng vì agent lặp lại cùng một hành động mà không có tiến triển. "
            "Hãy tạo phiên mới để tiếp tục."
        )
    return _agent_turn_limit_message()


def _agent_no_outcome_message() -> str:
    return (
        "Agent không tạo được hành động hoặc phản hồi hợp lệ cho lượt này, nên artifact chưa được cập nhật. "
        "Hãy gửi tin nhắn tiếp theo để tiếp tục từ phiên đã lưu."
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


async def expire_abandoned_session(db: AsyncSession, session: AgentSession) -> bool:
    """Lazily mark an abandoned ACTIVE/WAITING_FOR_HUMAN session EXPIRED.

    Takes a session row already loaded from the DB and mutates it in place if it has been
    inactive past session_abandoned_ttl. Does not commit — the caller owns the transaction
    boundary. TURN_FAILED is a resumable resting state, not an abandonment candidate, and is
    never touched here regardless of how stale updated_at is.

    Returns True if the session was marked EXPIRED, False otherwise. No turn context exists for
    this lazy-expiry path (it is not tied to any specific turn), so the terminal write never
    carries a `turn_id` and therefore never gets a `TurnOutcome` audit row (see
    `project_terminal_outcome`); only the compatibility status write happens, same as before.
    """
    if session.status not in (AgentSessionStatus.ACTIVE, AgentSessionStatus.WAITING_FOR_HUMAN):
        return False
    ttl = timedelta(hours=settings.session_abandoned_ttl)
    updated_at = session.updated_at
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    if datetime.now(UTC) - updated_at < ttl:
        return False
    await project_terminal_outcome(
        db, session, TurnOutcomeType.CANCELLED, "session_abandoned_ttl_exceeded"
    )
    session.interrupt_type = None
    logger.info(
        "session expired: reason_code=session_abandoned_ttl_exceeded session_id=%s", session.id
    )
    return True


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
    if status_val in (AgentSessionStatus.FAILED.value, AgentSessionStatus.TURN_FAILED.value):
        return "error"
    return "idle"
