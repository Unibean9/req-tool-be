import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import AgentRun, AgentSession
from app.models.artifact import (
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStep,
    WorkflowStepKey,
    WorkflowStepPhase,
    WorkflowStepStatus,
)
from app.schemas.workflow import (
    WorkflowProgressResponse,
    WorkflowRunCreateRequest,
    WorkflowRunResponse,
    WorkflowStepResponse,
)

STEP_PHASES: tuple[tuple[WorkflowStepKey, WorkflowStepPhase], ...] = (
    (WorkflowStepKey.INTENT_VISION, WorkflowStepPhase.BRD),
    (WorkflowStepKey.CAPABILITY_MAP, WorkflowStepPhase.BRD),
    (WorkflowStepKey.DOMAIN_MODEL, WorkflowStepPhase.PRD),
    (WorkflowStepKey.REQUIREMENTS_SPEC, WorkflowStepPhase.PRD),
    (WorkflowStepKey.REALIZATION_BACKLOG, WorkflowStepPhase.DELIVERY),
)


class ActiveWorkflowRunExistsError(Exception):
    pass


class WorkflowService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_run(
        self,
        *,
        project_id: uuid.UUID,
        body: WorkflowRunCreateRequest,
        created_by_id: uuid.UUID,
    ) -> WorkflowRunResponse:
        active_run = await self.get_current_run(project_id)
        if active_run is not None:
            raise ActiveWorkflowRunExistsError("Project already has an active workflow run")

        run = WorkflowRun(
            project_id=project_id,
            name=body.name,
            status=WorkflowRunStatus.ACTIVE,
            current_step_key=WorkflowStepKey.INTENT_VISION,
            created_by_id=created_by_id,
            extra_metadata=body.metadata,
        )
        self.db.add(run)
        await self.db.flush()

        steps = [
            WorkflowStep(
                run_id=run.id,
                project_id=project_id,
                step_key=step_key,
                phase=phase,
                status=WorkflowStepStatus.PENDING,
                extra_metadata={},
            )
            for step_key, phase in STEP_PHASES
        ]
        self.db.add_all(steps)
        await self.db.flush()
        return await self.run_to_response(run)

    async def get_current_run(self, project_id: uuid.UUID) -> WorkflowRun | None:
        result = await self.db.execute(
            select(WorkflowRun)
            .where(WorkflowRun.project_id == project_id, WorkflowRun.status == WorkflowRunStatus.ACTIVE)
            .order_by(WorkflowRun.created_at.desc(), WorkflowRun.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_current_run_response(self, project_id: uuid.UUID) -> WorkflowRunResponse | None:
        run = await self.get_current_run(project_id)
        if run is None:
            return None
        return await self.run_to_response(run)

    async def list_current_steps(self, project_id: uuid.UUID) -> list[WorkflowStepResponse] | None:
        run = await self.get_current_run(project_id)
        if run is None:
            return None
        steps = await self._steps_for_run(run.id)
        activity_by_step = await self._agent_activity_by_step(project_id)
        return [self.step_to_response(step, activity_by_step.get(step.step_key.value)) for step in steps]

    async def get_progress(self, project_id: uuid.UUID) -> WorkflowProgressResponse | None:
        run = await self.get_current_run(project_id)
        if run is None:
            return None
        steps = await self._steps_for_run(run.id)
        activity_by_step = await self._agent_activity_by_step(project_id)
        counts = {status.value: 0 for status in WorkflowStepStatus}
        for step in steps:
            counts[step.status.value] += 1
        return WorkflowProgressResponse(
            run_id=run.id,
            status=run.status,
            current_step_key=run.current_step_key,
            step_counts=counts,
            steps=[self.step_to_response(step, activity_by_step.get(step.step_key.value)) for step in steps],
        )

    async def update_step_status(
        self,
        *,
        project_id: uuid.UUID,
        step_id: uuid.UUID,
        status: WorkflowStepStatus,
    ) -> WorkflowStep:
        result = await self.db.execute(
            select(WorkflowStep).where(WorkflowStep.id == step_id, WorkflowStep.project_id == project_id)
        )
        step = result.scalar_one_or_none()
        if step is None:
            raise ValueError("Workflow step not found")
        step.status = status
        await self.db.flush()
        return step

    async def run_to_response(self, run: WorkflowRun) -> WorkflowRunResponse:
        steps = await self._steps_for_run(run.id)
        activity_by_step = await self._agent_activity_by_step(run.project_id)
        return WorkflowRunResponse(
            id=run.id,
            project_id=run.project_id,
            name=run.name,
            status=run.status,
            current_step_key=run.current_step_key,
            created_by_id=run.created_by_id,
            metadata=run.extra_metadata or {},
            created_at=run.created_at,
            steps=[self.step_to_response(step, activity_by_step.get(step.step_key.value)) for step in steps],
        )

    async def _steps_for_run(self, run_id: uuid.UUID) -> list[WorkflowStep]:
        result = await self.db.execute(
            select(WorkflowStep).where(WorkflowStep.run_id == run_id).order_by(WorkflowStep.created_at, WorkflowStep.id)
        )
        return list(result.scalars().all())

    async def _agent_activity_by_step(self, project_id: uuid.UUID) -> dict[str, dict[str, Any]]:
        sessions = (
            (
                await self.db.execute(
                    select(AgentSession)
                    .where(AgentSession.project_id == project_id, AgentSession.step_key.is_not(None))
                    .order_by(AgentSession.updated_at.desc(), AgentSession.id.desc())
                )
            )
            .scalars()
            .all()
        )
        latest_sessions: dict[str, AgentSession] = {}
        for session in sessions:
            if session.step_key and session.step_key not in latest_sessions:
                latest_sessions[session.step_key] = session

        activity: dict[str, dict[str, Any]] = {}
        for step_key, session in latest_sessions.items():
            run = (
                await self.db.execute(
                    select(AgentRun)
                    .where(AgentRun.session_id == session.id)
                    .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            analysis = dict(run.analysis_result or {}) if run is not None else {}
            activity[step_key] = {
                "input_snapshot": {
                    "agent_session_id": str(session.id),
                    "artifact_type": session.artifact_type,
                    "focused_artifact_id": str(session.focused_artifact_id) if session.focused_artifact_id else None,
                    "status": session.status.value,
                    "interrupt_type": session.interrupt_type.value if session.interrupt_type else None,
                    "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                },
                "output_snapshot": {
                    "agent_run_id": str(run.id) if run is not None else None,
                    "analysis_result": analysis,
                    "model_tool_calls": analysis.get("model_tool_calls") or [],
                    "dispatched_tool_calls": analysis.get("dispatched_tool_calls") or analysis.get("tools") or [],
                    "dropped_tool_calls": analysis.get("dropped_tool_calls") or [],
                },
                "metadata": {
                    "agent_session_id": str(session.id),
                    "agent_run_id": str(run.id) if run is not None else None,
                },
            }
        return activity

    def step_to_response(
        self,
        step: WorkflowStep,
        agent_activity: dict[str, Any] | None = None,
    ) -> WorkflowStepResponse:
        metadata = dict(step.extra_metadata or {})
        if agent_activity:
            metadata.setdefault("agent_activity", agent_activity.get("metadata") or {})
        return WorkflowStepResponse(
            id=step.id,
            run_id=step.run_id,
            project_id=step.project_id,
            step_key=step.step_key,
            phase=step.phase,
            status=step.status,
            input_snapshot=step.input_snapshot or (agent_activity or {}).get("input_snapshot"),
            output_snapshot=step.output_snapshot or (agent_activity or {}).get("output_snapshot"),
            approved_at=step.approved_at,
            approved_by_id=step.approved_by_id,
            metadata=metadata,
            created_at=step.created_at,
        )
