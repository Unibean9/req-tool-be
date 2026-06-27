import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
        return [self.step_to_response(step) for step in steps]

    async def get_progress(self, project_id: uuid.UUID) -> WorkflowProgressResponse | None:
        run = await self.get_current_run(project_id)
        if run is None:
            return None
        steps = await self._steps_for_run(run.id)
        counts = {status.value: 0 for status in WorkflowStepStatus}
        for step in steps:
            counts[step.status.value] += 1
        return WorkflowProgressResponse(
            run_id=run.id,
            status=run.status,
            current_step_key=run.current_step_key,
            step_counts=counts,
            steps=[self.step_to_response(step) for step in steps],
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
        return WorkflowRunResponse(
            id=run.id,
            project_id=run.project_id,
            name=run.name,
            status=run.status,
            current_step_key=run.current_step_key,
            created_by_id=run.created_by_id,
            metadata=run.extra_metadata or {},
            created_at=run.created_at,
            steps=[self.step_to_response(step) for step in steps],
        )

    async def _steps_for_run(self, run_id: uuid.UUID) -> list[WorkflowStep]:
        result = await self.db.execute(
            select(WorkflowStep).where(WorkflowStep.run_id == run_id).order_by(WorkflowStep.created_at, WorkflowStep.id)
        )
        return list(result.scalars().all())

    def step_to_response(self, step: WorkflowStep) -> WorkflowStepResponse:
        return WorkflowStepResponse(
            id=step.id,
            run_id=step.run_id,
            project_id=step.project_id,
            step_key=step.step_key,
            phase=step.phase,
            status=step.status,
            input_snapshot=step.input_snapshot,
            output_snapshot=step.output_snapshot,
            approved_at=step.approved_at,
            approved_by_id=step.approved_by_id,
            metadata=step.extra_metadata or {},
            created_at=step.created_at,
        )
