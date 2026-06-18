import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.guards import require_project_access
from app.core.responses import created, ok
from app.database import get_db
from app.deps import current_user
from app.models.user import User
from app.schemas.response import ApiResponse
from app.schemas.workflow import WorkflowProgressResponse, WorkflowRunCreateRequest, WorkflowRunResponse, WorkflowStepResponse
from app.services.workflow_service import ActiveWorkflowRunExistsError, WorkflowService

router = APIRouter(prefix="/projects/{project_id}", tags=["Workflow"])


@router.post("/workflow-runs", response_model=ApiResponse[WorkflowRunResponse], status_code=status.HTTP_201_CREATED)
async def create_workflow_run(
    project_id: uuid.UUID,
    body: WorkflowRunCreateRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        run = await WorkflowService(db).create_run(project_id=project_id, body=body, created_by_id=user.id)
    except ActiveWorkflowRunExistsError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return created(run)


@router.get("/workflow-runs/current", response_model=ApiResponse[WorkflowRunResponse])
async def get_current_workflow_run(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    run = await WorkflowService(db).get_current_run_response(project_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Không tìm thấy workflow run đang hoạt động")
    return ok(run)


@router.get("/workflow-steps", response_model=ApiResponse[list[WorkflowStepResponse]])
async def list_workflow_steps(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    steps = await WorkflowService(db).list_current_steps(project_id)
    if steps is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Không tìm thấy workflow run đang hoạt động")
    return ok(steps)


@router.get("/workflow-progress", response_model=ApiResponse[WorkflowProgressResponse])
async def get_workflow_progress(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    progress = await WorkflowService(db).get_progress(project_id)
    if progress is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Không tìm thấy workflow run đang hoạt động")
    return ok(progress)
