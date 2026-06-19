import uuid

from fastapi import APIRouter, Depends, status

from app.core.guards import require_org_member, require_org_owner
from app.core.responses import created, ok
from app.deps import current_user, get_project_service
from app.models.user import User
from app.schemas.project import ProjectCreateRequest, ProjectResponse, ProjectUpdateRequest
from app.schemas.response import ApiResponse
from app.services.project_service import ProjectService

router = APIRouter(prefix="/orgs/{org_id}/projects", tags=["Projects"])


@router.post("", response_model=ApiResponse[ProjectResponse], status_code=status.HTTP_201_CREATED)
async def create_project(
    org_id: uuid.UUID,
    body: ProjectCreateRequest,
    user: User = Depends(current_user),
    service: ProjectService = Depends(get_project_service),
):
    await require_org_member(org_id, user, service.db)
    return created(await service.create(org_id, body))


@router.get("", response_model=ApiResponse[list[ProjectResponse]])
async def list_projects(
    org_id: uuid.UUID,
    user: User = Depends(current_user),
    service: ProjectService = Depends(get_project_service),
):
    await require_org_member(org_id, user, service.db)
    return ok(await service.list(org_id))


@router.get("/{project_id}", response_model=ApiResponse[ProjectResponse])
async def get_project(
    org_id: uuid.UUID,
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    service: ProjectService = Depends(get_project_service),
):
    await require_org_member(org_id, user, service.db)
    return ok(await service.get(org_id, project_id))


@router.patch("/{project_id}", response_model=ApiResponse[ProjectResponse])
async def update_project(
    org_id: uuid.UUID,
    project_id: uuid.UUID,
    body: ProjectUpdateRequest,
    user: User = Depends(current_user),
    service: ProjectService = Depends(get_project_service),
):
    await require_org_member(org_id, user, service.db)
    return ok(await service.update(org_id, project_id, body))


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    org_id: uuid.UUID,
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    service: ProjectService = Depends(get_project_service),
):
    await require_org_owner(org_id, user, service.db)
    await service.delete(org_id, project_id)
