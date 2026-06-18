import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy import select

from app.core.guards import require_org_member, require_org_owner
from app.core.guards import require_project_access
from app.core.responses import created, ok
from app.deps import current_user, get_project_service
from app.models.organization import OrgMember
from app.models.project import Project
from app.models.user import User
from app.schemas.project import ProjectCreateRequest, ProjectCreateTopLevelRequest, ProjectResponse, ProjectUpdateRequest
from app.schemas.response import ApiResponse
from app.services.project_service import ProjectService

router = APIRouter(prefix="/orgs/{org_id}/projects", tags=["Projects"])
alias_router = APIRouter(prefix="/projects", tags=["Projects"])


@alias_router.post("", response_model=ApiResponse[ProjectResponse], status_code=status.HTTP_201_CREATED)
async def create_project_top_level(
    body: ProjectCreateTopLevelRequest,
    user: User = Depends(current_user),
    service: ProjectService = Depends(get_project_service),
):
    await require_org_member(body.org_id, user, service.db)
    return created(await service.create(body.org_id, body))


@alias_router.get("", response_model=ApiResponse[list[ProjectResponse]])
async def list_projects_top_level(
    user: User = Depends(current_user),
    service: ProjectService = Depends(get_project_service),
):
    org_ids = select(OrgMember.org_id).where(OrgMember.user_id == user.id)
    result = await service.db.execute(select(Project).where(Project.org_id.in_(org_ids)))
    return ok(list(result.scalars().all()))


@alias_router.get("/{project_id}", response_model=ApiResponse[ProjectResponse])
async def get_project_top_level(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    service: ProjectService = Depends(get_project_service),
):
    return ok(await require_project_access(project_id, user, service.db))


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
