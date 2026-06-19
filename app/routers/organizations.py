import uuid

from fastapi import APIRouter, Depends, Query, status

from app.core.guards import require_org_member, require_org_owner
from app.core.responses import created, ok
from app.deps import current_user, get_org_service
from app.models.user import User
from app.schemas.organization import (
    AddMemberRequest,
    BulkAddMemberResponse,
    OrgCreateRequest,
    OrgMemberResponse,
    OrgResponse,
)
from app.schemas.response import ApiResponse
from app.services.organization_service import OrgService

router = APIRouter(prefix="/orgs", tags=["Organizations"])
alias_router = APIRouter(prefix="/organizations", tags=["Organizations"])


@router.get("/me", response_model=ApiResponse[list[OrgResponse]])
@alias_router.get("", response_model=ApiResponse[list[OrgResponse]])
async def list_my_orgs(
    user: User = Depends(current_user),
    service: OrgService = Depends(get_org_service),
):
    return ok(await service.list_mine(user))


@router.post("", response_model=ApiResponse[OrgResponse], status_code=status.HTTP_201_CREATED)
async def create_org(
    body: OrgCreateRequest,
    user: User = Depends(current_user),
    service: OrgService = Depends(get_org_service),
):
    return created(await service.create(body, user))


@router.get("/{org_id}", response_model=ApiResponse[OrgResponse])
async def get_org(
    org_id: uuid.UUID,
    user: User = Depends(current_user),
    service: OrgService = Depends(get_org_service),
):
    await require_org_member(org_id, user, service.db)
    return ok(await service.get(org_id))


@router.get("/{org_id}/members", response_model=ApiResponse[list[OrgMemberResponse]])
async def list_members(
    org_id: uuid.UUID,
    q: str | None = Query(default=None, min_length=1, max_length=255),
    role: str | None = Query(default=None, pattern="^(owner|member)$"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(current_user),
    service: OrgService = Depends(get_org_service),
):
    await require_org_member(org_id, user, service.db)
    return ok(await service.list_members(org_id, q, role, limit, offset))


@router.post(
    "/{org_id}/members",
    response_model=ApiResponse[BulkAddMemberResponse],
    status_code=status.HTTP_201_CREATED,
)
async def add_member(
    org_id: uuid.UUID,
    body: AddMemberRequest,
    user: User = Depends(current_user),
    service: OrgService = Depends(get_org_service),
):
    await require_org_owner(org_id, user, service.db)
    return created(await service.add_members(org_id, body))


@router.delete("/{org_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    user: User = Depends(current_user),
    service: OrgService = Depends(get_org_service),
):
    await require_org_owner(org_id, user, service.db)
    await service.remove_member(org_id, user_id)
