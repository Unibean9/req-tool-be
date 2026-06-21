import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.responses import ok
from app.database import get_db
from app.deps import require_admin
from app.models.organization import Organization, OrgMember
from app.models.user import User
from app.schemas.organization import OrgResponse
from app.schemas.response import ApiResponse
from app.schemas.user import UserResponse

router = APIRouter(prefix="/admin", tags=["Admin"])


class UserUpdateAdminRequest(BaseModel):
    role: Literal["user", "admin"] | None = None
    is_active: bool | None = None


class OrgAdminResponse(OrgResponse):
    member_count: int


@router.get("/users", response_model=ApiResponse[list[UserResponse]])
async def list_users(
    q: str | None = Query(default=None, max_length=255),
    role: Literal["user", "admin"] | None = Query(default=None),
    is_active: bool | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(User)
    if q:
        pattern = f"%{q}%"
        from sqlalchemy import or_
        stmt = stmt.where(or_(User.email.ilike(pattern), User.full_name.ilike(pattern)))
    if role is not None:
        stmt = stmt.where(User.role == role)
    if is_active is not None:
        stmt = stmt.where(User.is_active == is_active)
    stmt = stmt.order_by(User.created_at.desc()).limit(limit).offset(offset)
    result = await db.execute(stmt)
    return ok(result.scalars().all())


@router.patch("/users/{user_id}", response_model=ApiResponse[UserResponse])
async def update_user(
    user_id: uuid.UUID,
    body: UserUpdateAdminRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    target = result.scalar_one_or_none()
    if not target:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Không tìm thấy người dùng")
    if target.id == admin.id and body.role == "user":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Không thể hạ cấp chính mình")
    if body.role is not None:
        target.role = body.role
    if body.is_active is not None:
        target.is_active = body.is_active
    return ok(target)


@router.get("/orgs", response_model=ApiResponse[list[OrgAdminResponse]])
async def list_all_orgs(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    # Org list with member count in one query
    rows = await db.execute(
        select(Organization, func.count(OrgMember.id).label("member_count"))
        .outerjoin(OrgMember, OrgMember.org_id == Organization.id)
        .group_by(Organization.id)
        .order_by(Organization.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    results = []
    for org, count in rows:
        results.append(OrgAdminResponse(
            id=org.id,
            name=org.name,
            slug=org.slug,
            owner_id=org.owner_id,
            created_at=org.created_at,
            member_count=count,
        ))
    return ok(results)
