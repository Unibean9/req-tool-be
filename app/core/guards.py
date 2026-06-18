import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.organization import OrgMember
from app.models.project import Project


async def require_org_member(org_id: uuid.UUID, user, db: AsyncSession) -> OrgMember:
    result = await db.execute(
        select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == user.id)
    )
    member = result.scalar_one_or_none()
    if not member:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Bạn không phải thành viên của tổ chức này")
    return member


async def require_org_owner(org_id: uuid.UUID, user, db: AsyncSession) -> OrgMember:
    member = await require_org_member(org_id, user, db)
    if member.role != "owner":
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Yêu cầu quyền owner")
    return member


async def require_project_access(project_id: uuid.UUID, user, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Không tìm thấy dự án")
    member = await db.execute(
        select(OrgMember).where(OrgMember.org_id == project.org_id, OrgMember.user_id == user.id)
    )
    if not member.scalar_one_or_none():
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Bạn không phải thành viên của tổ chức sở hữu dự án này")
    return project


async def require_project_owner(project_id: uuid.UUID, user, db: AsyncSession) -> Project:
    project = await require_project_access(project_id, user, db)
    member = await db.execute(
        select(OrgMember).where(OrgMember.org_id == project.org_id, OrgMember.user_id == user.id)
    )
    org_member = member.scalar_one()
    # Hiện model OrgMember chỉ có owner/member; nếu thêm admin role cần mở rộng guard này.
    if org_member.role != "owner":
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Yêu cầu quyền owner")
    return project
