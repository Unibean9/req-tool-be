import secrets
import uuid

from fastapi import HTTPException, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.utils import slugify
from app.models.organization import Organization, OrgMember
from app.models.project import Project
from app.models.user import User
from app.schemas.organization import (
    AddMemberRequest,
    BulkAddMemberResponse,
    OrgCreateRequest,
    OrgResponse,
    OrgStats,
)


class OrgService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def _unique_slug(self, base: str) -> str:
        slug = base
        for _ in range(10):
            if not (await self.db.execute(select(Organization).where(Organization.slug == slug))).scalar_one_or_none():
                return slug
            slug = f"{base}-{secrets.token_hex(3)}"
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not create a unique slug")

    async def _fetch_stats(self, org_ids: list) -> dict:
        member_q = (
            select(OrgMember.org_id, func.count().label("cnt"))
            .where(OrgMember.org_id.in_(org_ids))
            .group_by(OrgMember.org_id)
        )
        project_q = (
            select(Project.org_id, func.count().label("cnt"))
            .where(Project.org_id.in_(org_ids))
            .group_by(Project.org_id)
        )
        members = {r.org_id: r.cnt for r in (await self.db.execute(member_q)).all()}
        projects = {r.org_id: r.cnt for r in (await self.db.execute(project_q)).all()}
        return {oid: OrgStats(member_count=members.get(oid, 0), project_count=projects.get(oid, 0)) for oid in org_ids}

    async def list_mine(self, user: User) -> list[OrgResponse]:
        result = await self.db.execute(
            select(Organization)
            .join(OrgMember, OrgMember.org_id == Organization.id)
            .where(OrgMember.user_id == user.id)
        )
        orgs = list(result.scalars().all())
        stats = await self._fetch_stats([o.id for o in orgs])
        return [OrgResponse.model_validate(o).model_copy(update={"stats": stats[o.id]}) for o in orgs]

    async def create(self, body: OrgCreateRequest, user: User) -> Organization:
        slug = await self._unique_slug(slugify(body.name, fallback="org"))
        org = Organization(name=body.name, slug=slug, owner_id=user.id)
        self.db.add(org)
        await self.db.flush()
        membership = OrgMember(org_id=org.id, user_id=user.id, role="owner")
        self.db.add(membership)
        await self.db.flush()
        await self.db.commit()
        await self.db.refresh(org)
        return org

    async def get(self, org_id: uuid.UUID) -> OrgResponse:
        result = await self.db.execute(select(Organization).where(Organization.id == org_id))
        org = result.scalar_one_or_none()
        if not org:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Organization not found")
        stats = await self._fetch_stats([org.id])
        return OrgResponse.model_validate(org).model_copy(update={"stats": stats[org.id]})

    async def list_members(
        self,
        org_id: uuid.UUID,
        q: str | None = None,
        role: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[OrgMember]:
        stmt = (
            select(OrgMember)
            .where(OrgMember.org_id == org_id)
            .options(selectinload(OrgMember.user))
            .join(User, User.id == OrgMember.user_id)
        )
        if q:
            escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            stmt = stmt.where(
                or_(
                    User.email.ilike(pattern, escape="\\"),
                    User.github_login.ilike(pattern, escape="\\"),
                    User.full_name.ilike(pattern, escape="\\"),
                )
            )
        if role:
            stmt = stmt.where(OrgMember.role == role)
        stmt = stmt.order_by(OrgMember.created_at.desc()).limit(limit).offset(offset)
        return list((await self.db.execute(stmt)).scalars().all())

    async def add_members(self, org_id: uuid.UUID, body: AddMemberRequest) -> BulkAddMemberResponse:
        added = []
        skipped: list[str] = []
        not_found: list[str] = []
        for item in body.members:
            target = (
                await self.db.execute(
                    select(User).where(or_(User.email == item.identifier, User.github_login == item.identifier))
                )
            ).scalar_one_or_none()
            if not target:
                not_found.append(item.identifier)
                continue
            existing = (
                await self.db.execute(
                    select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == target.id)
                )
            ).scalar_one_or_none()
            if existing:
                skipped.append(item.identifier)
                continue
            member = OrgMember(org_id=org_id, user_id=target.id, role=item.role)
            self.db.add(member)
            await self.db.flush()
            member.user = target
            added.append(member)
        return BulkAddMemberResponse(added=added, skipped=skipped, not_found=not_found)

    async def remove_member(self, org_id: uuid.UUID, user_id: uuid.UUID) -> None:
        result = await self.db.execute(
            select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
        )
        member = result.scalar_one_or_none()
        if not member:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Member not found")
        if member.role == "owner":
            owners = await self.db.execute(
                select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.role == "owner")
            )
            if len(owners.scalars().all()) <= 1:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    detail="Cannot remove the organization's only owner",
                )
        await self.db.delete(member)
