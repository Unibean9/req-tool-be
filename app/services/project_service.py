from __future__ import annotations

import secrets
import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.utils import slugify
from app.models.project import Project
from app.schemas.project import ProjectCreateRequest, ProjectUpdateRequest


class ProjectService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def _unique_slug(self, org_id: uuid.UUID, base: str) -> str:
        slug = base
        for _ in range(10):
            if not (await self.db.execute(
                select(Project).where(Project.org_id == org_id, Project.slug == slug)
            )).scalar_one_or_none():
                return slug
            slug = f"{base}-{secrets.token_hex(3)}"
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Không thể tạo slug duy nhất")

    async def create(self, org_id: uuid.UUID, body: ProjectCreateRequest) -> Project:
        slug = await self._unique_slug(org_id, slugify(body.name, fallback="project"))
        project = Project(
            org_id=org_id,
            name=body.name,
            slug=slug,
            description=body.description,
        )
        self.db.add(project)
        await self.db.flush()
        return project

    async def list(self, org_id: uuid.UUID) -> list[Project]:
        result = await self.db.execute(select(Project).where(Project.org_id == org_id))
        return list(result.scalars().all())

    async def get(self, org_id: uuid.UUID, project_id: uuid.UUID) -> Project:
        result = await self.db.execute(
            select(Project).where(Project.id == project_id, Project.org_id == org_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Không tìm thấy dự án")
        return project

    async def update(
        self, org_id: uuid.UUID, project_id: uuid.UUID, body: ProjectUpdateRequest
    ) -> Project:
        project = await self.get(org_id, project_id)
        if body.name is not None:
            project.name = body.name
        if body.description is not None:
            project.description = body.description
        return project

    async def delete(self, org_id: uuid.UUID, project_id: uuid.UUID) -> None:
        project = await self.get(org_id, project_id)
        await self.db.delete(project)
