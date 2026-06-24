from __future__ import annotations

import secrets
import uuid

from fastapi import HTTPException, status
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.utils import slugify
from app.models.agent import AgentMessage, AgentRun, AgentSession, AgentToolCall
from app.models.artifact import (
    Artifact,
    ArtifactEvidence,
    ArtifactLink,
    ArtifactReview,
    ArtifactVersion,
    SourceDocument,
    WorkflowRun,
    WorkflowStep,
)
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
        await self._cascade_delete_children(project_id)
        await self.db.delete(project)

    async def _cascade_delete_children(self, project_id: uuid.UUID) -> None:
        artifact_ids = select(Artifact.id).where(Artifact.project_id == project_id)
        version_ids = select(ArtifactVersion.id).where(ArtifactVersion.artifact_id.in_(artifact_ids))
        run_ids = select(AgentRun.id).where(
            AgentRun.session_id.in_(select(AgentSession.id).where(AgentSession.project_id == project_id))
        )

        # 1. Break the cyclic cross-FKs before deleting the rows they reference.
        await self.db.execute(
            update(Artifact).where(Artifact.project_id == project_id).values(current_version_id=None)
        )
        await self.db.execute(
            update(ArtifactVersion)
            .where(ArtifactVersion.artifact_id.in_(artifact_ids))
            .values(parent_version_id=None)
        )
        await self.db.execute(
            update(AgentToolCall)
            .where(AgentToolCall.run_id.in_(run_ids))
            .values(created_artifact_id=None, created_version_id=None)
        )

        # 2. Delete children from leaves up to the root.
        await self.db.execute(delete(ArtifactEvidence).where(ArtifactEvidence.artifact_id.in_(artifact_ids)))
        await self.db.execute(delete(ArtifactReview).where(ArtifactReview.artifact_id.in_(artifact_ids)))
        await self.db.execute(delete(ArtifactLink).where(ArtifactLink.project_id == project_id))
        await self.db.execute(delete(ArtifactVersion).where(ArtifactVersion.id.in_(version_ids)))
        await self.db.execute(delete(Artifact).where(Artifact.project_id == project_id))

        await self.db.execute(delete(AgentToolCall).where(AgentToolCall.run_id.in_(run_ids)))
        await self.db.execute(
            delete(AgentMessage).where(
                AgentMessage.session_id.in_(select(AgentSession.id).where(AgentSession.project_id == project_id))
            )
        )
        await self.db.execute(delete(AgentRun).where(AgentRun.id.in_(run_ids)))
        await self.db.execute(delete(AgentSession).where(AgentSession.project_id == project_id))

        await self.db.execute(delete(SourceDocument).where(SourceDocument.project_id == project_id))
        await self.db.execute(delete(WorkflowStep).where(WorkflowStep.project_id == project_id))
        await self.db.execute(delete(WorkflowRun).where(WorkflowRun.project_id == project_id))
        await self.db.flush()
