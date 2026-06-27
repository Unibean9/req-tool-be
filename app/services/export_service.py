import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.documents.registry import children_of, get_config
from app.models.artifact import Artifact, ArtifactStatus, ArtifactType, ArtifactVersion, SourceDocument
from app.models.organization import OrgMember
from app.models.project import Project
from app.schemas.artifact_synthesis import strip_synthesis_assumptions


class ExportService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def render(self, *, project_id: uuid.UUID, user_id: uuid.UUID, export_type: str) -> str:
        await self._require_project_member(project_id, user_id)
        if export_type == "brd":
            return await render_brd(project_id, self.db)
        if export_type == "product-brief":
            return await render_product_brief(project_id, self.db)
        if export_type == "prd":
            return await render_prd(project_id, self.db)
        raise ValueError("Invalid export type")

    async def _require_project_member(self, project_id: uuid.UUID, user_id: uuid.UUID) -> None:
        org_id = (await self.db.execute(select(Project.org_id).where(Project.id == project_id))).scalar_one_or_none()
        if org_id is None:
            raise ValueError("Project not found")
        member = (
            await self.db.execute(select(OrgMember.id).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id))
        ).scalar_one_or_none()
        if member is None:
            raise PermissionError("User does not have access to the project")


async def render_product_brief(project_id: uuid.UUID, db: AsyncSession) -> str:
    sections = await _load_document_items(project_id, db, "brd")
    keys = ("vision_objectives", "problem_statement", "stakeholder_register", "scope_capabilities")
    return _document("Product Brief", [(get_config(key).label, sections.get(key)) for key in keys])


async def render_brd(project_id: uuid.UUID, db: AsyncSession) -> str:
    sections = await _load_document_items(project_id, db, "brd")
    doc_sections = [(get_config(key).label, sections.get(key)) for key in children_of("brd")]
    doc_sections.append(("Research Basis", await _load_research_basis(project_id, db)))
    return _document("Business Requirements Document", doc_sections)


async def render_prd(project_id: uuid.UUID, db: AsyncSession) -> str:
    items = await _load_document_items(project_id, db, "prd")
    doc_sections = [(get_config(key).label, items.get(key)) for key in children_of("prd")]
    return _document("Product Requirements Document", doc_sections)


async def _load_document_items(
    project_id: uuid.UUID,
    db: AsyncSession,
    document_type: str,
) -> dict[str, Any]:
    container_id = (
        await db.execute(
            select(Artifact.id).where(
                Artifact.project_id == project_id,
                Artifact.type == ArtifactType(document_type),
                Artifact.parent_id.is_(None),
            )
        )
    ).scalar_one_or_none()
    if container_id is None:
        return {}
    rows = (
        await db.execute(
            select(Artifact, ArtifactVersion)
            .join(ArtifactVersion, Artifact.current_version_id == ArtifactVersion.id)
            .where(
                Artifact.project_id == project_id,
                Artifact.parent_id == container_id,
                Artifact.type.in_([ArtifactType(item) for item in children_of(document_type)]),
                Artifact.status.in_((ArtifactStatus.DRAFT, ArtifactStatus.ACCEPTED)),
            )
            .order_by(Artifact.created_at, Artifact.id)
        )
    ).all()
    return {artifact.type.value: strip_synthesis_assumptions(version.body) for artifact, version in rows}


async def _load_research_basis(project_id: uuid.UUID, db: AsyncSession) -> str | None:
    rows = (
        (
            await db.execute(
                select(SourceDocument)
                .where(SourceDocument.project_id == project_id)
                .order_by(SourceDocument.created_at, SourceDocument.id)
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return None
    lines = []
    for doc in rows:
        excerpt = (doc.content_text or doc.locator or "").strip()
        label = f"{doc.title}: {excerpt}" if excerpt else doc.title
        lines.append(label)
    return "\n".join(lines)


def _document(title: str, sections: list[tuple[str, Any]]) -> str:
    lines = [f"# {title}", ""]
    for section_title, value in sections:
        lines.append(f"## {section_title}")
        lines.extend(_content_lines(value))
    return "\n".join(lines).strip() + "\n"


def _content_lines(value: Any) -> list[str]:
    if value is None or value == "":
        return ["_No content._"]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if not value:
            return ["_No content._"]
        return [f"- {_stringify(item)}" for item in value]
    if isinstance(value, dict):
        if not value:
            return ["_No content._"]
        return [f"- {key}: {_stringify(item)}" for key, item in value.items()]
    return [_stringify(value)]


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
