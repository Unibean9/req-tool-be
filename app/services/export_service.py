import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.graphs.section_schema import SECTION_SPECS
from app.models.artifact import Artifact, ArtifactStatus, ArtifactType, ArtifactVersion, SourceDocument
from app.models.organization import OrgMember
from app.models.project import Project

SECTION_TITLES = {
    "vision_objectives": "Vision and Objectives",
    "problem_statement": "Problem Statement",
    "stakeholder_register": "Stakeholder Register",
    "scope_capabilities": "Scope and Capabilities",
    "business_rules": "Business Rules",
    "constraints_assumptions": "Constraints and Assumptions",
    "risks_issues": "Risks and Issues",
}


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
        raise ValueError("Loại export không hợp lệ")

    async def _require_project_member(self, project_id: uuid.UUID, user_id: uuid.UUID) -> None:
        org_id = (await self.db.execute(select(Project.org_id).where(Project.id == project_id))).scalar_one_or_none()
        if org_id is None:
            raise ValueError("Không tìm thấy dự án")
        member = (
            await self.db.execute(
                select(OrgMember.id).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
            )
        ).scalar_one_or_none()
        if member is None:
            raise PermissionError("User không có quyền truy cập dự án")


async def render_product_brief(project_id: uuid.UUID, db: AsyncSession) -> str:
    sections = await _load_requirements_sections(project_id, db)
    keys = ("vision_objectives", "problem_statement", "stakeholder_register", "scope_capabilities")
    return _document("Product Brief", [(SECTION_TITLES[key], sections.get(key)) for key in keys])


async def render_brd(project_id: uuid.UUID, db: AsyncSession) -> str:
    sections = await _load_requirements_sections(project_id, db)
    doc_sections = [(SECTION_TITLES[key], sections.get(key)) for key in SECTION_SPECS]
    doc_sections.append(("Research Basis", await _load_research_basis(project_id, db)))
    return _document("Business Requirements Document", doc_sections)


async def render_prd(project_id: uuid.UUID, db: AsyncSession) -> str:  # noqa: ARG001
    # Stub skeleton only: the PRD draws on design-phase artifact types (FR/NFR/epic/story) that are
    # dormant in P1, so the delivery layer stays empty until that data exists (post-design phase).
    doc_sections = [
        ("Executive Summary", None),
        ("Problem and Users", None),
        ("Scope", None),
        ("Requirements Summary", None),
        ("Non-Functional Requirements", None),
        ("Backlog", None),
        ("Traceability", None),
    ]
    return _document("Product Requirements Document", doc_sections)


async def _load_requirements_sections(project_id: uuid.UUID, db: AsyncSession) -> dict[str, Any]:
    result = await db.execute(
        select(Artifact, ArtifactVersion)
        .join(ArtifactVersion, Artifact.current_version_id == ArtifactVersion.id)
        .where(
            Artifact.project_id == project_id,
            Artifact.type == ArtifactType.REQUIREMENTS,
            Artifact.status.in_((ArtifactStatus.DRAFT, ArtifactStatus.ACCEPTED)),
        )
        .order_by(Artifact.created_at.desc(), Artifact.id.desc())
        .limit(1)
    )
    row = result.first()
    if row is None:
        return {}
    _, version = row
    try:
        body = json.loads(version.body or "{}")
    except json.JSONDecodeError:
        return {}
    if isinstance(body, dict):
        return body
    return {}


async def _load_research_basis(project_id: uuid.UUID, db: AsyncSession) -> str | None:
    rows = (
        await db.execute(
            select(SourceDocument)
            .where(SourceDocument.project_id == project_id)
            .order_by(SourceDocument.created_at, SourceDocument.id)
        )
    ).scalars().all()
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
        return ["_Không có nội dung._"]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if not value:
            return ["_Không có nội dung._"]
        return [f"- {_stringify(item)}" for item in value]
    if isinstance(value, dict):
        if not value:
            return ["_Không có nội dung._"]
        return [f"- {key}: {_stringify(item)}" for key, item in value.items()]
    return [_stringify(value)]


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
