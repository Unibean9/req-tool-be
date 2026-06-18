import uuid
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.artifact import Artifact, ArtifactLink, ArtifactPriority, ArtifactStatus, ArtifactType


@dataclass(frozen=True)
class ExportArtifact:
    id: uuid.UUID
    type: ArtifactType
    title: str
    body: str
    priority: ArtifactPriority | None
    code: str | None
    nfr_category: str | None
    stakeholder_role: str | None
    metadata: dict


class ExportService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def render(self, *, project_id: uuid.UUID, export_type: str, include_wont: bool = False) -> str:
        artifacts = await self._load_artifacts(project_id, include_wont)
        links = await self._load_links(project_id)
        if export_type == "brd":
            return render_brd(artifacts)
        if export_type == "product-brief":
            return render_product_brief(artifacts)
        if export_type == "srs":
            return render_srs(artifacts, links)
        if export_type == "prd":
            return render_prd(artifacts, links)
        raise ValueError("Loại export không hợp lệ")

    async def _load_artifacts(self, project_id: uuid.UUID, include_wont: bool) -> list[ExportArtifact]:
        query = select(Artifact).where(Artifact.project_id == project_id, Artifact.status == ArtifactStatus.ACCEPTED)
        if not include_wont:
            query = query.where((Artifact.priority.is_(None)) | (Artifact.priority != ArtifactPriority.WONT))
        rows = (await self.db.execute(query.order_by(Artifact.created_at, Artifact.id))).scalars().all()
        result: list[ExportArtifact] = []
        from app.models.artifact import ArtifactVersion

        for artifact in rows:
            body = ""
            if artifact.current_version_id is not None:
                version = await self.db.get(ArtifactVersion, artifact.current_version_id)
                if version is not None:
                    body = version.body
            result.append(
                ExportArtifact(
                    id=artifact.id,
                    type=artifact.type,
                    title=artifact.title,
                    body=body,
                    priority=artifact.priority,
                    code=artifact.code,
                    nfr_category=artifact.nfr_category,
                    stakeholder_role=artifact.stakeholder_role,
                    metadata=artifact.extra_metadata or {},
                )
            )
        return result

    async def _load_links(self, project_id: uuid.UUID) -> list[ArtifactLink]:
        return list((await self.db.execute(select(ArtifactLink).where(ArtifactLink.project_id == project_id))).scalars().all())


def render_brd(items: list[ExportArtifact]) -> str:
    sections = [
        ("Business Objectives", _by_type(items, ArtifactType.GOAL)),
        ("Problem Statement", _by_type(items, ArtifactType.PROBLEM)),
        ("Stakeholder Register", _stakeholders(items, {"business_stakeholder", "both"})),
        ("Business Scope", _by_type(items, ArtifactType.CAPABILITY)),
        ("Business Rules", _by_type(items, ArtifactType.BUSINESS_RULE)),
        (
            "Constraints and Assumptions",
            _constraints(items, {"business", "both"}) + _by_type(items, ArtifactType.ASSUMPTION),
        ),
        ("Risks and Issues", _by_type(items, ArtifactType.RISK) + _by_type(items, ArtifactType.OPEN_QUESTION)),
        ("Research Basis", _by_type(items, ArtifactType.RESEARCH_OUTPUT)),
    ]
    return _document("Business Requirements Document", sections)


def render_product_brief(items: list[ExportArtifact]) -> str:
    lines = ["# Product Brief", ""]
    sections = [
        ("Problem", _by_type(items, ArtifactType.PROBLEM)),
        ("User Personas", _stakeholders(items, {"user_persona", "both"})),
        ("Objectives", _by_type(items, ArtifactType.GOAL)),
        ("Capabilities", _by_type(items, ArtifactType.CAPABILITY)),
        ("Risks", _by_type(items, ArtifactType.RISK)),
        ("Assumptions", _by_type(items, ArtifactType.ASSUMPTION)),
        ("Open Questions", _by_type(items, ArtifactType.OPEN_QUESTION)),
    ]
    for title, values in sections:
        lines.extend(_section(title, values))
    lines.append("## Research Summary")
    grouped = defaultdict(list)
    for item in _by_type(items, ArtifactType.RESEARCH_OUTPUT):
        grouped[item.metadata.get("research_type") or "uncategorized"].append(item)
    if not grouped:
        lines.append("_Không có nội dung._")
    for research_type in sorted(grouped):
        lines.append(f"### Research: {research_type}")
        lines.extend(_bullets(grouped[research_type]))
    return "\n".join(lines).strip() + "\n"


def render_srs(items: list[ExportArtifact], links: list[ArtifactLink]) -> str:
    lines = ["# Software Requirements Specification", ""]
    lines.extend(_section("User Personas", _stakeholders(items, {"user_persona", "both"})))
    lines.extend(_section("System Constraints", _constraints(items, {"system", "both"})))
    lines.append("## Functional Requirements")
    lines.extend(_frs_by_epic(items, links))
    lines.append("## Non-Functional Requirements")
    for category, values in _group_by_nfr_category(items).items():
        lines.append(f"### NFR: {category}")
        lines.extend(_bullets(values))
    lines.extend(_traceability_section(items, links))
    return "\n".join(lines).strip() + "\n"


def render_prd(items: list[ExportArtifact], links: list[ArtifactLink]) -> str:
    lines = ["# Product Requirements Document", ""]
    sections = [
        ("Executive Summary", _by_type(items, ArtifactType.GOAL)),
        ("Personas", _stakeholders(items, {"user_persona", "both"})),
        ("Capabilities", _by_type(items, ArtifactType.CAPABILITY)),
        ("Requirements Summary", _by_type(items, ArtifactType.FUNCTIONAL_REQUIREMENT)),
        ("NFRs", _by_type(items, ArtifactType.NON_FUNCTIONAL_REQUIREMENT)),
        ("Backlog", _by_type(items, ArtifactType.EPIC) + _by_type(items, ArtifactType.STORY)),
    ]
    for title, values in sections:
        lines.extend(_section(title, values))
    lines.extend(_traceability_section(items, links))
    return "\n".join(lines).strip() + "\n"


def sort_by_moscow(items: list[ExportArtifact]) -> list[ExportArtifact]:
    order = {
        ArtifactPriority.MUST: 0,
        ArtifactPriority.SHOULD: 1,
        ArtifactPriority.COULD: 2,
        None: 3,
        ArtifactPriority.WONT: 4,
    }
    return sorted(items, key=lambda item: (order[item.priority], item.code or "", item.title))


def _document(title: str, sections: list[tuple[str, list[ExportArtifact]]]) -> str:
    lines = [f"# {title}", ""]
    for section_title, values in sections:
        lines.extend(_section(section_title, values))
    return "\n".join(lines).strip() + "\n"


def _section(title: str, values: list[ExportArtifact]) -> list[str]:
    lines = [f"## {title}"]
    lines.extend(_bullets(values))
    return lines


def _bullets(values: list[ExportArtifact]) -> list[str]:
    sorted_values = sort_by_moscow(values)
    if not sorted_values:
        return ["_Không có nội dung._"]
    lines: list[str] = []
    current_priority: ArtifactPriority | None | object = object()
    for item in sorted_values:
        if item.priority is None and current_priority is not None:
            lines.append("### Unprioritized")
            current_priority = None
        label = f"{item.title}: {item.body}" if item.body else item.title
        lines.append(f"- {label}")
    return lines


def _by_type(items: list[ExportArtifact], artifact_type: ArtifactType) -> list[ExportArtifact]:
    return [item for item in items if item.type == artifact_type]


def _stakeholders(items: list[ExportArtifact], roles: set[str]) -> list[ExportArtifact]:
    return [item for item in _by_type(items, ArtifactType.STAKEHOLDER) if item.stakeholder_role in roles]


def _constraints(items: list[ExportArtifact], types: set[str]) -> list[ExportArtifact]:
    return [
        item
        for item in _by_type(items, ArtifactType.CONSTRAINT)
        if (item.metadata.get("constraint_type") or "business") in types
    ]


def _group_by_nfr_category(items: list[ExportArtifact]) -> dict[str, list[ExportArtifact]]:
    grouped: dict[str, list[ExportArtifact]] = defaultdict(list)
    for item in _by_type(items, ArtifactType.NON_FUNCTIONAL_REQUIREMENT):
        grouped[item.nfr_category or "uncategorized"].append(item)
    return dict(sorted(grouped.items()))


def _frs_by_epic(items: list[ExportArtifact], links: list[ArtifactLink]) -> list[str]:
    artifacts_by_id = {item.id: item for item in items}
    frs = _by_type(items, ArtifactType.FUNCTIONAL_REQUIREMENT)
    ac_by_parent: dict[uuid.UUID, list[ExportArtifact]] = defaultdict(list)
    epic_by_fr: dict[uuid.UUID, ExportArtifact] = {}
    for link in links:
        source = artifacts_by_id.get(link.source_artifact_id)
        target = artifacts_by_id.get(link.target_artifact_id)
        if source is None or target is None:
            continue
        if source.type == ArtifactType.ACCEPTANCE_CRITERIA and link.relation_type.value == "validates":
            ac_by_parent[target.id].append(source)
        if source.type == ArtifactType.FUNCTIONAL_REQUIREMENT and target.type == ArtifactType.EPIC:
            epic_by_fr[source.id] = target

    grouped: dict[str, list[ExportArtifact]] = defaultdict(list)
    for fr in frs:
        grouped[epic_by_fr[fr.id].title if fr.id in epic_by_fr else "Unassigned"].append(fr)

    lines: list[str] = []
    for epic_title in sorted(grouped):
        lines.append(f"### {epic_title}")
        for fr in sort_by_moscow(grouped[epic_title]):
            lines.append(f"- {fr.title}: {fr.body}")
            for ac in sort_by_moscow(ac_by_parent.get(fr.id, [])):
                lines.append(f"  - {ac.title}: {ac.body}")
    if not lines:
        lines.append("_Không có nội dung._")
    return lines


def _traceability_section(items: list[ExportArtifact], links: list[ArtifactLink]) -> list[str]:
    artifacts_by_id = {item.id: item for item in items}
    lines = ["## Traceability"]
    allowed = {"satisfies", "derives_from", "supports"}
    rows = []
    for link in links:
        if link.relation_type.value not in allowed:
            continue
        source = artifacts_by_id.get(link.source_artifact_id)
        target = artifacts_by_id.get(link.target_artifact_id)
        if source is not None and target is not None:
            rows.append(f"- {source.title} -> {target.title} ({link.relation_type.value})")
    lines.extend(sorted(rows) if rows else ["_Không có traceability links._"])
    return lines
