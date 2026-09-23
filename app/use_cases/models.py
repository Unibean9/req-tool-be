"""Typed source, model, validation, and render contracts for use-case modeling.

The contracts mirror the rules in the supplied SRS use-case guide.  In particular, the source
snapshot is made from the *stored project document components*; the BRD/PRD markdown files in the
repository are examples and are never read by this package.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

UseCaseLevel = Literal["L0", "L1", "L2"]
UseCaseAbstraction = Literal["summary", "user_goal", "subfunction"]
UseCaseStatus = Literal["confirmed", "inferred", "suggested"]
UseCasePriority = Literal["must", "should", "could"]
RelationKind = Literal["association", "include", "extend", "generalization"]
ActorKind = Literal["human_role", "external_system", "time"]
EvidenceKind = Literal[
    "component",
    "heading",
    "actor",
    "subsystem",
    "business_requirement",
    "business_capability",
    "functional_requirement",
    "non_functional_requirement",
    "business_rule",
    "scope",
    "out_of_scope",
    "traceability",
    "workflow",
]


class StoredComponentSnapshot(BaseModel):
    """One BRD/PRD item exactly as it exists in the project document registry."""

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["brd", "prd"]
    artifact_type: str
    label: str
    description: str = ""
    artifact_id: str | None = None
    parent_id: str | None = None
    status: str | None = None
    priority: str | None = None
    code: str | None = None
    title: str | None = None
    confidence: float | None = None
    current_version_id: str | None = None
    version_number: int | None = None
    body: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredDocumentSnapshot(BaseModel):
    """A stored BRD or PRD container plus every registry child, including missing slots."""

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["brd", "prd"]
    label: str
    description: str = ""
    artifact_id: str | None = None
    project_id: str | None = None
    status: str | None = None
    title: str | None = None
    current_version_id: str | None = None
    container_body: str = ""
    components: list[StoredComponentSnapshot] = Field(default_factory=list)


class SourceEvidence(BaseModel):
    """A stable citation target exposed to the agent and checked after generation."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    document_type: Literal["brd", "prd"]
    artifact_type: str
    artifact_id: str | None = None
    version_id: str | None = None
    kind: EvidenceKind
    locator: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    entity_id: str | None = None
    entity_name: str | None = None


class RequirementsSourceSnapshot(BaseModel):
    """Complete, immutable-in-memory input for use-case generation."""

    model_config = ConfigDict(extra="forbid")

    project_id: str | None = None
    brd: StoredDocumentSnapshot
    prd: StoredDocumentSnapshot
    components: list[StoredComponentSnapshot] = Field(default_factory=list)
    evidence: list[SourceEvidence] = Field(default_factory=list)
    source_hash: str = Field(min_length=1)

    def evidence_by_id(self) -> dict[str, SourceEvidence]:
        return {item.evidence_id: item for item in self.evidence}

    def render_full_source(self) -> str:
        """Render every stored component body in registry order without silent truncation."""

        blocks: list[str] = []
        for document in (self.brd, self.prd):
            blocks.append(
                "\n".join(
                    (
                        f"=== {document.document_type.upper()} CONTAINER ===",
                        f"label: {document.label}",
                        f"description: {document.description or '(none)'}",
                        f"artifact_id: {document.artifact_id or '(not created)'}",
                        f"project_id: {document.project_id or '(unknown)'}",
                        f"status: {document.status or '(unknown)'}",
                        f"title: {document.title or '(untitled)'}",
                        f"current_version_id: {document.current_version_id or '(no current version)'}",
                        "container_body:",
                        document.container_body or "(missing body)",
                    )
                )
            )
            for component in document.components:
                blocks.append(
                    "\n".join(
                        (
                            f"--- {document.document_type}.{component.artifact_type} ---",
                            f"label: {component.label}",
                            f"description: {component.description or '(none)'}",
                            f"artifact_id: {component.artifact_id or '(not created)'}",
                            f"parent_id: {component.parent_id or '(unknown)'}",
                            f"code: {component.code or '(unset)'}",
                            f"title: {component.title or '(untitled)'}",
                            f"version_id: {component.current_version_id or '(no current version)'}",
                            f"version_number: {component.version_number or '(unknown)'}",
                            f"status: {component.status or '(unknown)'}",
                            f"priority: {component.priority or '(unset)'}",
                            f"confidence: {component.confidence if component.confidence is not None else '(unset)'}",
                            f"metadata: {component.metadata or {}}",
                            "body:",
                            component.body or "(missing body)",
                        )
                    )
                )
        return "\n\n".join(blocks)

    def render_evidence_index(self) -> str:
        """Render citation identity and a short locator excerpt.

        ``render_full_source`` is the authoritative, complete BRD/PRD input.  The evidence
        index only exists to make stable ``source_refs`` easy to choose.  Including the full
        excerpt for every evidence item duplicates the source several times and can make a
        generation request exceed the provider's practical deadline, so keep this derived
        index bounded while retaining the evidence id and locator needed for traceability.
        """

        lines = [
            "evidence_id | document | component | kind | locator | excerpt",
            "---|---|---|---|---|---",
        ]
        for item in self.evidence:
            excerpt = " ".join(item.excerpt.split()).replace("|", "\\|")
            if len(excerpt) > 160:
                excerpt = excerpt[:159].rstrip() + "…"
            lines.append(
                "| ".join(
                    (
                        item.evidence_id,
                        item.document_type,
                        item.artifact_type,
                        item.kind,
                        item.locator,
                        excerpt,
                    )
                )
            )
        return "\n".join(lines)


class UseCaseActor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^ACT-[A-Z0-9-]+$")
    name: str = Field(min_length=1, max_length=120)
    kind: ActorKind = "human_role"
    source_refs: list[str] = Field(min_length=1)


class UseCaseSubsystem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^SUB-[A-Z0-9-]+$")
    name: str = Field(min_length=1, max_length=160)
    source_refs: list[str] = Field(min_length=1)


class UseCaseEntry(BaseModel):
    """One row in the Use Case List and one oval on any diagram where it appears."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^UC-[A-Z0-9]+(?:-[A-Z0-9]+)*$")
    name: str = Field(min_length=1, max_length=160)
    level: UseCaseLevel
    abstraction: UseCaseAbstraction
    primary_actor_id: str = Field(pattern=r"^ACT-[A-Z0-9-]+$")
    secondary_actor_ids: list[str] = Field(default_factory=list)
    subsystem_id: str = Field(pattern=r"^SUB-[A-Z0-9-]+$")
    parent_use_case_id: str | None = Field(default=None, pattern=r"^UC-[A-Z0-9]+(?:-[A-Z0-9]+)*$")
    description: str = Field(min_length=1, max_length=600)
    precondition: str = Field(min_length=1, max_length=400)
    relationship_ids: list[str] = Field(default_factory=list)
    priority: UseCasePriority
    status: UseCaseStatus
    source_refs: list[str] = Field(min_length=1)
    note: str | None = Field(default=None, max_length=400)


class UseCaseRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^REL-[A-Za-z0-9-]+$")
    kind: RelationKind
    source_id: str
    target_id: str
    condition: str | None = Field(default=None, max_length=300)
    source_refs: list[str] = Field(default_factory=list)


class UseCaseDiagramDefinition(BaseModel):
    """A semantic diagram definition; a renderer supplies positions and concrete UI nodes."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^DGM-L[012](?:-[A-Z0-9-]+)?$")
    level: UseCaseLevel
    system_boundary: str = Field(min_length=1, max_length=160)
    subsystem_id: str | None = Field(default=None, pattern=r"^SUB-[A-Z0-9-]+$")
    actor_ids: list[str] = Field(default_factory=list)
    use_case_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)


class UseCaseModel(BaseModel):
    """The only model a future API should persist or hand to the FE renderer."""

    model_config = ConfigDict(extra="forbid")

    system_name: str = Field(min_length=1, max_length=160)
    actors: list[UseCaseActor] = Field(default_factory=list)
    subsystems: list[UseCaseSubsystem] = Field(default_factory=list)
    use_cases: list[UseCaseEntry] = Field(default_factory=list)
    relations: list[UseCaseRelation] = Field(default_factory=list)
    diagrams: list[UseCaseDiagramDefinition] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["error", "warning"]
    code: str
    message: str
    path: str | None = None


class UseCaseValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[ValidationIssue] = Field(default_factory=list)
    eligible_for_srs: bool = False
    eligible_diagram_ids: list[str] = Field(default_factory=list)
    confirmed_use_case_ids: list[str] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]


class DiagramRenderNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["system_boundary", "actor", "use_case"]
    label: str
    shape: Literal["rectangle", "actor", "ellipse"]
    side: Literal["left", "right", "inside"] | None = None


class DiagramRenderEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_id: str
    target_id: str
    kind: RelationKind
    line_style: Literal["solid", "dashed"]
    directed: bool
    marker: Literal["none", "open_arrow", "open_triangle"]
    label: str | None = None
    condition: str | None = None


class DiagramRenderPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagram_id: str
    level: UseCaseLevel
    system_boundary: str
    nodes: list[DiagramRenderNode] = Field(default_factory=list)
    edges: list[DiagramRenderEdge] = Field(default_factory=list)
