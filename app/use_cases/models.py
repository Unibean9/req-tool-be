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


class UseCaseRelationshipDraft(BaseModel):
    """One include/extend/generalization relation proposed by the relationship-resolution pass.

    The LLM is not trusted with a stable, collision-free ``REL-*`` id here; the service assigns
    one deterministically after the draft passes the same structural checks as a manually created
    relationship.
    """

    model_config = ConfigDict(extra="forbid")

    kind: RelationKind
    source_id: str
    target_id: str
    condition: str | None = Field(default=None, max_length=300)


class UseCaseRelationshipDraftList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relations: list[UseCaseRelationshipDraft] = Field(default_factory=list)


class UseCaseActorDraft(BaseModel):
    """One actor proposed by the groups-generation pass (Phase 1 of the split generation flow).

    No stable id is trusted from the model -- same rationale as UseCaseRelationshipDraft: the
    service assigns a collision-free ACT-* id once the draft is merged, deduping by name across
    every group so a role used by several groups becomes one shared actor.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    kind: ActorKind = "human_role"
    source_refs: list[str] = Field(min_length=1)


class UseCaseGroupDraft(BaseModel):
    """One capability/domain group proposed by the groups-generation pass (Phase 1 of the split
    generation flow) -- an L0 row, not an individual use case. Read from the project's Business
    Capabilities content in whatever format it was actually written (no assumed ID/heading
    convention), unlike a regex-based parse of one fixed convention.

    No stable id is trusted from the model -- the service assigns SUB-*/UC-SUM-* ids
    deterministically once merged.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    goal: str = Field(min_length=1, max_length=600)
    user_segment: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(min_length=1)


class UseCaseGroupsDraftList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actors: list[UseCaseActorDraft] = Field(default_factory=list)
    groups: list[UseCaseGroupDraft] = Field(default_factory=list)


class UseCaseGroupDetailDraft(BaseModel):
    """One use case (an L1 user goal, or an L2 sub-use-case) proposed by the per-group detail
    pass (Phase 2 of the split generation flow). Scoped to a single capability group so the
    prompt/output stay a fraction of the size of generating the whole project's use cases in one
    call -- the actual fix for use-case generation timing out on a larger project.

    No stable UC-* id is trusted from the model, and neither is the L1 parent's id (it doesn't
    exist yet when the model writes this, since it too is proposed in this same call): each draft
    carries a small model-chosen `local_tag`, and an L2 draft points at its L1 parent via
    `parent_local_tag`. The service assigns real ids in two passes (L1 first, then L2) and resolves
    `parent_local_tag` through the tags it just minted -- same "never trust an invented id"
    rationale as UseCaseRelationshipDraft, just two levels instead of one.
    """

    model_config = ConfigDict(extra="forbid")

    level: Literal["L1", "L2"]
    local_tag: str = Field(min_length=1, max_length=20)
    parent_local_tag: str | None = Field(default=None, max_length=20)
    name: str = Field(min_length=1, max_length=160)
    primary_actor_id: str = Field(pattern=r"^ACT-[A-Z0-9-]+$")
    secondary_actor_ids: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1, max_length=600)
    precondition: str = Field(min_length=1, max_length=400)
    priority: UseCasePriority
    source_refs: list[str] = Field(min_length=1)
    note: str | None = Field(default=None, max_length=400)


class UseCaseGroupDetailDraftList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_cases: list[UseCaseGroupDetailDraft] = Field(default_factory=list)


class UseCaseDiagramDefinition(BaseModel):
    """Legacy semantic diagram definition kept only for backward-compatible stored records."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^DGM-L[012](?:-[A-Z0-9-]+)?$")
    level: UseCaseLevel
    system_boundary: str = Field(min_length=1, max_length=160)
    subsystem_id: str | None = Field(default=None, pattern=r"^SUB-[A-Z0-9-]+$")
    actor_ids: list[str] = Field(default_factory=list)
    use_case_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)


class UseCaseModel(BaseModel):
    """The source-backed table aggregate used to render the editable UML document."""

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
