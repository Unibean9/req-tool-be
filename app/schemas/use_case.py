"""HTTP contracts for the editable use-case table and PlantUML source."""

from __future__ import annotations

import enum
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class UseCaseLevel(enum.StrEnum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"


class UseCaseStatus(enum.StrEnum):
    CONFIRMED = "Confirmed"
    INFERRED = "Inferred"
    SUGGESTED = "Suggested"


class UseCasePriority(enum.StrEnum):
    MUST = "Must"
    SHOULD = "Should"
    COULD = "Could"


class ActorKind(enum.StrEnum):
    PRIMARY = "Primary actor"
    SUPPORTING = "Supporting actor"


class RelationshipType(enum.StrEnum):
    ASSOCIATION = "association"
    PART_OF = "part-of"
    INCLUDE = "include"
    EXTEND = "extend"
    GENERALIZATION = "generalization"


class UseCaseApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ActorResponse(UseCaseApiModel):
    id: str
    name: str
    kind: ActorKind


class UseCaseRelationshipResponse(UseCaseApiModel):
    id: str
    source_id: str = Field(alias="sourceId")
    target_id: str = Field(alias="targetId")
    type: RelationshipType
    condition: str | None = None


class UseCaseRowResponse(UseCaseApiModel):
    id: str
    level: UseCaseLevel
    title: str
    primary_actor_id: str = Field(alias="primaryActorId")
    supporting_actor_ids: list[str] = Field(default_factory=list, alias="supportingActorIds")
    subsystem: str
    status: UseCaseStatus
    priority: UseCasePriority
    parent_use_case_id: str | None = Field(default=None, alias="parentUseCaseId")
    description: str
    precondition: str
    source_trace: list[str] = Field(default_factory=list, alias="sourceTrace")


class UseCaseResponse(UseCaseRowResponse):
    relationships: list[UseCaseRelationshipResponse] = Field(default_factory=list)


class UseCaseDiagramResponse(UseCaseApiModel):
    id: str
    level: UseCaseLevel
    system_boundary: str = Field(alias="systemBoundary")
    subsystem: str | None = None
    actor_ids: list[str] = Field(default_factory=list, alias="actorIds")
    use_case_ids: list[str] = Field(default_factory=list, alias="useCaseIds")
    relation_ids: list[str] = Field(default_factory=list, alias="relationIds")


class DiagramNodeResponse(UseCaseApiModel):
    id: str
    kind: str
    label: str
    shape: str
    side: str | None = None


class DiagramEdgeResponse(UseCaseApiModel):
    id: str
    source_id: str = Field(alias="sourceId")
    target_id: str = Field(alias="targetId")
    kind: str
    line_style: str = Field(alias="lineStyle")
    directed: bool
    marker: str
    label: str | None = None
    condition: str | None = None


class DiagramPlanResponse(UseCaseApiModel):
    diagram_id: str = Field(alias="diagramId")
    level: UseCaseLevel
    system_boundary: str = Field(alias="systemBoundary")
    subsystem: str | None = None
    nodes: list[DiagramNodeResponse] = Field(default_factory=list)
    edges: list[DiagramEdgeResponse] = Field(default_factory=list)


class UseCaseValidationIssueResponse(UseCaseApiModel):
    severity: str
    code: str
    message: str
    path: str | None = None


class UseCaseValidationResponse(UseCaseApiModel):
    issues: list[UseCaseValidationIssueResponse] = Field(default_factory=list)
    eligible_for_srs: bool = Field(alias="eligibleForSrs", default=False)
    eligible_diagram_ids: list[str] = Field(default_factory=list, alias="eligibleDiagramIds")
    confirmed_use_case_ids: list[str] = Field(default_factory=list, alias="confirmedUseCaseIds")


class UseCasePlantUmlResponse(UseCaseApiModel):
    """Editable PlantUML source rendered from the current use-case table."""

    language: Literal["plantuml"] = "plantuml"
    source: str
    editable: bool = True
    stale: bool = False
    generated_from: Literal["use-case-table", "manual"] = Field(
        default="use-case-table", alias="generatedFrom"
    )


class UseCasePlantUmlUpdateRequest(UseCaseApiModel):
    source: str = Field(min_length=1, max_length=500_000)


class UseCaseModelResponse(UseCaseApiModel):
    project_id: str = Field(alias="projectId")
    project_name: str = Field(alias="projectName")
    actors: list[ActorResponse] = Field(default_factory=list)
    use_cases: list[UseCaseRowResponse] = Field(default_factory=list, alias="useCases")
    relationships: list[UseCaseRelationshipResponse] = Field(default_factory=list)
    diagrams: list[UseCaseDiagramResponse] = Field(default_factory=list)
    diagram_plans: list[DiagramPlanResponse] = Field(default_factory=list, alias="diagramPlans")
    source_hash: str | None = Field(default=None, alias="sourceHash")
    validation: UseCaseValidationResponse | None = None
    generation: dict[str, Any] | None = None
    plant_uml: UseCasePlantUmlResponse | None = Field(default=None, alias="plantUml")


class ActorCreateRequest(UseCaseApiModel):
    name: str = Field(min_length=1, max_length=120)
    kind: ActorKind = ActorKind.SUPPORTING


class ActorUpdateRequest(UseCaseApiModel):
    name: str = Field(min_length=1, max_length=120)


class UseCaseCreateRequest(UseCaseApiModel):
    level: UseCaseLevel
    title: str = Field(min_length=2, max_length=160)
    primary_actor_id: str = Field(alias="primaryActorId", min_length=1)
    supporting_actor_ids: list[str] = Field(default_factory=list, alias="supportingActorIds")
    subsystem: str = Field(min_length=1, max_length=160)
    status: UseCaseStatus = UseCaseStatus.SUGGESTED
    priority: UseCasePriority = UseCasePriority.SHOULD
    parent_use_case_id: str | None = Field(default=None, alias="parentUseCaseId")
    description: str = Field(min_length=1, max_length=600)
    precondition: str = Field(min_length=1, max_length=400)
    source_trace: list[str] = Field(min_length=1, alias="sourceTrace")


class UseCaseUpdateRequest(UseCaseApiModel):
    level: UseCaseLevel | None = None
    title: str | None = Field(default=None, min_length=2, max_length=160)
    primary_actor_id: str | None = Field(default=None, alias="primaryActorId")
    supporting_actor_ids: list[str] | None = Field(default=None, alias="supportingActorIds")
    subsystem: str | None = Field(default=None, min_length=1, max_length=160)
    status: UseCaseStatus | None = None
    priority: UseCasePriority | None = None
    parent_use_case_id: str | None = Field(default=None, alias="parentUseCaseId")
    description: str | None = Field(default=None, min_length=1, max_length=600)
    precondition: str | None = Field(default=None, min_length=1, max_length=400)
    source_trace: list[str] | None = Field(default=None, min_length=1, alias="sourceTrace")


class RelationshipCreateRequest(UseCaseApiModel):
    source_id: str = Field(alias="sourceId", min_length=1)
    target_id: str = Field(alias="targetId", min_length=1)
    type: RelationshipType
    condition: str | None = Field(default=None, max_length=300)


class UseCaseGenerateRequest(UseCaseApiModel):
    provider_config_id: uuid.UUID | None = Field(default=None, alias="providerConfigId")
    max_level: UseCaseLevel = Field(default=UseCaseLevel.L2, alias="maxLevel")
