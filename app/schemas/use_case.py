"""HTTP contracts for the generated Use Case Table and editable PlantUML source.

The canonical aggregate is System → Module/Capability → Use Case.  Use cases are generated from
stored BRD/PRD components; the only editable artifact is the PlantUML source.
"""

from __future__ import annotations

import enum
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class UseCaseApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class UseCasePriority(enum.StrEnum):
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class EvidenceType(enum.StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class ActorType(enum.StrEnum):
    HUMAN = "human"
    EXTERNAL_SYSTEM = "external_system"
    SCHEDULER = "scheduler"


ActorKind = ActorType


class ActorSide(enum.StrEnum):
    LEFT = "left"
    RIGHT = "right"


class RelationshipType(enum.StrEnum):
    INCLUDE = "include"
    EXTEND = "extend"
    GENERALIZATION = "generalization"


class FlowParticipantType(enum.StrEnum):
    ACTOR = "actor"
    SYSTEM = "system"
    EXTERNAL_SYSTEM = "external_system"


class FlowType(enum.StrEnum):
    MAIN = "main"
    ALTERNATIVE = "alternative"
    EXCEPTION = "exception"


class RequirementType(enum.StrEnum):
    FUNCTIONAL = "functional"
    BUSINESS_RULE = "business_rule"
    NON_FUNCTIONAL = "non_functional"


class RelationshipReviewState(enum.StrEnum):
    ACCEPTED = "accepted"
    REVIEW_REQUIRED = "review_required"
    REJECTED = "rejected"


class ActorResponse(UseCaseApiModel):
    id: str
    name: str
    kind: ActorKind
    side: ActorSide | None = None


class ModuleResponse(UseCaseApiModel):
    id: str
    name: str
    goal: str | None = None
    source_trace: list[str] = Field(default_factory=list, alias="sourceTrace")


class SystemResponse(UseCaseApiModel):
    id: str
    name: str
    description: str | None = None
    source_trace: list[str] = Field(default_factory=list, alias="sourceTrace")


class UseCaseRelationshipResponse(UseCaseApiModel):
    id: str
    source_id: str = Field(alias="sourceId")
    target_id: str = Field(alias="targetId")
    type: RelationshipType
    condition: str | None = None
    reason: str | None = None
    confidence: float | None = None
    review_state: RelationshipReviewState | None = Field(default=None, alias="reviewState")
    source_trace: list[str] = Field(default_factory=list, alias="sourceTrace")


class FlowStepResponse(UseCaseApiModel):
    step: int
    participant_type: FlowParticipantType = Field(alias="participantType")
    participant_id: str | None = Field(default=None, alias="participantId")
    action: str


class FlowResponse(UseCaseApiModel):
    flow_type: Literal["alternative", "exception"] = Field(alias="flowType")
    label: str
    branch_at_step: int | None = Field(default=None, alias="branchAtStep")
    steps: list[FlowStepResponse] = Field(default_factory=list)


class RequirementLinkResponse(UseCaseApiModel):
    id: str
    type: RequirementType
    title: str | None = None
    source_trace: list[str] = Field(default_factory=list, alias="sourceTrace")


class UseCaseRowResponse(UseCaseApiModel):
    id: str
    name: str
    module_id: str = Field(alias="moduleId")
    primary_actor_id: str = Field(alias="primaryActorId")
    secondary_actor_ids: list[str] = Field(default_factory=list, alias="secondaryActorIds")
    relationship_ids: list[str] = Field(default_factory=list, alias="relationshipIds")
    evidence: EvidenceType
    priority: UseCasePriority
    description: str
    trigger: str | None = None
    preconditions: list[str] = Field(default_factory=list)
    main_flow: list[FlowStepResponse] = Field(default_factory=list, alias="mainFlow")
    alternative_flows: list[FlowResponse] = Field(default_factory=list, alias="alternativeFlows")
    exception_flows: list[FlowResponse] = Field(default_factory=list, alias="exceptionFlows")
    postconditions_success: list[str] = Field(default_factory=list, alias="postconditionsSuccess")
    postconditions_failure: list[str] = Field(default_factory=list, alias="postconditionsFailure")
    business_rules: list[str] = Field(default_factory=list, alias="businessRules")
    related_requirements: list[RequirementLinkResponse] = Field(default_factory=list, alias="relatedRequirements")
    source_trace: list[str] = Field(default_factory=list, alias="sourceTrace")
    note: str | None = None


class UseCaseResponse(UseCaseRowResponse):
    relationships: list[UseCaseRelationshipResponse] = Field(default_factory=list)


class UseCasePlantUmlResponse(UseCaseApiModel):
    language: Literal["plantuml"] = "plantuml"
    source: str
    editable: bool = True
    stale: bool = False
    generated_from: Literal["use-case-table", "manual"] = Field(default="use-case-table", alias="generatedFrom")


class UseCasePlantUmlUpdateRequest(UseCaseApiModel):
    source: str = Field(min_length=1, max_length=500_000)


class UseCaseValidationIssueResponse(UseCaseApiModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    path: str | None = None


class UseCaseValidationResponse(UseCaseApiModel):
    issues: list[UseCaseValidationIssueResponse] = Field(default_factory=list)
    eligible_for_srs: bool = Field(alias="eligibleForSrs", default=False)
    eligible_diagram_ids: list[str] = Field(default_factory=list, alias="eligibleDiagramIds")
    # Kept as an empty compatibility field; the new workflow has no confirm/edit gate.
    confirmed_use_case_ids: list[str] = Field(default_factory=list, alias="confirmedUseCaseIds")


class UseCaseModelResponse(UseCaseApiModel):
    project_id: str = Field(alias="projectId")
    project_name: str = Field(alias="projectName")
    system: SystemResponse = Field(default_factory=lambda: SystemResponse(id="SYSTEM", name="Requirements System"))
    actors: list[ActorResponse] = Field(default_factory=list)
    modules: list[ModuleResponse] = Field(default_factory=list)
    use_cases: list[UseCaseRowResponse] = Field(default_factory=list, alias="useCases")
    relationships: list[UseCaseRelationshipResponse] = Field(default_factory=list)
    source_hash: str | None = Field(default=None, alias="sourceHash")
    validation: UseCaseValidationResponse | None = None
    generation: dict[str, Any] | None = None
    plant_uml: UseCasePlantUmlResponse | None = Field(default=None, alias="plantUml")


class ActorCreateRequest(UseCaseApiModel):
    name: str = Field(min_length=1, max_length=120)
    kind: ActorKind = ActorKind.HUMAN


class ActorUpdateRequest(UseCaseApiModel):
    name: str = Field(min_length=1, max_length=120)


# Legacy request classes remain import-compatible for clients that still have the old routes. The
# new UI never calls them; generated use cases are the canonical write path.
class UseCaseLevel(enum.StrEnum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"


class UseCaseStatus(enum.StrEnum):
    CONFIRMED = "Confirmed"
    INFERRED = "Inferred"
    SUGGESTED = "Suggested"


class UseCaseCreateRequest(UseCaseApiModel):
    name: str | None = Field(default=None, min_length=2, max_length=160)
    title: str | None = Field(default=None, min_length=2, max_length=160)
    module_id: str = Field(alias="moduleId", min_length=1)
    primary_actor_id: str = Field(alias="primaryActorId", min_length=1)
    secondary_actor_ids: list[str] = Field(default_factory=list, alias="secondaryActorIds")
    description: str = Field(min_length=1, max_length=600)
    preconditions: list[str] = Field(default_factory=list)
    priority: UseCasePriority = UseCasePriority.RECOMMENDED
    evidence: EvidenceType = EvidenceType.INFERRED
    source_trace: list[str] = Field(default_factory=list, alias="sourceTrace")


class UseCaseUpdateRequest(UseCaseApiModel):
    name: str | None = Field(default=None, min_length=2, max_length=160)
    title: str | None = Field(default=None, min_length=2, max_length=160)
    description: str | None = Field(default=None, min_length=1, max_length=600)
    trigger: str | None = Field(default=None, max_length=300)
    preconditions: list[str] | None = None
    priority: UseCasePriority | None = None


class RelationshipCreateRequest(UseCaseApiModel):
    source_id: str = Field(alias="sourceId", min_length=1)
    target_id: str = Field(alias="targetId", min_length=1)
    type: RelationshipType
    condition: str | None = Field(default=None, max_length=300)
    reason: str | None = Field(default=None, max_length=500)


class UseCaseGenerateRequest(UseCaseApiModel):
    provider_config_id: uuid.UUID | None = Field(default=None, alias="providerConfigId")
