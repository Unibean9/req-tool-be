"""Typed source, model, validation, and render contracts for use-case modeling.

The contracts mirror the rules in the supplied SRS use-case guide.  In particular, the source
snapshot is made from the *stored project document components*; the BRD/PRD markdown files in the
repository are examples and are never read by this package.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

UseCaseLevel = Literal["L0", "L1", "L2"]  # legacy type for old imports only
UseCaseAbstraction = Literal["summary", "user_goal", "subfunction"]  # legacy type
UseCaseStatus = Literal["confirmed", "inferred", "suggested"]  # legacy type
EvidenceType = Literal["explicit", "inferred"]
UseCasePriority = Literal["required", "recommended", "optional"]
RelationshipType = Literal["include", "extend", "generalization"]
LegacyRelationKind = Literal["association", "include", "extend", "generalization"]
RelationshipDraftKind = Literal["include", "extend", "generalization"]
ActorKind = Literal["human", "external_system", "scheduler"]
ActorType = ActorKind
ActorSide = Literal["left", "right"]
FlowType = Literal["main", "alternative", "exception"]
FlowParticipantType = Literal["actor", "system", "external_system"]
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


class UseCaseSystem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="SYSTEM", min_length=1)
    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=600)
    source_refs: list[str] = Field(default_factory=list)


class UseCaseActor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    kind: ActorKind = "human"
    side: ActorSide | None = None
    source_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_type_alias(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "kind" not in data and data.get("type") is not None:
            data["kind"] = data["type"]
        kind = str(data.get("kind") or "human").lower()
        data["kind"] = {
            "human_role": "human",
            "primary actor": "human",
            "supporting actor": "human",
            "time": "scheduler",
        }.get(kind, kind)
        if isinstance(data.get("source_refs"), str):
            data["source_refs"] = [data["source_refs"]]
        if isinstance(data.get("sourceTrace"), str):
            data["sourceTrace"] = [data["sourceTrace"]]
        if "source_refs" not in data and "sourceTrace" in data:
            data["source_refs"] = data["sourceTrace"]
        data.pop("sourceTrace", None)
        data.pop("parent_actor_id", None)
        data.pop("parentActorId", None)
        data.pop("type", None)
        return data


class UseCaseModule(BaseModel):
    """A BRD/PRD capability group. A module is never a use case."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=160)
    goal: str | None = Field(default=None, max_length=600)
    source_refs: list[str] = Field(default_factory=list)


UseCaseSubsystem = UseCaseModule


class UseCaseFlowStep(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    step: int = Field(ge=1)
    participant_type: FlowParticipantType
    participant_id: str | None = Field(default=None, min_length=1)
    action: str = Field(min_length=1, max_length=300)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.setdefault("participant_type", data.get("participantType"))
        data.setdefault("participant_id", data.get("participantId"))
        # extra="forbid" rejects the now-redundant camelCase keys if they are left in place --
        # a source of "Extra inputs are not permitted" failures for a step the LLM sent in
        # (correct, FE-facing) camelCase.
        data.pop("participantType", None)
        data.pop("participantId", None)
        return data


class UseCaseFlow(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    flow_type: Literal["alternative", "exception"]
    label: str = Field(min_length=1, max_length=160)
    branch_at_step: int | None = Field(default=None, ge=1)
    steps: list[UseCaseFlowStep] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.setdefault("flow_type", data.get("flowType"))
        data.setdefault("branch_at_step", data.get("branchAtStep"))
        data.pop("flowType", None)
        data.pop("branchAtStep", None)
        return data


class UseCaseRequirementLink(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1, max_length=120)
    type: Literal["functional", "business_rule", "non_functional"]
    title: str | None = Field(default=None, max_length=300)
    source_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            if isinstance(value, str):
                return {"id": value, "type": "functional"}
            return value
        data = dict(value)
        data.setdefault("source_refs", data.get("sourceTrace", []))
        if isinstance(data.get("source_refs"), str):
            data["source_refs"] = [data["source_refs"]]
        data.pop("sourceTrace", None)
        return data


class UseCaseEntry(BaseModel):
    """One generated use case in the canonical System → Module → Use Case hierarchy."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=160)
    module_id: str = Field(min_length=1)
    primary_actor_id: str = Field(min_length=1)
    secondary_actor_ids: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1, max_length=600)
    trigger: str | None = Field(default=None, max_length=300)
    preconditions: list[str] = Field(default_factory=list)
    main_flow: list[UseCaseFlowStep] = Field(default_factory=list)
    alternative_flows: list[UseCaseFlow] = Field(default_factory=list)
    exception_flows: list[UseCaseFlow] = Field(default_factory=list)
    postconditions_success: list[str] = Field(default_factory=list)
    postconditions_failure: list[str] = Field(default_factory=list)
    business_rules: list[str] = Field(default_factory=list)
    related_requirements: list[UseCaseRequirementLink] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    priority: UseCasePriority = "recommended"
    evidence: EvidenceType = "inferred"
    source_refs: list[str] = Field(default_factory=list)
    note: str | None = Field(default=None, max_length=400)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "name" not in data and data.get("title"):
            data["name"] = data["title"]
        if "module_id" not in data:
            module = data.get("module_id") or data.get("moduleId") or data.get("subsystem_id")
            if not module and data.get("subsystem"):
                slug = re.sub(r"[^A-Z0-9]+", "-", str(data["subsystem"]).upper()).strip("-") or "GENERAL"
                module = f"SUB-{slug}"
            if module:
                data["module_id"] = module
        if "secondary_actor_ids" not in data:
            if "supportingActorIds" in data:
                data["secondary_actor_ids"] = data["supportingActorIds"]
            elif "secondaryActorIds" in data:
                data["secondary_actor_ids"] = data["secondaryActorIds"]
        if "primary_actor_id" not in data and data.get("primaryActorId"):
            data["primary_actor_id"] = data["primaryActorId"]
        if "primary_actor_id" not in data and data.get("primary_actors"):
            primary_actors = data.get("primary_actors")
            if isinstance(primary_actors, list) and primary_actors:
                data["primary_actor_id"] = primary_actors[0]
        if "secondary_actor_ids" not in data and data.get("secondary_actors") is not None:
            data["secondary_actor_ids"] = data["secondary_actors"]
        if "description" not in data:
            data["description"] = data.get("goal") or data.get("name") or data.get("title")
        if "preconditions" not in data:
            legacy = data.get("precondition")
            data["preconditions"] = [legacy] if isinstance(legacy, str) and legacy.strip() else []
        aliases = {
            "module_id": "moduleId",
            "secondary_actor_ids": "supportingActorIds",
            "trigger": "trigger",
            "main_flow": "mainFlow",
            "alternative_flows": "alternativeFlows",
            "exception_flows": "exceptionFlows",
            "postconditions_success": "postconditionsSuccess",
            "postconditions_failure": "postconditionsFailure",
            "business_rules": "businessRules",
            "relationship_ids": "relationshipIds",
            "source_refs": "sourceTrace",
        }
        for target, alias in aliases.items():
            if target not in data and alias in data:
                data[target] = data[alias]
        if "evidence" not in data:
            status = str(data.get("status", "")).lower()
            data["evidence"] = "explicit" if status in {"confirmed", "explicit"} else "inferred"
        priority = str(data.get("priority", "")).lower()
        data["priority"] = {"must": "required", "should": "recommended", "could": "optional"}.get(
            priority, priority or "recommended"
        )
        refs = data.get("related_requirements")
        if refs is None:
            refs = data.get("relatedRequirements")
        if refs is None:
            refs = data.get("requirements")
        if refs is not None:
            if isinstance(refs, str):
                refs = [refs]
            data["related_requirements"] = refs
        if isinstance(data.get("source_refs"), str):
            data["source_refs"] = [data["source_refs"]]
        if isinstance(data.get("sourceTrace"), str):
            data["sourceTrace"] = [data["sourceTrace"]]
        for key in (
            "title",
            "primaryActorId",
            "primary_actors",
            "supportingActorIds",
            "secondaryActorIds",
            "secondary_actors",
            "moduleId",
            "subsystem",
            "subsystem_id",
            "level",
            "abstraction",
            "status",
            "precondition",
            "goal",
            "requirements",
            "parentUseCaseId",
            "parent_use_case_id",
            "mainFlow",
            "alternativeFlows",
            "exceptionFlows",
            "postconditionsSuccess",
            "postconditionsFailure",
            "businessRules",
            "relationshipIds",
            "sourceTrace",
            "relatedRequirements",
        ):
            data.pop(key, None)
        return data

    @property
    def subsystem_id(self) -> str:
        return self.module_id

    @property
    def parent_use_case_id(self) -> None:
        return None

    @property
    def level(self) -> str:
        return "L1"

    @property
    def abstraction(self) -> str:
        return "user_goal"

    @property
    def status(self) -> str:
        return "confirmed" if self.evidence == "explicit" else "inferred"

    @property
    def precondition(self) -> str:
        return "\n".join(self.preconditions)


class UseCaseRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1)
    # Actor association is derived from actor IDs and is intentionally not part of the
    # canonical semantic relationship schema sent to the LLM or exposed by the API.
    kind: RelationshipType
    source_id: str
    target_id: str
    condition: str | None = Field(default=None, max_length=300)
    reason: str | None = Field(default=None, max_length=500)
    confidence: float | None = Field(default=None, ge=0, le=1)
    review_state: Literal["accepted", "review_required", "rejected"] | None = None
    source_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.setdefault("kind", data.get("type"))
        data.setdefault("source_id", data.get("sourceId") or data.get("source"))
        data.setdefault("target_id", data.get("targetId") or data.get("target"))
        data.setdefault("source_refs", data.get("sourceTrace") or data.get("evidence", []))
        if isinstance(data.get("source_refs"), str):
            data["source_refs"] = [data["source_refs"]]
        data.setdefault("review_state", data.get("reviewState"))
        data.pop("sourceId", None)
        data.pop("targetId", None)
        data.pop("source", None)
        data.pop("target", None)
        data.pop("type", None)
        data.pop("sourceTrace", None)
        data.pop("evidence", None)
        data.pop("reviewState", None)
        return data


class UseCaseRelationshipDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: RelationshipDraftKind
    source_id: str
    target_id: str
    condition: str | None = Field(default=None, max_length=300)
    reason: str | None = Field(default=None, max_length=500)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_spec_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.setdefault("kind", data.get("type"))
        data.setdefault("source_id", data.get("sourceId") or data.get("source"))
        data.setdefault("target_id", data.get("targetId") or data.get("target"))
        data.setdefault("evidence", data.get("sourceTrace", []))
        if isinstance(data.get("evidence"), str):
            data["evidence"] = [data["evidence"]]
        for key in ("type", "sourceId", "targetId", "source", "target", "sourceTrace"):
            data.pop(key, None)
        return data


class UseCaseRelationshipDraftList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relations: list[UseCaseRelationshipDraft] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_relationships_alias(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "relations" not in data and "relationships" in data:
            data["relations"] = data["relationships"]
        data.pop("relationships", None)
        return data


class UseCaseActorDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    kind: ActorKind = "human"
    source_refs: list[str] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_kind(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        kind = str(data.get("kind") or data.get("type") or "human").lower()
        data["kind"] = {"human_role": "human", "primary actor": "human", "supporting actor": "human"}.get(kind, kind)
        return data


class UseCaseGroupDraft(BaseModel):
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
    """AI output for one module. No level or parent row is emitted."""

    model_config = ConfigDict(extra="forbid")
    local_tag: str | None = Field(default=None, max_length=40)
    # 70, not the 160 every other name field in this file allows: a hard backstop for the naming
    # convention the system prompt asks for (Verb + Noun, no actor, no implementation detail --
    # "Send Notification", not "Trưởng nhóm cấu hình và gửi thông báo task qua Slack webhook").
    # Long enough for a real Verb + Noun name in Vietnamese (more syllables per word than
    # English) without being long enough to fit a whole sentence back in.
    name: str = Field(min_length=1, max_length=70)
    # One actor per use case, deliberately -- a second (supporting) actor was a real UML concept
    # but made the diagram's actor<->use-case association lines dense enough to read as noise on
    # a concept-stage project, and it doubled the surface the weight-balance/coverage logic in
    # diagram_layout/layout.mjs had to reason about for comparatively little benefit. Dropping
    # the field from this draft schema (not just ignoring it after the fact) means a structured-
    # output provider is constrained to never propose one in the first place.
    primary_actor_id: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=600)
    trigger: str | None = Field(default=None, max_length=300)
    preconditions: list[str] = Field(default_factory=list)
    main_flow: list[UseCaseFlowStep] = Field(default_factory=list)
    alternative_flows: list[UseCaseFlow] = Field(default_factory=list)
    exception_flows: list[UseCaseFlow] = Field(default_factory=list)
    postconditions_success: list[str] = Field(default_factory=list)
    postconditions_failure: list[str] = Field(default_factory=list)
    business_rules: list[str] = Field(default_factory=list)
    related_requirements: list[UseCaseRequirementLink] = Field(default_factory=list)
    priority: UseCasePriority = "recommended"
    evidence: EvidenceType = "inferred"
    source_refs: list[str] = Field(min_length=1)
    note: str | None = Field(default=None, max_length=400)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_draft(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "preconditions" not in data and data.get("precondition"):
            data["preconditions"] = [data["precondition"]]
        if "evidence" not in data:
            data["evidence"] = (
                "explicit" if str(data.get("status", "")).lower() in {"confirmed", "explicit"} else "inferred"
            )
        priority = str(data.get("priority", "")).lower()
        data["priority"] = {"must": "required", "should": "recommended", "could": "optional"}.get(
            priority, priority or "recommended"
        )
        if "primary_actor_id" not in data and data.get("primaryActorId"):
            data["primary_actor_id"] = data["primaryActorId"]
        for key in (
            "level",
            "parent_local_tag",
            "precondition",
            "status",
            "primaryActorId",
            "supportingActorIds",
            "secondary_actor_ids",
            "secondaryActorIds",
            "parentUseCaseId",
        ):
            data.pop(key, None)
        return data


class UseCaseGroupDetailDraftList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_cases: list[UseCaseGroupDetailDraft] = Field(default_factory=list)


class UseCaseModel(BaseModel):
    """Canonical structured model: System → Module → Use Case + relationships."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    system: UseCaseSystem
    modules: list[UseCaseModule] = Field(default_factory=list)
    actors: list[UseCaseActor] = Field(default_factory=list)
    use_cases: list[UseCaseEntry] = Field(default_factory=list)
    relationships: list[UseCaseRelation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_model(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "system" not in data:
            system_name = data.pop("system_name", None) or data.pop("projectName", None) or "Requirements System"
            data["system"] = {"id": "SYSTEM", "name": system_name}
        elif isinstance(data["system"], str):
            data["system"] = {"id": "SYSTEM", "name": data["system"]}
        if "modules" not in data:
            data["modules"] = data.pop("subsystems", [])
        if "use_cases" not in data:
            data["use_cases"] = data.pop("useCases", data.get("use_cases", []))
        if "relationships" not in data:
            data["relationships"] = data.pop("relations", data.get("relationships", []))
        data.pop("diagrams", None)
        return data

    @property
    def system_name(self) -> str:
        return self.system.name

    @system_name.setter
    def system_name(self, value: str) -> None:
        self.system = UseCaseSystem(
            id=self.system.id, name=value, description=self.system.description, source_refs=self.system.source_refs
        )

    @property
    def subsystems(self) -> list[UseCaseModule]:
        return self.modules

    @property
    def relations(self) -> list[UseCaseRelation]:
        return self.relationships

    @relations.setter
    def relations(self, value: list[UseCaseRelation]) -> None:
        self.relationships = value


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


# Kept only so legacy imports do not break while all new generation output uses one PlantUML
# document. The canonical UseCaseModel deliberately does not contain this collection.
class UseCaseDiagramDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    system_boundary: str
    actor_ids: list[str] = Field(default_factory=list)
    use_case_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)


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
    kind: LegacyRelationKind
    line_style: Literal["solid", "dashed"]
    directed: bool
    marker: Literal["none", "open_arrow", "open_triangle"]
    label: str | None = None
    condition: str | None = None


class DiagramRenderPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diagram_id: str
    system_boundary: str
    nodes: list[DiagramRenderNode] = Field(default_factory=list)
    edges: list[DiagramRenderEdge] = Field(default_factory=list)
