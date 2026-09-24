"""Prompt and structured-output boundaries for use-case generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.use_cases.models import (
    RequirementsSourceSnapshot,
    UseCaseCandidateDraftList,
    UseCaseDetailDraftList,
    UseCaseGroupDetailDraftList,
    UseCaseGroupsDraftList,
    UseCaseModel,
    UseCaseRelationshipDraftList,
    UseCaseValidationReport,
)
from app.use_cases.rules import validate_use_case_model

USE_CASE_AGENT_SYSTEM_PROMPT = """You are the ReqTool Use-Case Modeling Agent.

Use only the CURRENT PROJECT REQUIREMENTS SOURCE SNAPSHOT. The repository markdown files are
examples, not runtime input. Extract actors, capability modules, atomic business-goal use cases,
use-case detail, semantic relationships, evidence and priority. Never invent a feature, actor,
module, integration, UI, API, database or report that is not supported by the snapshot.

CANONICAL HIERARCHY
System -> Module/Capability -> Use Case. A module is a grouping and must never be emitted as a
use case. Do not emit L0/L1/L2, parent use-case rows, React Flow coordinates, diagram plans or
PlantUML arrow syntax.

SEMANTICS
- Use Case names are active verb + business object from the actor's point of view.
- `evidence` is `explicit` only when directly stated; otherwise `inferred`.
- `priority` is one of required, recommended, optional.
- include: base -> reusable included use case, only when behavior is mandatory and shared.
- extend: optional extension -> complete base use case, with a business condition.
- generalization: child -> parent for a real is-a relationship.
- Actor associations are represented by primary_actor_id/secondary_actor_ids. Do not emit an
  association relation in the relationship pass.
- Every generated element cites exact evidence_id values from the snapshot.

Return JSON only matching the supplied schema. The backend validator and renderer are authoritative.
"""


@dataclass(frozen=True)
class UseCaseGenerationHarness:
    source: RequirementsSourceSnapshot
    language: str = "match_source"
    include_relations: bool = True

    def build_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.build_system_instruction()},
            {"role": "user", "content": self.build_user_prompt()},
        ]

    def build_system_instruction(self) -> str:
        return USE_CASE_AGENT_SYSTEM_PROMPT

    def build_user_prompt(self) -> str:
        relation_rule = (
            ""
            if self.include_relations
            else " Return an empty relationships array; relationships are resolved in a separate call."
        )
        return "\n\n".join(
            (
                "Generate the canonical use-case model for this stored project snapshot." + relation_rule,
                f"source_hash: {self.source.source_hash}",
                f"language_policy: {self.language}",
                "Reason over every component internally and return only the final JSON object.",
                "--- FULL STORED BRD/PRD COMPONENTS ---\n" + self.source.render_full_source(),
                "--- COMPLETE EVIDENCE INDEX ---\n" + self.source.render_evidence_index(),
                "--- JSON SCHEMA ---\n" + json.dumps(self.output_schema(), ensure_ascii=False, indent=2),
            )
        )

    def output_schema(self) -> dict[str, Any]:
        return UseCaseModel.model_json_schema()

    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {"name": "reqtool_use_case_model", "strict": True, "schema": self.output_schema()},
        }

    def parse_and_validate(
        self, payload: UseCaseModel | dict[str, Any]
    ) -> tuple[UseCaseModel, UseCaseValidationReport]:
        model = payload if isinstance(payload, UseCaseModel) else UseCaseModel.model_validate(payload)
        return model, validate_use_case_model(model, self.source)


USE_CASE_RELATIONSHIP_SYSTEM_PROMPT = (
    "You are the ReqTool relationship resolver. The supplied actors, modules and use cases are final.\n"
    "Propose only evidence-backed include, extend or generalization relationships. Do not add, rename or\n"
    "remove elements. Do not return actor associations. include direction is base -> included; extend\n"
    "direction is extension -> base and requires a condition; generalization direction is child -> parent.\n"
    "The `evidence` array must contain only exact `evidence_id` values from the evidence index, without\n"
    "display labels, line explanations or free text. Omit a relationship when no exact citation exists.\n"
    "Return JSON only with a `relations` array and no PlantUML syntax."
)


@dataclass(frozen=True)
class UseCaseRelationshipHarness:
    source: RequirementsSourceSnapshot
    use_cases: list[dict[str, Any]]
    actors: list[dict[str, Any]]
    source_text: str | None = None
    evidence_text: str | None = None

    def build_system_instruction(self) -> str:
        return USE_CASE_RELATIONSHIP_SYSTEM_PROMPT

    def build_user_prompt(self) -> str:
        return "\n\n".join(
            (
                "Resolve semantic use-case relationships for this completed table.",
                f"source_hash: {self.source.source_hash}",
                "Return only the final JSON object.",
                "--- ACTORS ---\n" + json.dumps(self.actors, ensure_ascii=False, indent=2),
                "--- USE CASES ---\n" + json.dumps(self.use_cases, ensure_ascii=False, indent=2),
                "--- STORED BRD/PRD SOURCE ---\n" + (self.source_text or self.source.render_full_source()),
                "--- EVIDENCE INDEX ---\n" + (self.evidence_text or self.source.render_evidence_index()),
                "--- JSON SCHEMA ---\n" + json.dumps(self.output_schema(), ensure_ascii=False, indent=2),
            )
        )

    def output_schema(self) -> dict[str, Any]:
        return UseCaseRelationshipDraftList.model_json_schema()

    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "reqtool_use_case_relationships",
                "strict": True,
                "schema": self.output_schema(),
            },
        }


USE_CASE_GROUPS_SYSTEM_PROMPT = (
    "Extract only capability/domain modules and the complete actor roster from the current stored BRD/PRD "
    "snapshot. A module is not a use case. Do not invent names. Every module and actor cites exact evidence_id "
    "values. Return JSON with `actors` and `groups` only."
)


@dataclass(frozen=True)
class UseCaseGroupsHarness:
    source: RequirementsSourceSnapshot

    def build_system_instruction(self) -> str:
        return USE_CASE_GROUPS_SYSTEM_PROMPT

    def build_user_prompt(self) -> str:
        return "\n\n".join(
            (
                "Extract modules and actors.",
                f"source_hash: {self.source.source_hash}",
                "Return only JSON.",
                "--- SOURCE ---\n" + self.source.render_full_source(),
                "--- EVIDENCE ---\n" + self.source.render_evidence_index(),
                "--- SCHEMA ---\n" + json.dumps(self.output_schema(), ensure_ascii=False, indent=2),
            )
        )

    def output_schema(self) -> dict[str, Any]:
        return UseCaseGroupsDraftList.model_json_schema()

    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "reqtool_use_case_groups",
                "strict": True,
                "schema": self.output_schema(),
            },
        }


_NAMING_RULES = (
    "NAMING (name, max 70 characters): Verb + Noun, e.g. 'Send Notification', 'Process Order', 'Book Flight' -- "
    "not 'Order' (not specific enough) and not a full sentence. Do not include the actor in the name; it has "
    "its own field. Use an active verb naming the action performed (Calculate, Validate, Send, Assign...), "
    "not a passive or vague one. State the user's goal, not the system mechanism behind it: 'Send Notification', "
    "not 'Trigger Webhook Call'; 'Assign Task Owner', not 'Update Assignee Field'. Never name a concrete "
    "integration, protocol, or technology in the name (Slack, webhook, API, database, SMTP, ...) -- that detail "
    "belongs in the description or flow steps, not the name. Every name in the whole module must be unique and "
    "unambiguous, and use plain business language a non-technical stakeholder would recognize. "
)

USE_CASE_GROUP_DETAIL_SYSTEM_PROMPT = (
    "Generate atomic business-goal use cases for exactly the supplied capability module. "
    "Do not emit a module as a use case. Do not emit L0/L1/L2, parent IDs, React Flow nodes or PlantUML. "
    "Choose exactly one primary_actor_id from the supplied roster per use case -- the schema has no field "
    "for a second (supporting) actor, so if more than one role is involved, name the primary actor as the "
    "one who initiates the use case and mention the other role's part in the flow steps or description "
    "instead. Include complete detail fields when the source supports them, and cite exact evidence IDs. "
    + _NAMING_RULES
    + "The `sourceUseCases` list is the deterministic floor: enrich those rows only, preserve each row's ID "
    "in `local_tag`, and do not add or remove rows. "
    "Every source_refs value must be an exact evidence_id from the supplied index, never a rendered label "
    "or explanation. "
    "Return JSON with a `use_cases` array only."
)


@dataclass(frozen=True)
class UseCaseGroupUseCasesHarness:
    source: RequirementsSourceSnapshot
    group: dict[str, Any]
    actors: list[dict[str, Any]]
    # The source snapshot remains the authority for citations.  A module generation pass may use
    # a compact, source-backed excerpt so one large BRD/PRD cannot make every provider request
    # exceed its deadline.  Legacy callers omit these fields and retain the full snapshot.
    source_text: str | None = None
    evidence_text: str | None = None

    def build_system_instruction(self) -> str:
        return USE_CASE_GROUP_DETAIL_SYSTEM_PROMPT

    def build_user_prompt(self) -> str:
        return "\n\n".join(
            (
                f"Generate use cases for module: {self.group.get('name')}.",
                f"source_hash: {self.source.source_hash}",
                "Return only JSON.",
                "--- MODULE ---\n" + json.dumps(self.group, ensure_ascii=False, indent=2),
                "--- ACTORS ---\n" + json.dumps(self.actors, ensure_ascii=False, indent=2),
                "--- SOURCE ---\n" + (self.source_text or self.source.render_full_source()),
                "--- EVIDENCE ---\n" + (self.evidence_text or self.source.render_evidence_index()),
                "--- SCHEMA ---\n" + json.dumps(self.output_schema(), ensure_ascii=False, indent=2),
            )
        )

    def output_schema(self) -> dict[str, Any]:
        return UseCaseGroupDetailDraftList.model_json_schema()

    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {"name": "reqtool_use_case_group_detail", "strict": True, "schema": self.output_schema()},
        }


USE_CASE_CANDIDATES_SYSTEM_PROMPT = (
    "Propose candidate use cases for exactly the supplied capability module, using ONLY behavior the supplied "
    "BRD/PRD excerpt states or directly requires. Never invent a feature, actor, integration or report the "
    "excerpt does not support; if it supports nothing for this module, return an empty list. "
    "Return at most {limit} candidates, most important first. This is a shortlist: do not write flows, "
    "preconditions or any other detail. For each candidate give: `name`; `primary_actor_id` (exactly one id "
    "from the supplied roster -- the role that initiates it); `description` (one sentence, max 240 "
    "characters); `priority` (required = core to the module's stated goal or marked Must; recommended = "
    "stated but secondary; optional = nice-to-have); `evidence` (`explicit` only when the excerpt states this "
    "behavior directly, `inferred` when it is a necessary consequence of stated requirements); `source_refs` "
    "(at least one exact evidence_id from the supplied index that supports it -- never a rendered label or "
    "explanation; a candidate without a real citation is discarded). "
    + _NAMING_RULES
    + "Return JSON with a `candidates` array only."
)


@dataclass(frozen=True)
class UseCaseCandidatesHarness:
    source: RequirementsSourceSnapshot
    group: dict[str, Any]
    actors: list[dict[str, Any]]
    limit: int
    source_text: str | None = None
    evidence_text: str | None = None

    def build_system_instruction(self) -> str:
        return USE_CASE_CANDIDATES_SYSTEM_PROMPT.format(limit=self.limit)

    def build_user_prompt(self) -> str:
        return "\n\n".join(
            (
                f"Propose up to {self.limit} use cases for module: {self.group.get('name')}.",
                f"source_hash: {self.source.source_hash}",
                "Return only JSON.",
                "--- MODULE ---\n" + json.dumps(self.group, ensure_ascii=False, indent=2),
                "--- ACTORS ---\n" + json.dumps(self.actors, ensure_ascii=False, indent=2),
                "--- SOURCE ---\n" + (self.source_text or self.source.render_full_source()),
                "--- EVIDENCE ---\n" + (self.evidence_text or self.source.render_evidence_index()),
                "--- SCHEMA ---\n" + json.dumps(self.output_schema(), ensure_ascii=False, indent=2),
            )
        )

    def output_schema(self) -> dict[str, Any]:
        return UseCaseCandidateDraftList.model_json_schema()

    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {"name": "reqtool_use_case_candidates", "strict": True, "schema": self.output_schema()},
        }


USE_CASE_DETAIL_SYSTEM_PROMPT = (
    "Write the detail for each supplied use case. Its name, primary actor, module, priority and evidence are "
    "already fixed: do not rename, add, drop or merge use cases, and return exactly one `details` entry per "
    "supplied `use_case_id`. Use ONLY the supplied BRD/PRD excerpt; where it says nothing that supports a "
    "field, leave that field empty instead of inventing content. Keep it concise: a main flow of at most 8 "
    "steps, at most 2 alternative and 2 exception flows. Flow steps use participantType actor/system/"
    "external_system; an actor step's participantId is the use case's primary actor id. "
    "`related_requirements` may only reference requirement codes that appear in the excerpt, and their "
    "source_refs must be exact evidence_id values from the supplied index. "
    "Return JSON with a `details` array only."
)


@dataclass(frozen=True)
class UseCaseDetailHarness:
    source: RequirementsSourceSnapshot
    use_cases: list[dict[str, Any]]
    actors: list[dict[str, Any]]
    source_text: str
    evidence_text: str

    def build_system_instruction(self) -> str:
        return USE_CASE_DETAIL_SYSTEM_PROMPT

    def build_user_prompt(self) -> str:
        return "\n\n".join(
            (
                f"Write the detail for these {len(self.use_cases)} use cases.",
                f"source_hash: {self.source.source_hash}",
                "Return only JSON.",
                "--- USE CASES ---\n" + json.dumps(self.use_cases, ensure_ascii=False, indent=2),
                "--- ACTORS ---\n" + json.dumps(self.actors, ensure_ascii=False, indent=2),
                "--- SOURCE ---\n" + self.source_text,
                "--- EVIDENCE ---\n" + self.evidence_text,
                "--- SCHEMA ---\n" + json.dumps(self.output_schema(), ensure_ascii=False, indent=2),
            )
        )

    def output_schema(self) -> dict[str, Any]:
        return UseCaseDetailDraftList.model_json_schema()

    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {"name": "reqtool_use_case_details", "strict": True, "schema": self.output_schema()},
        }
