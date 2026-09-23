"""AI-agent harness for deriving a traceable use-case model from stored BRD/PRD components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.use_cases.diagram import build_diagram_render_plan
from app.use_cases.models import (
    DiagramRenderPlan,
    RequirementsSourceSnapshot,
    UseCaseGroupDetailDraftList,
    UseCaseGroupsDraftList,
    UseCaseModel,
    UseCaseRelationshipDraftList,
    UseCaseValidationReport,
)
from app.use_cases.rules import validate_use_case_model

USE_CASE_AGENT_SYSTEM_PROMPT = """You are the ReqTool Use-Case Modeling Agent.

Your only source of truth is the CURRENT PROJECT REQUIREMENTS SOURCE SNAPSHOT supplied in the user
message. It is assembled from the project's stored BRD container, PRD container, and every BRD/PRD
registry component/version. Repository markdown examples, general product knowledge, and chat
assumptions are not source evidence.

MISSION
1. Read every stored BRD and PRD component before proposing a model. Preserve missing or partial
   components as missing evidence; never silently fill a blank with a feature.
2. Extract actors, subsystems, business goals, functional requirements, constraints, business
   rules, scope boundaries, and BRD-to-PRD traceability. Every generated actor, subsystem, use case,
   and non-association relation MUST cite one or more exact evidence_id values from the snapshot.
3. Use status=confirmed only when the source explicitly supports the item. Use inferred when the
   item is a conservative normalization of cited evidence. Use suggested only for a reviewable
   candidate. Human review is required before an inferred/suggested row is treated as ready for SRS.

NO INVENTED FEATURES
- Do not add a chatbot, dashboard, notification, report, integration, actor, workflow, or feature
  just because it is common in similar products.
- Do not treat a UI screen, API, database, backend, server, AI model, or implementation step as a
  use case or actor.
- Do not use conversation text as evidence. If BRD/PRD is insufficient, keep the row suggested or
  return a missing-evidence issue in note; do not fabricate.
- Out-of-scope/non-goal statements are hard exclusions.

SOURCE-TO-MODEL PROCEDURE
1. Goal Analysis: make the complete Use Case List first. Actors come from BRD stakeholders,
   external entities, and explicitly named PRD roles/systems. Subsystems come from BRD scope/flows
   and PRD business capabilities or feature modules. Goals come from business requirements and
   functional behavior, not from technical nouns.
2. Language: Actor is a concrete role/external-system noun. System/subsystem is a noun phrase.
   Use Case is active Verb + business Object from the actor's point of view. Avoid passive voice,
   screen names, button names, API/database terms, and vague 'Handle/Process Data' names.
3. Abstraction: L0 is the system overview with Summary use cases (Actor × subsystem). L1 contains
   User Goal use cases under the relevant capability. L2 is optional detail only for a genuinely
   shared subfunction or a complex user goal. Do not use L2 for ordinary step-by-step decomposition.
   Use parent_use_case_id to make the table hierarchy explicit; an L2 shared subfunction may have
   multiple include parents instead. A 'Manage X' summary is allowed only when its description
   lists at least two concrete CRUD-style operations.
4. IDs: use one stable, never-reused UC-01/UC-001 or UC-SUB-01 format consistently. Use the same
   id and name in the table, the generated UML source, and traceability. Do not put actor, priority,
   sprint, or status in an id.

UML/SRS NOTATION
- System Boundary is a named rectangle. Actors are outside it; use cases are inside ellipses.
- Association is a solid, unnamed, undirected line and may connect only Actor–Use Case.
- «include» is a dashed directed arrow from the base use case to the mandatory shared included use
  case. Use it only for reusable behavior shared by at least two use cases; never for steps.
- «extend» is a dashed directed arrow from the optional extension to the base use case and MUST
  include a business condition/extension point. The base remains complete without it.
- Generalization is a solid directed line with an open triangle from child to parent and is only
  for a real is-a relationship between two actors or two use cases.
- Never include Log In/Authenticate in every use case; put authentication in a standalone use case
  or a precondition. Never use an association to show a data flow or a sequence.
- The backend renders one editable PlantUML document from the completed table. Do not return
  coordinates, React Flow nodes, diagram plans, or a second diagram representation. Keep
  include/extend relations evidence-backed and limited to reusable behavior or a real business
  extension point.

OUTPUT
Return JSON only matching the supplied UseCaseModel schema. Leave the legacy `diagrams` field empty;
the backend generates PlantUML after the table is completed. Do not return Markdown, Mermaid,
coordinates, or prose. The backend validator is authoritative and will reject unsupported refs,
invalid names, wrong relation direction/notation, out-of-scope items, or missing actors. Human
review is required before any inferred/suggested row becomes confirmed.
"""


@dataclass(frozen=True)
class UseCaseGenerationHarness:
    """Prompt + validation boundary shared by the API and a future graph node."""

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
        parts = [
            "Generate the use-case model for this stored project source snapshot.",
            f"source_hash: {self.source.source_hash}",
            f"language_policy: {self.language} (keep diagram labels consistent with the source documents)",
            "First reason over every component internally. Return only the final JSON object.",
        ]
        if not self.include_relations:
            parts.append(
                "RELATIONS PASS: this call only establishes actors, subsystems, and use cases. "
                "Return an empty `relations` list. include/extend/generalization relations are "
                "resolved in a separate follow-up call against this same completed table; do not "
                "guess at them here."
            )
        parts.extend(
            (
                "\n--- FULL STORED BRD/PRD COMPONENTS ---\n" + self.source.render_full_source(),
                "\n--- COMPLETE EVIDENCE INDEX ---\n" + self.source.render_evidence_index(),
                "\n--- JSON SCHEMA ---\n" + _schema_text(),
            )
        )
        return "\n\n".join(parts)

    def output_schema(self) -> dict[str, Any]:
        return UseCaseModel.model_json_schema()

    def response_format(self) -> dict[str, Any]:
        """Return the structured-output contract accepted by the existing LLM clients."""

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "reqtool_use_case_model",
                "strict": True,
                "schema": self.output_schema(),
            },
        }

    def parse_and_validate(
        self,
        payload: UseCaseModel | dict[str, Any],
        *,
        require_confirmed: bool = False,
    ) -> tuple[UseCaseModel, UseCaseValidationReport]:
        model = payload if isinstance(payload, UseCaseModel) else UseCaseModel.model_validate(payload)
        return model, validate_use_case_model(model, self.source, require_confirmed=require_confirmed)

    def render_diagram(
        self,
        model: UseCaseModel,
        diagram_id: str,
        *,
        require_confirmed: bool = True,
    ) -> DiagramRenderPlan:
        report = validate_use_case_model(model, self.source, require_confirmed=require_confirmed)
        if any(issue.severity == "error" for issue in report.issues):
            raise ValueError("Cannot render a semantically invalid use-case model")
        return build_diagram_render_plan(model, diagram_id, require_confirmed=require_confirmed)


USE_CASE_RELATIONSHIP_SYSTEM_PROMPT = """You are the ReqTool Use-Case Relationship Agent.

The actors, subsystems, and use cases below are already final for this pass; do not rename,
add, remove, or re-scope any of them. Your only job is to propose include/extend/generalization
relations between them, grounded in the same CURRENT PROJECT REQUIREMENTS SOURCE SNAPSHOT you are
given. Do not return association relations: every actor-to-use-case association is already
established by each use case's primary/supporting actors and must not be repeated here.

RELATION SEMANTICS
- «include» is a dashed directed arrow from the base use case to the mandatory shared included use
  case. Use it only for reusable behavior shared by at least two use cases; never for a single-use
  step, and never invent an included use case that is not already in the supplied list.
- «extend» is a dashed directed arrow from the optional extension use case to the base use case and
  MUST include a business condition/extension point. The base remains complete without it.
- Generalization is a directed is-a relation (child -> parent) between two actors or two use cases
  from the supplied lists; use it only for a genuine specialization, not a loose similarity.
- Never invent a relation to justify padding the output. An empty or short relations list is
  correct when the source does not support reusable/optional/is-a behavior.

OUTPUT
Return JSON only matching the supplied schema: a `relations` array of
`{kind, source_id, target_id, condition}` objects, where `source_id`/`target_id` are existing
actor or use-case ids from the supplied lists. Do not return an `id`, Markdown, or prose.
"""


@dataclass(frozen=True)
class UseCaseRelationshipHarness:
    """Prompt boundary for the follow-up call that resolves include/extend/generalization only."""

    source: RequirementsSourceSnapshot
    use_cases: list[dict[str, Any]]
    actors: list[dict[str, Any]]

    def build_system_instruction(self) -> str:
        return USE_CASE_RELATIONSHIP_SYSTEM_PROMPT

    def build_user_prompt(self) -> str:
        import json

        return "\n\n".join(
            (
                "Propose include/extend/generalization relations for this already-completed use-case table.",
                f"source_hash: {self.source.source_hash}",
                "First reason over every component internally. Return only the final JSON object.",
                "\n--- ACTORS ---\n" + json.dumps(self.actors, ensure_ascii=False, indent=2),
                "\n--- USE CASES ---\n" + json.dumps(self.use_cases, ensure_ascii=False, indent=2),
                "\n--- FULL STORED BRD/PRD COMPONENTS ---\n" + self.source.render_full_source(),
                "\n--- COMPLETE EVIDENCE INDEX ---\n" + self.source.render_evidence_index(),
                "\n--- JSON SCHEMA ---\n" + _relationship_schema_text(),
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


USE_CASE_GROUPS_SYSTEM_PROMPT = """You are the ReqTool Use-Case Modeling Agent, extracting only
the top-level capability/domain groups and the actor roster for this project -- NOT individual use
cases yet (those come from a separate, smaller follow-up call per group).

Your only source of truth is the CURRENT PROJECT REQUIREMENTS SOURCE SNAPSHOT supplied in the user
message. The project's Business Capabilities content may use ANY heading/table/ID convention (for
example numbered domains with a capability table, or "BC-xx" headings, or something else) -- read
it exactly as written; do not assume a specific numbering scheme or reject content that doesn't
match one.

MISSION
1. Identify the distinct capability/domain groups the source's Business Capabilities content
   already presents (however it labels or organizes them) -- do not invent a group the source does
   not support, and do not split or merge a group the source already presents as one.
2. For each group, extract: name, goal (one sentence, business terms), and user_segment (the
   role(s) who use it, exactly as named in the source -- from the Stakeholder Register or the
   capability's own text).
3. Separately extract the full actor roster (every distinct role/external system the source
   names), so a role shared by several groups becomes one actor instead of being repeated.
4. Every group and actor MUST cite one or more exact evidence_id values from the snapshot.

NO INVENTED FEATURES (same rule as the full-model pass)
- Do not add a UI screen, API, database, notification, report, or integration as a group.
- Do not use conversation text as evidence.

OUTPUT
Return JSON only matching the supplied schema: `actors` and `groups` arrays. Do not return an id
for either (the service assigns ids) and do not enumerate individual use cases here -- that is a
separate, later call per group. Do not return Markdown or prose.
"""


@dataclass(frozen=True)
class UseCaseGroupsHarness:
    """Prompt boundary for Phase 1 of the split generation flow: capability groups + the actor
    roster only, nothing else.

    Its output is small regardless of project size (a handful of groups, not every use case), so
    this call is what actually avoids the single-shot generate()'s timeout risk: the heavier
    per-group use-case detail is a separate, smaller call per group (UseCaseGroupUseCasesHarness).
    Reads the project's Business Capabilities content in whatever format it was actually written --
    unlike a regex parse tied to one fixed ID convention, this generalizes across projects.
    """

    source: RequirementsSourceSnapshot

    def build_system_instruction(self) -> str:
        return USE_CASE_GROUPS_SYSTEM_PROMPT

    def build_user_prompt(self) -> str:
        return "\n\n".join(
            (
                "Extract the capability/domain groups and actor roster for this stored project source snapshot.",
                f"source_hash: {self.source.source_hash}",
                "First reason over every component internally. Return only the final JSON object.",
                "\n--- FULL STORED BRD/PRD COMPONENTS ---\n" + self.source.render_full_source(),
                "\n--- COMPLETE EVIDENCE INDEX ---\n" + self.source.render_evidence_index(),
                "\n--- JSON SCHEMA ---\n" + _groups_schema_text(),
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


USE_CASE_GROUP_DETAIL_SYSTEM_PROMPT = """You are the ReqTool Use-Case Modeling Agent, generating
detail for ONE capability group at a time instead of the whole project in one call.

The group below (and the actor roster) is already final for this pass -- do not rename, add,
remove, or re-scope it. Your only job is to propose this group's own L1 user-goal use cases, and
L2 sub-use-cases where genuinely warranted, grounded in the CURRENT PROJECT REQUIREMENTS SOURCE
SNAPSHOT supplied in the user message.

MISSION
1. Propose the L1 user-goal use cases that belong under this group only -- do not propose a use
   case for a different group's scope.
2. For each L1 goal, decide whether it needs L2 detail at all. Most L1 goals do not -- add an L2
   row only for a genuinely shared subfunction (reused by more than one goal) or a complex goal
   whose steps are independently meaningful use cases. Do not decompose an ordinary sequential
   workflow into steps; that belongs in the L1 description, not as L2 rows.
3. Every generated use case MUST cite one or more exact evidence_id values from the snapshot.
4. primary_actor_id/secondary_actor_ids MUST be chosen from the supplied actor list only -- never
   invent a new actor here.
5. Each draft carries a `local_tag` you choose (e.g. "G1", "G2") so an L2 draft can reference its
   L1 parent via `parent_local_tag` -- neither this group's own id nor any use case's real id
   exists yet; the service assigns them after this call. An L1 draft leaves `parent_local_tag`
   null (its parent is this group itself).

NO INVENTED FEATURES (same rule as the full-model pass)
- Do not add a UI screen, API, database, notification, report, or integration as a use case.
- Do not use conversation text as evidence.
- An empty use_cases list is correct when this group genuinely has no distinct user goals beyond
  its own summary.

LANGUAGE
Active Verb + business Object from the actor's point of view -- no passive voice, no
screen/button/API/database terms.

OUTPUT
Return JSON only matching the supplied schema: a `use_cases` array. Do not return a real `id`, a
`subsystem_id`, a `status`, or `relationship_ids` -- the service fills those in. Do not return
Markdown or prose.
"""


@dataclass(frozen=True)
class UseCaseGroupUseCasesHarness:
    """Prompt boundary for the per-group follow-up call (Phase 2) that proposes one capability
    group's own L1/L2 use cases only.

    The output is scoped to one group regardless of project size, so it stays a fraction of the
    size of generating every group's use cases in one call -- the actual fix for generation timing
    out on a larger project. Unlike an earlier version of this harness, the source is NOT narrowed
    by a regex slice (that assumed one fixed capability-ID convention and silently produced an
    empty slice for any project that uses a different one) -- it gets the same full source the
    groups pass saw, and relies on the model to scope its own output to the given group.
    """

    source: RequirementsSourceSnapshot
    group: dict[str, Any]
    actors: list[dict[str, Any]]

    def build_system_instruction(self) -> str:
        return USE_CASE_GROUP_DETAIL_SYSTEM_PROMPT

    def build_user_prompt(self) -> str:
        import json

        return "\n\n".join(
            (
                f"Generate use cases for this one capability group only: {self.group.get('name')}.",
                f"source_hash: {self.source.source_hash}",
                "First reason over the source internally. Return only the final JSON object.",
                "\n--- THIS GROUP ---\n" + json.dumps(self.group, ensure_ascii=False, indent=2),
                "\n--- ACTORS ---\n" + json.dumps(self.actors, ensure_ascii=False, indent=2),
                "\n--- FULL STORED BRD/PRD COMPONENTS ---\n" + self.source.render_full_source(),
                "\n--- COMPLETE EVIDENCE INDEX ---\n" + self.source.render_evidence_index(),
                "\n--- JSON SCHEMA ---\n" + _group_detail_schema_text(),
            )
        )

    def output_schema(self) -> dict[str, Any]:
        return UseCaseGroupDetailDraftList.model_json_schema()

    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "reqtool_use_case_group_detail",
                "strict": True,
                "schema": self.output_schema(),
            },
        }


def _schema_text() -> str:
    import json

    return json.dumps(UseCaseModel.model_json_schema(), ensure_ascii=False, indent=2)


def _relationship_schema_text() -> str:
    import json

    return json.dumps(UseCaseRelationshipDraftList.model_json_schema(), ensure_ascii=False, indent=2)


def _groups_schema_text() -> str:
    import json

    return json.dumps(UseCaseGroupsDraftList.model_json_schema(), ensure_ascii=False, indent=2)


def _group_detail_schema_text() -> str:
    import json

    return json.dumps(UseCaseGroupDetailDraftList.model_json_schema(), ensure_ascii=False, indent=2)
