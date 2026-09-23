"""AI-agent harness for deriving a traceable use-case model from stored BRD/PRD components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.use_cases.diagram import build_diagram_render_plan
from app.use_cases.models import (
    DiagramRenderPlan,
    RequirementsSourceSnapshot,
    UseCaseModel,
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
   candidate. Do not make inferred/suggested items eligible for the SRS diagram.

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
3. Abstraction: L0 is the single system overview with Summary use cases (Actor × subsystem). L1
   contains one diagram per subsystem with User Goal use cases. L2 is optional detail only for a
   genuinely shared subfunction or a complex user goal. Do not use L2 for ordinary step-by-step
   decomposition. Use parent_use_case_id to make the table hierarchy explicit; an L2 shared
   subfunction may have multiple include parents instead. A 'Manage X' summary is allowed only
   when its description lists at least two concrete CRUD-style operations.
4. IDs: use one stable, never-reused UC-01/UC-001 or UC-SUB-01 format consistently. Use the same
   id and name in the table, every diagram, and traceability. Do not put actor, priority, sprint,
   or status in an id.

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
- Keep each diagram readable: approximately 5–12 use cases, at most 6 actors, and no more than 5
  include/extend relations. Produce exactly one L0 and one L1 per represented subsystem; produce
  L2 only when needed.

OUTPUT
Return JSON only matching the supplied UseCaseModel schema. Do not return Markdown, Mermaid,
coordinates, or prose. The backend validator is authoritative and will reject unsupported refs,
invalid names, wrong relation direction/notation, out-of-scope items, missing actors, and diagram
size/level violations. Human review is required before any inferred/suggested row becomes confirmed.
"""


@dataclass(frozen=True)
class UseCaseGenerationHarness:
    """Prompt + validation boundary shared by the API and a future graph node."""

    source: RequirementsSourceSnapshot
    language: str = "match_source"

    def build_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.build_system_instruction()},
            {"role": "user", "content": self.build_user_prompt()},
        ]

    def build_system_instruction(self) -> str:
        return USE_CASE_AGENT_SYSTEM_PROMPT

    def build_user_prompt(self) -> str:
        return "\n\n".join(
            (
                "Generate the use-case model for this stored project source snapshot.",
                f"source_hash: {self.source.source_hash}",
                f"language_policy: {self.language} (keep diagram labels consistent with the source documents)",
                "First reason over every component internally. Return only the final JSON object.",
                "\n--- FULL STORED BRD/PRD COMPONENTS ---\n" + self.source.render_full_source(),
                "\n--- COMPLETE EVIDENCE INDEX ---\n" + self.source.render_evidence_index(),
                "\n--- JSON SCHEMA ---\n" + _schema_text(),
            )
        )

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


def _schema_text() -> str:
    import json

    return json.dumps(UseCaseModel.model_json_schema(), ensure_ascii=False, indent=2)
