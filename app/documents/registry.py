from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentTypeConfig:
    artifact_type: str
    label: str
    children: tuple[str, ...]
    is_container: bool
    description: str
    sub_dimensions: tuple[tuple[str, str], ...] = ()
    threshold: float = 1.0


# Shown verbatim in any table cell the agent left empty. It is also the gate's signal: a body
# containing it has an unfilled required column, so candidate_readiness refuses to mark it SUFFICIENT.
INCOMPLETE_CELL_PLACEHOLDER = "_(cần bổ sung)_"


@dataclass(frozen=True)
class ArtifactOutputContract:
    artifact_type: str
    format: str
    required_headings: tuple[str, ...]
    guidance: str
    table_columns: tuple[str, ...] = ()
    # id_prefix turns the first "id" column into an auto-numbered trace tag (e.g. FR-01) so other
    # artifacts cross-reference a requirement by tag instead of restating it. render_style selects how
    # an id-tagged item projects: "table" (one row per entry) or "entries" (a per-entry sub-section,
    # used when a field like a multi-step flow does not fit a cell).
    id_prefix: str = ""
    render_style: str = "table"
    confirmation_note: str = "(agent-inferred, needs confirmation)"
    # Per-artifact-type behavior WITHIN the universal phases (the DRAFT scaffold is already
    # required_headings). All optional: an empty tuple/string means the type falls back to the generic
    # phase behavior, so unmapped types and legacy callers are unchanged.
    #   elicit_checklist  — key topics/questions to gather for this type (ELICIT phase).
    #   elicit_technique  — a suggested BMAD technique name (must exist in agent_tools.ELICIT_TECHNIQUES).
    #   review_criteria   — per-type critique criteria (REVIEW phase).
    elicit_checklist: tuple[str, ...] = ()
    elicit_technique: str = ""
    review_criteria: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "format": self.format,
            "required_headings": list(self.required_headings),
            "guidance": self.guidance,
            "table_columns": list(self.table_columns),
            "id_prefix": self.id_prefix,
            "render_style": self.render_style,
            "confirmation_note": self.confirmation_note,
            "elicit_checklist": list(self.elicit_checklist),
            "elicit_technique": self.elicit_technique,
            "review_criteria": list(self.review_criteria),
        }


def _item(
    artifact_type: str,
    label: str,
    description: str,
    *,
    sub_dimensions: dict[str, str] | None = None,
    threshold: float = 1.0,
) -> DocumentTypeConfig:
    return DocumentTypeConfig(
        artifact_type=artifact_type,
        label=label,
        children=(),
        is_container=False,
        description=description,
        sub_dimensions=tuple((sub_dimensions or {}).items()),
        threshold=threshold,
    )


_CONFIGS: tuple[DocumentTypeConfig, ...] = (
    DocumentTypeConfig(
        artifact_type="brd",
        label="Business Requirements Document",
        children=(
            "vision_objectives",
            "problem_statement",
            "stakeholder_register",
            "scope_capabilities",
            "business_rules",
            "constraints_assumptions",
        ),
        is_container=True,
        description="Business context, scope, rules, constraints, and risks.",
    ),
    DocumentTypeConfig(
        artifact_type="prd",
        label="Product Requirements Document",
        children=(
            "use_case",
            "functional_requirement",
            "non_functional_requirement",
        ),
        is_container=True,
        description="Product behavior, quality attributes, and use cases.",
    ),
    DocumentTypeConfig(
        artifact_type="sad",
        label="Software Architecture Document",
        children=("tech_stack", "domain_entity", "component", "interface", "tech_decision"),
        is_container=True,
        description="Architecture domains, components, interfaces, and technical decisions.",
    ),
    DocumentTypeConfig(
        artifact_type="event_storming",
        label="Event Storming",
        children=("domain_event", "actor_command", "policy", "aggregate"),
        is_container=True,
        description="Domain events, actors/commands, policies, and aggregates discovered via event storming.",
    ),
    # executive_summary is retired as an elicited item — it is synthesized from
    # vision/problem/scope and promoted to a project-level field. The enum value
    # is kept for historical rows; it is no longer a BRD child.
    _item(
        "vision_objectives",
        "Vision and Objectives",
        "Vision and objectives: why this exists, what success looks like, and how it is measured.",
        sub_dimensions={
            "business_goal": "Specific business goal to achieve.",
            "user_goal": "Outcome the user wants to achieve.",
            "metric": "Metric used to measure success.",
            "target": "Target threshold or outcome to achieve.",
            "timeframe": "Timeline or completion deadline.",
            "intent": "Reason and motivation for the initiative.",
            "success_definition": "Ideal state when the initiative succeeds.",
        },
        threshold=0.8,
    ),
    _item(
        "problem_statement",
        "Problem Statement",
        "Problem statement: who faces which obstacle, root cause, frequency, and impact.",
        sub_dimensions={
            "who": "Audience directly affected by the problem.",
            "obstacle": "Specific obstacle the audience faces.",
            "root_cause": "Root cause after problem exploration.",
            "frequency": "Frequency or repeated scale of the problem.",
            "impact": "Impact on users, process, or business.",
        },
        threshold=0.8,
    ),
    _item(
        "stakeholder_register",
        "Stakeholder Register",
        "Stakeholder register: primary users, secondary stakeholders, decision makers, and operators.",
        sub_dimensions={
            "primary_user": "Direct end user or primary persona.",
            "secondary_stakeholders": "Indirect stakeholders and how they are affected.",
            "decision_maker": "Person with approval or decision authority.",
            "operator": "Person who deploys, operates, or supports after launch.",
        },
        threshold=0.75,
    ),
    _item(
        "scope_capabilities",
        "Scope and Capabilities",
        "Scope and capabilities: what is in scope, out of scope, required capabilities, and priority.",
        sub_dimensions={
            "in_scope": "Capabilities and items in scope for this iteration.",
            "out_of_scope": "Items explicitly excluded from scope.",
            "capability": "Required product or system capability.",
            "priority": "Priority such as Must, Should, or Could.",
        },
        threshold=0.75,
    ),
    _item(
        "business_rules",
        "Business Rules",
        "Business rules: conditions, outcomes, triggers, and applicability scope.",
        sub_dimensions={
            "condition": "Condition that triggers the business rule.",
            "outcome": "Outcome or action when the condition is satisfied.",
            "trigger": "Event or time when the rule applies.",
            "scope": "Applicability scope of the rule.",
        },
        threshold=0.75,
    ),
    _item(
        "constraints_assumptions",
        "Constraints and Assumptions",
        "Constraints and assumptions: hard limits, relied-on assumptions, and validation approach.",
        sub_dimensions={
            "constraint": "Hard constraint on time, budget, technology, or legal scope.",
            "assumption": "Assumption currently relied on for decisions.",
            "validation": "How the assumption will be validated before build.",
            "dependency": "External dependency, vendor, or another team.",
        },
        threshold=0.75,
    ),
    # risks_issues is retired as a standalone item — risks and mitigation are
    # merged into constraints_assumptions ("Constraints, Assumptions & Risks").
    # The enum value is kept for historical rows.
    _item("functional_requirement", "Functional Requirements", "A testable product behavior."),
    _item(
        "use_case",
        "Business Capabilities",
        "A business capability: the boundary of a domain or a major business flow.",
        sub_dimensions={
            "business_value": "Business value the capability delivers, not the mechanics of using it.",
            "user_segment": "Target user segment or persona the capability serves.",
        },
    ),
    _item("non_functional_requirement", "Non-Functional Requirements", "A measurable quality constraint."),
    # acceptance_criteria is retired as a standalone item — Given/When/Then
    # acceptance is merged into the functional_requirement contract. The enum
    # value is kept for historical rows.
    _item(
        "tech_stack",
        "Tech Stack",
        "The definitive technology selection: options considered, the chosen technology, and pinned version.",
    ),
    _item("domain_entity", "Domain Entity", "A core domain concept and its responsibilities."),
    _item("component", "Component", "A deployable or logical architecture component."),
    _item("interface", "Interface", "A contract between architecture components."),
    _item("tech_decision", "Technical Decision", "A technical choice, rationale, and consequences."),
    _item("domain_event", "Domain Events", "A past-tense fact that happened in the domain, grouped by business flow."),
    _item("actor_command", "Actors and Commands", "An actor-initiated intent that triggers a domain event."),
    _item(
        "policy",
        "Policies",
        "A 'Whenever {event}, then {command/event}' reaction, often crossing an aggregate boundary.",
    ),
    _item("aggregate", "Aggregates", "A consistency boundary that handles commands and emits events."),
)

_BY_TYPE = {config.artifact_type: config for config in _CONFIGS}
_CONTAINER_BY_ITEM = {
    child: config.artifact_type for config in _CONFIGS if config.is_container for child in config.children
}

_OUTPUT_CONTRACTS: dict[str, ArtifactOutputContract] = {
    "vision_objectives": ArtifactOutputContract(
        artifact_type="vision_objectives",
        format="markdown",
        required_headings=("## Vision", "## Objectives", "## Success Metrics"),
        guidance="Document the vision, objectives, and success measurement.",
        table_columns=("goal", "user/business value", "metric", "target", "timeframe"),
    ),
    "problem_statement": ArtifactOutputContract(
        artifact_type="problem_statement",
        format="markdown",
        required_headings=(
            "## Problem Statement",
            "## Affected Users",
            "## Impact",
            "## Root Cause / Contributing Factors",
        ),
        guidance="State who faces which problem, its frequency, cause, and impact.",
    ),
    "stakeholder_register": ArtifactOutputContract(
        artifact_type="stakeholder_register",
        format="markdown",
        required_headings=("## Stakeholders",),
        guidance="List stakeholders and their authority/responsibilities.",
        table_columns=("role", "responsibility", "decision authority", "needs/concerns", "involvement"),
    ),
    "scope_capabilities": ArtifactOutputContract(
        artifact_type="scope_capabilities",
        format="markdown",
        required_headings=("## Scope", "## Capabilities", "## Out of Scope"),
        guidance="Separate scope, required capabilities, and exclusions clearly.",
        table_columns=("capability", "priority", "rationale", "dependency"),
        elicit_checklist=(
            "Which capabilities are in scope vs explicitly out of scope?",
            "For each capability: priority and the rationale for that priority.",
            "What dependencies exist between capabilities?",
            "Where is the boundary that separates this release from later ones?",
        ),
        elicit_technique="moscow",
        review_criteria=(
            "Every capability carries a defensible priority (Must/Should/Could/Won't).",
            "Out-of-scope items are stated explicitly, not merely omitted.",
            "Dependencies between capabilities are captured.",
        ),
    ),
    "business_rules": ArtifactOutputContract(
        artifact_type="business_rules",
        format="markdown",
        required_headings=("## Business Rules",),
        guidance="Each rule must include condition, trigger, outcome, scope, and exceptions.",
        table_columns=("rule id", "condition", "trigger", "outcome", "scope", "exception"),
        elicit_checklist=(
            "For each rule: the condition, the triggering event, and the resulting outcome.",
            "What is the scope of the rule and who owns it?",
            "What exceptions or edge cases override the rule?",
            "Are any rules in conflict with each other?",
        ),
        elicit_technique="socratic_questioning",
        review_criteria=(
            "Each rule is testable: condition, trigger, and outcome are unambiguous.",
            "Exceptions and edge cases are documented, not implied.",
            "No two rules contradict each other.",
        ),
    ),
    # Constraints, Assumptions & Risks — risks_issues was merged in here. Risks
    # live in a second table under ## Risks with strategy under ## Mitigation Plan.
    "constraints_assumptions": ArtifactOutputContract(
        artifact_type="constraints_assumptions",
        format="markdown",
        required_headings=(
            "## Constraints",
            "## Assumptions",
            "## Validation Plan",
            "## Risks",
            "## Mitigation Plan",
        ),
        guidance=(
            "Separate hard constraints from assumptions and state the validation approach. "
            "Capture risks in a second table under ## Risks (risk, likelihood, impact, mitigation, "
            "status) and record the handling strategy under ## Mitigation Plan."
        ),
        table_columns=("constraint/assumption", "impact", "owner/source", "validation"),
        elicit_checklist=(
            "What hard constraints (time, budget, technology, legal) bound this work?",
            "Which assumptions are relied on, and how will each be validated before build?",
            "What could cause this initiative to fail or slip, and how likely is each risk?",
            "For each top risk: its impact, an owner, and a concrete mitigation or contingency.",
        ),
        elicit_technique="pre_mortem",
        review_criteria=(
            "Constraints are separated from assumptions, each with a validation approach.",
            "Every risk has a likelihood, an impact, and a concrete mitigation (not 'monitor').",
            "No high-impact risk is left without an owner or contingency.",
        ),
    ),
    # Functional requirement — acceptance_criteria was merged in here: each row's
    # "acceptance signal" carries a Given/When/Then condition instead of a separate
    # AC artifact.
    "functional_requirement": ArtifactOutputContract(
        artifact_type="functional_requirement",
        format="markdown",
        required_headings=("## Functional Requirements",),
        guidance=(
            "One row per testable behavior; reference business capabilities by their BC id where "
            "relevant. State each requirement's acceptance as a Given/When/Then condition in the "
            "acceptance signal column. The dependencies column lists FR ids this requirement "
            "depends on, as free text."
        ),
        table_columns=(
            "id",
            "requirement",
            "behavior",
            "inputs/outputs",
            "acceptance signal",
            "priority",
            "dependencies",
        ),
        id_prefix="FR",
        elicit_checklist=(
            "For each behavior: the exact input, the expected output, and the observable acceptance signal.",
            "Which business capability (BC id) does each requirement serve?",
            "State acceptance as Given/When/Then, including negative/failure cases, not only the happy path.",
            "Are the acceptance results measurable rather than subjective?",
            "Priority of each requirement and its dependencies on other FRs.",
        ),
        elicit_technique="first_principles",
        review_criteria=(
            "Each requirement is independently testable via its Given/When/Then acceptance signal.",
            "Inputs and outputs are concrete, not vague ('handle X properly').",
            "Failure/negative cases are covered, not only the happy path.",
        ),
    ),
    "use_case": ArtifactOutputContract(
        artifact_type="use_case",
        format="markdown",
        required_headings=("## Business Capabilities",),
        guidance=(
            "List 3-8 business capabilities. Describe each briefly: the goal it solves, the user segment "
            "it serves, the business value it delivers, and its main scope. Do not detail flows, "
            "preconditions, or exception branches here — those belong to user stories and event storming."
        ),
        table_columns=("goal", "user_segment", "business_value", "scope"),
        id_prefix="BC",
        render_style="entries",
    ),
    "non_functional_requirement": ArtifactOutputContract(
        artifact_type="non_functional_requirement",
        format="markdown",
        required_headings=("## Non-Functional Requirements",),
        guidance="One row per quality attribute with a measurable target and verification method.",
        table_columns=("id", "quality attribute", "requirement", "measurement", "scope/tradeoff"),
        id_prefix="NFR",
    ),
    "tech_stack": ArtifactOutputContract(
        artifact_type="tech_stack",
        format="markdown",
        required_headings=("## Tech Stack",),
        guidance=(
            "One row per technology category (e.g. language, framework, database). List the options "
            "considered, the chosen technology, the pinned version, and the rationale for the choice."
        ),
        table_columns=("category", "options considered", "choice", "version", "rationale"),
    ),
    "domain_entity": ArtifactOutputContract(
        artifact_type="domain_entity",
        format="markdown",
        required_headings=(
            "## Domain Entity",
            "## Responsibilities",
            "## Attributes",
            "## Relationships",
            "## Lifecycle / States",
        ),
        guidance="Describe domain concepts, attributes, relationships, and lifecycle.",
    ),
    "component": ArtifactOutputContract(
        artifact_type="component",
        format="markdown",
        required_headings=("## Component", "## Responsibilities", "## Interfaces", "## Dependencies", "## Constraints"),
        guidance="Describe architecture components, responsibilities, and dependencies.",
    ),
    "interface": ArtifactOutputContract(
        artifact_type="interface",
        format="markdown",
        required_headings=(
            "## Interface Contract",
            "## Provider and Consumers",
            "## Data Exchanged",
            "## Error Cases",
            "## Compatibility Notes",
        ),
        guidance="Describe provider/consumer contract, data, errors, and compatibility.",
    ),
    "tech_decision": ArtifactOutputContract(
        artifact_type="tech_decision",
        format="markdown",
        required_headings=(
            "## Technical Decision",
            "## Context",
            "## Options Considered",
            "## Decision",
            "## Consequences",
            "## Event Storming Reference",
        ),
        guidance=(
            "ADR-style Markdown for technical decisions, options, and consequences. Under "
            "`## Event Storming Reference`, name the specific event storming event or actor that "
            "drove the decision by its stable id (EVT-/CMD-/POL-/AGG-/HS-), not just by prose name."
        ),
        review_criteria=(
            "Verify the decision cites a concrete event/actor from the Event Storming draft by its "
            "ES id (e.g. EVT-04, HS-01), not a placeholder or generic justification.",
        ),
    ),
    "domain_event": ArtifactOutputContract(
        artifact_type="domain_event",
        format="markdown",
        required_headings=("## Domain Events", "## Hotspots"),
        guidance=(
            "Group domain events by business flow: one `### Flow: {BC-id} {name}` sub-heading per "
            "flow, cross-referencing the PRD's `use_case` Business Capability id, each with its own "
            "small `flowchart LR` mermaid diagram showing that flow's command -> event -> policy -> "
            "aggregate chain only, not one global diagram covering every flow. Record only Trigger, "
            "Triggered By, and Downstream Effects for each event — no structured data/schema table; "
            "that is SAD's `domain_entity`, derived later, not an event storming input. Events are "
            "past-tense facts: a `...Requested` event records that a request was received, and "
            "commands are not restated as events. Under `## Hotspots`, list one row per unresolved "
            "point of confusion or conflict surfaced during the timeline walkthrough, with columns "
            "id (HS-prefixed), description, raised during, flow, and status."
        ),
        table_columns=("trigger", "triggered by", "downstream effects"),
        id_prefix="EVT",
        render_style="entries",
        elicit_checklist=(
            "For each business flow (PRD use_case BC id): walk the domain events in chronological order.",
            "For each event: what trigger (a command or another event) caused it?",
            "For each event: what downstream effects (events, policies, aggregates) does it cause?",
            "What hotspots (unresolved confusion, conflict, or risk) surfaced during the timeline walkthrough?",
        ),
        elicit_technique="event_storming",
        review_criteria=(
            "An event names its trigger and actor with no data schema.",
            "Every `### Flow:` sub-heading names a BC id present in the PRD draft.",
            "Each flow has its own `flowchart LR` diagram.",
            "Events are past-tense facts.",
            "Every hotspot has an HS id, the flow it blocks, and a status.",
        ),
    ),
    "actor_command": ArtifactOutputContract(
        artifact_type="actor_command",
        format="markdown",
        required_headings=("## Actors and Commands",),
        guidance=(
            "One row per command: the actor who issues it, the command itself, its precondition, "
            "and the resulting event. Commands are often shared across flows, so each row notes the "
            "flow/BC-id it participates in rather than being grouped under a per-flow sub-heading."
        ),
        table_columns=("id", "actor", "command", "precondition", "resulting event", "flow"),
        id_prefix="CMD",
        render_style="table",
        elicit_checklist=(
            "Who are the actors (users, systems, schedulers) that initiate commands?",
            "For each actor: what commands do they issue?",
            "For each command: what precondition must hold before it can be accepted?",
            "For each command: what domain event does it result in?",
        ),
        elicit_technique="event_storming",
        review_criteria=("A command names its precondition and resulting event.",),
    ),
    "policy": ArtifactOutputContract(
        artifact_type="policy",
        format="markdown",
        required_headings=("## Policies",),
        guidance=(
            "A policy is a 'Whenever {event}, then {command/event}' reaction — the automatic "
            "connection between events and commands, often crossing an aggregate boundary. Policies "
            "are often shared across flows, so each entry notes the flow/BC-id it participates in "
            "rather than being grouped under a per-flow sub-heading."
        ),
        table_columns=("when", "then", "crosses aggregate boundary", "flow"),
        id_prefix="POL",
        render_style="entries",
        elicit_checklist=(
            "For each event: does it trigger a 'whenever {event}, then {command/event}' reaction?",
            "For each policy: which aggregate emits the event, and which aggregate handles the reaction?",
            "Does the reaction cross an aggregate boundary, or stay within the same aggregate?",
        ),
        elicit_technique="event_storming",
        review_criteria=(
            "A policy names the event(s) it reacts to and the command/event it triggers, and states "
            "whether it crosses an aggregate boundary.",
            "Event-to-event edges without a policy are only acceptable as documented same-aggregate "
            "emissions.",
        ),
    ),
    "aggregate": ArtifactOutputContract(
        artifact_type="aggregate",
        format="markdown",
        required_headings=("## Aggregates",),
        guidance=(
            "One entry per aggregate: its responsibilities, the commands it handles, the events it "
            "emits, and its invariants. An aggregate shared across flows appears once with a "
            "`Flows:` note listing every flow/BC-id it participates in — never duplicated per flow."
        ),
        table_columns=("responsibilities", "commands handled", "events emitted", "invariants", "flows"),
        id_prefix="AGG",
        render_style="entries",
        elicit_checklist=(
            "For each aggregate: what is its core responsibility and consistency boundary?",
            "Which commands does it handle?",
            "Which events does it emit?",
            "What invariants must it always enforce?",
        ),
        elicit_technique="event_storming",
        review_criteria=(
            "An aggregate names the commands it handles, the events it emits, and its invariants.",
            "An aggregate shared across flows appears once with a `Flows:` note, never duplicated per flow.",
        ),
    ),
}


def get_config(artifact_type: str) -> DocumentTypeConfig:
    try:
        return _BY_TYPE[artifact_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported document artifact type: {artifact_type}") from exc


def output_contract(item_type: str) -> ArtifactOutputContract:
    if item_type not in all_item_types():
        raise ValueError(f"{item_type} is not a document item")
    try:
        return _OUTPUT_CONTRACTS[item_type]
    except KeyError as exc:
        raise ValueError(f"Document item has no output contract: {item_type}") from exc


def container_for(item_type: str) -> str | None:
    return _CONTAINER_BY_ITEM.get(item_type)


def children_of(container_type: str) -> tuple[str, ...]:
    config = get_config(container_type)
    if not config.is_container:
        raise ValueError(f"{container_type} is not a document container")
    return config.children


def all_container_types() -> tuple[str, ...]:
    return tuple(config.artifact_type for config in _CONFIGS if config.is_container)


def all_item_types() -> tuple[str, ...]:
    return tuple(config.artifact_type for config in _CONFIGS if not config.is_container)


def item_configs(container_type: str) -> tuple[DocumentTypeConfig, ...]:
    return tuple(get_config(item_type) for item_type in children_of(container_type))


def item_description(item_type: str) -> str:
    return get_config(item_type).description


def item_label(item_type: str) -> str:
    return get_config(item_type).label


def status_score(status: Any) -> float:
    if status == "filled":
        return 1.0
    if status in {"partial", "needs_review"}:
        return 0.5
    return 0.0
