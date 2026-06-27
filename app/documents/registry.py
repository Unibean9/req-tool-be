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


@dataclass(frozen=True)
class ArtifactOutputContract:
    artifact_type: str
    format: str
    required_headings: tuple[str, ...]
    guidance: str
    table_columns: tuple[str, ...] = ()
    confirmation_note: str = "(agent-inferred, needs confirmation)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "format": self.format,
            "required_headings": list(self.required_headings),
            "guidance": self.guidance,
            "table_columns": list(self.table_columns),
            "confirmation_note": self.confirmation_note,
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
            "risks_issues",
        ),
        is_container=True,
        description="Business context, scope, rules, constraints, and risks.",
    ),
    DocumentTypeConfig(
        artifact_type="prd",
        label="Product Requirements Document",
        children=(
            "functional_requirement",
            "use_case",
            "non_functional_requirement",
            "acceptance_criteria",
        ),
        is_container=True,
        description="Product behavior, quality attributes, use cases, and acceptance criteria.",
    ),
    DocumentTypeConfig(
        artifact_type="sad",
        label="Software Architecture Document",
        children=("domain_entity", "component", "interface", "tech_decision"),
        is_container=True,
        description="Architecture domains, components, interfaces, and technical decisions.",
    ),
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
    _item(
        "risks_issues",
        "Risks and Issues",
        "Risks and issues: adverse events, likelihood, mitigation, and tracking status.",
        sub_dimensions={
            "risk": "Adverse event or condition that may occur.",
            "likelihood": "Likelihood of the risk occurring.",
            "mitigation": "Risk mitigation or handling strategy.",
            "status": "Tracking status for the risk or open issue.",
        },
        threshold=0.75,
    ),
    _item("functional_requirement", "Functional Requirement", "A testable product behavior."),
    _item("use_case", "Use Case", "An actor-goal interaction flow."),
    _item("non_functional_requirement", "Non-Functional Requirement", "A measurable quality constraint."),
    _item("acceptance_criteria", "Acceptance Criteria", "Conditions that prove a requirement is met."),
    _item("domain_entity", "Domain Entity", "A core domain concept and its responsibilities."),
    _item("component", "Component", "A deployable or logical architecture component."),
    _item("interface", "Interface", "A contract between architecture components."),
    _item("tech_decision", "Technical Decision", "A technical choice, rationale, and consequences."),
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
    ),
    "business_rules": ArtifactOutputContract(
        artifact_type="business_rules",
        format="markdown",
        required_headings=("## Business Rules",),
        guidance="Each rule must include condition, trigger, outcome, scope, and exceptions.",
        table_columns=("rule id", "condition", "trigger", "outcome", "scope", "exception"),
    ),
    "constraints_assumptions": ArtifactOutputContract(
        artifact_type="constraints_assumptions",
        format="markdown",
        required_headings=("## Constraints", "## Assumptions", "## Validation Plan"),
        guidance="Separate hard constraints from assumptions and state validation approach.",
        table_columns=("constraint/assumption", "impact", "owner/source", "validation"),
    ),
    "risks_issues": ArtifactOutputContract(
        artifact_type="risks_issues",
        format="markdown",
        required_headings=("## Risks", "## Issues", "## Mitigation Plan"),
        guidance="Track risks/issues with likelihood, impact, and mitigation.",
        table_columns=("risk", "likelihood", "impact", "mitigation", "status"),
    ),
    "functional_requirement": ArtifactOutputContract(
        artifact_type="functional_requirement",
        format="markdown",
        required_headings=(
            "## Functional Requirement",
            "## Behavior",
            "## Inputs and Outputs",
            "## Acceptance Signals",
        ),
        guidance="Write testable behavior, using system shall/should where appropriate.",
    ),
    "use_case": ArtifactOutputContract(
        artifact_type="use_case",
        format="markdown",
        required_headings=(
            "## Use Case",
            "## Actors",
            "## Preconditions",
            "## Main Flow",
            "## Alternate / Exception Flows",
            "## Postconditions",
        ),
        guidance="Describe actor-goal flow and error/exception branches.",
    ),
    "non_functional_requirement": ArtifactOutputContract(
        artifact_type="non_functional_requirement",
        format="markdown",
        required_headings=("## Quality Attribute", "## Requirement", "## Measurement", "## Scope and Tradeoffs"),
        guidance="State quality attributes with measurable targets and verification method.",
    ),
    "acceptance_criteria": ArtifactOutputContract(
        artifact_type="acceptance_criteria",
        format="markdown",
        required_headings=("## Acceptance Criteria",),
        guidance="Use Given/When/Then or a testable checklist.",
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
        ),
        guidance="ADR-style Markdown for technical decisions, options, and consequences.",
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
