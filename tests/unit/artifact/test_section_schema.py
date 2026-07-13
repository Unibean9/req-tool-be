"""Document registry contract tests."""

import pytest

from app.documents.registry import (
    all_container_types,
    all_item_types,
    children_of,
    container_for,
    get_config,
)


def test_registry_has_all_document_containers():
    assert all_container_types() == ("brd", "prd", "event_storming", "add")


def test_brd_registry_has_six_items():
    # executive_summary promoted to a project field; risks_issues merged into
    # constraints_assumptions.
    assert children_of("brd") == (
        "problem_statement",
        "vision_objectives",
        "stakeholder_register",
        "scope_capabilities",
        "business_rules",
        "constraints_assumptions",
    )


def test_prd_registry_has_three_active_items():
    # acceptance_criteria merged into functional_requirement.
    assert children_of("prd") == (
        "use_case",
        "functional_requirement",
        "non_functional_requirement",
    )


def test_add_registry_has_five_scaffold_items():
    assert children_of("add") == (
        "tech_stack",
        "domain_entity",
        "component",
        "interface",
        "tech_decision",
    )


def test_registry_maps_items_to_containers():
    assert container_for("vision_objectives") == "brd"
    assert container_for("functional_requirement") == "prd"
    assert container_for("component") == "add"
    assert container_for("epic") is None


def test_registry_item_metadata_is_complete():
    assert set(all_item_types()) == {
        *children_of("brd"),
        *children_of("prd"),
        *children_of("add"),
        *children_of("event_storming"),
    }
    for item_type in all_item_types():
        config = get_config(item_type)
        assert config.is_container is False
        assert config.label
        assert config.description


def test_registry_uses_standard_artifact_display_metadata():
    expected = {
        "brd": (
            "Business Requirements Document",
            "Business requirements: problem, vision, stakeholders, scope, rules, constraints, assumptions, and risks.",
        ),
        "prd": (
            "Product Requirements Document",
            "Product requirements: business capabilities, functional behavior, and measurable quality attributes.",
        ),
        "event_storming": (
            "Event Storming Canvas",
            "Domain discovery canvas for events, commands, policies, aggregates, and hotspots.",
        ),
        "add": (
            "Architecture Design Document",
            "Architecture design: technology selections, domain model, components, interfaces, and architecture decisions.",
        ),
        "problem_statement": (
            "Problem Statement",
            "Defines the affected audience, obstacle, root cause, frequency, and impact.",
        ),
        "vision_objectives": (
            "Vision and Objectives",
            "Defines the product vision, measurable objectives, success metrics, targets, and timeframe.",
        ),
        "stakeholder_register": (
            "Stakeholder Register",
            "Identifies users, stakeholders, decision makers, operators, responsibilities, and concerns.",
        ),
        "scope_capabilities": (
            "Scope and Capabilities",
            "Defines in-scope and out-of-scope capabilities, priorities, rationale, and dependencies.",
        ),
        "business_rules": (
            "Business Rules",
            "Captures testable business policies with conditions, triggers, outcomes, scope, and exceptions.",
        ),
        "constraints_assumptions": (
            "Constraints, Assumptions, and Risks",
            "Captures hard constraints, assumptions to validate, dependencies, risks, and mitigations.",
        ),
        "functional_requirement": (
            "Functional Requirements",
            "Testable product behaviors with inputs, outputs, acceptance signals, priority, and dependencies.",
        ),
        "use_case": (
            "Business Capabilities",
            "Business capabilities that define domain boundaries or major business flows.",
        ),
        "non_functional_requirement": (
            "Non-Functional Requirements",
            "Measurable quality attributes, constraints, and verification methods.",
        ),
        "tech_stack": (
            "Technology Stack",
            "Technology selections by category, alternatives considered, pinned versions, and rationale.",
        ),
        "domain_entity": (
            "Domain Entities",
            "Core domain concepts, responsibilities, attributes, relationships, and lifecycle.",
        ),
        "component": (
            "Architecture Components",
            "Deployable or logical architecture components, responsibilities, interfaces, dependencies, and constraints.",
        ),
        "interface": (
            "Interface Contracts",
            "Provider/consumer contracts, data exchanged, error cases, and compatibility notes.",
        ),
        "tech_decision": (
            "Architecture Decisions",
            "ADR-style architecture decisions with context, options, rationale, consequences, and event-storming references.",
        ),
        "domain_event": (
            "Domain Events",
            "Past-tense domain facts grouped by business flow, with triggers, downstream effects, and hotspots.",
        ),
        "actor_command": (
            "Actors and Commands",
            "Actors, commands, preconditions, resulting events, and participating business flows.",
        ),
        "policy": (
            "Policies",
            "Event-triggered reactions that connect events to commands or events and note aggregate-boundary crossings.",
        ),
        "aggregate": (
            "Aggregates",
            "Consistency boundaries, responsibilities, handled commands, emitted events, invariants, and flows.",
        ),
    }

    for artifact_type, (label, description) in expected.items():
        config = get_config(artifact_type)
        assert config.label == label
        assert config.description == description


def test_all_item_types_preserves_brd_order_first():
    brd_children = children_of("brd")
    assert all_item_types()[: len(brd_children)] == brd_children


def test_children_of_rejects_non_container():
    with pytest.raises(ValueError):
        children_of("vision_objectives")
