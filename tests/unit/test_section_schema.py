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
    assert all_container_types() == ("brd", "prd", "sad", "event_storming")


def test_brd_registry_has_six_items():
    # executive_summary promoted to a project field; risks_issues merged into
    # constraints_assumptions.
    assert children_of("brd") == (
        "vision_objectives",
        "problem_statement",
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


def test_sad_registry_has_five_scaffold_items():
    assert children_of("sad") == (
        "tech_stack",
        "domain_entity",
        "component",
        "interface",
        "tech_decision",
    )


def test_registry_maps_items_to_containers():
    assert container_for("vision_objectives") == "brd"
    assert container_for("functional_requirement") == "prd"
    assert container_for("component") == "sad"
    assert container_for("epic") is None


def test_registry_item_metadata_is_complete():
    assert set(all_item_types()) == {
        *children_of("brd"),
        *children_of("prd"),
        *children_of("sad"),
        *children_of("event_storming"),
    }
    for item_type in all_item_types():
        config = get_config(item_type)
        assert config.is_container is False
        assert config.label
        assert config.description


def test_children_of_rejects_non_container():
    with pytest.raises(ValueError):
        children_of("vision_objectives")
