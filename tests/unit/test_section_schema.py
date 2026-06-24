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
    assert all_container_types() == ("brd", "prd", "sad")


def test_brd_registry_has_seven_items():
    assert children_of("brd") == (
        "vision_objectives",
        "problem_statement",
        "stakeholder_register",
        "scope_capabilities",
        "business_rules",
        "constraints_assumptions",
        "risks_issues",
    )


def test_prd_registry_has_four_active_items():
    assert children_of("prd") == (
        "functional_requirement",
        "use_case",
        "non_functional_requirement",
        "acceptance_criteria",
    )


def test_sad_registry_has_four_scaffold_items():
    assert children_of("sad") == (
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
    }
    for item_type in all_item_types():
        config = get_config(item_type)
        assert config.is_container is False
        assert config.label
        assert config.description


def test_children_of_rejects_non_container():
    with pytest.raises(ValueError):
        children_of("vision_objectives")
