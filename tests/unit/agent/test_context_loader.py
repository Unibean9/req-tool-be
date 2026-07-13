"""Turn context loading invariants."""

from app.graphs.analysis.context_loader import _context_artifact_types
from app.graphs.policy import ancestor_types


def test_context_artifact_types_covers_same_container_and_transitive_ancestors():
    vision_objectives_types = _context_artifact_types("vision_objectives")
    assert vision_objectives_types[0] == "vision_objectives"
    assert "scope_capabilities" in vision_objectives_types  # same-container BRD sibling
    assert len(vision_objectives_types) == len(set(vision_objectives_types))
    assert ancestor_types("vision_objectives") == ["problem_statement"]

    functional_requirement_types = _context_artifact_types("functional_requirement")
    assert functional_requirement_types[:3] == ["functional_requirement", "use_case", "business_rules"]
    assert "scope_capabilities" in functional_requirement_types
