"""Turn context loading invariants."""

from app.graphs.analysis.context_loader import _context_artifact_types
from app.graphs.policy import ancestor_types


def test_context_types_include_same_container_sources_for_vision_objectives():
    types = _context_artifact_types("vision_objectives")

    assert types[0] == "vision_objectives"
    assert "executive_summary" in types
    assert len(types) == len(set(types))
    assert ancestor_types("vision_objectives") == []


def test_context_types_keep_transitive_ancestors_for_derived_artifacts():
    types = _context_artifact_types("functional_requirement")

    assert types[:2] == ["functional_requirement", "brd"]
    assert "use_case" in types
