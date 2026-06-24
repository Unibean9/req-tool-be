"""Tests for the BMAD method-layer state fields (addendum §8)."""

from app.graphs.state import (
    DEFAULT_ARTIFACT_CHAIN,
    DEFAULT_METHOD_PROFILE,
    DEFAULT_READINESS,
    WorkflowState,
)


def test_method_profile_fields_exist_in_workflow_state():
    for key in ("method_profile", "artifact_chain", "readiness"):
        assert key in WorkflowState.__annotations__


def test_method_profile_default_values():
    assert DEFAULT_METHOD_PROFILE["method"] == "bmad_inspired"
    assert DEFAULT_METHOD_PROFILE["planning_track"] == "quick"
    assert DEFAULT_METHOD_PROFILE["current_workflow"] == "brainstorm"
    assert DEFAULT_METHOD_PROFILE["recommended_next_workflow"] is None
    assert DEFAULT_METHOD_PROFILE["project_type"] == "unknown"


def test_artifact_chain_default_values():
    assert DEFAULT_ARTIFACT_CHAIN == {
        "brainstorming": "missing",
        "product_brief": "missing",
        "prd": "missing",
    }


def test_readiness_default_values():
    assert DEFAULT_READINESS["requirements_ready"] is False
    assert DEFAULT_READINESS["architecture_needed"] == "unknown"
    assert DEFAULT_READINESS["blocking_gaps"] == []
