"""Tests for workflow_mode / planning_track assignment in analyze_node (addendum §5, §11, §12)."""

from app.graphs.nodes import _infer_workflow_mode
from tests.factories import _state


def test_infer_workflow_mode_low_coverage_returns_brainstorm():
    state = _state(artifact_type="intent")
    state["section_coverage"] = {"vision_objectives": "missing", "problem_statement": "missing"}
    assert _infer_workflow_mode(state) == "brainstorm"


def test_infer_workflow_mode_partial_coverage_by_artifact_type():
    partial = {"vision_objectives": "partial", "problem_statement": "partial"}
    brief_state = _state(artifact_type="brainstorming")
    brief_state["section_coverage"] = partial
    assert _infer_workflow_mode(brief_state) == "brief"

    prd_state = _state(artifact_type="product_brief")
    prd_state["section_coverage"] = partial
    assert _infer_workflow_mode(prd_state) == "prd"


def test_planning_track_quick_is_default():
    state = _state(artifact_type="intent")
    assert state["method_profile"]["planning_track"] == "quick"
