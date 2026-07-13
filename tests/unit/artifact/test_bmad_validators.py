"""Tests for BMAD workflow validators (addendum §17)."""

from app.graphs.validators import validate_proposal


def test_invalid_workflow_mode_raises_violation():
    r = validate_proposal("workflow_state", {"title": "T", "body": "B",
                                             "workflow_mode": "unknown_mode", "planning_track": "quick"})
    assert any("workflow_mode" in v for v in r.violations)


def test_invalid_planning_track_raises_violation():
    r = validate_proposal("workflow_state", {"title": "T", "body": "B",
                                             "workflow_mode": "brief", "planning_track": "mega"})
    assert any("planning_track" in v for v in r.violations)


def test_valid_workflow_state_passes():
    r = validate_proposal("workflow_state", {"title": "T", "body": "B",
                                             "workflow_mode": "brief", "planning_track": "quick"})
    assert not any("workflow_mode" in v or "planning_track" in v for v in r.violations)


def test_epic_story_readiness_is_out_of_scope():
    r = validate_proposal("workflow_recommendation", {
        "title": "T", "body": "B",
        "recommended_next_workflow": "epic_story_readiness", "prd_coverage": 0.3,
    })
    assert any("outside BMAD MVP scope" in v for v in r.violations)


def test_cannot_recommend_implementation_with_critical_risks():
    r = validate_proposal("workflow_recommendation", {
        "title": "T", "body": "B",
        "recommended_next_workflow": "architecture_readiness",
        "unresolved_critical_risks": ["data_loss"],
    })
    assert any("unresolved_critical_risks" in v for v in r.violations)
