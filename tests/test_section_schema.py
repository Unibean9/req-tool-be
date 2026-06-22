"""Tests for the 7-section requirements taxonomy (spec §3, §7.3–7.4)."""

from app.graphs.section_schema import (
    SECTION_SPECS,
    SECTION_STATUSES,
    compute_section_coverage,
)

EXPECTED_SECTIONS = {
    "vision_objectives",
    "problem_statement",
    "stakeholder_register",
    "scope_capabilities",
    "business_rules",
    "constraints_assumptions",
    "risks_issues",
}


def test_section_keys_are_exactly_7():
    assert set(SECTION_SPECS) == EXPECTED_SECTIONS


def test_each_section_has_sub_dimensions():
    for section, spec in SECTION_SPECS.items():
        assert len(spec["sub_dimensions"]) >= 2, section


def test_business_rules_section_has_condition_and_outcome():
    subs = SECTION_SPECS["business_rules"]["sub_dimensions"]
    assert "condition" in subs
    assert "outcome" in subs


def test_scope_capabilities_has_out_of_scope_sub_dimension():
    subs = SECTION_SPECS["scope_capabilities"]["sub_dimensions"]
    assert "in_scope" in subs
    assert "out_of_scope" in subs


def test_compute_coverage_returns_7_section_keys():
    result = compute_section_coverage({})
    assert set(result["section_coverage"]) == EXPECTED_SECTIONS


def test_compute_coverage_missing_all_returns_zero():
    result = compute_section_coverage({})
    assert result["coverage_ratio"] == 0.0
    assert result["coverage_complete"] is False


def test_compute_coverage_all_filled_returns_complete():
    assessment = {section: "filled" for section in EXPECTED_SECTIONS}
    result = compute_section_coverage(assessment)
    assert result["coverage_complete"] is True
    assert result["coverage_ratio"] == 1.0


def test_compute_coverage_partial_ratio():
    assessment = {
        "vision_objectives": "filled",
        "problem_statement": "filled",
        "stakeholder_register": "filled",
    }
    result = compute_section_coverage(assessment)
    assert result["coverage_ratio"] == 3 / 7
    assert result["coverage_complete"] is False


def test_section_status_enum():
    assert set(SECTION_STATUSES) == {"missing", "partial", "filled", "needs_review"}


def test_coverage_threshold_per_section():
    for section, spec in SECTION_SPECS.items():
        assert 0.0 < spec["threshold"] <= 1.0, section


def test_vision_objectives_has_goal_sub_dimensions():
    subs = SECTION_SPECS["vision_objectives"]["sub_dimensions"]
    for key in ("business_goal", "user_goal", "metric", "target", "timeframe"):
        assert key in subs, key


def test_vision_objectives_has_intent_sub_dimensions():
    subs = SECTION_SPECS["vision_objectives"]["sub_dimensions"]
    assert "intent" in subs
    assert "success_definition" in subs


def test_compute_section_coverage_scores_sub_dimension_presence():
    assessment = {
        "vision_objectives": {
            "business_goal": "filled",
            "user_goal": "partial",
            "metric": "missing",
            "target": "missing",
            "timeframe": "missing",
            "intent": "filled",
            "success_definition": "missing",
        }
    }
    result = compute_section_coverage(assessment)
    assert result["section_coverage"]["vision_objectives"] != "filled"
