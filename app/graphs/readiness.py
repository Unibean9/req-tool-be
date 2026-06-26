"""BMAD readiness rubric (addendum §15) — 10 dimensions over the 7-section coverage.

Distinct from rubric.py (artifact *quality*): this assesses readiness to advance the planning
lifecycle from document-registry item coverage.

Scope Stability reads the `out_of_scope` sub-dimension of scope_capabilities: when
section_coverage carries a granular dict for that section and out_of_scope is absent, the
dimension scores 0 and "scope_stability" is flagged.
"""

from typing import Any

from app.documents.registry import children_of, status_score

# dimension key -> (description, source sections). Implementation Readiness is special (min of all).
READINESS_DIMENSIONS: dict[str, dict[str, Any]] = {
    "goal_clarity": {"description": "Mục tiêu rõ ràng.", "sections": ["vision_objectives"]},
    "problem_depth": {"description": "Vấn đề được đào đủ sâu.", "sections": ["problem_statement"]},
    "stakeholder_coverage": {"description": "Bên liên quan đầy đủ.", "sections": ["stakeholder_register"]},
    "scope_stability": {"description": "Ranh giới scope ổn định.", "sections": ["scope_capabilities"]},
    "capability_clarity": {"description": "Năng lực rõ ràng.", "sections": ["scope_capabilities"]},
    "business_rule_testability": {"description": "Quy tắc nghiệp vụ kiểm chứng được.", "sections": ["business_rules"]},
    "constraint_visibility": {"description": "Ràng buộc hiển thị.", "sections": ["constraints_assumptions"]},
    "risk_exposure": {"description": "Rủi ro được nhận diện.", "sections": ["risks_issues"]},
    "architecture_readiness": {
        "description": "Sẵn sàng cho kiến trúc.",
        "sections": ["constraints_assumptions", "risks_issues"],
    },
    "implementation_readiness": {"description": "Sẵn sàng triển khai.", "sections": ["__all__"]},
}

_ALL_SECTIONS = list(children_of("brd"))

_READY_THRESHOLD = 0.7
_DIMENSION_PASS = 0.5


def _value_score(value: Any) -> float:
    """Score a section value that may be a flat status or a granular sub-dimension dict."""
    if isinstance(value, dict):
        return sum(status_score(v) for v in value.values()) / len(value) if value else 0.0
    return status_score(value)


def _scope_stability_score(scope_value: Any) -> float:
    """Scope is stable only when the out_of_scope boundary is drawn (granular) or the section is filled."""
    if isinstance(scope_value, dict):
        return status_score(scope_value.get("out_of_scope"))
    return _value_score(scope_value)


def compute_readiness_score(section_coverage: dict[str, Any] | None, state: Any = None) -> dict[str, Any]:  # noqa: ARG001
    """Score the 10 readiness dimensions and return the readiness report (addendum §10.2)."""
    cov = section_coverage or {}
    section_scores = {section: _value_score(cov.get(section)) for section in _ALL_SECTIONS}

    dimension_scores: dict[str, float] = {}
    for key, spec in READINESS_DIMENSIONS.items():
        if key == "scope_stability":
            dimension_scores[key] = _scope_stability_score(cov.get("scope_capabilities"))
        elif spec["sections"] == ["__all__"]:
            # Implementation readiness is strict: the weakest section bounds it.
            dimension_scores[key] = min(section_scores.values())
        else:
            scores = [section_scores[s] for s in spec["sections"]]
            dimension_scores[key] = sum(scores) / len(scores)

    readiness_score = sum(dimension_scores.values()) / len(dimension_scores)
    blocking_gaps = [key for key, score in dimension_scores.items() if score < _DIMENSION_PASS]
    warnings = [key for key, score in dimension_scores.items() if _DIMENSION_PASS <= score < _READY_THRESHOLD]
    ready = readiness_score >= _READY_THRESHOLD and not blocking_gaps

    if ready:
        recommended_next_step = "architecture_readiness"
    elif "constraint_visibility" in blocking_gaps:
        recommended_next_step = "complete_constraints"
    else:
        recommended_next_step = "address_blocking_gaps"

    return {
        "ready": ready,
        "readiness_score": readiness_score,
        "dimension_scores": dimension_scores,
        "blocking_gaps": blocking_gaps,
        "warnings": warnings,
        "recommended_next_step": recommended_next_step,
    }
