import inspect

from tests.eval import rubric

# The 6 ISO/IEC/IEEE 29148 criteria + 2 story/goal criteria
_CORE_29148 = {"unambiguous", "verifiable", "complete", "consistent", "traceable", "feasible"}
_OPTIONAL = {"invest", "smart"}


def test_rubric_criteria_keys_are_exactly_the_expected_11():
    # Together, this replaces separate presence checks for each of the 6 ISO/IEC/IEEE 29148
    # criteria, the 2 story/goal criteria (invest/smart), business_alignment, risk_awareness and
    # scope_control, plus the count==11 check: set equality proves every expected key is present
    # AND that there are no unexpected extras.
    expected = _CORE_29148 | _OPTIONAL | {"business_alignment", "risk_awareness", "scope_control"}
    assert set(rubric.RUBRIC_CRITERIA.keys()) == expected


def test_every_criterion_has_name_and_description():
    for key, spec in rubric.RUBRIC_CRITERIA.items():
        assert spec["name"], f"{key} missing name"
        assert spec["description"], f"{key} missing description"


def test_business_alignment_has_guidance():
    guidance = rubric.RUBRIC_CRITERIA["business_alignment"]["guidance"]
    assert isinstance(guidance, str) and guidance.strip()


def test_risk_awareness_references_constraints_assumptions_section():
    # risks_issues merged into constraints_assumptions.
    assert "constraints_assumptions" in rubric.RUBRIC_CRITERIA["risk_awareness"]["guidance"]


def test_rubric_is_pure_no_llm_dependency():
    source = inspect.getsource(rubric)
    # Rubric must be pure Python: no LLM client import, no generate() call
    assert "llm_clients" not in source
    assert ".generate(" not in source
