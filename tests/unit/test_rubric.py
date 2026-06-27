import inspect

from tests.eval import rubric

# The 6 ISO/IEC/IEEE 29148 criteria + 2 story/goal criteria
_CORE_29148 = {"unambiguous", "verifiable", "complete", "consistent", "traceable", "feasible"}
_OPTIONAL = {"invest", "smart"}


def test_rubric_has_all_29148_criteria():
    for key in _CORE_29148:
        assert key in rubric.RUBRIC_CRITERIA


def test_rubric_has_invest_and_smart_criteria():
    for key in _OPTIONAL:
        assert key in rubric.RUBRIC_CRITERIA


def test_every_criterion_has_name_and_description():
    for key, spec in rubric.RUBRIC_CRITERIA.items():
        assert spec["name"], f"{key} missing name"
        assert spec["description"], f"{key} missing description"


def test_rubric_has_business_alignment():
    assert "business_alignment" in rubric.RUBRIC_CRITERIA


def test_rubric_has_risk_awareness():
    assert "risk_awareness" in rubric.RUBRIC_CRITERIA


def test_rubric_has_scope_control():
    assert "scope_control" in rubric.RUBRIC_CRITERIA


def test_rubric_criteria_count_is_11():
    assert len(rubric.RUBRIC_CRITERIA) == 11


def test_business_alignment_has_guidance():
    guidance = rubric.RUBRIC_CRITERIA["business_alignment"]["guidance"]
    assert isinstance(guidance, str) and guidance.strip()


def test_risk_awareness_references_risks_issues_section():
    assert "risks_issues" in rubric.RUBRIC_CRITERIA["risk_awareness"]["guidance"]


def test_rubric_is_pure_no_llm_dependency():
    source = inspect.getsource(rubric)
    # Rubric must be pure Python: no LLM client import, no generate() call
    assert "llm_clients" not in source
    assert ".generate(" not in source
