import inspect

from tests.eval import rubric

# 6 tiêu chí ISO/IEC/IEEE 29148 + 2 tiêu chí story/goal
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
        assert spec["name"], f"{key} thiếu name"
        assert spec["description"], f"{key} thiếu description"


def test_rubric_is_pure_no_llm_dependency():
    source = inspect.getsource(rubric)
    # Rubric phải thuần Python: không import client LLM, không gọi generate()
    assert "llm_clients" not in source
    assert ".generate(" not in source
