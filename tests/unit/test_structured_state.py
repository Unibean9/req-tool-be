"""Tests for structured analytical objects in state (spec §5.4, §7.1)."""

from app.graphs.note_parser import extract_structured_objects
from app.graphs.state import (
    AssumptionObject,
    OpenQuestionObject,
    RiskObject,
    WorkflowState,
)


def test_assumption_object_shape():
    assert set(AssumptionObject.__annotations__) == {
        "statement", "source", "confidence", "impact", "owner", "status",
    }


def test_risk_object_shape():
    assert set(RiskObject.__annotations__) == {
        "statement", "likelihood", "impact", "mitigation", "owner", "status",
    }


def test_open_question_object_shape():
    assert set(OpenQuestionObject.__annotations__) == {
        "question", "domain", "decision_needed", "status",
    }


def test_state_has_assumptions_field():
    assert "assumptions" in WorkflowState.__annotations__


def test_state_has_risks_field():
    assert "risks" in WorkflowState.__annotations__


def test_state_has_open_questions_field():
    assert "open_questions" in WorkflowState.__annotations__


def test_extract_assumption_from_note_content():
    content = (
        "ASSUMPTION: MVP target small garages | source: inferred | confidence: medium "
        "| impact: high | owner: PM | status: unconfirmed"
    )
    result = extract_structured_objects(content)

    assert len(result["assumptions"]) == 1
    obj = result["assumptions"][0]
    assert obj["statement"] == "MVP target small garages"
    assert obj["source"] == "inferred"
    assert obj["confidence"] == "medium"
    assert obj["impact"] == "high"
    assert obj["owner"] == "PM"
    assert obj["status"] == "unconfirmed"


def test_extract_risk_from_note_content():
    content = (
        "RISK: vendor API may change | likelihood: medium | impact: high "
        "| mitigation: pin version | owner: tech lead | status: open"
    )
    result = extract_structured_objects(content)

    assert len(result["risks"]) == 1
    obj = result["risks"][0]
    assert obj["statement"] == "vendor API may change"
    assert obj["likelihood"] == "medium"
    assert obj["mitigation"] == "pin version"
    assert obj["status"] == "open"


def test_extract_open_question_from_note_content():
    content = "OPEN_QUESTION: which payment gateway? | domain: technical | decision_needed: gateway choice | status: open"
    result = extract_structured_objects(content)

    assert len(result["open_questions"]) == 1
    obj = result["open_questions"][0]
    assert obj["question"] == "which payment gateway?"
    assert obj["domain"] == "technical"
    assert obj["decision_needed"] == "gateway choice"
    assert obj["status"] == "open"


def test_note_without_prefix_returns_empty_lists():
    result = extract_structured_objects("Just a free-form thought about the project, no tags here.")

    assert result == {"assumptions": [], "risks": [], "open_questions": [], "key_facts": []}


def test_extract_handles_missing_optional_fields():
    """A tagged line with only a statement parses, with absent fields defaulting to empty."""
    result = extract_structured_objects("ASSUMPTION: users have smartphones")

    assert len(result["assumptions"]) == 1
    obj = result["assumptions"][0]
    assert obj["statement"] == "users have smartphones"
    assert obj["confidence"] == ""


def test_extract_multiple_objects_across_lines():
    content = (
        "ASSUMPTION: A | confidence: low\n"
        "RISK: B | likelihood: high\n"
        "OPEN_QUESTION: C? | status: open"
    )
    result = extract_structured_objects(content)

    assert len(result["assumptions"]) == 1
    assert len(result["risks"]) == 1
    assert len(result["open_questions"]) == 1
