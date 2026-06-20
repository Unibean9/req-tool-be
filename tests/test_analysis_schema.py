from app.graphs.nodes import ANALYSIS_SCHEMA


def test_analysis_schema_has_slot_assessment_optional():
    slot_assessment = ANALYSIS_SCHEMA["properties"]["slot_assessment"]

    assert slot_assessment["type"] == "object"
    assert slot_assessment["additionalProperties"]["enum"] == ["filled", "partial", "empty"]
    assert "slot_assessment" not in ANALYSIS_SCHEMA["required"]
