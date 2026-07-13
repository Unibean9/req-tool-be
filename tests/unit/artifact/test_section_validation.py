"""Section-scoped structural validation + validated coverage numerator."""

from app.documents.registry import INCOMPLETE_CELL_PLACEHOLDER
from app.graphs.analysis.section_validation import validate_section, validated_coverage

# --- validate_section -----------------------------------------------------


def test_weasel_word_in_a_section_is_a_warning():
    findings = validate_section("brd", "## Objectives", "Deliver a fast and flexible platform.")
    assert findings, "weasel words should surface at least one finding"
    assert all(f["severity"] == "warning" for f in findings)
    assert all(f["section"] == "## Objectives" for f in findings)


def test_business_rule_missing_outcome_is_a_violation():
    # A rules section carries the condition+outcome completeness rule. Condition present, outcome absent.
    findings = validate_section("brd", "## Business Rules", "When the cart is empty")
    assert any(f["severity"] == "violation" for f in findings)


def test_complete_business_rule_passes():
    findings = validate_section(
        "brd", "## Business Rules", "When the cart is empty then the checkout button must be disabled."
    )
    assert [f for f in findings if f["severity"] == "violation"] == []


def test_unfilled_required_cell_is_a_violation():
    findings = validate_section("prd", "## Functional Requirements", f"The user can log in {INCOMPLETE_CELL_PLACEHOLDER}")
    assert any(f["severity"] == "violation" for f in findings)


def test_clean_prose_section_has_no_findings():
    assert validate_section("brd", "## Stakeholders", "The product owner approves scope changes.") == []


# --- validated_coverage ---------------------------------------------------


def test_section_with_violation_finding_does_not_count_as_covered():
    coverage = {"## Business Rules": "filled", "## Objectives": "filled"}
    findings = {"## Business Rules": [{"section": "## Business Rules", "severity": "violation", "message": "x"}]}
    result = validated_coverage(coverage, findings)
    assert result["## Business Rules"] == "missing"
    assert result["## Objectives"] == "filled"


def test_fixing_the_section_restores_the_count():
    coverage = {"## Business Rules": "filled"}
    # Re-validated clean -> stored as [] -> no downgrade.
    assert validated_coverage(coverage, {"## Business Rules": []}) == {"## Business Rules": "filled"}


def test_warning_only_findings_do_not_downgrade_coverage():
    coverage = {"## Objectives": "filled"}
    findings = {"## Objectives": [{"section": "## Objectives", "severity": "warning", "message": "weasel"}]}
    assert validated_coverage(coverage, findings) == {"## Objectives": "filled"}


def test_unaligned_or_absent_findings_leave_coverage_untouched():
    coverage = {"vision_objectives": "filled", "stakeholders": "missing"}
    assert validated_coverage(coverage, None) == coverage
    assert validated_coverage(coverage, {"## Some Heading": [{"severity": "violation"}]}) == coverage
