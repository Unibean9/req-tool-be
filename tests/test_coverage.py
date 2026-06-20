import pytest


def test_all_required_filled_gives_complete():
    from app.graphs.slot_schema import BRD_SLOTS, compute_coverage

    assessment = {slot: "filled" for slot in BRD_SLOTS["problem"]["required"]}

    result = compute_coverage("problem", assessment)

    assert result["coverage_ratio"] == 1.0
    assert result["coverage_complete"] is True


def test_empty_assessment_gives_zero_ratio():
    from app.graphs.slot_schema import compute_coverage

    result = compute_coverage("problem", {})

    assert result["coverage_ratio"] == 0.0
    assert result["coverage_complete"] is False


def test_partial_counts_as_half():
    from app.graphs.slot_schema import compute_coverage

    result = compute_coverage(
        "problem",
        {
            "who": "filled",
            "obstacle": "filled",
            "root_cause": "partial",
            "frequency": "empty",
            "impact": "empty",
        },
    )

    assert result["coverage_ratio"] == 0.5
    assert result["coverage_complete"] is False


def test_optional_slots_ignored_in_ratio():
    from app.graphs.slot_schema import compute_coverage

    result = compute_coverage("problem", {"workaround": "filled"})

    assert result["slot_coverage"]["workaround"] == "filled"
    assert result["coverage_ratio"] == 0.0
    assert result["coverage_complete"] is False


def test_threshold_boundary_complete_true():
    from app.graphs.slot_schema import compute_coverage

    result = compute_coverage(
        "problem",
        {
            "who": "filled",
            "obstacle": "filled",
            "root_cause": "filled",
            "frequency": "filled",
            "impact": "empty",
        },
    )

    assert result["coverage_ratio"] == 0.8
    assert result["coverage_complete"] is True


def test_threshold_boundary_below_incomplete():
    from app.graphs.slot_schema import compute_coverage

    result = compute_coverage(
        "problem",
        {
            "who": "filled",
            "obstacle": "filled",
            "root_cause": "filled",
            "frequency": "partial",
            "impact": "empty",
        },
    )

    assert result["coverage_ratio"] == 0.7
    assert result["coverage_complete"] is False


def test_unknown_brd_key_returns_complete_true():
    from app.graphs.slot_schema import compute_coverage

    result = compute_coverage("functional_requirement", {"actor": "empty"})

    assert result["slot_coverage"] == {"actor": "empty"}
    assert result["coverage_ratio"] == 1.0
    assert result["coverage_complete"] is True


def test_missing_slot_key_in_assessment_treated_as_empty():
    from app.graphs.slot_schema import compute_coverage

    result = compute_coverage("problem", {"who": "filled"})

    assert result["coverage_ratio"] == 0.2
    assert result["coverage_complete"] is False


@pytest.mark.parametrize(
    "non_brd_key",
    ["functional_requirement", "epic", "story", "non_functional_requirement"],
)
def test_gate_fail_open_for_non_brd_keys(non_brd_key):
    from app.graphs.slot_schema import compute_coverage

    # Artifact types outside the BRD scope must never be gated by coverage.
    result = compute_coverage(non_brd_key, {})

    assert result["coverage_ratio"] == 1.0
    assert result["coverage_complete"] is True
