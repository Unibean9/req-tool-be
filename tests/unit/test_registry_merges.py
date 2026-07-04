"""Registry-merge invariants.

Locks the FR-absorbs-AC and constraints_assumptions-absorbs-risks_issues merges,
the completeness-sweep exclusion of the retired types, and the recommendation
score denominator against silent drift when BRD children change count.
"""

from app.documents.registry import all_item_types, children_of, output_contract
from app.graphs.agent_tools import _BRIEF_SECTIONS, _compute_recommendation
from app.graphs.decision_graph import completeness_sweep
from app.schemas.artifact_synthesis import evaluate_candidate_readiness


def test_retired_types_are_not_registry_items():
    items = set(all_item_types())
    for retired in ("executive_summary", "risks_issues", "acceptance_criteria"):
        assert retired not in items
        assert retired not in children_of("brd")
        assert retired not in children_of("prd")


def test_functional_requirement_absorbs_acceptance_criteria():
    contract = output_contract("functional_requirement")
    blob = " ".join((contract.guidance, *contract.elicit_checklist, *contract.review_criteria))
    assert "BC id" in blob
    assert "UC id" not in blob
    assert "Given/When/Then" in blob
    # Acceptance signal column carries the merged acceptance condition.
    assert "acceptance signal" in contract.table_columns
    # AC's negative/measurable checklist and failure-case review criterion carried over.
    assert any("negative" in item.lower() or "failure" in item.lower() for item in contract.elicit_checklist)
    assert any("measurable" in item.lower() for item in contract.elicit_checklist)
    assert any("failure" in c.lower() or "negative" in c.lower() for c in contract.review_criteria)


def test_constraints_assumptions_absorbs_risks_issues():
    contract = output_contract("constraints_assumptions")
    assert "## Risks" in contract.required_headings
    assert "## Mitigation Plan" in contract.required_headings
    assert contract.elicit_technique == "pre_mortem"
    assert any("risk" in item.lower() for item in contract.elicit_checklist)
    assert any("mitigation" in c.lower() for c in contract.review_criteria)


def test_completeness_sweep_omits_retired_types():
    """No gap questions reference the retired types once they leave `children`."""
    joined = " ".join(completeness_sweep({}, "prd") + completeness_sweep({}, "brd"))
    for retired_label in ("Acceptance Criteria", "Risks and Issues", "Executive Summary"):
        assert retired_label not in joined


def test_recommendation_denominator_tracks_trimmed_brd_children():
    """Score denominator == len(children_of('brd')) (now 6), so a full brief (4/6)
    stays below the readiness threshold instead of inflating past it."""
    assert len(children_of("brd")) == 6

    brief_only = {s: "filled" for s in _BRIEF_SECTIONS}
    assert _compute_recommendation(brief_only, "standard")["recommended_next_workflow"] == "prd"

    all_six = {s: "filled" for s in children_of("brd")}
    assert _compute_recommendation(all_six, "standard")["recommended_next_workflow"] == "readiness_check"


def test_candidate_readiness_survives_retired_type_without_contract():
    """An in-flight draft of a retired type (enum kept, registry contract gone) must not
    raise — evaluate_candidate_readiness degrades to no required headings."""
    import uuid

    for retired in ("risks_issues", "acceptance_criteria", "executive_summary"):
        metadata = {"artifact_type": retired, "focused_artifact_id": str(uuid.uuid4())}
        readiness = evaluate_candidate_readiness(
            artifact_type=retired, body="## Anything\nsome content", synthesis_metadata=metadata
        )
        # No contract → no missing-heading block; the call completes instead of raising ValueError.
        assert readiness.missing == []
