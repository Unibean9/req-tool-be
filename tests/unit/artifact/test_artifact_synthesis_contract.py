import uuid

import pytest
from pydantic import ValidationError

from app.documents.registry import all_item_types, output_contract
from app.schemas.artifact_synthesis import (
    ArtifactCandidateReadiness,
    ArtifactReadinessState,
    ArtifactSynthesisMetadata,
    evaluate_candidate_readiness,
    strip_synthesis_assumptions,
)


def test_strip_removes_synthesis_block_by_structure():
    body = "## Problem\nSlow.\n\n## Assumptions\n### Needs Confirmation\n- Target ⚠️ needs confirmation"

    stripped = strip_synthesis_assumptions(body)

    assert stripped == "## Problem\nSlow."


def test_strip_cleans_leftover_marker_from_legacy_body():
    legacy = "## Problem\nSlow.\n\n<!-- synthesis-assumptions -->\n## Assumptions\n### Confirmed\n- Done"

    stripped = strip_synthesis_assumptions(legacy)

    assert stripped == "## Problem\nSlow."
    assert "<!-- synthesis-assumptions -->" not in stripped


def test_strip_preserves_real_assumptions_content_heading():
    body = (
        "## Constraints\n- Mobile only.\n\n## Assumptions\n- Students own a smartphone.\n\n## Validation Plan\n- Pilot."
    )
    assert strip_synthesis_assumptions(body) == body


def test_candidate_readiness_blocks_when_table_has_empty_cells():
    metadata = ArtifactSynthesisMetadata(
        artifact_type="functional_requirement",
        focused_artifact_id=uuid.uuid4(),
    )
    body = (
        "## Functional Requirements\n"
        "| id | requirement | behavior | inputs/outputs | acceptance signal | priority |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| FR-01 | Scan QR | _(cần bổ sung)_ | _(cần bổ sung)_ | within 30s | Must |"
    )

    readiness = evaluate_candidate_readiness(
        artifact_type="functional_requirement",
        body=body,
        synthesis_metadata=metadata,
    )

    assert readiness.state == ArtifactReadinessState.WELL_STRUCTURED_BUT_INCOMPLETE
    assert readiness.can_persist is False
    assert readiness.blocking_reasons


def test_all_document_items_have_markdown_output_contracts():
    for item_type in all_item_types():
        contract = output_contract(item_type)
        assert contract.format == "markdown"
        assert contract.required_headings
        assert contract.confirmation_note == "(agent-inferred, needs confirmation)"


def test_vision_objectives_contract_is_requirements_specific():
    contract = output_contract("vision_objectives")

    assert contract.required_headings == ("## Vision", "## Objectives", "## Success Metrics")
    assert "goal" in contract.table_columns
    assert "timeframe" in contract.table_columns


def test_synthesis_metadata_requires_stable_provenance_shape():
    metadata = ArtifactSynthesisMetadata(
        artifact_type="vision_objectives",
        focused_artifact_id=uuid.uuid4(),
        evidence_refs=["agent_run:run-1"],
        pending_assumptions=["Retention target needs confirmation"],
    )

    dumped = metadata.model_dump(mode="json")
    assert "contract_version" not in dumped
    assert dumped["synthesis_source"] == "bmad_synthesis"
    assert dumped["inference_level"] == "medium"


def test_synthesis_metadata_rejects_missing_required_fields():
    with pytest.raises(ValidationError):
        ArtifactSynthesisMetadata.model_validate({"artifact_type": "vision_objectives"})


def test_candidate_readiness_schema_exposes_persistence_contract():
    readiness = ArtifactCandidateReadiness(
        state=ArtifactReadinessState.NEEDS_CONFIRMATION,
        missing=["target"],
        needs_confirmation=["Retention target needs confirmation"],
        inferred=["Retention metric inferred by the agent"],
        blocking_reasons=["Remaining assumption needs user confirmation"],
    )

    dumped = readiness.model_dump(mode="json")

    assert dumped["state"] == "needs_confirmation"
    assert dumped["can_persist"] is False
    assert dumped["missing"] == ["target"]
    assert dumped["needs_confirmation"] == ["Retention target needs confirmation"]
    assert dumped["inferred"] == ["Retention metric inferred by the agent"]
    assert dumped["blocking_reasons"] == ["Remaining assumption needs user confirmation"]


def test_candidate_readiness_is_poorly_structured_when_required_heading_missing():
    metadata = ArtifactSynthesisMetadata(
        artifact_type="vision_objectives",
        focused_artifact_id=uuid.uuid4(),
    )

    readiness = evaluate_candidate_readiness(
        artifact_type="vision_objectives",
        body="## Vision\nTang retention.\n\n## Objectives\n- Cai thien activation.",
        synthesis_metadata=metadata,
    )

    assert readiness.state == ArtifactReadinessState.POORLY_STRUCTURED
    assert readiness.can_persist is False
    assert "## Success Metrics" in readiness.missing


def test_candidate_readiness_is_incomplete_when_pending_assumption_has_no_marker():
    metadata = ArtifactSynthesisMetadata(
        artifact_type="vision_objectives",
        focused_artifact_id=uuid.uuid4(),
        pending_assumptions=["Target retention 15%"],
    )
    body = "\n\n".join(
        [
            "## Vision\nTang retention.",
            "## Objectives\n- Cai thien activation.",
            "## Success Metrics\n- Retention target 15%.",
        ]
    )

    readiness = evaluate_candidate_readiness(
        artifact_type="vision_objectives",
        body=body,
        synthesis_metadata=metadata,
    )

    assert readiness.state == ArtifactReadinessState.WELL_STRUCTURED_BUT_INCOMPLETE
    assert readiness.can_persist is False
    assert readiness.needs_confirmation == []
    assert readiness.blocking_reasons


def test_candidate_readiness_needs_confirmation_when_pending_assumption_is_marked():
    metadata = ArtifactSynthesisMetadata(
        artifact_type="vision_objectives",
        focused_artifact_id=uuid.uuid4(),
        pending_assumptions=["Target retention 15%"],
    )
    body = "\n\n".join(
        [
            "## Vision\nTang retention.",
            "## Objectives\n- Cai thien activation.",
            "## Success Metrics\n- Retention target 15% (agent-inferred, needs confirmation).",
        ]
    )

    readiness = evaluate_candidate_readiness(
        artifact_type="vision_objectives",
        body=body,
        synthesis_metadata=metadata,
    )

    assert readiness.state == ArtifactReadinessState.NEEDS_CONFIRMATION
    assert readiness.can_persist is True
    assert readiness.needs_confirmation == ["Target retention 15%"]
    assert readiness.blocking_reasons == []


def test_candidate_readiness_needs_confirmation_accepts_locale_marker():
    metadata = ArtifactSynthesisMetadata(
        artifact_type="vision_objectives",
        focused_artifact_id=uuid.uuid4(),
        pending_assumptions=["Target retention 15%"],
    )
    body = "\n\n".join(
        [
            "## Vision\nTang retention.",
            "## Objectives\n- Cai thien activation.",
            "## Success Metrics\n- Retention target 15% ⚠️ cần xác nhận.",
        ]
    )

    readiness = evaluate_candidate_readiness(
        artifact_type="vision_objectives",
        body=body,
        synthesis_metadata=metadata,
    )

    assert readiness.state == ArtifactReadinessState.NEEDS_CONFIRMATION
    assert readiness.can_persist is True
    assert readiness.needs_confirmation == ["Target retention 15%"]


def test_candidate_readiness_is_sufficient_when_structure_and_assumptions_are_confirmed():
    metadata = ArtifactSynthesisMetadata(
        artifact_type="vision_objectives",
        focused_artifact_id=uuid.uuid4(),
        confirmed_assumptions=["Target retention 15%"],
    )
    body = "\n\n".join(
        [
            "## Vision\nTang retention.",
            "## Objectives\n- Cai thien activation.",
            "## Success Metrics\n- Retention target 15%.",
        ]
    )

    readiness = evaluate_candidate_readiness(
        artifact_type="vision_objectives",
        body=body,
        synthesis_metadata=metadata,
    )

    assert readiness.state == ArtifactReadinessState.SUFFICIENT
    assert readiness.can_persist is True
    assert readiness.blocking_reasons == []
