import uuid

import pytest
from pydantic import ValidationError

from app.documents.registry import all_item_types, output_contract
from app.schemas.artifact_synthesis import (
    ArtifactCandidateReadiness,
    ArtifactReadinessState,
    ArtifactSynthesisMetadata,
    evaluate_candidate_readiness,
)


def test_all_document_items_have_markdown_output_contracts():
    for item_type in all_item_types():
        contract = output_contract(item_type)
        assert contract.format == "markdown"
        assert contract.required_headings
        assert contract.confirmation_note == "(agent suy diễn, cần xác nhận)"


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
        pending_assumptions=["Target retention cần xác nhận"],
    )

    dumped = metadata.model_dump(mode="json")
    assert dumped["synthesis_source"] == "bmad_synthesis"
    assert dumped["inference_level"] == "medium"


def test_synthesis_metadata_rejects_missing_required_fields():
    with pytest.raises(ValidationError):
        ArtifactSynthesisMetadata.model_validate({"artifact_type": "vision_objectives"})


def test_candidate_readiness_schema_exposes_persistence_contract():
    readiness = ArtifactCandidateReadiness(
        state=ArtifactReadinessState.NEEDS_CONFIRMATION,
        missing=["target"],
        needs_confirmation=["Target retention cần xác nhận"],
        inferred=["Metric retention được agent suy luận"],
        blocking_reasons=["Còn assumption cần user xác nhận"],
    )

    dumped = readiness.model_dump(mode="json")

    assert dumped["state"] == "needs_confirmation"
    assert dumped["can_persist"] is False
    assert dumped["missing"] == ["target"]
    assert dumped["needs_confirmation"] == ["Target retention cần xác nhận"]
    assert dumped["inferred"] == ["Metric retention được agent suy luận"]
    assert dumped["blocking_reasons"] == ["Còn assumption cần user xác nhận"]


def test_candidate_readiness_is_poorly_structured_when_required_heading_missing():
    metadata = ArtifactSynthesisMetadata(
        artifact_type="vision_objectives",
        focused_artifact_id=uuid.uuid4(),
    )

    readiness = evaluate_candidate_readiness(
        artifact_type="vision_objectives",
        body="## Vision\nTăng retention.\n\n## Objectives\n- Cải thiện activation.",
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
            "## Vision\nTăng retention.",
            "## Objectives\n- Cải thiện activation.",
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
            "## Vision\nTăng retention.",
            "## Objectives\n- Cải thiện activation.",
            "## Success Metrics\n- Retention target 15% (agent suy diễn, cần xác nhận).",
        ]
    )

    readiness = evaluate_candidate_readiness(
        artifact_type="vision_objectives",
        body=body,
        synthesis_metadata=metadata,
    )

    assert readiness.state == ArtifactReadinessState.NEEDS_CONFIRMATION
    assert readiness.can_persist is False
    assert readiness.needs_confirmation == ["Target retention 15%"]


def test_candidate_readiness_is_sufficient_when_structure_and_assumptions_are_confirmed():
    metadata = ArtifactSynthesisMetadata(
        artifact_type="vision_objectives",
        focused_artifact_id=uuid.uuid4(),
        confirmed_assumptions=["Target retention 15%"],
    )
    body = "\n\n".join(
        [
            "## Vision\nTăng retention.",
            "## Objectives\n- Cải thiện activation.",
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
