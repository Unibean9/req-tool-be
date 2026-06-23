import uuid

import pytest
from pydantic import ValidationError

from app.documents.registry import all_item_types, output_contract
from app.schemas.artifact_synthesis import ArtifactSynthesisMetadata


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
