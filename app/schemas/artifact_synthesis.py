import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from app.documents.registry import output_contract

InferenceLevel = Literal["low", "medium", "high"]
SynthesisSource = Literal["bmad_synthesis"]


class ArtifactSynthesisMetadata(BaseModel):
    contract_version: str = "2026-06-23"
    artifact_type: str = Field(min_length=1, max_length=100)
    focused_artifact_id: uuid.UUID
    base_version_id: uuid.UUID | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    inference_level: InferenceLevel = "medium"
    confirmed_assumptions: list[str] = Field(default_factory=list)
    pending_assumptions: list[str] = Field(default_factory=list)
    synthesis_source: SynthesisSource = "bmad_synthesis"


class ArtifactReadinessState(StrEnum):
    POORLY_STRUCTURED = "poorly_structured"
    WELL_STRUCTURED_BUT_INCOMPLETE = "well_structured_but_incomplete"
    NEEDS_CONFIRMATION = "needs_confirmation"
    SUFFICIENT = "sufficient"


class ArtifactCandidateReadiness(BaseModel):
    state: ArtifactReadinessState
    missing: list[str] = Field(default_factory=list)
    needs_confirmation: list[str] = Field(default_factory=list)
    inferred: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def can_persist(self) -> bool:
        return self.state == ArtifactReadinessState.SUFFICIENT


def synthesis_metadata_from_snapshot(snapshot: dict[str, Any]) -> ArtifactSynthesisMetadata:
    metadata = snapshot.get("synthesis_metadata")
    if not isinstance(metadata, dict):
        metadata = {
            "artifact_type": snapshot.get("artifact_type"),
            "focused_artifact_id": snapshot.get("focused_artifact_id"),
            "base_version_id": snapshot.get("base_version_id"),
            "evidence_refs": snapshot.get("evidence_refs") or [],
            "confirmed_assumptions": snapshot.get("confirmed_assumptions") or [],
            "pending_assumptions": snapshot.get("pending_assumptions") or [],
        }
    return ArtifactSynthesisMetadata.model_validate(metadata)


def synthesis_metadata_dict(snapshot: dict[str, Any]) -> dict[str, Any]:
    return synthesis_metadata_from_snapshot(snapshot).model_dump(mode="json")


def evaluate_candidate_readiness(
    *,
    artifact_type: str,
    body: str,
    synthesis_metadata: ArtifactSynthesisMetadata | dict[str, Any],
) -> ArtifactCandidateReadiness:
    metadata = (
        synthesis_metadata
        if isinstance(synthesis_metadata, ArtifactSynthesisMetadata)
        else ArtifactSynthesisMetadata.model_validate(synthesis_metadata)
    )
    contract = output_contract(artifact_type)
    missing_headings = [heading for heading in contract.required_headings if heading not in body]
    if missing_headings:
        return ArtifactCandidateReadiness(
            state=ArtifactReadinessState.POORLY_STRUCTURED,
            missing=missing_headings,
            blocking_reasons=[
                f"Candidate is missing required headings from the output contract: {', '.join(missing_headings)}"
            ],
        )

    inferred = _extract_marked_lines(body, ("agent-inferred", "inferred"))
    marked_confirmations = _extract_marked_lines(
        body, ("needs confirmation", "needs user confirmation", "needs_confirmation")
    )
    pending = list(metadata.pending_assumptions)
    if pending and not marked_confirmations:
        return ArtifactCandidateReadiness(
            state=ArtifactReadinessState.WELL_STRUCTURED_BUT_INCOMPLETE,
            needs_confirmation=[],
            inferred=inferred,
            blocking_reasons=[
                "Candidate has pending assumptions but the body does not mark content needing confirmation."
            ],
        )
    if pending:
        return ArtifactCandidateReadiness(
            state=ArtifactReadinessState.NEEDS_CONFIRMATION,
            needs_confirmation=pending,
            inferred=inferred,
            blocking_reasons=["Candidate still has assumptions requiring user confirmation before persistence."],
        )

    return ArtifactCandidateReadiness(
        state=ArtifactReadinessState.SUFFICIENT,
        inferred=inferred,
    )


def _extract_marked_lines(body: str, markers: tuple[str, ...]) -> list[str]:
    marked: list[str] = []
    for line in body.splitlines():
        normalized = line.lower()
        if any(marker in normalized for marker in markers):
            marked.append(line.strip())
    return marked
