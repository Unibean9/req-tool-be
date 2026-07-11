import re
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from app.documents.registry import INCOMPLETE_CELL_PLACEHOLDER, output_contract

InferenceLevel = Literal["low", "medium", "high"]
SynthesisSource = Literal["bmad_synthesis"]


class ArtifactSynthesisMetadata(BaseModel):
    artifact_type: str = Field(min_length=1, max_length=100)
    focused_artifact_id: uuid.UUID
    base_version_id: uuid.UUID | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    inference_level: InferenceLevel = "medium"
    confirmed_assumptions: list[str] = Field(default_factory=list)
    pending_assumptions: list[str] = Field(default_factory=list)
    # Non-blocking findings (e.g. deterministic proposal gate warnings, stale-predecessor notices) —
    # violations that would block the proposal never reach here. Read by the human reviewer at
    # approval time via the FE snapshot, and by the model at its next turn via the write_draft
    # ToolMessage content.
    deterministic_warnings: list[str] = Field(default_factory=list)
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
        if self.state == ArtifactReadinessState.SUFFICIENT:
            return True
        return (
            self.state == ArtifactReadinessState.NEEDS_CONFIRMATION
            and not self.missing
            and not self.blocking_reasons
        )


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


# Retained only to strip the marker/appendix out of bodies persisted before assumptions/facts were
# dropped as a separate synthesis appendix in favor of the real "constraints_assumptions" section.
SYNTHESIS_ASSUMPTIONS_MARKER = "<!-- synthesis-assumptions -->"


def strip_synthesis_assumptions(body: str) -> str:
    """Remove the synthesis assumptions appendix so an exported report omits internal tracking."""
    text = str(body or "").replace(SYNTHESIS_ASSUMPTIONS_MARKER + "\n", "").replace(SYNTHESIS_ASSUMPTIONS_MARKER, "")
    # The appendix is the trailing "## Assumptions" section opening with a "### Confirmed" /
    # "### Needs Confirmation" subsection — matched by structure so a real content "## Assumptions"
    # (followed by a table or prose) is left intact.
    synthesis = re.search(r"\n##\s+Assumptions\s*\n###\s+(?:Confirmed|Needs Confirmation)", text)
    if synthesis:
        return text[: synthesis.start()].rstrip()
    return text


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
    # Retired item types (e.g. risks_issues/acceptance_criteria/executive_summary) keep their enum
    # value for historical rows but no longer have a registry contract; degrade to no required
    # headings so an in-flight draft of such a type stays workable instead of raising.
    try:
        required_headings = output_contract(artifact_type).required_headings
    except ValueError:
        required_headings = ()
    missing_headings = [heading for heading in required_headings if heading not in body]
    if missing_headings:
        return ArtifactCandidateReadiness(
            state=ArtifactReadinessState.POORLY_STRUCTURED,
            missing=missing_headings,
            blocking_reasons=[
                f"Candidate is missing required headings from the output contract: {', '.join(missing_headings)}"
            ],
        )

    if INCOMPLETE_CELL_PLACEHOLDER in body:
        return ArtifactCandidateReadiness(
            state=ArtifactReadinessState.WELL_STRUCTURED_BUT_INCOMPLETE,
            blocking_reasons=[
                "Candidate has empty required table cells. Fill every column, or mark an unknown value "
                "as needing confirmation — do not leave it blank."
            ],
        )

    inferred = _extract_marked_lines(body, ("agent-inferred", "inferred"))
    marked_confirmations = _extract_marked_lines(
        body,
        (
            "needs confirmation",
            "needs user confirmation",
            "needs_confirmation",
            "cần xác nhận",
            "can xac nhan",
            "⚠️",
        ),
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
