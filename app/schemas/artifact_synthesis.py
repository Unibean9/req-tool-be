import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

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
