from typing import Any, Literal

from pydantic import BaseModel, Field


class WorkspaceItemResponse(BaseModel):
    key: str
    title: str
    description: str | None = None
    order: int
    status: Literal["missing", "partial", "filled", "needs_review"]
    body: str | None = None
    assessment: dict[str, Any] | None = None
    artifact_id: str | None = None
    artifact_type: str | None = None
    version_number: int | None = None
    updated_at: str | None = None


class WorkspaceContainerResponse(BaseModel):
    key: str
    kind: Literal["document", "artifact_group"]
    status: Literal["active", "pending", "disabled"]
    phase: Literal["brd", "prd", "delivery"] | None = None
    step_key: str | None = None
    primary_artifact_type: str | None = None
    artifact_types: list[str] = Field(default_factory=list)
    artifact_id: str | None = None
    current_version_id: str | None = None
    version_number: int | None = None
    active_item_key: str | None = None
    coverage_ratio: float | None = None
    coverage_complete: bool | None = None
    items: list[WorkspaceItemResponse] = Field(default_factory=list)
