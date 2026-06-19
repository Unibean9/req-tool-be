import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.artifact import WorkflowRunStatus, WorkflowStepKey, WorkflowStepPhase, WorkflowStepStatus


class WorkflowRunCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowStepResponse(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    project_id: uuid.UUID
    step_key: WorkflowStepKey
    phase: WorkflowStepPhase
    status: WorkflowStepStatus
    input_snapshot: dict[str, Any] | None = None
    output_snapshot: dict[str, Any] | None = None
    approved_at: datetime | None = None
    approved_by_id: uuid.UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class WorkflowRunResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    status: WorkflowRunStatus
    current_step_key: WorkflowStepKey | None = None
    created_by_id: uuid.UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    steps: list[WorkflowStepResponse] = Field(default_factory=list)


class WorkflowProgressResponse(BaseModel):
    run_id: uuid.UUID
    status: WorkflowRunStatus
    current_step_key: WorkflowStepKey | None = None
    step_counts: dict[str, int]
    steps: list[WorkflowStepResponse]
