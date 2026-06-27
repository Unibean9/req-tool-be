import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.models.agent import (
    AgentMessageRole,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCallStatus,
)
from app.schemas.document import DocumentView


class AgentSessionCreate(BaseModel):
    artifact_type: str = Field(max_length=100)
    step_key: str | None = Field(default=None, max_length=100)
    workflow_area: str = Field(default="analysis", max_length=50)
    agent_role: str | None = Field(default=None, max_length=100)
    provider_config_id: uuid.UUID | None = None
    focused_artifact_id: uuid.UUID | None = None


class AgentSessionResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    artifact_type: str
    workflow_area: str
    step_key: str | None
    status: AgentSessionStatus
    ui_status: str | None = None
    interrupt_type: AgentSessionInterruptType | None
    missing_context: Any | None
    focused_artifact_id: uuid.UUID | None = None
    document: DocumentView | None = None
    agent_role: str | None
    provider_config_id: uuid.UUID | None
    created_by_id: uuid.UUID | None
    created_at: datetime | None
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class AgentSessionCreateResponse(BaseModel):
    session_id: str
    missing_context: list[str]
    artifact_type: str | None = None
    focused_artifact_id: uuid.UUID | None = None
    document_type: str | None = None


class AgentMessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    # Optional one-shot steer: switch the agent's angle for this turn only.
    # Constrained to a fixed enum so user input can never be injected into the prompt.
    mode_hint: Literal["qa", "critique", "explore", "draft"] | None = None


class AgentMessageResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: AgentMessageRole
    content: str
    payload: Any | None = None
    created_at: datetime | None
    updated_at: datetime | None

    model_config = {"from_attributes": True}


def public_tool_call_input_snapshot(snapshot: Any) -> Any:
    if not isinstance(snapshot, dict):
        return snapshot
    sanitized = dict(snapshot)
    metadata = sanitized.get("synthesis_metadata")
    if isinstance(metadata, dict):
        public_metadata = dict(metadata)
        public_metadata.pop("contract_version", None)
        sanitized["synthesis_metadata"] = public_metadata
    return sanitized


class AgentToolCallResponse(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    tool_name: str
    input_snapshot: Any
    status: AgentToolCallStatus
    created_artifact_id: uuid.UUID | None
    created_version_id: uuid.UUID | None
    resolved_at: datetime | None
    created_at: datetime | None
    updated_at: datetime | None

    @field_validator("input_snapshot", mode="before")
    @classmethod
    def sanitize_input_snapshot(cls, value: Any) -> Any:
        return public_tool_call_input_snapshot(value)

    model_config = {"from_attributes": True}


class ToolCallEditRequest(BaseModel):
    note: str = Field(min_length=1, max_length=8000)
    base_version_id: uuid.UUID | None = None
