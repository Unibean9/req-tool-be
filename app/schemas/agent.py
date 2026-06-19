import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.agent import (
    AgentMessageRole,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCallStatus,
)
from app.models.artifact import ArtifactType


class AgentSessionCreate(BaseModel):
    artifact_type: ArtifactType
    step_key: str | None = Field(default=None, max_length=100)
    workflow_area: str = Field(default="analysis", max_length=50)
    agent_role: str | None = Field(default=None, max_length=100)
    provider_config_id: uuid.UUID | None = None


class AgentSessionResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    artifact_type: str
    workflow_area: str
    step_key: str | None
    status: AgentSessionStatus
    interrupt_type: AgentSessionInterruptType | None
    missing_context: Any | None
    agent_role: str | None
    provider_config_id: uuid.UUID | None
    created_by_id: uuid.UUID | None
    created_at: datetime | None
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class AgentSessionCreateResponse(BaseModel):
    session_id: str
    missing_context: list[str]


class AgentMessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class AgentMessageResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: AgentMessageRole
    content: str
    created_at: datetime | None
    updated_at: datetime | None

    model_config = {"from_attributes": True}


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

    model_config = {"from_attributes": True}


class ToolCallEditRequest(BaseModel):
    note: str = Field(min_length=1, max_length=8000)
