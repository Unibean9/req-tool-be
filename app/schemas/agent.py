import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.graphs.section_schema import SECTION_SPECS
from app.models.agent import (
    AgentMessageRole,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCallStatus,
)
from app.schemas.workspace import WorkspaceContainerResponse


class AgentSessionCreate(BaseModel):
    artifact_type: str = Field(max_length=100)
    step_key: str | None = Field(default=None, max_length=100)
    workflow_area: str = Field(default="analysis", max_length=50)
    agent_role: str | None = Field(default=None, max_length=100)
    provider_config_id: uuid.UUID | None = None
    focus_section: str | None = Field(default=None, max_length=100)

    @field_validator("focus_section")
    @classmethod
    def validate_focus_section(cls, value: str | None) -> str | None:
        if value is not None and value not in SECTION_SPECS:
            raise ValueError("focus_section không hợp lệ")
        return value


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
    focus_section: str | None = None
    workspace_container: WorkspaceContainerResponse | None = None
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
    focus_section: str | None = None
    container_key: str | None = None
    active_item_key: str | None = None


class AgentMessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    # Optional one-shot steer: switch the agent's angle for this turn only.
    # Constrained to the active_mode enum so user input can never be injected into the prompt.
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
