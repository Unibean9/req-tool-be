import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from app.models.artifact import ArtifactPriority, ArtifactStatus, ArtifactType, ChangeSource
from app.schemas.artifact import ArtifactVersionResponse


class DocumentItemWriteRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    body: str = Field(min_length=1)
    status: ArtifactStatus = ArtifactStatus.DRAFT
    priority: ArtifactPriority | None = None
    code: str | None = Field(default=None, max_length=100)
    confidence: Decimal | None = Field(default=None, ge=0, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)
    change_source: ChangeSource = ChangeSource.MANUAL
    change_summary: str | None = None


class DocumentItemView(BaseModel):
    artifact_type: ArtifactType
    label: str
    description: str
    artifact_id: uuid.UUID | None = None
    parent_id: uuid.UUID | None = None
    status: ArtifactStatus | None = None
    priority: ArtifactPriority | None = None
    code: str | None = None
    title: str | None = None
    confidence: Decimal | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    current_version_id: uuid.UUID | None = None
    current_version: ArtifactVersionResponse | None = None
    versions: list[ArtifactVersionResponse] = Field(default_factory=list)
    created_at: datetime | None = None


class DocumentView(BaseModel):
    document_type: ArtifactType
    label: str
    description: str
    artifact_id: uuid.UUID | None = None
    project_id: uuid.UUID
    status: ArtifactStatus | None = None
    title: str | None = None
    current_version_id: uuid.UUID | None = None
    items: list[DocumentItemView] = Field(default_factory=list)


class DocumentTypeView(BaseModel):
    artifact_type: str
    label: str
    description: str
    children: list[str] = Field(default_factory=list)
    is_container: bool
    output_contract: dict[str, Any] | None = None


class DocumentTypesResponse(BaseModel):
    containers: list[DocumentTypeView]
    items: list[DocumentTypeView]
