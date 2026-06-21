import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from app.models.artifact import (
    ArtifactPriority,
    ArtifactStatus,
    ArtifactType,
    ChangeSource,
    EvidenceSourceType,
    RelationType,
    ReviewStatus,
    SourceType,
    VersionStatus,
    WorkflowStepKey,
    WorkflowStepPhase,
)


class ArtifactCreateRequest(BaseModel):
    type: ArtifactType
    title: str = Field(min_length=1, max_length=255)
    body: str = Field(min_length=1)
    status: ArtifactStatus = ArtifactStatus.DRAFT
    priority: ArtifactPriority | None = None
    code: str | None = Field(default=None, max_length=100)
    confidence: Decimal | None = Field(default=None, ge=0, le=100)
    nfr_category: str | None = Field(default=None, max_length=100)
    stakeholder_role: str | None = Field(default=None, max_length=100)
    metadata: dict[str, Any] = Field(default_factory=dict)
    change_source: ChangeSource = ChangeSource.MANUAL
    change_summary: str | None = None
    source_document_id: uuid.UUID | None = None


class ArtifactUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    body: str | None = Field(default=None, min_length=1)
    status: ArtifactStatus | None = None
    priority: ArtifactPriority | None = None
    code: str | None = Field(default=None, max_length=100)
    confidence: Decimal | None = Field(default=None, ge=0, le=100)
    nfr_category: str | None = Field(default=None, max_length=100)
    stakeholder_role: str | None = Field(default=None, max_length=100)
    metadata: dict[str, Any] | None = None
    change_source: ChangeSource = ChangeSource.MANUAL
    change_summary: str | None = None
    source_document_id: uuid.UUID | None = None


class ArtifactVersionResponse(BaseModel):
    id: uuid.UUID
    artifact_id: uuid.UUID
    version_number: int
    title: str
    body: str
    status: VersionStatus
    parent_version_id: uuid.UUID | None = None
    change_source: ChangeSource
    change_summary: str | None = None
    review_status: ReviewStatus | None = None
    created_by_id: uuid.UUID | None = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    current_version_id: uuid.UUID | None = None
    type: ArtifactType
    status: ArtifactStatus
    priority: ArtifactPriority | None = None
    code: str | None = None
    title: str
    confidence: Decimal | None = None
    nfr_category: str | None = None
    stakeholder_role: str | None = None
    created_by_id: uuid.UUID | None = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    current_version: ArtifactVersionResponse | None = None


class ArtifactReviewRequest(BaseModel):
    review_status: ReviewStatus
    comment: str | None = None


class ArtifactReviewResponse(BaseModel):
    id: uuid.UUID
    artifact_id: uuid.UUID
    artifact_version_id: uuid.UUID | None = None
    reviewed_by_id: uuid.UUID | None = None
    review_status: ReviewStatus
    comment: str | None = None
    created_at: datetime


class SourceDocumentCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    source_type: SourceType
    locator: str | None = Field(default=None, max_length=512)
    content_text: str | None = None
    mime_type: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceDocumentResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    uploaded_by_id: uuid.UUID | None = None
    title: str
    source_type: SourceType
    locator: str | None = None
    content_text: str | None = None
    content_hash: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ArtifactLinkCreateRequest(BaseModel):
    source_artifact_id: uuid.UUID
    target_artifact_id: uuid.UUID
    relation_type: RelationType
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactLinkResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    source_artifact_id: uuid.UUID
    target_artifact_id: uuid.UUID
    relation_type: RelationType
    created_by_id: uuid.UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ArtifactEvidenceCreateRequest(BaseModel):
    artifact_version_id: uuid.UUID | None = None
    source_document_id: uuid.UUID | None = None
    source_type: EvidenceSourceType
    locator: str = Field(min_length=1, max_length=255)
    excerpt: str | None = None
    confidence: Decimal | None = Field(default=None, ge=0, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactEvidenceResponse(BaseModel):
    id: uuid.UUID
    artifact_id: uuid.UUID
    artifact_version_id: uuid.UUID | None = None
    source_document_id: uuid.UUID | None = None
    source_type: EvidenceSourceType
    locator: str
    excerpt: str | None = None
    confidence: Decimal | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class GraphWarningType(enum.StrEnum):
    ORPHAN_ARTIFACT = "orphan_artifact"
    MISSING_UPSTREAM_TRACE = "missing_upstream_trace"
    MISSING_DOWNSTREAM_REALIZATION = "missing_downstream_realization"
    CONFLICTING_ARTIFACTS = "conflicting_artifacts"
    NEEDS_CLARIFICATION = "needs_clarification"


class GraphWarning(BaseModel):
    type: GraphWarningType
    artifact_id: uuid.UUID


class ArtifactNode(BaseModel):
    id: uuid.UUID
    type: ArtifactType
    status: ArtifactStatus
    title: str
    current_version_id: uuid.UUID | None = None
    current_version: ArtifactVersionResponse | None = None


class ArtifactGraphResponse(BaseModel):
    nodes: list[ArtifactNode]
    links: list[ArtifactLinkResponse]
    warnings: list[GraphWarning]


class ArtifactListFilters(BaseModel):
    type: ArtifactType | None = None
    status: ArtifactStatus | None = None
    step_key: WorkflowStepKey | None = None
    phase: WorkflowStepPhase | None = None
    priority: ArtifactPriority | None = None
    current_version_status: VersionStatus | None = None
