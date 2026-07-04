import enum
import uuid
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint, text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.models.base import AuditMixin, Base


class ArtifactStatus(enum.StrEnum):
    DRAFT = "draft"
    NEEDS_CLARIFICATION = "needs_clarification"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class ArtifactPriority(enum.StrEnum):
    MUST = "must"
    SHOULD = "should"
    COULD = "could"
    WONT = "wont"


class ArtifactType(enum.StrEnum):
    BRD = "brd"
    PRD = "prd"
    SAD = "sad"
    EXECUTIVE_SUMMARY = "executive_summary"
    VISION_OBJECTIVES = "vision_objectives"
    PROBLEM_STATEMENT = "problem_statement"
    STAKEHOLDER_REGISTER = "stakeholder_register"
    SCOPE_CAPABILITIES = "scope_capabilities"
    BUSINESS_RULES = "business_rules"
    CONSTRAINTS_ASSUMPTIONS = "constraints_assumptions"
    RISKS_ISSUES = "risks_issues"
    DOMAIN_ENTITY = "domain_entity"
    FUNCTIONAL_REQUIREMENT = "functional_requirement"
    NON_FUNCTIONAL_REQUIREMENT = "non_functional_requirement"
    USE_CASE = "use_case"
    COMPONENT = "component"
    INTERFACE = "interface"
    TECH_DECISION = "tech_decision"
    TECH_STACK = "tech_stack"
    EVENT_STORMING = "event_storming"
    DOMAIN_EVENT = "domain_event"
    ACTOR_COMMAND = "actor_command"
    POLICY = "policy"
    AGGREGATE = "aggregate"
    # Legacy delivery values retained because PostgreSQL enum members cannot be dropped in place.
    EPIC = "epic"
    STORY = "story"
    ACCEPTANCE_CRITERIA = "acceptance_criteria"


class ChangeSource(enum.StrEnum):
    MANUAL = "manual"
    IMPORT = "import"
    AI_GENERATION = "ai_generation"
    TOOL_CALL = "tool_call"
    SYSTEM = "system"


class VersionStatus(enum.StrEnum):
    DRAFT = "draft"
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class SourceType(enum.StrEnum):
    MARKDOWN_UPLOAD = "markdown_upload"
    TEXT_PASTE = "text_paste"
    REPO_FILE = "repo_file"
    URL = "url"
    EXTERNAL_DOC = "external_doc"


class RelationType(enum.StrEnum):
    DERIVES_FROM = "derives_from"
    DECOMPOSES_TO = "decomposes_to"
    SATISFIES = "satisfies"
    SUPPORTS = "supports"
    DEPENDS_ON = "depends_on"
    BLOCKS = "blocks"
    CONFLICTS_WITH = "conflicts_with"
    CLARIFIES = "clarifies"
    CONSTRAINS = "constrains"
    INFORMS = "informs"
    VALIDATES = "validates"


class EvidenceSourceType(enum.StrEnum):
    CHAT = "chat"
    REPO_FILE = "repo_file"
    DOCUMENT = "document"
    URL = "url"
    USER_INPUT = "user_input"
    AI_OUTPUT = "ai_output"


class ReviewStatus(enum.StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class WorkflowRunStatus(enum.StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class WorkflowStepKey(enum.StrEnum):
    INTENT_VISION = "intent_vision"
    CAPABILITY_MAP = "capability_map"
    DOMAIN_MODEL = "domain_model"
    REQUIREMENTS_SPEC = "requirements_spec"
    REALIZATION_BACKLOG = "realization_backlog"


class WorkflowStepPhase(enum.StrEnum):
    BRD = "brd"
    PRD = "prd"
    DELIVERY = "delivery"


class WorkflowStepStatus(enum.StrEnum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    WAITING_REVIEW = "waiting_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


def enum_column(enum_class: type[enum.Enum], **kwargs):
    return mapped_column(
        SAEnum(
            enum_class,
            values_callable=lambda values: [item.value for item in values],
            validate_strings=True,
        ),
        **kwargs,
    )


def jsonb_column(*args, **kwargs):
    return mapped_column(*args, JSON().with_variant(postgresql.JSONB, "postgresql"), **kwargs)


class SourceDocument(AuditMixin, Base):
    __tablename__ = "source_documents"

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True
    )
    uploaded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[SourceType] = enum_column(SourceType, nullable=False)
    locator: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra_metadata: Mapped[Any] = jsonb_column("metadata", nullable=False, default=dict)


class WorkflowRun(AuditMixin, Base):
    __tablename__ = "workflow_runs"

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[WorkflowRunStatus] = enum_column(WorkflowRunStatus, nullable=False, default=WorkflowRunStatus.DRAFT)
    current_step_key: Mapped[WorkflowStepKey | None] = enum_column(WorkflowStepKey, nullable=True)
    extra_metadata: Mapped[Any] = jsonb_column("metadata", nullable=False, default=dict)
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True
    )

    steps: Mapped[list["WorkflowStep"]] = relationship(back_populates="workflow_run", cascade="all, delete-orphan")


class Artifact(AuditMixin, Base):
    __tablename__ = "artifacts"
    __table_args__ = (
        Index(
            "uq_artifacts_project_brd",
            "project_id",
            unique=True,
            postgresql_where=text("type = 'brd'"),
            sqlite_where=text("type = 'brd'"),
        ),
        Index(
            "uq_artifacts_project_prd",
            "project_id",
            unique=True,
            postgresql_where=text("type = 'prd'"),
            sqlite_where=text("type = 'prd'"),
        ),
        Index(
            "uq_artifacts_project_sad",
            "project_id",
            unique=True,
            postgresql_where=text("type = 'sad'"),
            sqlite_where=text("type = 'sad'"),
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("artifacts.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workflow_runs.id"), nullable=True, index=True
    )
    step_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workflow_steps.id"), nullable=True, index=True
    )
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("artifact_versions.id", use_alter=True, name="fk_artifact_current_version"),
        nullable=True,
    )
    type: Mapped[ArtifactType] = enum_column(ArtifactType, nullable=False, index=True)
    status: Mapped[ArtifactStatus] = enum_column(
        ArtifactStatus, nullable=False, default=ArtifactStatus.DRAFT, index=True
    )
    priority: Mapped[ArtifactPriority | None] = enum_column(ArtifactPriority, nullable=True, index=True)
    code: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    extra_metadata: Mapped[Any] = jsonb_column("metadata", nullable=False, default=dict)
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True
    )

    current_version: Mapped["ArtifactVersion | None"] = relationship(
        foreign_keys=[current_version_id],
        post_update=True,
    )
    versions: Mapped[list["ArtifactVersion"]] = relationship(
        back_populates="artifact",
        foreign_keys="ArtifactVersion.artifact_id",
        cascade="all, delete-orphan",
    )
    parent: Mapped["Artifact | None"] = relationship(
        back_populates="children",
        remote_side="Artifact.id",
        foreign_keys=[parent_id],
    )
    children: Mapped[list["Artifact"]] = relationship(
        back_populates="parent",
        foreign_keys=[parent_id],
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )


class ArtifactVersion(AuditMixin, Base):
    __tablename__ = "artifact_versions"
    __table_args__ = (UniqueConstraint("artifact_id", "version_number", name="uq_artifact_versions_artifact_version"),)

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.id"), nullable=False, index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    change_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[VersionStatus] = enum_column(VersionStatus, nullable=False, default=VersionStatus.DRAFT)
    change_source: Mapped[ChangeSource] = enum_column(ChangeSource, nullable=False)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifact_versions.id"), nullable=True, index=True
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True
    )
    source_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_documents.id"), nullable=True, index=True
    )
    agent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_runs.id"), nullable=True, index=True
    )
    tool_call_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_tool_calls.id", use_alter=True, name="fk_artifact_versions_tool_call"),
        nullable=True,
        index=True,
    )
    provider_config_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("llm_provider_configs.id"), nullable=True, index=True
    )
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    extra_metadata: Mapped[Any] = jsonb_column("metadata", nullable=False, default=dict)

    artifact: Mapped[Artifact] = relationship(back_populates="versions", foreign_keys=[artifact_id])


class ArtifactLink(AuditMixin, Base):
    __tablename__ = "artifact_links"
    __table_args__ = (
        UniqueConstraint(
            "source_artifact_id", "target_artifact_id", "relation_type", name="uq_artifact_links_pair_type"
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True
    )
    source_artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.id"), nullable=False, index=True
    )
    target_artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.id"), nullable=False, index=True
    )
    relation_type: Mapped[RelationType] = enum_column(RelationType, nullable=False, index=True)
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True
    )
    extra_metadata: Mapped[Any] = jsonb_column("metadata", nullable=False, default=dict)


class ArtifactEvidence(AuditMixin, Base):
    __tablename__ = "artifact_evidence"

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.id"), nullable=False, index=True
    )
    artifact_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifact_versions.id"), nullable=True, index=True
    )
    source_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_documents.id"), nullable=True, index=True
    )
    source_type: Mapped[EvidenceSourceType] = enum_column(EvidenceSourceType, nullable=False)
    locator: Mapped[str] = mapped_column(String(255), nullable=False)
    excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    extra_metadata: Mapped[Any] = jsonb_column("metadata", nullable=False, default=dict)


class ArtifactReview(AuditMixin, Base):
    __tablename__ = "artifact_reviews"

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.id"), nullable=False, index=True
    )
    artifact_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifact_versions.id"), nullable=True, index=True
    )
    reviewed_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True
    )
    review_status: Mapped[ReviewStatus] = enum_column(ReviewStatus, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)


class WorkflowStep(AuditMixin, Base):
    __tablename__ = "workflow_steps"
    __table_args__ = (UniqueConstraint("run_id", "step_key", name="uq_workflow_steps_run_key"),)

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workflow_runs.id"), nullable=False, index=True
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True
    )
    step_key: Mapped[WorkflowStepKey] = enum_column(WorkflowStepKey, nullable=False)
    phase: Mapped[WorkflowStepPhase] = enum_column(WorkflowStepPhase, nullable=False)
    status: Mapped[WorkflowStepStatus] = enum_column(
        WorkflowStepStatus, nullable=False, default=WorkflowStepStatus.PENDING
    )
    input_snapshot: Mapped[Any] = jsonb_column(nullable=True)
    output_snapshot: Mapped[Any] = jsonb_column(nullable=True)
    approved_at: Mapped[Any] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True
    )
    extra_metadata: Mapped[Any] = jsonb_column("metadata", nullable=False, default=dict)

    workflow_run: Mapped[WorkflowRun] = relationship(back_populates="steps")
