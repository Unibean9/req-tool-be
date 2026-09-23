"""Persistence for the project-level use-case model shown by the FE."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.models.base import AuditMixin, Base


def jsonb_column(*args, **kwargs):
    return mapped_column(*args, JSON().with_variant(postgresql.JSONB, "postgresql"), **kwargs)


class UseCaseModelRecord(AuditMixin, Base):
    """One current, editable use-case model per project.

    The model is intentionally stored as one JSON aggregate.  Actors, use cases, relationships,
    and semantic diagrams are edited together, so a FE update cannot leave half of a diagram in a
    different revision from its table.  The generated core model remains separately validated
    before this aggregate is written.
    """

    __tablename__ = "use_case_models"
    __table_args__ = (UniqueConstraint("project_id", name="use_case_models_project_id_key"),)

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    model_data: Mapped[Any] = jsonb_column(nullable=False, default=dict)
    source_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    generated_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    last_generation_error: Mapped[str | None] = mapped_column(Text, nullable=True)
