import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import AuditMixin, Base

if TYPE_CHECKING:
    from app.models.user import User


class Organization(AuditMixin, Base):
    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Relationships
    members: Mapped[list["OrgMember"]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    projects: Mapped[list["Project"]] = relationship(back_populates="organization", cascade="all, delete-orphan")  # noqa: F821


class OrgMember(AuditMixin, Base):
    __tablename__ = "org_members"
    __table_args__ = (UniqueConstraint("org_id", "user_id", name="uq_org_members_org_user"),)

    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="member")  # owner | member

    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="members")
    user: Mapped["User"] = relationship(back_populates="org_memberships")
