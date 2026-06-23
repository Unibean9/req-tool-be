from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import AuditMixin, Base


class User(AuditMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="user")  # "user" | "admin"

    # GitHub identity (linked via OAuth)
    github_id: Mapped[str | None] = mapped_column(String(100), unique=True, nullable=True, index=True)
    github_login: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Stored Fernet-encrypted; use app.core.crypto to read/write
    github_access_token: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Relationships
    org_memberships: Mapped[list["OrgMember"]] = relationship(back_populates="user")  # noqa: F821
