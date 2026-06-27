import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


@compiles(UUID, "sqlite")
def _compile_postgresql_uuid_for_sqlite(_type, _compiler, **_kwargs) -> str:
    # SQLite gives an unknown "UUID" declaration NUMERIC affinity. A rare UUID
    # containing digits only is then coerced to float and cannot round-trip.
    return "CHAR(32)"


class Base(DeclarativeBase):
    pass


class AuditMixin:
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
