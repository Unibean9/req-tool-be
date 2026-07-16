# Inherits all fixtures from tests/conftest.py (in-memory SQLite + HTTP client).
# Add integration-specific overrides here when needed.

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

# Single source of truth for the migration revision the Postgres-backed integration tests
# require. Bump this whenever a new migration lands so all `*_postgres.py` fixtures pick up
# the new expectation from one place instead of a grep-and-replace across every file.
EXPECTED_ALEMBIC_REVISION = "d2e5c8ecc7e0"


async def assert_postgres_schema_contract(
    connection: AsyncConnection,
    *,
    table_name: str | None = None,
    column_name: str | None = None,
) -> None:
    """Fail loudly if the connected database's schema does not match what CI produces.

    This does not call `Base.metadata.create_all()`: the schema contract must be produced
    only by `alembic upgrade head`, the same way CI provisions it. Checking the stamped
    revision here catches a local database that was stamped incorrectly instead of masking
    the mismatch behind ORM-created tables.

    Pass `table_name` alone to additionally assert that table exists (e.g. a table added by
    the same migration). Pass both `table_name` and `column_name` to assert a specific column
    exists on that table instead.
    """
    assert connection.dialect.name == "postgresql"
    revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
    assert revision == EXPECTED_ALEMBIC_REVISION

    if table_name and column_name:
        result = await connection.scalar(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = :table_name AND column_name = :column_name"
            ),
            {"table_name": table_name, "column_name": column_name},
        )
        assert result == 1
    elif table_name:
        result = await connection.scalar(
            text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_name = :table_name"
            ),
            {"table_name": table_name},
        )
        assert result == 1
