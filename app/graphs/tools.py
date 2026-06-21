import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.graphs.policy import governed
from app.models.artifact import Artifact, ArtifactLink


@governed
async def read_artifacts(
    *,
    db: AsyncSession,
    project_id: uuid.UUID,
    artifact_type: str | None = None,
) -> list[dict]:
    query = select(Artifact).where(Artifact.project_id == project_id)
    if artifact_type:
        query = query.where(Artifact.type == artifact_type)
    rows = (await db.execute(query)).scalars().all()
    return [
        {
            "id": str(row.id),
            "type": row.type,
            "title": row.title,
            "status": row.status,
        }
        for row in rows
    ]


@governed
async def read_artifact_graph(
    *,
    db: AsyncSession,
    project_id: uuid.UUID,
) -> list[dict]:
    rows = (
        await db.execute(
            select(ArtifactLink)
            .join(Artifact, ArtifactLink.source_id == Artifact.id)
            .where(Artifact.project_id == project_id)
        )
    ).scalars().all()
    return [
        {"source_id": str(r.source_id), "target_id": str(r.target_id), "relation_type": r.relation_type}
        for r in rows
    ]


async def read_current_body(
    *, db: AsyncSession, project_id: uuid.UUID, artifact_type: str
) -> dict | None:
    """Return the current version body of an artifact of this type, or None.

    Title-only read_artifacts is not enough for M7: analyze_node needs the draft
    content so it does not re-ask what is already recorded. Plain async (no @governed)
    because this is an internal context-load that analyze_node calls directly like a
    repository query, not an LLM-exposed tool.
    """
    row = (
        await db.execute(
            select(Artifact)
            .where(Artifact.project_id == project_id, Artifact.type == artifact_type)
            .where(Artifact.current_version_id.is_not(None))
            .order_by(Artifact.created_at.desc())
            .options(selectinload(Artifact.current_version))
            .limit(1)
        )
    ).scalars().first()
    if row is None or row.current_version is None:
        return None
    return {"artifact_id": str(row.id), "title": row.title, "body": row.current_version.body}


# The parameters below define each tool's schema (introspected by the agent);
# the body is never executed because @governed intercepts the call. ARG001 is
# therefore a false positive on these signature-only stubs.
@governed
async def create_artifact(
    *,
    artifact_type: str,  # noqa: ARG001
    title: str,  # noqa: ARG001
    body: str,  # noqa: ARG001
    rationale: str = "",  # noqa: ARG001
) -> dict:  # pragma: no cover — always intercepted by governed
    return {}


@governed
async def update_artifact(
    *,
    artifact_id: str,  # noqa: ARG001
    title: str | None = None,  # noqa: ARG001
    body: str | None = None,  # noqa: ARG001
) -> dict:  # pragma: no cover
    return {}


@governed
async def create_artifact_link(
    *,
    source_id: str,  # noqa: ARG001
    target_id: str,  # noqa: ARG001
    relation_type: str,  # noqa: ARG001
) -> dict:  # pragma: no cover
    return {}
