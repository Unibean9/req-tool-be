import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
        await db.execute(select(ArtifactLink).join(Artifact, ArtifactLink.source_id == Artifact.id).where(Artifact.project_id == project_id))
    ).scalars().all()
    return [
        {"source_id": str(r.source_id), "target_id": str(r.target_id), "relation_type": r.relation_type}
        for r in rows
    ]


@governed
async def create_artifact(
    *,
    artifact_type: str,
    title: str,
    body: str,
    rationale: str = "",
) -> dict:  # pragma: no cover — always intercepted by governed
    return {}


@governed
async def update_artifact(
    *,
    artifact_id: str,
    title: str | None = None,
    body: str | None = None,
) -> dict:  # pragma: no cover
    return {}


@governed
async def create_artifact_link(
    *,
    source_id: str,
    target_id: str,
    relation_type: str,
) -> dict:  # pragma: no cover
    return {}
