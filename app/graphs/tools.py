import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.graphs.policy import governed
from app.models.artifact import Artifact, ArtifactLink, ArtifactType, SourceDocument


def _is_known_artifact_type(artifact_type: str) -> bool:
    return artifact_type in {item.value for item in ArtifactType}


@governed
async def read_artifacts(
    *,
    db: AsyncSession,
    project_id: uuid.UUID,
    artifact_type: str | list[str] | None = None,
) -> list[dict]:
    query = select(Artifact).where(Artifact.project_id == project_id)
    if isinstance(artifact_type, list):
        # Batch the whole ancestor-type chain into one round trip instead of one query per type.
        known_types = [item for item in artifact_type if _is_known_artifact_type(item)]
        if not known_types:
            return []
        query = query.where(Artifact.type.in_(known_types))
    elif artifact_type:
        if not _is_known_artifact_type(artifact_type):
            return []
        query = query.where(Artifact.type == artifact_type)
    rows = (await db.execute(query)).scalars().all()
    return [
        {
            "id": str(row.id),
            "type": row.type.value if hasattr(row.type, "value") else str(row.type),
            "title": row.title,
            "status": row.status.value if hasattr(row.status, "value") else str(row.status),
            "current_version_id": str(row.current_version_id) if row.current_version_id else None,
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
        (
            await db.execute(
                select(ArtifactLink)
                .join(Artifact, ArtifactLink.source_id == Artifact.id)
                .where(Artifact.project_id == project_id)
            )
        )
        .scalars()
        .all()
    )
    return [
        {"source_id": str(r.source_id), "target_id": str(r.target_id), "relation_type": r.relation_type} for r in rows
    ]


async def read_current_body(
    *,
    db: AsyncSession,
    project_id: uuid.UUID,
    artifact_type: str | None = None,
    artifact_id: uuid.UUID | None = None,
) -> dict | None:
    """Return the current version body of an artifact (by id, or latest of a type), or None.

    Title-only read_artifacts is not enough for M7: analyze_node needs the draft
    content so it does not re-ask what is already recorded. Plain async (no @governed)
    because this is an internal context-load that analyze_node calls directly like a
    repository query, not an LLM-exposed tool.

    Two read modes share one query. By id (read_artifact tool): artifact_type is irrelevant,
    so it is not required. By type (analyze_node context-load): the type is the only filter and
    must be known.
    """
    if artifact_id is None and not _is_known_artifact_type(artifact_type or ""):
        return None
    query = (
        select(Artifact)
        .where(Artifact.project_id == project_id)
        .where(Artifact.current_version_id.is_not(None))
        .options(selectinload(Artifact.current_version))
    )
    if artifact_id is not None:
        query = query.where(Artifact.id == artifact_id)
    else:
        query = query.where(Artifact.type == artifact_type).order_by(Artifact.created_at.desc()).limit(1)
    row = (await db.execute(query)).scalars().first()
    if row is None or row.current_version is None:
        return None
    return {
        "artifact_id": str(row.id),
        "artifact_type": row.type.value if hasattr(row.type, "value") else str(row.type),
        "current_version_id": str(row.current_version.id),
        "title": row.title,
        "body": row.current_version.body,
    }


@governed
async def read_source_documents(
    *,
    db: AsyncSession,
    project_id: uuid.UUID,
    source_document_ids: list[uuid.UUID] | None = None,
    limit: int = 3,
    max_chars: int = 8000,
) -> list[dict]:
    """Return bounded source-document excerpts on demand.

    Source documents can be large, so analyzer context gets title/id discovery
    from the tool surface and content only when this helper is called.
    """
    safe_limit = max(1, min(int(limit or 1), 10))
    safe_max_chars = max(1, int(max_chars or 1))
    ids = list(source_document_ids or [])[:safe_limit]
    query = select(SourceDocument).where(SourceDocument.project_id == project_id)
    if ids:
        query = query.where(SourceDocument.id.in_(ids))
    else:
        query = query.order_by(SourceDocument.created_at.desc(), SourceDocument.id).limit(safe_limit)
    rows = (await db.execute(query)).scalars().all()
    if ids:
        order = {item: index for index, item in enumerate(ids)}
        rows = sorted(rows, key=lambda row: order.get(row.id, len(order)))
    documents: list[dict] = []
    for row in rows[:safe_limit]:
        content = row.content_text or ""
        excerpt = content[:safe_max_chars]
        documents.append(
            {
                "id": str(row.id),
                "title": row.title,
                "source_type": row.source_type.value if hasattr(row.source_type, "value") else str(row.source_type),
                "locator": row.locator,
                "excerpt": excerpt,
                "truncated": len(content) > safe_max_chars,
                "content_hash": row.content_hash,
                "size_bytes": row.size_bytes,
            }
        )
    return documents
