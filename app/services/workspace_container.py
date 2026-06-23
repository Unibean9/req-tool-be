import json
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from app.graphs.checkpointer import AgentSessionCheckpointer
from app.graphs.workspace import WorkspaceContainerDefinition, workspace_container_for_artifact_type
from app.models.artifact import Artifact, ArtifactType, ArtifactVersion
from app.schemas.workspace import WorkspaceContainerResponse, WorkspaceItemResponse


def parse_workspace_items(artifact_type: str | None, body: str | None) -> dict[str, str] | None:
    definition = workspace_container_for_artifact_type(artifact_type)
    if definition is None or definition.body_shape != "json_items":
        return None
    try:
        parsed = json.loads(body or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    allowed = {item.key for item in definition.item_definitions}
    return {key: value for key, value in parsed.items() if key in allowed and isinstance(value, str)}


async def checkpoint_values_for_session(
    *,
    session_id: uuid.UUID,
    session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
) -> dict[str, Any] | None:
    checkpointer = AgentSessionCheckpointer(session_id=str(session_id), session_factory=session_factory)
    try:
        checkpoint = await checkpointer.aget_tuple({"configurable": {"thread_id": str(session_id)}})
    except NoResultFound:
        return None
    if checkpoint is None:
        return None
    values = checkpoint.checkpoint.get("channel_values") or {}
    return values if isinstance(values, dict) else None


async def build_workspace_container(
    *,
    db: AsyncSession,
    project_id: uuid.UUID,
    artifact_type: str,
    active_item_key: str | None = None,
    state_values: dict[str, Any] | None = None,
) -> WorkspaceContainerResponse | None:
    definition = workspace_container_for_artifact_type(artifact_type)
    if definition is None:
        return None
    artifact, version = await _load_primary_artifact(db, project_id, definition)
    parsed_items = parse_workspace_items(artifact_type, version.body if version else None) or {}
    state_values = state_values or {}
    coverage = state_values.get("section_coverage") if isinstance(state_values.get("section_coverage"), dict) else {}
    assessment = (
        state_values.get("section_assessment") if isinstance(state_values.get("section_assessment"), dict) else {}
    )

    return WorkspaceContainerResponse(
        key=definition.key,
        kind=definition.kind,
        status=definition.status,
        phase=definition.phase,
        step_key=definition.step_key,
        primary_artifact_type=definition.primary_artifact_type,
        artifact_types=list(definition.artifact_types),
        artifact_id=str(artifact.id) if artifact else None,
        current_version_id=str(artifact.current_version_id) if artifact and artifact.current_version_id else None,
        version_number=version.version_number if version else None,
        active_item_key=active_item_key,
        coverage_ratio=state_values.get("coverage_ratio"),
        coverage_complete=state_values.get("coverage_complete"),
        items=[
            WorkspaceItemResponse(
                key=item.key,
                title=item.title,
                description=item.description,
                order=item.order,
                status=coverage.get(item.key, "missing"),
                body=parsed_items.get(item.key),
                assessment=assessment.get(item.key) if isinstance(assessment.get(item.key), dict) else None,
                artifact_id=str(artifact.id) if artifact else None,
                artifact_type=definition.primary_artifact_type,
                version_number=version.version_number if version else None,
                updated_at=version.updated_at.isoformat() if version and version.updated_at else None,
            )
            for item in definition.item_definitions
        ],
    )


async def _load_primary_artifact(
    db: AsyncSession,
    project_id: uuid.UUID,
    definition: WorkspaceContainerDefinition,
) -> tuple[Artifact | None, ArtifactVersion | None]:
    if definition.primary_artifact_type is None:
        return None, None
    artifact_type = ArtifactType(definition.primary_artifact_type)
    artifact = (
        await db.execute(
            select(Artifact)
            .where(Artifact.project_id == project_id, Artifact.type == artifact_type)
            .order_by(Artifact.created_at.desc(), Artifact.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if artifact is None or artifact.current_version_id is None:
        return artifact, None
    version = await db.get(ArtifactVersion, artifact.current_version_id)
    return artifact, version
