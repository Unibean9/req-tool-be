import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.guards import require_project_access
from app.core.responses import created, ok
from app.database import get_db
from app.deps import current_user
from app.models.artifact import (
    ArtifactPriority,
    ArtifactStatus,
    ArtifactType,
    VersionStatus,
    WorkflowStepKey,
    WorkflowStepPhase,
)
from app.models.user import User
from app.schemas.artifact import (
    ArtifactCreateRequest,
    ArtifactEvidenceCreateRequest,
    ArtifactEvidenceResponse,
    ArtifactResponse,
    ArtifactReviewRequest,
    ArtifactReviewResponse,
    ArtifactUpdateRequest,
)
from app.schemas.response import ApiResponse
from app.services.artifact_service import ArtifactService, ArtifactVersionService

router = APIRouter(prefix="/projects/{project_id}/artifacts", tags=["Artifacts"])


@router.post("", response_model=ApiResponse[ArtifactResponse], status_code=status.HTTP_201_CREATED)
async def create_artifact(
    project_id: uuid.UUID,
    body: ArtifactCreateRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    return created(await ArtifactService(db).create(project_id=project_id, body=body, created_by_id=user.id))


@router.get("", response_model=ApiResponse[list[ArtifactResponse]])
async def list_artifacts(
    project_id: uuid.UUID,
    artifact_type: ArtifactType | None = Query(default=None, alias="type"),
    status_filter: ArtifactStatus | None = Query(default=None, alias="status"),
    step_key: WorkflowStepKey | None = None,
    phase: WorkflowStepPhase | None = None,
    priority: ArtifactPriority | None = None,
    current_version_status: VersionStatus | None = None,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    return ok(
        await ArtifactService(db).list(
            project_id=project_id,
            user_id=user.id,
            artifact_type=artifact_type,
            status=status_filter,
            step_key=step_key,
            phase=phase,
            priority=priority,
            current_version_status=current_version_status,
        )
    )


@router.patch("/{artifact_id}", response_model=ApiResponse[ArtifactResponse])
async def update_artifact(
    project_id: uuid.UUID,
    artifact_id: uuid.UUID,
    body: ArtifactUpdateRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        artifact = await ArtifactService(db).update(
            project_id=project_id,
            artifact_id=artifact_id,
            body=body,
            updated_by_id=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ok(artifact)


@router.delete("/{artifact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_artifact(
    project_id: uuid.UUID,
    artifact_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        await ArtifactService(db).delete(project_id=project_id, artifact_id=artifact_id, user_id=user.id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/{artifact_id}/evidence",
    response_model=ApiResponse[ArtifactEvidenceResponse],
    status_code=status.HTTP_201_CREATED,
)
async def add_artifact_evidence(
    project_id: uuid.UUID,
    artifact_id: uuid.UUID,
    body: ArtifactEvidenceCreateRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        evidence = await ArtifactService(db).add_evidence(
            project_id=project_id,
            artifact_id=artifact_id,
            body=body,
            created_by_id=user.id,
        )
    except PermissionError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return created(evidence)


@router.get("/{artifact_id}/evidence", response_model=ApiResponse[list[ArtifactEvidenceResponse]])
async def list_artifact_evidence(
    project_id: uuid.UUID,
    artifact_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        evidence = await ArtifactService(db).list_evidence(
            project_id=project_id, artifact_id=artifact_id, user_id=user.id
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ok(evidence)


@router.post(
    "/{artifact_id}/versions/{version_id}/review",
    response_model=ApiResponse[ArtifactReviewResponse],
    status_code=status.HTTP_201_CREATED,
)
async def review_artifact_version(
    project_id: uuid.UUID,
    artifact_id: uuid.UUID,
    version_id: uuid.UUID,
    body: ArtifactReviewRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        review = await ArtifactVersionService(db).review(
            project_id=project_id,
            artifact_id=artifact_id,
            version_id=version_id,
            body=body,
            reviewed_by_id=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return created(review)


@router.post("/{artifact_id}/versions/{version_id}/restore", response_model=ApiResponse[ArtifactResponse])
async def restore_artifact_version(
    project_id: uuid.UUID,
    artifact_id: uuid.UUID,
    version_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        artifact = await ArtifactVersionService(db).restore(
            project_id=project_id,
            artifact_id=artifact_id,
            version_id=version_id,
            user_id=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ok(artifact)
