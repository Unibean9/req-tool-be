import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.guards import require_project_access
from app.core.responses import created, ok
from app.database import get_db
from app.deps import current_user
from app.models.user import User
from app.schemas.artifact import ArtifactGraphResponse, ArtifactLinkCreateRequest, ArtifactLinkResponse
from app.schemas.response import ApiResponse
from app.services.artifact_service import ArtifactLinkService

router = APIRouter(prefix="/projects/{project_id}", tags=["Artifact Links"])


@router.get("/artifact-graph", response_model=ApiResponse[ArtifactGraphResponse])
async def get_artifact_graph(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    return ok(await ArtifactLinkService(db).graph(project_id=project_id, user_id=user.id))


@router.post("/artifact-links", response_model=ApiResponse[ArtifactLinkResponse], status_code=status.HTTP_201_CREATED)
async def create_artifact_link(
    project_id: uuid.UUID,
    body: ArtifactLinkCreateRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        link = await ArtifactLinkService(db).create(project_id=project_id, body=body, created_by_id=user.id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return created(link)


@router.delete("/artifact-links/{link_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_artifact_link(
    project_id: uuid.UUID,
    link_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        await ArtifactLinkService(db).delete(project_id=project_id, link_id=link_id, user_id=user.id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
