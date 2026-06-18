import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.guards import require_project_access
from app.core.responses import created
from app.database import get_db
from app.deps import current_user
from app.models.user import User
from app.schemas.artifact import SourceDocumentCreateRequest, SourceDocumentResponse
from app.schemas.response import ApiResponse
from app.services.artifact_service import SourceDocumentInUseError, SourceDocumentService

router = APIRouter(prefix="/projects/{project_id}/source-documents", tags=["Source Documents"])


@router.post("", response_model=ApiResponse[SourceDocumentResponse], status_code=status.HTTP_201_CREATED)
async def upload_source_document(
    project_id: uuid.UUID,
    body: SourceDocumentCreateRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    return created(await SourceDocumentService(db).upload(project_id=project_id, body=body, uploaded_by_id=user.id))


@router.delete("/{source_document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source_document(
    project_id: uuid.UUID,
    source_document_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        await SourceDocumentService(db).delete(
            project_id=project_id,
            source_document_id=source_document_id,
            user_id=user.id,
        )
    except SourceDocumentInUseError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail={
                "message": "Source document đang được dùng làm evidence",
                "artifact_ids": [str(artifact_id) for artifact_id in exc.artifact_ids],
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
