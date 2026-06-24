import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.guards import require_project_access
from app.core.responses import created, ok
from app.database import get_db
from app.deps import current_user
from app.models.user import User
from app.schemas.document import (
    DocumentItemView,
    DocumentItemWriteRequest,
    DocumentTypesResponse,
    DocumentView,
)
from app.schemas.response import ApiResponse
from app.services.document_service import DocumentService

router = APIRouter(tags=["Documents"])


@router.get("/documents/types", response_model=ApiResponse[DocumentTypesResponse])
async def list_document_types(
    user: User = Depends(current_user),  # noqa: ARG001 - authentication gate
    db: AsyncSession = Depends(get_db),
):
    return ok(DocumentService(db).types())


@router.get(
    "/projects/{project_id}/documents/{document_type}",
    response_model=ApiResponse[DocumentView],
)
async def get_document(
    project_id: uuid.UUID,
    document_type: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        document = await DocumentService(db).get_document(
            project_id=project_id,
            document_type=document_type,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ok(document)


@router.post(
    "/projects/{project_id}/documents/{document_type}",
    response_model=ApiResponse[DocumentView],
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    project_id: uuid.UUID,
    document_type: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        document = await DocumentService(db).create_document(
            project_id=project_id,
            document_type=document_type,
            created_by_id=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return created(document)


@router.get(
    "/projects/{project_id}/documents/{document_type}/{item_type}",
    response_model=ApiResponse[DocumentItemView],
)
async def get_document_item(
    project_id: uuid.UUID,
    document_type: str,
    item_type: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        item = await DocumentService(db).get_item(
            project_id=project_id,
            document_type=document_type,
            item_type=item_type,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ok(item)


@router.post(
    "/projects/{project_id}/documents/{document_type}/{item_type}",
    response_model=ApiResponse[DocumentItemView],
    status_code=status.HTTP_201_CREATED,
)
async def upsert_document_item(
    project_id: uuid.UUID,
    document_type: str,
    item_type: str,
    body: DocumentItemWriteRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        item = await DocumentService(db).upsert_item(
            project_id=project_id,
            document_type=document_type,
            item_type=item_type,
            body=body,
            created_by_id=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return created(item)
