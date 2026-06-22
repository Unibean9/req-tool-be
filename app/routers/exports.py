import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.guards import require_project_access
from app.database import get_db
from app.deps import current_user
from app.models.user import User
from app.services.export_service import ExportService

router = APIRouter(prefix="/projects/{project_id}/exports", tags=["Exports"])


@router.get("/brd.md", response_class=PlainTextResponse)
async def export_brd(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _render(project_id, "brd", user, db)


@router.get("/prd.md", response_class=PlainTextResponse)
async def export_prd(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _render(project_id, "prd", user, db)


@router.get("/product-brief.md", response_class=PlainTextResponse)
async def export_product_brief(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _render(project_id, "product-brief", user, db)


async def _render(
    project_id: uuid.UUID,
    export_type: str,
    user: User,
    db: AsyncSession,
) -> PlainTextResponse:
    await require_project_access(project_id, user, db)
    try:
        content = await ExportService(db).render(
            project_id=project_id, user_id=user.id, export_type=export_type
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PlainTextResponse(content, media_type="text/plain; charset=utf-8")
