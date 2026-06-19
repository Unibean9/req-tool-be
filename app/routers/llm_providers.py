import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.guards import require_project_access, require_project_owner
from app.core.responses import created, ok
from app.database import get_db
from app.deps import current_user
from app.models.user import User
from app.schemas.llm_provider import LLMProviderConfigRead, LLMProviderHealthCheckResult, LLMProviderKeyRequest
from app.schemas.response import ApiResponse
from app.services.llm_provider_service import CooldownError, LLMProviderService, ProviderUnavailableError

router = APIRouter(prefix="/projects/{project_id}/llm-provider-configs", tags=["llm-providers"])


@router.post("", response_model=ApiResponse[LLMProviderConfigRead], status_code=status.HTTP_201_CREATED)
async def create_llm_provider_config(
    project_id: uuid.UUID,
    body: LLMProviderKeyRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    return created(await LLMProviderService(db).create(project_id=project_id, body=body))


@router.get("", response_model=ApiResponse[list[LLMProviderConfigRead]])
async def list_llm_provider_configs(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    return ok(await LLMProviderService(db).list(project_id=project_id))


@router.get("/{config_id}", response_model=ApiResponse[LLMProviderConfigRead])
async def get_llm_provider_config(
    project_id: uuid.UUID,
    config_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    return ok(await LLMProviderService(db).get(project_id=project_id, config_id=config_id))


@router.patch("/{config_id}", response_model=ApiResponse[LLMProviderConfigRead])
async def update_llm_provider_config(
    project_id: uuid.UUID,
    config_id: uuid.UUID,
    body: LLMProviderKeyRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_owner(project_id, user, db)
    return ok(await LLMProviderService(db).update(project_id=project_id, config_id=config_id, body=body))


@router.delete("/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_llm_provider_config(
    project_id: uuid.UUID,
    config_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_owner(project_id, user, db)
    await LLMProviderService(db).delete(project_id=project_id, config_id=config_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{config_id}/health-check", response_model=ApiResponse[LLMProviderHealthCheckResult])
async def health_check_llm_provider_config(
    project_id: uuid.UUID,
    config_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    await require_project_access(project_id, user, db)
    try:
        result = await LLMProviderService(db).health_check(project_id=project_id, config_id=config_id)
    except CooldownError as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc))
    except ProviderUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    return ok(result)
