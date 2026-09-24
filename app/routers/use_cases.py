"""Use-case table, React Flow layout, and generation endpoints."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.guards import require_project_access
from app.core.responses import created, ok
from app.database import get_db
from app.deps import current_user
from app.models.user import User
from app.schemas.response import ApiResponse
from app.schemas.use_case import (
    ActorCreateRequest,
    ActorResponse,
    ActorUpdateRequest,
    RelationshipCreateRequest,
    UseCaseCreateRequest,
    UseCaseGenerateRequest,
    UseCaseModelResponse,
    UseCasePlantUmlResponse,
    UseCasePlantUmlUpdateRequest,
    UseCaseRelationshipResponse,
    UseCaseResponse,
    UseCaseUpdateRequest,
)
from app.services.use_case_service import UseCaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects/{project_id}", tags=["Use Cases"])


@router.get(
    "/use-case-model",
    response_model=ApiResponse[UseCaseModelResponse],
    response_model_by_alias=True,
)
async def get_use_case_model(
    project_id: uuid.UUID,
    include_actors: bool = Query(default=True, alias="includeActors"),
    include_relationships: bool = Query(default=True, alias="includeRelationships"),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_access(project_id, user, db)
    return ok(
        await UseCaseService(db).get_model(
            project_id=project_id,
            include_actors=include_actors,
            include_relationships=include_relationships,
        )
    )


@router.post(
    "/use-case-model/generate",
    response_model=ApiResponse[UseCaseModelResponse],
    response_model_by_alias=True,
)
async def generate_use_case_model(
    project_id: uuid.UUID,
    body: UseCaseGenerateRequest = Body(default_factory=UseCaseGenerateRequest),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_access(project_id, user, db)
    service = UseCaseService(db)
    try:
        return ok(await service.generate(project_id=project_id, user_id=user.id, body=body))
    except HTTPException:
        raise
    except Exception as exc:
        # The generation marker is committed before the long-running provider calls. If an
        # unexpected exception escapes after that point, release it so the next request can
        # retry instead of leaving the FE permanently disabled.
        await service.mark_running_generation_failed(
            project_id=project_id,
            message=f"Use-case generation failed: {str(exc)[:400] or 'unexpected server error'}",
        )
        raise


@router.post(
    "/use-case-model/relations/generate",
    response_model=ApiResponse[UseCaseModelResponse],
    response_model_by_alias=True,
)
async def generate_use_case_relations(
    project_id: uuid.UUID,
    body: UseCaseGenerateRequest = Body(default_factory=UseCaseGenerateRequest),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Resolve include/extend/generalization for an already-generated table (step 2 of 2)."""

    await require_project_access(project_id, user, db)
    service = UseCaseService(db)
    try:
        return ok(await service.generate_relations(project_id=project_id, user_id=user.id, body=body))
    except HTTPException:
        raise
    except Exception as exc:
        # Same rationale as /generate's handler below: release the running marker on an
        # exception that escaped generate_relations' own handling, so a bug here cannot leave
        # the Generate button (and a fresh generate_groups call) permanently locked out.
        await service.mark_running_generation_failed(
            project_id=project_id,
            message=f"Relationship generation failed: {str(exc)[:400] or 'unexpected server error'}",
        )
        raise


@router.post(
    "/use-case-model/diagram/generate",
    response_model=ApiResponse[UseCaseModelResponse],
    response_model_by_alias=True,
)
async def generate_use_case_diagram(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Recompute the ELK diagram layout from the already-generated table. No LLM call; requires
    at least one use case to already exist (409 otherwise)."""

    await require_project_access(project_id, user, db)
    return ok(await UseCaseService(db).generate_diagram(project_id=project_id))


@router.post(
    "/use-case-model/groups/generate",
    response_model=ApiResponse[UseCaseModelResponse],
    response_model_by_alias=True,
)
async def generate_use_case_groups(
    project_id: uuid.UUID,
    body: UseCaseGenerateRequest = Body(default_factory=UseCaseGenerateRequest),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Split generation, phase 1 of 2: extracts modules/actors only (``useCases`` stays empty).

    This is the primary generation entrypoint the frontend drives -- each phase is its own
    bounded request instead of one request blocking for the whole pipeline, which is what
    ``/generate`` (the monolithic endpoint) does and why it can exceed a client/proxy timeout
    on projects with several modules.
    """

    await require_project_access(project_id, user, db)
    service = UseCaseService(db)
    try:
        return ok(await service.generate_groups(project_id=project_id, user_id=user.id, body=body))
    except HTTPException:
        raise
    except Exception as exc:
        await service.mark_running_generation_failed(
            project_id=project_id,
            message=f"Module extraction failed: {str(exc)[:400] or 'unexpected server error'}",
        )
        raise


@router.post(
    "/use-case-model/groups/{group_id}/use-cases/generate",
    response_model=ApiResponse[UseCaseModelResponse],
    response_model_by_alias=True,
)
async def generate_use_case_group_use_cases(
    project_id: uuid.UUID,
    group_id: str,
    body: UseCaseGenerateRequest = Body(default_factory=UseCaseGenerateRequest),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Split generation, phase 2 of 2: generates one module's use cases (call once per module
    returned by ``/groups/generate``). Idempotently replaces that module's previously generated
    rows, so retrying a single failed/timed-out module is safe.
    """

    await require_project_access(project_id, user, db)
    try:
        return ok(
            await UseCaseService(db).generate_group_use_cases(
                project_id=project_id, user_id=user.id, group_id=group_id, body=body
            )
        )
    except HTTPException:
        raise
    except Exception:
        # Deliberately does not touch the run's running marker (unlike /generate,
        # /groups/generate and /relations/generate): one module failing must not block the
        # caller's remaining queued-up modules or the closing relations call. Logged here so
        # a failure that also escaped generate_group_use_cases' own handling is still visible.
        logger.exception("Module use-case generation request failed for %s/%s", project_id, group_id)
        raise


@router.patch(
    "/use-case-model/uml",
    response_model=ApiResponse[UseCasePlantUmlResponse],
    response_model_by_alias=True,
)
async def update_use_case_uml(
    project_id: uuid.UUID,
    body: UseCasePlantUmlUpdateRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Compatibility route for old clients; the current Use Case screen does not use PlantUML."""

    await require_project_access(project_id, user, db)
    return ok(await UseCaseService(db).update_plant_uml(project_id=project_id, body=body))


@router.get(
    "/use-cases/{use_case_id}",
    response_model=ApiResponse[UseCaseResponse],
    response_model_by_alias=True,
)
async def get_use_case(
    project_id: uuid.UUID,
    use_case_id: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_access(project_id, user, db)
    return ok(await UseCaseService(db).get_use_case(project_id=project_id, use_case_id=use_case_id))


@router.post(
    "/actors",
    response_model=ApiResponse[ActorResponse],
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
async def create_actor(
    project_id: uuid.UUID,
    body: ActorCreateRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_access(project_id, user, db)
    return created(await UseCaseService(db).create_actor(project_id=project_id, body=body))


@router.patch(
    "/actors/{actor_id}",
    response_model=ApiResponse[ActorResponse],
    response_model_by_alias=True,
)
async def update_actor(
    project_id: uuid.UUID,
    actor_id: str,
    body: ActorUpdateRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_access(project_id, user, db)
    return ok(await UseCaseService(db).update_actor(project_id=project_id, actor_id=actor_id, body=body))


@router.delete("/actors/{actor_id}", response_model=ApiResponse[dict[str, Any]])
async def delete_actor(
    project_id: uuid.UUID,
    actor_id: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_access(project_id, user, db)
    return ok(await UseCaseService(db).delete_actor(project_id=project_id, actor_id=actor_id))


@router.post(
    "/use-cases",
    response_model=ApiResponse[UseCaseResponse],
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
async def create_use_case(
    project_id: uuid.UUID,
    body: UseCaseCreateRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_access(project_id, user, db)
    return created(await UseCaseService(db).create_use_case(project_id=project_id, body=body))


@router.patch(
    "/use-cases/{use_case_id}",
    response_model=ApiResponse[UseCaseResponse],
    response_model_by_alias=True,
)
async def update_use_case(
    project_id: uuid.UUID,
    use_case_id: str,
    body: UseCaseUpdateRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_access(project_id, user, db)
    return ok(
        await UseCaseService(db).update_use_case(
            project_id=project_id,
            use_case_id=use_case_id,
            body=body,
        )
    )


@router.delete("/use-cases/{use_case_id}", response_model=ApiResponse[dict[str, Any]])
async def delete_use_case(
    project_id: uuid.UUID,
    use_case_id: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_access(project_id, user, db)
    return ok(await UseCaseService(db).delete_use_case(project_id=project_id, use_case_id=use_case_id))


@router.post(
    "/use-case-relationships",
    response_model=ApiResponse[UseCaseRelationshipResponse],
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
async def create_relationship(
    project_id: uuid.UUID,
    body: RelationshipCreateRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_access(project_id, user, db)
    return created(await UseCaseService(db).create_relationship(project_id=project_id, body=body))


@router.delete(
    "/use-case-relationships/{relationship_id}",
    response_model=ApiResponse[dict[str, Any]],
)
async def delete_relationship(
    project_id: uuid.UUID,
    relationship_id: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_access(project_id, user, db)
    return ok(
        await UseCaseService(db).delete_relationship(
            project_id=project_id,
            relationship_id=relationship_id,
        )
    )
