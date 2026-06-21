import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.guards import require_project_member_404
from app.core.responses import created, ok
from app.database import async_session_factory, get_db
from app.deps import current_user
from app.models.user import User
from app.schemas.agent import (
    AgentMessageCreate,
    AgentMessageResponse,
    AgentSessionCreate,
    AgentSessionCreateResponse,
    AgentSessionResponse,
    AgentToolCallResponse,
    ToolCallEditRequest,
)
from app.schemas.response import ApiResponse
from app.services.agent_event_service import AgentEventService
from app.services.agent_service import AgentService

router = APIRouter(tags=["Agent Sessions"])

_SESSION_PREFIX = "/projects/{project_id}/agent-sessions"
_TC_PREFIX = "/projects/{project_id}/agent-tool-calls"


def _graph(request: Request):
    return getattr(request.app.state, "compiled_graph", None)


def _require_graph(request: Request):
    graph = _graph(request)
    if graph is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent service chưa sẵn sàng")
    return graph


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

@router.post(_SESSION_PREFIX, response_model=ApiResponse[AgentSessionCreateResponse], status_code=status.HTTP_201_CREATED)
async def create_session(
    request: Request,
    project_id: uuid.UUID,
    body: AgentSessionCreate,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_member_404(project_id, user, db)
    result = await AgentService(db=db, graph=_require_graph(request), session_factory=async_session_factory).create_session(
        project_id=project_id,
        artifact_type=body.artifact_type,
        step_key=body.step_key,
        workflow_area=body.workflow_area,
        agent_role=body.agent_role,
        provider_config_id=body.provider_config_id,
        created_by_id=user.id,
    )
    return created(result)


@router.get(_SESSION_PREFIX + "/{session_id}", response_model=ApiResponse[AgentSessionResponse])
async def get_session(
    request: Request,
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_member_404(project_id, user, db)
    session = await AgentService(db=db, graph=_graph(request), session_factory=async_session_factory).get_session(
        project_id=project_id, session_id=session_id, user_id=user.id
    )
    return ok(AgentSessionResponse.model_validate(session))


@router.delete(_SESSION_PREFIX + "/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    request: Request,
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    await require_project_member_404(project_id, user, db)
    await AgentService(db=db, graph=_graph(request), session_factory=async_session_factory).delete_session(
        project_id=project_id, session_id=session_id, user_id=user.id
    )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@router.post(_SESSION_PREFIX + "/{session_id}/messages", response_model=ApiResponse[AgentMessageResponse])
async def send_message(
    request: Request,
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    body: AgentMessageCreate,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_member_404(project_id, user, db)
    msg = await AgentService(db=db, graph=_require_graph(request), session_factory=async_session_factory).handle_user_message(
        project_id=project_id, session_id=session_id, content=body.content, user_id=user.id, mode_hint=body.mode_hint
    )
    return ok(AgentMessageResponse.model_validate(msg))


@router.get(_SESSION_PREFIX + "/{session_id}/messages", response_model=ApiResponse[list[AgentMessageResponse]])
async def list_messages(
    request: Request,
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_member_404(project_id, user, db)
    msgs = await AgentService(db=db, graph=_graph(request), session_factory=async_session_factory).list_messages(
        project_id=project_id, session_id=session_id, user_id=user.id
    )
    return ok([AgentMessageResponse.model_validate(m) for m in msgs])


@router.get(_SESSION_PREFIX + "/{session_id}/events")
async def stream_session_events(
    request: Request,
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    await require_project_member_404(project_id, user, db)
    service = AgentEventService(db)
    await service.build_snapshot(project_id=project_id, session_id=session_id, user_id=user.id)
    return StreamingResponse(
        service.stream_session_events(
            project_id=project_id,
            session_id=session_id,
            user_id=user.id,
            request=request,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------

@router.get(_SESSION_PREFIX + "/{session_id}/tool-calls", response_model=ApiResponse[list[AgentToolCallResponse]])
async def list_tool_calls(
    request: Request,
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_member_404(project_id, user, db)
    tcs = await AgentService(db=db, graph=_graph(request), session_factory=async_session_factory).list_tool_calls(
        project_id=project_id, session_id=session_id, user_id=user.id
    )
    return ok([AgentToolCallResponse.model_validate(tc) for tc in tcs])


@router.post(_TC_PREFIX + "/{tool_call_id}/approve", response_model=ApiResponse[AgentToolCallResponse])
async def approve_tool_call(
    request: Request,
    project_id: uuid.UUID,
    tool_call_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_member_404(project_id, user, db)
    tc = await AgentService(db=db, graph=_require_graph(request), session_factory=async_session_factory).approve_tool_call(
        project_id=project_id, tool_call_id=tool_call_id, created_by_id=user.id, user_id=user.id
    )
    return ok(AgentToolCallResponse.model_validate(tc))


@router.post(_TC_PREFIX + "/{tool_call_id}/reject", response_model=ApiResponse[AgentToolCallResponse])
async def reject_tool_call(
    request: Request,
    project_id: uuid.UUID,
    tool_call_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_member_404(project_id, user, db)
    tc = await AgentService(db=db, graph=_require_graph(request), session_factory=async_session_factory).reject_tool_call(
        project_id=project_id, tool_call_id=tool_call_id, user_id=user.id
    )
    return ok(AgentToolCallResponse.model_validate(tc))


@router.post(_TC_PREFIX + "/{tool_call_id}/request-edit", response_model=ApiResponse[AgentToolCallResponse])
async def request_edit(
    request: Request,
    project_id: uuid.UUID,
    tool_call_id: uuid.UUID,
    body: ToolCallEditRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await require_project_member_404(project_id, user, db)
    tc = await AgentService(db=db, graph=_require_graph(request), session_factory=async_session_factory).request_edit(
        project_id=project_id, tool_call_id=tool_call_id, note=body.note, user_id=user.id
    )
    return ok(AgentToolCallResponse.model_validate(tc))
