import os
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import settings
from app.core.errors import http_exception_handler, unhandled_exception_handler, validation_exception_handler
from app.database import engine
from app.routers import (
    admin,
    agent_sessions,
    artifact_links,
    artifacts,
    auth,
    documents,
    exports,
    github_auth,
    llm_providers,
    organizations,
    projects,
    source_documents,
    users,
    workflow,
)


def _run_migrations() -> None:
    from alembic.config import Config

    from alembic import command

    ini_path = os.path.join(os.path.dirname(__file__), "..", "alembic.ini")
    cfg = Config(os.path.normpath(ini_path))
    command.upgrade(cfg, "head")


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    if settings.auto_migrate:
        await asyncio.get_event_loop().run_in_executor(None, _run_migrations)
    if settings.app_env == "development":
        from scripts.seed_dev_users import seed

        await seed()

    from app.instructions import load_instructions, loaded_roles

    load_instructions()
    if loaded_roles():
        print(f"[instructions] loaded: {loaded_roles()}")
    else:
        print("[instructions] no instruction files found — system=None for all sessions")

    from app.database import async_session_factory
    from app.graphs.checkpointer import DelegatingCheckpointer
    from app.graphs.graph import build_graph

    app.state.compiled_graph = build_graph(DelegatingCheckpointer(async_session_factory))

    yield
    await engine.dispose()


app = FastAPI(
    title="ReqFlow API",
    version="1.0.0",
    debug=settings.app_debug,
    lifespan=lifespan,
)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(admin.router)
api_v1.include_router(auth.router)
api_v1.include_router(github_auth.router)
api_v1.include_router(users.router)
api_v1.include_router(organizations.router)
api_v1.include_router(organizations.alias_router)
api_v1.include_router(projects.router)
api_v1.include_router(artifacts.router)
api_v1.include_router(documents.router)
api_v1.include_router(artifact_links.router)
api_v1.include_router(source_documents.router)
api_v1.include_router(workflow.router)
api_v1.include_router(exports.router)
api_v1.include_router(llm_providers.router)
api_v1.include_router(agent_sessions.router)

app.include_router(api_v1)


@app.get("/", tags=["System Health"], include_in_schema=False)
async def root():
    return {"name": "ReqFlow API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health", tags=["System Health"])
async def health():
    from sqlalchemy import text

    from app.database import async_session_factory

    async with async_session_factory() as session:
        await session.execute(text("SELECT 1"))
    return {"status": "ok", "db": "connected"}
