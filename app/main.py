import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import settings
from app.database import engine
from app.core.errors import http_exception_handler, validation_exception_handler, unhandled_exception_handler
from app.routers import admin, auth, github_auth, users, organizations, projects


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
api_v1.include_router(projects.alias_router)

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


@app.get("/health/bedrock", tags=["System Health"])
async def health_bedrock():
    import asyncio
    from app.config import settings

    if not settings.aws_access_key_id or not settings.aws_secret_access_key:
        return {
            "status": "disabled",
            "model": settings.bedrock_notation_model,
            "region": settings.aws_region,
            "message": "AWS credentials not configured — rule-based notation only",
        }

    def _ping():
        import boto3
        client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        resp = client.converse(
            modelId=settings.bedrock_notation_model,
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            inferenceConfig={"maxTokens": 5, "temperature": 0.0},
        )
        return resp["output"]["message"]["content"][0]["text"].strip()

    try:
        reply = await asyncio.to_thread(_ping)
        return {"status": "ok", "model": settings.bedrock_notation_model, "region": settings.aws_region, "reply": reply}
    except Exception as exc:
        return {"status": "error", "model": settings.bedrock_notation_model, "region": settings.aws_region, "message": str(exc)}
