import logging
from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _problem(
    status_code: int,
    title: str,
    detail: str | dict[str, Any],
    instance: str | None = None,
    request_id: str | None = None,
    errors: list[dict] | None = None,
) -> JSONResponse:
    body: dict = {
        "type": "about:blank",
        "title": title,
        "status": status_code,
        "detail": detail,
    }
    if instance:
        body["instance"] = instance
    if request_id:
        body["request_id"] = request_id
    if errors:
        body["errors"] = errors
    return JSONResponse(
        status_code=status_code,
        content=body,
        media_type="application/problem+json",
    )


_HTTP_TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    409: "Conflict",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
}


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    title = _HTTP_TITLES.get(exc.status_code, "Error")
    return _problem(
        exc.status_code,
        title,
        exc.detail,
        str(request.url),
        _request_id(request),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = []
    for e in exc.errors():
        # Strip leading "body", "query", "path" location prefix from field path
        loc = e["loc"]
        if loc and loc[0] in ("body", "query", "path", "header", "cookie"):
            loc = loc[1:]
        field = ".".join(str(p) for p in loc) if loc else "request"
        errors.append({"field": field, "message": e["msg"]})

    return _problem(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "Validation Error",
        "Dữ liệu đầu vào không hợp lệ",
        str(request.url),
        _request_id(request),
        errors,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url, exc_info=exc)
    # Never expose internal detail outside development
    detail = str(exc) if settings.app_debug else "Đã xảy ra lỗi không mong muốn."
    return _problem(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Internal Server Error",
        detail,
        str(request.url),
        _request_id(request),
    )
