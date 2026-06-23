import hashlib
import hmac
import html
import json
import os
import urllib.parse
import uuid

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, EmailStr

from app.config import settings
from app.core.responses import ok
from app.deps import current_user, get_auth_service
from app.models.user import User
from app.schemas.auth import RefreshRequest, TokenResponse
from app.schemas.response import ApiResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth/github", tags=["Auth"])

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
_LOGIN_COOKIE = "oauth_nonce"
_CONNECT_COOKIE = "gh_connect_nonce"
_COOKIE_MAX_AGE = 600


# --- State helpers ---

def _state_hmac_key() -> bytes:
    if settings.github_state_secret:
        return settings.github_state_secret.encode()
    return (settings.jwt_secret_key + ":github-oauth-csrf").encode()


def _sign(payload: str) -> str:
    return hmac.new(_state_hmac_key(), payload.encode(), hashlib.sha256).hexdigest()


def _make_login_state(nonce: str) -> str:
    return f"{nonce}.{_sign(nonce)}"


def _verify_login_state(state: str, cookie_nonce: str) -> bool:
    try:
        nonce, sig = state.rsplit(".", 1)
    except ValueError:
        return False
    return hmac.compare_digest(_sign(nonce), sig) and hmac.compare_digest(nonce, cookie_nonce)


_CONNECT_PREFIX = "connect"


def make_connect_state(project_id: uuid.UUID, user_id: uuid.UUID, nonce: str) -> str:
    payload = f"{_CONNECT_PREFIX}:{project_id}:{user_id}:{nonce}"
    return f"{payload}.{_sign(payload)}"


def verify_connect_state(state: str, cookie_nonce: str) -> tuple[uuid.UUID, uuid.UUID] | None:
    try:
        payload, sig = state.rsplit(".", 1)
        prefix, project_id_str, user_id_str, nonce = payload.split(":", 3)
    except ValueError:
        return None
    if prefix != _CONNECT_PREFIX:
        return None
    if not hmac.compare_digest(_sign(payload), sig):
        return None
    if not hmac.compare_digest(nonce, cookie_nonce):
        return None
    try:
        return uuid.UUID(project_id_str), uuid.UUID(user_id_str)
    except ValueError:
        return None


def is_connect_state(state: str) -> bool:
    payload = state.rsplit(".", 1)[0]
    return payload.startswith(f"{_CONNECT_PREFIX}:")


def _error_html(target_origin: str, error: str, description: str | None = None) -> str:
    payload = json.dumps({"type": "github_error", "error": error, "description": description})
    return f"""<!doctype html><html><body><script>
window.opener && window.opener.postMessage({payload}, {json.dumps(target_origin)});
window.close();
</script><p>Lỗi xác thực: {html.escape(error)}</p></body></html>"""


# --- Endpoints ---

@router.get("")
async def github_authorize():
    nonce = os.urandom(16).hex()
    state = _make_login_state(nonce)
    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": settings.github_redirect_uri,
        "scope": "read:user user:email",
        "state": state,
    }
    redirect = RedirectResponse(f"{GITHUB_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}")
    redirect.set_cookie(
        _LOGIN_COOKIE,
        nonce,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        secure=settings.app_env != "development",
        samesite="lax",
    )
    return redirect


@router.get("/callback", response_class=HTMLResponse)
async def github_callback(
    state: str,
    response: Response,
    code: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
    auth_service: AuthService = Depends(get_auth_service),
    oauth_nonce: str | None = Cookie(default=None, alias=_LOGIN_COOKIE),
):
    target_origin = settings.cors_origins[0] if settings.cors_origins else "*"
    _secure = settings.app_env != "development"

    if error:
        return HTMLResponse(_error_html(target_origin, error, error_description))

    if not code:
        return HTMLResponse(_error_html(target_origin, "missing_code", "GitHub không trả về authorization code"))

    if not oauth_nonce or not _verify_login_state(state, oauth_nonce):
        return HTMLResponse(
            _error_html(target_origin, "invalid_state", "OAuth state không hợp lệ — có thể là tấn công CSRF")
        )
    response.delete_cookie(_LOGIN_COOKIE, httponly=True, samesite="lax", secure=_secure)

    try:
        access_token, refresh_token = await auth_service.process_github_callback(code)
    except HTTPException as exc:
        return HTMLResponse(_error_html(target_origin, "auth_failed", exc.detail))

    payload = json.dumps({
        "type": "github_oauth",
        "access_token": access_token,
        "refresh_token": refresh_token,
    })
    return HTMLResponse(f"""<!doctype html><html><body><script>
window.opener && window.opener.postMessage({payload}, {json.dumps(target_origin)});
window.close();
</script><p>Đang đóng cửa sổ...</p></body></html>""")


@router.post("/refresh", response_model=ApiResponse[TokenResponse])
async def refresh(
    body: RefreshRequest,
    service: AuthService = Depends(get_auth_service),
):
    access_token, refresh_token = await service.refresh_tokens(body.refresh_token)
    return ok(TokenResponse(access_token=access_token, refresh_token=refresh_token))


class DevLoginRequest(BaseModel):
    email: EmailStr


@router.post(
    "/dev-login",
    response_model=ApiResponse[TokenResponse],
    include_in_schema=settings.app_env == "development",
)
async def dev_login(
    body: DevLoginRequest,
    service: AuthService = Depends(get_auth_service),
):
    if settings.app_env != "development":
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    user = await service.get_user_for_dev_login(body.email)
    access_token, refresh_token = service.create_token_pair(user)
    return ok(TokenResponse(access_token=access_token, refresh_token=refresh_token))


@router.delete("/unlink", status_code=status.HTTP_204_NO_CONTENT)
async def github_unlink(
    user: User = Depends(current_user),
    service: AuthService = Depends(get_auth_service),
):
    await service.unlink_github(user)
