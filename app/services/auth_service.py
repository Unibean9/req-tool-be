import asyncio
import uuid

import httpx
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.crypto import encrypt_token
from app.core.security import create_access_token, create_refresh_token, decode_token
from app.models.user import User

_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GITHUB_USER_URL = "https://api.github.com/user"
_GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def process_github_callback(self, code: str) -> tuple[str, str]:
        """Exchange GitHub OAuth code, upsert user, return (access_token, refresh_token)."""
        async with httpx.AsyncClient() as client:
            try:
                token_resp = await client.post(
                    _GITHUB_TOKEN_URL,
                    data={
                        "client_id": settings.github_client_id,
                        "client_secret": settings.github_client_secret,
                        "code": code,
                        "redirect_uri": settings.github_redirect_uri,
                    },
                    headers={"Accept": "application/json"},
                    timeout=10,
                )
                token_resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise HTTPException(
                    status.HTTP_502_BAD_GATEWAY,
                    detail=f"GitHub token exchange failed: {exc.response.status_code}",
                ) from exc
            except httpx.RequestError as exc:
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="Could not connect to GitHub") from exc

        gh_access_token = token_resp.json().get("access_token")
        if not gh_access_token:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail="GitHub OAuth failed: no access token received",
            )

        gh_headers = {"Authorization": f"Bearer {gh_access_token}", "Accept": "application/vnd.github+json"}
        async with httpx.AsyncClient() as client:
            try:
                user_resp, emails_resp = await asyncio.gather(
                    client.get(_GITHUB_USER_URL, headers=gh_headers, timeout=10),
                    client.get(_GITHUB_EMAILS_URL, headers=gh_headers, timeout=10),
                )
                user_resp.raise_for_status()
                emails_resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise HTTPException(
                    status.HTTP_502_BAD_GATEWAY,
                    detail=f"GitHub API error: {exc.response.status_code}",
                ) from exc
            except httpx.RequestError as exc:
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="Could not connect to GitHub") from exc

        gh_user = user_resp.json()
        emails = emails_resp.json()

        verified_primary = next(
            (e["email"] for e in emails if e.get("verified") and e.get("primary")),
            None,
        )
        if not verified_primary:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail="GitHub account has no verified primary email. Add and verify an email on GitHub before linking.",  # noqa: E501 — single message string
            )

        github_id = str(gh_user["id"])
        github_login = gh_user.get("login")
        encrypted_token = encrypt_token(gh_access_token)

        result = await self.db.execute(select(User).where(User.github_id == github_id))
        user = result.scalar_one_or_none()

        if not user:
            result = await self.db.execute(select(User).where(User.email == verified_primary))
            user = result.scalar_one_or_none()

        if user:
            user.github_id = github_id
            user.github_login = github_login
            user.github_access_token = encrypted_token
        else:
            user = User(
                email=verified_primary,
                github_id=github_id,
                github_login=github_login,
                github_access_token=encrypted_token,
                full_name=gh_user.get("name"),
            )
            self.db.add(user)

        await self.db.flush()
        return create_access_token(str(user.id), user.role), create_refresh_token(str(user.id))

    async def refresh_tokens(self, refresh_token: str) -> tuple[str, str]:
        payload = decode_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Refresh token is invalid")
        try:
            uid = uuid.UUID(payload["sub"])
        except (KeyError, ValueError):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Refresh token is invalid") from None
        result = await self.db.execute(select(User).where(User.id == uid))
        user = result.scalar_one_or_none()
        if not user or not user.is_active:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return create_access_token(str(user.id), user.role), create_refresh_token(str(user.id))

    async def get_user_for_dev_login(self, email: str) -> User:
        result = await self.db.execute(select(User).where(User.email == email, User.is_active.is_(True)))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                detail="No active user found with this email",
            )
        return user

    def create_token_pair(self, user: User) -> tuple[str, str]:
        return create_access_token(str(user.id), user.role), create_refresh_token(str(user.id))

    async def unlink_github(self, user: User) -> None:
        user.github_id = None
        user.github_login = None
        user.github_access_token = None
        await self.db.flush()
