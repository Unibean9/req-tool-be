import base64
import hashlib
import hmac as _hmac
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from jose import JWTError, jwt

from app.config import settings


def _prehash(password: str) -> bytes:
    # HMAC-SHA256 with a server-side pepper; base64 output is 44 bytes (well under the 72-byte bcrypt limit).
    # Pepper defaults to jwt_secret_key in dev so no extra config is needed locally.
    # Set PASSWORD_PEPPER independently in production so JWT and password secrets can rotate separately.
    pepper = settings.password_pepper or settings.jwt_secret_key
    digest = _hmac.new(pepper.encode("utf-8"), password.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(_prehash(plain), hashed.encode())


def _create_token(subject: Any, expires_delta: timedelta, token_type: str, extra: dict | None = None) -> str:
    expire = datetime.now(UTC) + expires_delta
    payload = {"sub": str(subject), "exp": expire, "type": token_type, **(extra or {})}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: str, role: str = "user") -> str:
    return _create_token(
        user_id,
        timedelta(minutes=settings.jwt_access_token_expire_minutes),
        "access",
        {"role": role},
    )


def create_refresh_token(user_id: str) -> str:
    return _create_token(
        user_id,
        timedelta(days=settings.jwt_refresh_token_expire_days),
        "refresh",
    )


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return {}
