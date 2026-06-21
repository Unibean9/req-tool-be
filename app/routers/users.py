from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.responses import ok
from app.database import get_db
from app.deps import current_user
from app.models.user import User
from app.schemas.organization import UserSearchResult
from app.schemas.response import ApiResponse
from app.schemas.user import UserResponse, UserUpdateRequest

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/search", response_model=ApiResponse[list[UserSearchResult]])
async def search_users(
    q: str = Query(min_length=1, max_length=255),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Search users by email or GitHub username — use to find members to invite."""
    pattern = f"%{q}%"
    q_stmt = select(User).where(
        or_(User.email.ilike(pattern), User.github_login.ilike(pattern), User.full_name.ilike(pattern))
    )
    if _user.role != "admin":
        q_stmt = q_stmt.where(User.role == "user")
    result = await db.execute(q_stmt.order_by(User.full_name).limit(limit).offset(offset))
    return ok(result.scalars().all())


@router.get("/me", response_model=ApiResponse[UserResponse])
async def get_me(user: User = Depends(current_user)):
    return ok(user)


@router.patch("/me", response_model=ApiResponse[UserResponse])
async def update_me(
    body: UserUpdateRequest,
    user: User = Depends(current_user),
):
    if body.full_name is not None:
        user.full_name = body.full_name
    return ok(user)
