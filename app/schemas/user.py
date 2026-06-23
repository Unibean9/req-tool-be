import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, computed_field


class UserResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    email: EmailStr
    full_name: str | None
    is_active: bool
    role: str
    github_id: str | None
    github_login: str | None
    created_at: datetime

    @computed_field
    @property
    def github_avatar_url(self) -> str | None:
        if self.github_id:
            return f"https://avatars.githubusercontent.com/u/{self.github_id}"
        return None


class UserUpdateRequest(BaseModel):
    full_name: str | None = None
