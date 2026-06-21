import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class OrgCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class OrgStats(BaseModel):
    member_count: int
    project_count: int


class OrgResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    name: str
    slug: str
    owner_id: uuid.UUID
    created_at: datetime
    stats: OrgStats | None = None


class MemberUserInfo(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    email: str
    full_name: str | None
    github_id: str | None
    github_login: str | None

    @computed_field
    @property
    def github_avatar_url(self) -> str | None:
        if self.github_id:
            return f"https://avatars.githubusercontent.com/u/{self.github_id}"
        return None


class OrgMemberResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    org_id: uuid.UUID
    user_id: uuid.UUID
    role: str
    created_at: datetime
    user: MemberUserInfo | None = None


class AddMemberItem(BaseModel):
    identifier: str = Field(min_length=1, max_length=255)
    role: Literal["owner", "member"] = "member"


class AddMemberRequest(BaseModel):
    members: list[AddMemberItem] = Field(min_length=1, max_length=50)


class BulkAddMemberResponse(BaseModel):
    added: list[OrgMemberResponse]
    skipped: list[str]
    not_found: list[str]


class UserSearchResult(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    email: str
    full_name: str | None
    github_id: str | None
    github_login: str | None

    @computed_field
    @property
    def github_avatar_url(self) -> str | None:
        if self.github_id:
            return f"https://avatars.githubusercontent.com/u/{self.github_id}"
        return None
