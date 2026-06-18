import uuid
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field, model_validator


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    org_id: uuid.UUID | None = None
    description: str | None = None
    context: str | None = None
    problems: list[str] = []
    proposed_solutions: list[str] | None = None
    start_date: date | None = None
    end_date: date | None = None
    budget: Decimal | None = Field(None, ge=0)
    executive_summary: str | None = None
    roi_notes: str | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> "ProjectCreateRequest":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")
        return self


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    context: str | None = None
    problems: list[str] | None = None
    proposed_solutions: list[str] | None = None
    start_date: date | None = None
    end_date: date | None = None
    budget: Decimal | None = Field(None, ge=0)
    executive_summary: str | None = None
    roi_notes: str | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> "ProjectUpdateRequest":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")
        return self


class ProjectResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    org_id: uuid.UUID
    name: str
    slug: str
    description: str | None
    context: str | None = None
    problems: list[str] = []
    proposed_solutions: list[str] = []
    start_date: date | None = None
    end_date: date | None = None
    budget: Decimal | None = None
    executive_summary: str | None = None
    roi_notes: str | None = None
    created_at: datetime
