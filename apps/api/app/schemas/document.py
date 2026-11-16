from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    issuing_authority: str | None = Field(default=None, max_length=255)
    document_type: str | None = Field(default=None, max_length=100)


class DocumentRead(BaseModel):
    id: UUID
    title: str
    issuing_authority: str | None
    document_type: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentVersionRead(BaseModel):
    id: UUID
    document_id: UUID
    version_label: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)