from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    version_id: UUID | None = None
    limit: int = Field(default=5, ge=1, le=20)


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    version_id: UUID | None = None
    top_k: int = Field(default=5, ge=1, le=10)    