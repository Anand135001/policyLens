
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import DateTime, ForeignKey, String, Text, Uuid, func

from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    issuing_authority: Mapped[str | None] = mapped_column(String(255))
    document_type: Mapped[str | None] = mapped_column(String(100))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    version_label: Mapped[str | None] = mapped_column(String(100))
    source_url: Mapped[str | None] = mapped_column(Text)
    storage_path: Mapped[str | None] = mapped_column(String(1024))

    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    effective_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    effective_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    document: Mapped["Document"] = relationship(
        back_populates="versions",
    )