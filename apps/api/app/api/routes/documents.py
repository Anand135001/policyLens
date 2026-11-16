from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, status, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Document
from app.models import DocumentVersion
from app.schemas.document import DocumentCreate, DocumentRead
from app.schemas.document import DocumentVersionRead

from app.core.config import PROJECT_ROOT

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/", response_model=DocumentRead, status_code=status.HTTP_201_CREATED,)
def create_document(
    payload: DocumentCreate,
    db: Session = Depends(get_db),
):
    document = Document(**payload.model_dump())

    db.add(document)
    db.commit()
    db.refresh(document)

    return document


@router.get("/", response_model=list[DocumentRead])
def list_documents(db: Session = Depends(get_db)):
    documents = db.scalars(
        select(Document).order_by(Document.created_at.desc())
    ).all()

    return documents



@router.post(
    "/{document_id}/versions",
    response_model=DocumentVersionRead,
    status_code=201,
)
def upload_document_version(
    document_id: UUID,
    file: UploadFile = File(...),
    version_label: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    # Confirm parent document exists.
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(404, "Document not found")

    # Validate PDF format.
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported")

    content = file.file.read(25 * 1024 * 1024 + 1)

    if not content.startswith(b"%PDF-"):
        raise HTTPException(400, "Invalid PDF file")

    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(413, "PDF must be 25 MB or smaller")

    # Generate a safe storage path.
    version_id = uuid4()
    relative_path = (
        Path("uploads")
        / str(document_id)
        / f"{version_id}.pdf"
    )

    absolute_path = PROJECT_ROOT / relative_path
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_bytes(content)

    try:
        version = DocumentVersion(
            id=version_id,
            document=document,
            version_label=version_label,
            storage_path=relative_path.as_posix(),
        )

        db.add(version)
        db.commit()
        db.refresh(version)

        return version

    except Exception:
        db.rollback()
        absolute_path.unlink(missing_ok=True)
        raise

    finally:
        file.file.close()
