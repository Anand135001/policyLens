from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Document
from app.schemas.document import DocumentCreate, DocumentRead

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
