from google.genai.errors import ServerError

from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, status, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Document, DocumentVersion, DocumentChunk
from app.schemas.document import DocumentCreate, DocumentRead, DocumentVersionRead
from app.schemas.search import SearchRequest
from app.schemas.search import AskRequest
from app.services.pdf_parser import parse_pdf
from app.services.chunker import chunk_docling_json
from app.services.embedding import generate_embedding, generate_query_embedding
from app.services.answer import generate_grounded_answer
from app.services.retrieval import retrieve_chunks

from app.core.config import PROJECT_ROOT

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_COSINE_DISTANCE = 0.45

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



@router.post(
    "/{document_id}/versions/{version_id}/parse",
    tags=["Document Processing"],
)
def parse_document_version(
    document_id: UUID,
    version_id: UUID,
    db: Session = Depends(get_db),
):
    version = db.scalar(
        select(DocumentVersion).where(
            DocumentVersion.id == version_id,
            DocumentVersion.document_id == document_id,
        )
    )

    if version is None:
        raise HTTPException(404, "Document version not found")

    if not version.storage_path:
        raise HTTPException(400, "No PDF stored for this version")

    pdf_path = (PROJECT_ROOT / version.storage_path).resolve()
    uploads_root = (PROJECT_ROOT / "uploads").resolve()

    if uploads_root not in pdf_path.parents:
        raise HTTPException(400, "Invalid PDF storage path")

    try:
        # 1. Extract PDF content.
        parse_result = parse_pdf(pdf_path)

        # 2. Generate section-aware chunks.
        chunks = chunk_docling_json(Path(parse_result["json_path"]))

        if not chunks:
            raise HTTPException(422, "No extractable content found")

        # 3. Replace existing chunks for this version.
        db.query(DocumentChunk).filter(
            DocumentChunk.version_id == version.id
        ).delete(synchronize_session=False)

        # 4. Store new chunks.
        db.add_all([
            DocumentChunk(
                version_id=version.id,
                chunk_index=chunk["chunk_index"],
                content=chunk["text"],
                section_path=chunk["metadata"]["section_path"],
                page_numbers=chunk["metadata"]["page_numbers"],
                element_labels=chunk["metadata"]["element_labels"],
            )
            for chunk in chunks
        ])

        db.commit()

        return {
            "version_id": str(version.id),
            "status": "processed",
            "page_count": parse_result["page_count"],
            "chunk_count": len(chunks),
        }

    except Exception:
        db.rollback()
        raise



@router.post(
    "/{document_id}/versions/{version_id}/embed",
    tags=["Document Processing"],
)
def embed_document_version(
    document_id: UUID,
    version_id: UUID,
    db: Session = Depends(get_db),
):
    version = db.scalar(
        select(DocumentVersion).where(
            DocumentVersion.id == version_id,
            DocumentVersion.document_id == document_id,
        )
    )

    if version is None:
        raise HTTPException(404, "Document version not found")

    document = db.get(Document, document_id)

    chunks = db.scalars(
        select(DocumentChunk)
        .where(
            DocumentChunk.version_id == version_id,
            DocumentChunk.embedding.is_(None),
        )
        .order_by(DocumentChunk.chunk_index)
    ).all()

    try:
        for chunk in chunks:
            chunk.embedding = generate_embedding(
                text=chunk.content,
                title=document.title,
            )

        db.commit()

        return {
            "status": "completed",
            "embedded_chunks": len(chunks),
            "dimensions": 768,
        }

    except Exception:
        db.rollback()
        raise



@router.post("/search", tags=["Semantic Search"])
def semantic_search(
    payload: SearchRequest,
    db: Session = Depends(get_db),
):
    results = retrieve_chunks(
        db=db,
        query=payload.query,
        version_id=payload.version_id,
        limit=payload.limit,
    )

    return {
        "query": payload.query,
        "results": [
            {
                "document_title": item["document"].title,
                "version_id": str(item["version"].id),
                "version_label": item["version"].version_label,
                "chunk_id": str(item["chunk"].id),
                "content": item["chunk"].content,
                "section": item["chunk"].section_path,
                "pages": item["chunk"].page_numbers,
                "cosine_distance": round(item["distance"], 4),
            }
            for item in results
        ],
    }



@router.post("/ask", tags=["RAG"])
def ask_documents(
    payload: AskRequest,
    db: Session = Depends(get_db),
):
    # Retrieve relevant document chunks.
    results = retrieve_chunks(
        db=db,
        query=payload.query,
        version_id=payload.version_id,
        limit=payload.top_k,
    )

    # Abstain when no supporting evidence is found.
    if not results:
        return {
            "question": payload.query,
            "answer": (
                "I couldn't find sufficient evidence in the "
                "uploaded documents to answer this question."
            ),
            "sources": [],
        }

    # Prepare source references.
    sources = []

    for index, item in enumerate(results, start=1):
        chunk = item["chunk"]
        document = item["document"]
        version = item["version"]

        sources.append({
            "source_id": f"S{index}",
            "document_title": document.title,
            "version_label": version.version_label,
            "pages": chunk.page_numbers,
            "section": chunk.section_path,
            "content": chunk.content,
        })

    # Generate an evidence-grounded answer.
    try:
        answer = generate_grounded_answer(
            question=payload.query,
            sources=sources,
        )
    except ServerError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is temporarily unavailable. "
                "Please retry shortly."
            ),
        ) from exc

    # Return answer and source metadata.
    return {
        "question": payload.query,
        "answer": answer,
        "sources": [
            {
                "citation": f"[S{index}]",
                "document_title": source["document_title"],
                "version_label": source["version_label"],
                "pages": source["pages"],
                "section": source["section"],
            }
            for index, source in enumerate(sources, start=1)
        ],
    }
