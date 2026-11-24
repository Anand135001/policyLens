from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk, DocumentVersion
from app.services.embedding import generate_query_embedding


MAX_COSINE_DISTANCE = 0.45


def retrieve_chunks(
    db: Session,
    query: str,
    version_id: UUID | None = None,
    limit: int = 5,
) -> list[dict]:

    query_embedding = generate_query_embedding(query)

    distance = DocumentChunk.embedding.cosine_distance(
        query_embedding
    ).label("distance")

    statement = (
        select(DocumentChunk, Document, DocumentVersion, distance)
        .join(
            DocumentVersion,
            DocumentChunk.version_id == DocumentVersion.id,
        )
        .join(
            Document,
            DocumentVersion.document_id == Document.id,
        )
        .where(
            DocumentChunk.embedding.is_not(None),
            DocumentChunk.embedding.cosine_distance(
                query_embedding
            ) <= MAX_COSINE_DISTANCE,
        )
        .order_by(distance)
        .limit(limit)
    )

    if version_id:
        statement = statement.where(
            DocumentVersion.id == version_id
        )

    rows = db.execute(statement).all()

    return [
        {
            "chunk": chunk,
            "document": document,
            "version": version,
            "distance": float(distance_value),
        }
        for chunk, document, version, distance_value in rows
    ]
