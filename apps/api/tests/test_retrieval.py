import pytest

from app.models import Document, DocumentVersion, DocumentChunk
from app.services import retrieval


def make_embedding(first: float, second: float) -> list[float]:
    """Create a 768-dimensional test embedding."""
    return [first, second] + [0.0] * 766


def add_test_document(db_session):
    document = Document(title="Retrieval Test Policy")
    db_session.add(document)
    db_session.flush()
    return document


def add_test_chunk(
    db_session,
    version_id,
    content: str,
    embedding: list[float],
):
    chunk = DocumentChunk(
        version_id=version_id,
        chunk_index=0,
        content=content,
        section_path=["Test Section"],
        page_numbers=[1],
        element_labels=["text"],
        embedding=embedding,
    )
    db_session.add(chunk)
    db_session.flush()
    return chunk


def test_retrieval_returns_most_similar_chunks_first(
    db_session,
    monkeypatch,
):
    document = add_test_document(db_session)

    version = DocumentVersion(
        document=document,
        version_label="2025",
    )
    db_session.add(version)
    db_session.flush()

    add_test_chunk(
        db_session,
        version.id,
        "Exact match",
        make_embedding(1.0, 0.0),
    )

    add_test_chunk(
        db_session,
        version.id,
        "Related match",
        make_embedding(0.8, 0.6),
    )

    add_test_chunk(
        db_session,
        version.id,
        "Unrelated content",
        make_embedding(0.0, 1.0),
    )

    monkeypatch.setattr(
        retrieval,
        "generate_query_embedding",
        lambda query: make_embedding(1.0, 0.0),
    )

    results = retrieval.retrieve_chunks(
        db=db_session,
        query="Test query",
        limit=5,
    )

    assert len(results) == 2

    assert results[0]["chunk"].content == "Exact match"
    assert results[1]["chunk"].content == "Related match"

    assert results[0]["distance"] < results[1]["distance"]


def test_retrieval_respects_version_filter(
    db_session,
    monkeypatch,
):
    document = add_test_document(db_session)

    old_version = DocumentVersion(
        document=document,
        version_label="January 2025",
    )

    new_version = DocumentVersion(
        document=document,
        version_label="August 2025",
    )

    db_session.add_all([old_version, new_version])
    db_session.flush()

    add_test_chunk(
        db_session,
        old_version.id,
        "Old policy content",
        make_embedding(1.0, 0.0),
    )

    add_test_chunk(
        db_session,
        new_version.id,
        "New policy content",
        make_embedding(0.8, 0.6),
    )

    monkeypatch.setattr(
        retrieval,
        "generate_query_embedding",
        lambda query: make_embedding(1.0, 0.0),
    )

    results = retrieval.retrieve_chunks(
        db=db_session,
        query="Test query",
        version_id=new_version.id,
        limit=5,
    )

    assert len(results) == 1
    assert results[0]["chunk"].content == "New policy content"
    assert results[0]["version"].id == new_version.id
