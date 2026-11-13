from sqlalchemy.orm import Session

from app.models import Document, DocumentVersion


def test_document_version_relationship_and_cascade_delete(
    db_session: Session,
):
    document = Document(
        title="Test Government Policy",
        issuing_authority="Test Authority",
        document_type="Guideline",
    )

    version = DocumentVersion(
        version_label="2026",
        source_url="https://example.com/policy",
        document=document,
    )

    db_session.add(document)
    db_session.flush()

    document_id = document.id
    version_id = version.id

    # Verify document and version relationship.
    saved_document = db_session.get(Document, document_id)

    assert saved_document is not None
    assert len(saved_document.versions) == 1
    assert saved_document.versions[0].version_label == "2026"

    # Verify cascade deletion.
    db_session.delete(saved_document)
    db_session.flush()

    assert db_session.get(Document, document_id) is None
    assert db_session.get(DocumentVersion, version_id) is None
