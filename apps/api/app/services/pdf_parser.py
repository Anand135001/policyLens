
import json
from pathlib import Path

from docling.document_converter import DocumentConverter


# Reuse one converter instead of initializing it for every PDF.
converter = DocumentConverter()


def parse_pdf(pdf_path: Path) -> dict:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Convert PDF into Docling's structured document format.
    result = converter.convert(pdf_path, max_file_size=25 * 1024 * 1024)
    document = result.document

    # Generate readable Markdown.
    markdown = document.export_to_markdown()

    # Preserve complete document structure in JSON.
    structured_data = document.export_to_dict()

    # Save both outputs beside the original PDF.
    markdown_path = pdf_path.with_suffix(".md")
    json_path = pdf_path.with_suffix(".docling.json")

    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(structured_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "status": result.status.value,
        "page_count": len(document.pages),
        "markdown_path": markdown_path.as_posix(),
        "json_path": json_path.as_posix(),
        "markdown_preview": markdown[:500],
    }
