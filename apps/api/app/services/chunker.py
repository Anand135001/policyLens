import re
import json
from pathlib import Path
from typing import Any


def _is_fragment_heading(text: str) -> bool:
    patterns = [
        r"version\s*\d+",
        r"commit\s+[0-9a-f]{20,40}",
    ]

    return any(
        re.fullmatch(pattern, text.strip(), re.IGNORECASE)
        for pattern in patterns
    )


def _resolve_ref(data: dict, ref: str) -> dict | None:
    """Resolve a Docling reference such as #/texts/4."""
    try:
        section, index = ref.lstrip("#/").split("/")
        return data[section][int(index)]
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def _walk_elements(data: dict):
    """Traverse document elements in their original reading order."""
    def walk(ref: str):
        node = _resolve_ref(data, ref)

        if not node:
            return

        if ref.startswith("#/groups/") or ref.startswith("#/pictures/"):
            for child in node.get("children", []):
                yield from walk(child["$ref"])

        elif ref.startswith("#/texts/"):
            yield node

        elif ref.startswith("#/tables/"):
            yield node

    for child in data["body"].get("children", []):
        yield from walk(child["$ref"])


def _render_table(table: dict) -> str:
    """Convert Docling table cells into readable rows."""
    cells = table.get("data", {}).get("table_cells", [])
    rows: dict[int, list[tuple[int, str]]] = {}

    for cell in cells:
        row = cell.get("start_row_offset_idx", 0)
        col = cell.get("start_col_offset_idx", 0)
        value = cell.get("text", "").strip()

        if value:
            rows.setdefault(row, []).append((col, value))

    rendered_rows = []

    for row in sorted(rows):
        values = [
            value
            for _, value in sorted(rows[row], key=lambda item: item[0])
        ]
        rendered_rows.append(" | ".join(values))

    return "[TABLE]\n" + "\n".join(rendered_rows)


def chunk_docling_json(
    json_path: Path,
    max_chars: int = 1800,
) -> list[dict[str, Any]]:
    """Create section-aware chunks with page provenance."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    with json_path.open(encoding="utf-8") as file:
        data = json.load(file)

    section_path: list[str] = []
    chunks = []

    current_parts: list[str] = []
    current_pages: set[int] = set()
    current_labels: set[str] = set()
    current_length = 0

    def flush():
        nonlocal current_parts, current_pages
        nonlocal current_labels, current_length

        content = "\n\n".join(current_parts).strip()

        if not content:
            return

        chunks.append({
            "chunk_index": len(chunks),
            "text": content,
            "metadata": {
                "section_path": section_path.copy(),
                "page_numbers": sorted(current_pages),
                "element_labels": sorted(current_labels),
            },
        })

        current_parts = []
        current_pages = set()
        current_labels = set()
        current_length = 0

    for element in _walk_elements(data):
        label = element.get("label", "")

        # Ignore repeated page furniture.
        if label in {"page_footer", "page_header"}:
            continue

        if label == "table":
            content = _render_table(element)
        else:
            content = element.get("text", "").strip()

        if not content:
            continue

        # A heading starts a new semantic section.
        if label == "section_header" and not _is_fragment_heading(content):
            flush()

            level = max(1, int(element.get("level", 1)))

            while len(section_path) >= level:
                section_path.pop()

            section_path.append(content)

        # Keep a complete element together whenever possible.
        if current_parts and current_length + len(content) > max_chars:
            flush()

        current_parts.append(content)
        current_length += len(content)
        current_labels.add(label)

        for provenance in element.get("prov", []):
            page = provenance.get("page_no")

            if page is not None:
                current_pages.add(page)

    flush()

    return chunks