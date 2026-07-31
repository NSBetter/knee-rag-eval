"""Audit automatic table detection in the local guideline corpus.

For every PDF page, this script:

1. Runs PyMuPDF table detection.
2. Records the number and shape of detected tables.
3. Saves each detected table as local Markdown.
4. Creates a page-level CSV audit report.

The generated Markdown files stay local under data/processed/.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pymupdf


ROOT_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT_DIR / "docs" / "source_registry.csv"
CORPUS_DIR = ROOT_DIR / "data" / "corpus"

TABLE_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "tables"
AUDIT_OUTPUT_PATH = ROOT_DIR / "docs" / "table_detection_audit.csv"


def read_included_sources() -> list[dict[str, str]]:
    """Read included source records from the registry."""

    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    included_rows = [
        row
        for row in rows
        if row["include_status"].strip().lower() == "included"
    ]

    if not included_rows:
        raise ValueError("No included sources found in source registry.")

    return included_rows


def safe_table_filename(
    source_id: str,
    page_number: int,
    table_number: int,
) -> str:
    """Build a stable filename for one detected table."""

    return (
        f"{source_id.lower()}_"
        f"page_{page_number:03d}_"
        f"table_{table_number:02d}.md"
    )


def extract_page_tables(
    source_id: str,
    title: str,
    page: pymupdf.Page,
    page_number: int,
) -> dict[str, Any]:
    """Detect and save all tables found on one PDF page."""

    result: dict[str, Any] = {
        "source_id": source_id,
        "title": title,
        "pdf_page_number": page_number,
        "detected_table_count": 0,
        "table_shapes": "",
        "table_bounding_boxes": "",
        "markdown_files": "",
        "detection_error": "",
        "manual_review": "",
    }

    try:
        table_finder = page.find_tables()
        tables = list(table_finder.tables)

        shapes: list[str] = []
        bounding_boxes: list[str] = []
        markdown_files: list[str] = []

        for table_index, table in enumerate(tables, start=1):
            filename = safe_table_filename(
                source_id=source_id,
                page_number=page_number,
                table_number=table_index,
            )
            output_path = TABLE_OUTPUT_DIR / filename

            # The table content must be copied while the Page object
            # is still available.
            markdown = table.to_markdown(
                clean=False,
                fill_empty=True,
            )

            document_text = "\n".join(
                [
                    f"# Detected Table",
                    "",
                    f"- Source ID: {source_id}",
                    f"- Document: {title}",
                    f"- PDF page: {page_number}",
                    f"- Table number on page: {table_index}",
                    f"- Rows: {table.row_count}",
                    f"- Columns: {table.col_count}",
                    f"- Bounding box: {tuple(round(x, 2) for x in table.bbox)}",
                    "",
                    "## Extracted content",
                    "",
                    markdown.strip(),
                    "",
                ]
            )

            output_path.write_text(
                document_text,
                encoding="utf-8",
            )

            shapes.append(
                f"{table.row_count}x{table.col_count}"
            )
            bounding_boxes.append(
                str(tuple(round(x, 2) for x in table.bbox))
            )
            markdown_files.append(filename)

        result.update(
            {
                "detected_table_count": len(tables),
                "table_shapes": "; ".join(shapes),
                "table_bounding_boxes": "; ".join(bounding_boxes),
                "markdown_files": "; ".join(markdown_files),
            }
        )

    except Exception as exc:
        result["detection_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


def main() -> None:
    """Run table detection over the entire included corpus."""

    TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_rows = read_included_sources()
    audit_rows: list[dict[str, Any]] = []

    print("\nAutomatic table detection audit")
    print("=" * 100)

    total_detected_tables = 0

    for source_row in source_rows:
        source_id = source_row["source_id"].strip()
        title = source_row["title"].strip()
        local_filename = source_row["local_filename"].strip()

        pdf_path = CORPUS_DIR / local_filename

        if not pdf_path.exists():
            print(f"{source_id}: PDF not found: {local_filename}")
            continue

        source_table_count = 0
        source_table_pages: list[int] = []

        with pymupdf.open(pdf_path) as document:
            for page_index in range(document.page_count):
                page_number = page_index + 1

                result = extract_page_tables(
                    source_id=source_id,
                    title=title,
                    page=document[page_index],
                    page_number=page_number,
                )

                audit_rows.append(result)

                detected_count = int(
                    result["detected_table_count"]
                )

                if detected_count > 0:
                    source_table_count += detected_count
                    source_table_pages.append(page_number)

        total_detected_tables += source_table_count

        print(
            f"{source_id}: "
            f"tables={source_table_count} | "
            f"pages={source_table_pages or 'none'}"
        )

    fieldnames = [
        "source_id",
        "title",
        "pdf_page_number",
        "detected_table_count",
        "table_shapes",
        "table_bounding_boxes",
        "markdown_files",
        "detection_error",
        "manual_review",
    ]

    with AUDIT_OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    print("=" * 100)
    print(f"Total detected tables: {total_detected_tables}")
    print(f"Audit CSV: {AUDIT_OUTPUT_PATH}")
    print(f"Extracted Markdown tables: {TABLE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()