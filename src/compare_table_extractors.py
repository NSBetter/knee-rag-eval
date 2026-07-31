"""Compare table extraction methods on the guideline corpus.

Compared methods:

1. PyMuPDF find_tables() default strategy.
2. PyMuPDF find_tables(strategy="text").
3. PyMuPDF4LLM with PyMuPDF Layout.

The script saves page-level results for manual review. Detection counts
are engineering measurements and must be checked against the manually
labelled table pages.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pymupdf

# Importing layout before pymupdf4llm explicitly enables layout analysis
# for package versions that require this import order.
try:
    import pymupdf.layout  # noqa: F401
except ImportError:
    pass

import pymupdf4llm


ROOT_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT_DIR / "docs" / "source_registry.csv"
CORPUS_DIR = ROOT_DIR / "data" / "corpus"

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "table_extractor_comparison"
)

RESULT_PATH = (
    ROOT_DIR
    / "docs"
    / "table_extractor_comparison.csv"
)


def read_included_sources() -> list[dict[str, str]]:
    """Read sources marked as included."""

    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    included = [
        row
        for row in rows
        if row["include_status"].strip().lower() == "included"
    ]

    if not included:
        raise ValueError("No included sources found.")

    return included


def is_markdown_separator(line: str) -> bool:
    """Return whether a line is a Markdown table separator row."""

    stripped = line.strip().strip("|")

    if "|" not in stripped:
        return False

    cells = [
        cell.strip()
        for cell in stripped.split("|")
    ]

    return (
        len(cells) >= 2
        and all(
            re.fullmatch(r":?-{3,}:?", cell) is not None
            for cell in cells
        )
    )


def count_markdown_tables(markdown: str) -> int:
    """Estimate the number of Markdown tables.

    Each GitHub-style Markdown table normally contains one separator row.
    This is a heuristic and must be manually reviewed.
    """

    return sum(
        is_markdown_separator(line)
        for line in markdown.splitlines()
    )


def save_pymupdf_tables(
    tables: list[Any],
    source_id: str,
    page_number: int,
    method_name: str,
) -> list[str]:
    """Save detected PyMuPDF tables as Markdown."""

    method_dir = OUTPUT_DIR / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []

    for table_index, table in enumerate(tables, start=1):
        filename = (
            f"{source_id.lower()}_"
            f"page_{page_number:03d}_"
            f"table_{table_index:02d}.md"
        )

        output_path = method_dir / filename

        try:
            markdown = table.to_markdown(
                clean=False,
                fill_empty=True,
            )
        except Exception as exc:
            markdown = (
                "# Table extraction error\n\n"
                f"{type(exc).__name__}: {exc}\n"
            )

        output_path.write_text(
            markdown,
            encoding="utf-8",
        )

        saved_files.append(filename)

    return saved_files


def extract_document(
    source_row: dict[str, str],
) -> list[dict[str, Any]]:
    """Run all extraction methods on one document."""

    source_id = source_row["source_id"].strip()
    title = source_row["title"].strip()
    filename = source_row["local_filename"].strip()
    pdf_path = CORPUS_DIR / filename

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    layout_output_dir = OUTPUT_DIR / "pymupdf4llm_layout"
    layout_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    with pymupdf.open(pdf_path) as document:
        layout_chunks = pymupdf4llm.to_markdown(
            document,
            page_chunks=True,
            header=False,
            footer=False,
            use_ocr=False,
            force_text=True,
            write_images=False,
            embed_images=False,
            show_progress=False,
        )

        if not isinstance(layout_chunks, list):
            raise TypeError(
                "Expected page_chunks=True to return a list."
            )

        if len(layout_chunks) != document.page_count:
            raise ValueError(
                f"Page count mismatch for {source_id}: "
                f"PDF={document.page_count}, "
                f"layout_chunks={len(layout_chunks)}"
            )

        for page_index in range(document.page_count):
            page = document[page_index]
            page_number = page_index + 1

            default_error = ""
            text_error = ""

            try:
                default_tables = list(
                    page.find_tables().tables
                )
            except Exception as exc:
                default_tables = []
                default_error = (
                    f"{type(exc).__name__}: {exc}"
                )

            try:
                text_tables = list(
                    page.find_tables(
                        strategy="text",
                    ).tables
                )
            except Exception as exc:
                text_tables = []
                text_error = (
                    f"{type(exc).__name__}: {exc}"
                )

            default_files = save_pymupdf_tables(
                tables=default_tables,
                source_id=source_id,
                page_number=page_number,
                method_name="pymupdf_default",
            )

            text_files = save_pymupdf_tables(
                tables=text_tables,
                source_id=source_id,
                page_number=page_number,
                method_name="pymupdf_text",
            )

            layout_markdown = str(
                layout_chunks[page_index].get("text", "")
            )

            layout_filename = (
                f"{source_id.lower()}_"
                f"page_{page_number:03d}.md"
            )

            (
                layout_output_dir
                / layout_filename
            ).write_text(
                layout_markdown,
                encoding="utf-8",
            )

            rows.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "pdf_page_number": page_number,
                    "default_detected_count": len(
                        default_tables
                    ),
                    "default_markdown_files": "; ".join(
                        default_files
                    ),
                    "default_error": default_error,
                    "text_strategy_detected_count": len(
                        text_tables
                    ),
                    "text_strategy_markdown_files": "; ".join(
                        text_files
                    ),
                    "text_strategy_error": text_error,
                    "layout_markdown_table_markers":
                        count_markdown_tables(
                            layout_markdown
                        ),
                    "layout_markdown_file": layout_filename,
                    "manual_has_table": "",
                    "manual_best_method": "",
                    "manual_readability": "",
                    "review_notes": "",
                }
            )

    return rows


def main() -> None:
    """Run the table extractor comparison."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    print("\nTable extractor comparison")
    print("=" * 100)

    for source_row in read_included_sources():
        source_id = source_row["source_id"].strip()

        source_rows = extract_document(source_row)
        all_rows.extend(source_rows)

        default_total = sum(
            int(row["default_detected_count"])
            for row in source_rows
        )

        text_total = sum(
            int(row["text_strategy_detected_count"])
            for row in source_rows
        )

        layout_total = sum(
            int(row["layout_markdown_table_markers"])
            for row in source_rows
        )

        default_pages = [
            int(row["pdf_page_number"])
            for row in source_rows
            if int(row["default_detected_count"]) > 0
        ]

        text_pages = [
            int(row["pdf_page_number"])
            for row in source_rows
            if int(
                row["text_strategy_detected_count"]
            ) > 0
        ]

        layout_pages = [
            int(row["pdf_page_number"])
            for row in source_rows
            if int(
                row["layout_markdown_table_markers"]
            ) > 0
        ]

        print(
            f"{source_id}: "
            f"default={default_total} {default_pages or 'none'} | "
            f"text={text_total} {text_pages or 'none'} | "
            f"layout_md={layout_total} "
            f"{layout_pages or 'none'}"
        )

    fieldnames = [
        "source_id",
        "title",
        "pdf_page_number",
        "default_detected_count",
        "default_markdown_files",
        "default_error",
        "text_strategy_detected_count",
        "text_strategy_markdown_files",
        "text_strategy_error",
        "layout_markdown_table_markers",
        "layout_markdown_file",
        "manual_has_table",
        "manual_best_method",
        "manual_readability",
        "review_notes",
    ]

    with RESULT_PATH.open(
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
        writer.writerows(all_rows)

    print("=" * 100)
    print(f"Compared {len(all_rows)} PDF pages.")
    print(f"Result CSV: {RESULT_PATH}")
    print(f"Local Markdown output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()