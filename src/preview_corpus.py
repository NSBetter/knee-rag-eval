"""Generate human-readable text previews for guideline PDFs.

The purpose is to inspect extraction quality before chunking the corpus.
The generated previews remain local and are ignored by Git.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pymupdf


ROOT_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT_DIR / "docs" / "source_registry.csv"
CORPUS_DIR = ROOT_DIR / "data" / "corpus"
PREVIEW_DIR = ROOT_DIR / "data" / "processed" / "previews"


def select_preview_pages(page_count: int) -> list[int]:
    """Select representative zero-based page indexes."""

    candidates = [
        0,                  # 首页
        1,                  # 第二页
        page_count // 2,    # 中间页
        page_count - 1,     # 最后一页
    ]

    return sorted(
        {
            page_index
            for page_index in candidates
            if 0 <= page_index < page_count
        }
    )


def safe_filename(value: str) -> str:
    """Convert a source identifier to a safe lowercase filename."""

    return value.strip().lower().replace(" ", "_")


def generate_preview(
    source_id: str,
    title: str,
    pdf_path: Path,
) -> Path:
    """Extract representative pages and save them as local text."""

    with pymupdf.open(pdf_path) as document:
        preview_pages = select_preview_pages(document.page_count)

        sections = [
            f"Source ID: {source_id}",
            f"Title: {title}",
            f"Local file: {pdf_path.name}",
            f"Total PDF pages: {document.page_count}",
            f"Preview pages: {', '.join(str(page + 1) for page in preview_pages)}",
            "",
            "=" * 100,
            "",
        ]

        for page_index in preview_pages:
            page = document[page_index]

            # sort=True attempts to reconstruct a top-to-bottom,
            # left-to-right reading order.
            text = page.get_text("text", sort=True).strip()

            sections.extend(
                [
                    f"PDF PAGE {page_index + 1}",
                    "-" * 100,
                    text if text else "[NO EXTRACTABLE TEXT]",
                    "",
                    "=" * 100,
                    "",
                ]
            )

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PREVIEW_DIR / f"{safe_filename(source_id)}_preview.txt"
    output_path.write_text("\n".join(sections), encoding="utf-8")

    return output_path


def main() -> None:
    """Generate previews for every included document."""

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Source registry not found: {REGISTRY_PATH}"
        )

    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        registry_rows = list(csv.DictReader(file))

    included_rows = [
        row
        for row in registry_rows
        if row["include_status"].strip().lower() == "included"
    ]

    if not included_rows:
        raise ValueError("No included documents were found in the registry.")

    print("\nGenerating corpus previews")
    print("=" * 80)

    created_count = 0

    for row in included_rows:
        pdf_path = CORPUS_DIR / row["local_filename"].strip()

        if not pdf_path.exists():
            print(
                f"{row['source_id']}: skipped; "
                f"file not found: {pdf_path.name}"
            )
            continue

        output_path = generate_preview(
            source_id=row["source_id"],
            title=row["title"],
            pdf_path=pdf_path,
        )

        created_count += 1
        print(
            f"{row['source_id']}: "
            f"{pdf_path.name} -> {output_path.name}"
        )

    print("=" * 80)
    print(f"Created {created_count} preview files.")
    print(f"Preview directory: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()