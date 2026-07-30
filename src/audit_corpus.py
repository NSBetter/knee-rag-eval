"""Audit local guideline PDFs before ingestion into the RAG corpus.

This script checks:

1. Whether every registered PDF exists.
2. File size and SHA-256 hash.
3. Number of pages.
4. Whether text can be directly extracted.
5. Whether the document may require OCR.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import pymupdf


ROOT_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT_DIR / "docs" / "source_registry.csv"
CORPUS_DIR = ROOT_DIR / "data" / "corpus"
OUTPUT_PATH = ROOT_DIR / "docs" / "source_audit.csv"


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def classify_text_extractability(
    page_count: int,
    nonempty_pages: int,
    total_characters: int,
) -> str:
    """Classify whether a PDF contains directly extractable text.

    These thresholds are engineering heuristics for initial screening.
    They are not medical or publishing quality standards.
    """

    if page_count <= 0:
        return "invalid"

    nonempty_ratio = nonempty_pages / page_count
    average_characters = total_characters / page_count

    if total_characters < 500 or nonempty_ratio < 0.30:
        return "ocr_required"

    if average_characters < 200 or nonempty_ratio < 0.70:
        return "partial"

    return "ok"


def audit_pdf(registry_row: dict[str, str]) -> dict[str, Any]:
    """Audit one registered PDF and return the result."""

    result: dict[str, Any] = dict(registry_row)

    local_filename = registry_row["local_filename"].strip()
    pdf_path = CORPUS_DIR / local_filename

    result.update(
        {
            "file_exists": False,
            "file_size_mb": "",
            "sha256": "",
            "pdf_page_count": "",
            "pdf_metadata_title": "",
            "nonempty_text_pages": "",
            "extractable_text_characters": "",
            "average_characters_per_page": "",
            "text_status": "missing",
            "audit_error": "",
        }
    )

    if not pdf_path.exists():
        result["audit_error"] = f"File not found: {pdf_path}"
        return result

    result["file_exists"] = True
    result["file_size_mb"] = round(pdf_path.stat().st_size / 1024 / 1024, 3)
    result["sha256"] = calculate_sha256(pdf_path)

    document = None

    try:
        document = pymupdf.open(pdf_path)

        if document.needs_pass:
            result["text_status"] = "encrypted"
            result["audit_error"] = "PDF requires a password."
            return result

        page_count = document.page_count
        metadata = document.metadata or {}

        total_characters = 0
        nonempty_pages = 0

        for page in document:
            page_text = page.get_text("text").strip()
            character_count = len(page_text)

            total_characters += character_count

            if character_count >= 20:
                nonempty_pages += 1

        average_characters = (
            round(total_characters / page_count, 1)
            if page_count > 0
            else 0
        )

        result.update(
            {
                "pdf_page_count": page_count,
                "pdf_metadata_title": metadata.get("title", ""),
                "nonempty_text_pages": nonempty_pages,
                "extractable_text_characters": total_characters,
                "average_characters_per_page": average_characters,
                "text_status": classify_text_extractability(
                    page_count=page_count,
                    nonempty_pages=nonempty_pages,
                    total_characters=total_characters,
                ),
            }
        )

    except Exception as exc:
        result["text_status"] = "error"
        result["audit_error"] = f"{type(exc).__name__}: {exc}"

    finally:
        if document is not None:
            document.close()

    return result


def main() -> None:
    """Read the registry, audit all PDFs, and save a CSV report."""

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

    if not registry_rows:
        raise ValueError("The source registry contains no records.")

    audit_rows = [audit_pdf(row) for row in registry_rows]

    audit_fields = [
        *registry_rows[0].keys(),
        "file_exists",
        "file_size_mb",
        "sha256",
        "pdf_page_count",
        "pdf_metadata_title",
        "nonempty_text_pages",
        "extractable_text_characters",
        "average_characters_per_page",
        "text_status",
        "audit_error",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=audit_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    print("\nCorpus audit results")
    print("=" * 80)

    for row in audit_rows:
        print(
            f"{row['source_id']}: "
            f"status={row['text_status']} | "
            f"pages={row['pdf_page_count']} | "
            f"characters={row['extractable_text_characters']} | "
            f"file={row['local_filename']}"
        )

        if row["audit_error"]:
            print(f"  error: {row['audit_error']}")

    print("=" * 80)
    print(f"Audit report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()