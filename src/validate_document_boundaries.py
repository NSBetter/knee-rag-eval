"""Validate document-level body and tail boundaries.

This script verifies that:

1. Configured PDF page numbers exist.
2. Start and stop markers occur on the configured pages.
3. Markers are unique enough for deterministic trimming.
4. Body, end, and reference page numbers are logically consistent.

A public summary containing no guideline excerpts is written to docs/.
A local detailed report containing short text snippets is written under
data/processed/ and remains excluded from Git.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]

BOUNDARY_PATH = (
    ROOT_DIR / "docs" / "document_boundaries.csv"
)

PAGE_DIR = (
    ROOT_DIR / "data" / "processed" / "pages"
)

PUBLIC_SUMMARY_PATH = (
    ROOT_DIR / "docs" / "document_boundary_validation.csv"
)

LOCAL_DETAIL_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "reviews"
    / "document_boundary_validation_details.csv"
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def read_page_records() -> dict[
    str,
    dict[int, dict[str, Any]],
]:
    """Read extracted page JSONL records by source and page."""

    records: dict[
        str,
        dict[int, dict[str, Any]],
    ] = defaultdict(dict)

    for jsonl_path in sorted(
        PAGE_DIR.glob("*_pages.jsonl")
    ):
        with jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {jsonl_path}, "
                        f"line {line_number}: {exc}"
                    ) from exc

                source_id = str(row["source_id"])
                page_number = int(
                    row["pdf_page_number"]
                )

                if page_number in records[source_id]:
                    raise ValueError(
                        f"Duplicate page record: "
                        f"{source_id} page {page_number}"
                    )

                records[source_id][page_number] = row

    if not records:
        raise FileNotFoundError(
            f"No page JSONL files found under {PAGE_DIR}"
        )

    return dict(records)


def marker_status(
    text: str,
    marker: str,
) -> tuple[str, int]:
    """Classify how often an exact marker occurs."""

    if not marker:
        return "empty_marker", 0

    count = text.count(marker)

    if count == 0:
        return "not_found", 0

    if count == 1:
        return "found_unique", 1

    return "found_multiple", count


def marker_snippet(
    text: str,
    marker: str,
    context_characters: int = 100,
) -> str:
    """Return a short local context around the first marker."""

    if not marker:
        return ""

    position = text.find(marker)

    if position < 0:
        return ""

    start = max(
        0,
        position - context_characters,
    )
    end = min(
        len(text),
        position + len(marker) + context_characters,
    )

    return text[start:end].replace(
        "\n",
        " ",
    ).strip()


def validate_row(
    row: dict[str, str],
    page_records: dict[
        str,
        dict[int, dict[str, Any]],
    ],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one source boundary record."""

    source_id = row["source_id"].strip()

    body_start_page = int(
        row["body_start_page"]
    )
    reference_start_page = int(
        row["reference_start_page"]
    )
    body_end_page = int(
        row["body_end_page"]
    )

    body_start_marker = row[
        "body_start_marker"
    ].strip()

    reference_start_marker = row[
        "reference_start_marker"
    ].strip()

    errors: list[str] = []
    warnings: list[str] = []

    source_pages = page_records.get(source_id)

    if source_pages is None:
        errors.append(
            "Source not found in extracted page records."
        )

        public_result = {
            "source_id": source_id,
            "page_range_status": "error",
            "body_start_marker_status": "not_checked",
            "body_start_marker_count": 0,
            "tail_marker_status": "not_checked",
            "tail_marker_count": 0,
            "boundary_mode": "not_checked",
            "overall_status": "error",
            "issues": "; ".join(errors),
        }

        return public_result, dict(public_result)

    maximum_page = max(source_pages)

    for field_name, page_number in [
        ("body_start_page", body_start_page),
        (
            "reference_start_page",
            reference_start_page,
        ),
        ("body_end_page", body_end_page),
    ]:
        if not 1 <= page_number <= maximum_page:
            errors.append(
                f"{field_name}={page_number} is outside "
                f"the PDF range 1-{maximum_page}."
            )

    if body_start_page > body_end_page:
        errors.append(
            "body_start_page is after body_end_page."
        )

    if reference_start_page < body_start_page:
        errors.append(
            "reference_start_page is before "
            "body_start_page."
        )

    if reference_start_page < body_end_page:
        warnings.append(
            "Tail marker starts before the configured "
            "last body page; verify whether this is intended."
        )

    start_text = str(
        source_pages.get(
            body_start_page,
            {},
        ).get("text", "")
    )

    tail_text = str(
        source_pages.get(
            reference_start_page,
            {},
        ).get("text", "")
    )

    start_status, start_count = marker_status(
        text=start_text,
        marker=body_start_marker,
    )

    tail_status, tail_count = marker_status(
        text=tail_text,
        marker=reference_start_marker,
    )

    if start_status == "not_found":
        errors.append(
            "Body start marker was not found on its "
            "configured page."
        )

    elif start_status == "found_multiple":
        warnings.append(
            "Body start marker occurs multiple times."
        )

    if tail_status == "not_found":
        errors.append(
            "Tail marker was not found on its "
            "configured page."
        )

    elif tail_status == "found_multiple":
        warnings.append(
            "Tail marker occurs multiple times."
        )

    if reference_start_page <= body_end_page:
        boundary_mode = "trim_tail_page_at_marker"
    else:
        boundary_mode = "exclude_pages_after_body_end"

    if errors:
        overall_status = "error"
    elif warnings:
        overall_status = "review"
    else:
        overall_status = "pass"

    issue_text = "; ".join(
        [*errors, *warnings]
    )

    public_result = {
        "source_id": source_id,
        "maximum_pdf_page": maximum_page,
        "body_start_page": body_start_page,
        "body_end_page": body_end_page,
        "reference_start_page":
            reference_start_page,
        "page_range_status":
            "error" if errors else "valid",
        "body_start_marker_status":
            start_status,
        "body_start_marker_count":
            start_count,
        "tail_marker_status":
            tail_status,
        "tail_marker_count":
            tail_count,
        "boundary_mode": boundary_mode,
        "overall_status": overall_status,
        "issues": issue_text,
    }

    local_result = {
        **public_result,
        "body_start_marker":
            body_start_marker,
        "body_start_context":
            marker_snippet(
                text=start_text,
                marker=body_start_marker,
            ),
        "tail_marker":
            reference_start_marker,
        "tail_context":
            marker_snippet(
                text=tail_text,
                marker=reference_start_marker,
            ),
        "notes": row.get(
            "notes",
            "",
        ).strip(),
    }

    return public_result, local_result


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write rows to CSV."""

    if not rows:
        raise ValueError(
            f"No rows to write: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Validate all configured document boundaries."""

    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError(
            f"Boundary configuration not found: "
            f"{BOUNDARY_PATH}"
        )

    boundary_rows = read_csv_rows(
        BOUNDARY_PATH
    )
    page_records = read_page_records()

    public_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []

    print("\nDocument boundary validation")
    print("=" * 100)

    for row in boundary_rows:
        public_result, local_result = validate_row(
            row=row,
            page_records=page_records,
        )

        public_rows.append(public_result)
        local_rows.append(local_result)

        print(
            f"{public_result['source_id']}: "
            f"overall={public_result['overall_status']} | "
            f"start="
            f"{public_result['body_start_marker_status']} "
            f"({public_result['body_start_marker_count']}) | "
            f"tail="
            f"{public_result['tail_marker_status']} "
            f"({public_result['tail_marker_count']}) | "
            f"mode={public_result['boundary_mode']}"
        )

        if public_result["issues"]:
            print(
                f"  issues: "
                f"{public_result['issues']}"
            )

    write_csv(
        path=PUBLIC_SUMMARY_PATH,
        rows=public_rows,
    )

    write_csv(
        path=LOCAL_DETAIL_PATH,
        rows=local_rows,
    )

    print("=" * 100)
    print(
        f"Public summary: {PUBLIC_SUMMARY_PATH}"
    )
    print(
        f"Local details: {LOCAL_DETAIL_PATH}"
    )


if __name__ == "__main__":
    main()