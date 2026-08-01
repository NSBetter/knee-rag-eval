"""Validate the manually curated document structure map.

Validation covers:

1. Required columns and allowed field values.
2. Unique node IDs and document-level order values.
3. Parent-child hierarchy and node levels.
4. Inclusion status and extraction mode consistency.
5. Marker existence on the configured PDF page.
6. Selection of a specific marker occurrence when the same text
   appears multiple times on one page.
7. Marker ordering among nodes located on the same PDF page.

A public validation report without guideline excerpts is written to docs/.
A local detailed report with short marker contexts is written under
data/processed/ and remains excluded from Git.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]

STRUCTURE_PATH = (
    ROOT_DIR
    / "docs"
    / "document_structure_map.csv"
)

BOUNDARY_PATH = (
    ROOT_DIR
    / "docs"
    / "document_boundaries.csv"
)

PAGE_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "pages"
)

PUBLIC_OUTPUT_PATH = (
    ROOT_DIR
    / "docs"
    / "document_structure_validation.csv"
)

LOCAL_OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "reviews"
    / "document_structure_validation_details.csv"
)


REQUIRED_COLUMNS = [
    "source_id",
    "node_id",
    "parent_id",
    "order",
    "node_type",
    "level",
    "pdf_page",
    "marker_text",
    "marker_occurrence",
    "display_title",
    "include_status",
    "notes",
    "extraction_mode",
]


ALLOWED_NODE_TYPES = {
    "section",
    "subsection",
    "question",
    "recommendation",
}


ALLOWED_INCLUDE_STATUS = {
    "included",
    "excluded",
}


ALLOWED_EXTRACTION_MODES = {
    "auto_text",
    "manual_table",
    "hybrid",
    "exclude",
}


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    """Read a UTF-8 CSV and normalize empty values."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        return [
            {
                key: (value or "").strip()
                for key, value in row.items()
            }
            for row in reader
        ]


def read_pages() -> dict[str, dict[int, str]]:
    """Read extracted page text indexed by source and PDF page."""

    pages: dict[
        str,
        dict[int, str],
    ] = defaultdict(dict)

    for path in sorted(
        PAGE_DIR.glob("*_pages.jsonl")
    ):
        with path.open(
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
                        f"Invalid JSON in {path}, "
                        f"line {line_number}: {exc}"
                    ) from exc

                source_id = str(
                    row["source_id"]
                )

                page_number = int(
                    row["pdf_page_number"]
                )

                if page_number in pages[source_id]:
                    raise ValueError(
                        f"Duplicate extracted page: "
                        f"{source_id} page {page_number}"
                    )

                pages[source_id][page_number] = str(
                    row.get("text", "")
                )

    if not pages:
        raise FileNotFoundError(
            f"No extracted page JSONL files found "
            f"under {PAGE_DIR}"
        )

    return dict(pages)


def read_boundaries() -> dict[
    str,
    tuple[int, int],
]:
    """Read validated body page ranges for each document."""

    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError(
            f"Document boundary file not found: "
            f"{BOUNDARY_PATH}"
        )

    boundaries: dict[
        str,
        tuple[int, int],
    ] = {}

    for row in read_csv(BOUNDARY_PATH):
        source_id = row["source_id"]

        if source_id in boundaries:
            raise ValueError(
                f"Duplicate document boundary: "
                f"{source_id}"
            )

        boundaries[source_id] = (
            int(row["body_start_page"]),
            int(row["body_end_page"]),
        )

    return boundaries


def find_nth_occurrence(
    text: str,
    marker: str,
    occurrence: int,
) -> int:
    """Return the position of the requested marker occurrence.

    The occurrence number is one-based:

    - occurrence=1 means the first occurrence.
    - occurrence=2 means the second occurrence.

    The search uses non-overlapping occurrences, consistent with
    Python's str.count() behavior.

    Returns:
        Character position of the requested occurrence, or -1 when
        it cannot be found.
    """

    if occurrence < 1:
        return -1

    if not marker:
        return -1

    search_start = 0
    position = -1

    for _ in range(occurrence):
        position = text.find(
            marker,
            search_start,
        )

        if position < 0:
            return -1

        search_start = (
            position
            + len(marker)
        )

    return position


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write validation records to a UTF-8 CSV."""

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
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_required_integer(
    raw_value: str,
    field_name: str,
    node_id: str,
) -> int:
    """Parse a required integer field with a clear error message."""

    try:
        return int(raw_value)

    except ValueError as exc:
        raise ValueError(
            f"Invalid integer in {field_name}: "
            f"node_id={node_id!r}, "
            f"value={raw_value!r}"
        ) from exc


def main() -> None:
    """Validate structure, hierarchy, markers, and document order."""

    if not STRUCTURE_PATH.exists():
        raise FileNotFoundError(
            f"Structure map not found: "
            f"{STRUCTURE_PATH}"
        )

    rows = read_csv(
        STRUCTURE_PATH
    )

    if not rows:
        raise ValueError(
            "The structure map is empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in rows[0]
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    pages = read_pages()
    boundaries = read_boundaries()

    node_map = {
        row["node_id"]: row
        for row in rows
    }

    node_id_counts: dict[
        str,
        int,
    ] = defaultdict(int)

    order_counts: dict[
        tuple[str, int],
        int,
    ] = defaultdict(int)

    parsed_orders: dict[
        str,
        int,
    ] = {}

    for row in rows:
        node_id = row["node_id"]

        order = parse_required_integer(
            raw_value=row["order"],
            field_name="order",
            node_id=node_id,
        )

        parsed_orders[node_id] = order

        node_id_counts[node_id] += 1

        order_counts[
            (
                row["source_id"],
                order,
            )
        ] += 1

    public_rows: list[
        dict[str, Any]
    ] = []

    local_rows: list[
        dict[str, Any]
    ] = []

    print(
        "\nDocument structure validation"
    )
    print("=" * 100)

    for row in rows:
        issues: list[str] = []
        warnings: list[str] = []

        source_id = row["source_id"]
        node_id = row["node_id"]
        parent_id = row["parent_id"]

        order = parse_required_integer(
            raw_value=row["order"],
            field_name="order",
            node_id=node_id,
        )

        level = parse_required_integer(
            raw_value=row["level"],
            field_name="level",
            node_id=node_id,
        )

        pdf_page = parse_required_integer(
            raw_value=row["pdf_page"],
            field_name="pdf_page",
            node_id=node_id,
        )

        marker = row["marker_text"]

        raw_marker_occurrence = row[
            "marker_occurrence"
        ].strip()

        marker_occurrence_explicit = bool(
            raw_marker_occurrence
        )

        try:
            marker_occurrence = int(
                raw_marker_occurrence
                or "1"
            )

        except ValueError:
            marker_occurrence = 1

            issues.append(
                "invalid_marker_occurrence"
            )

        if marker_occurrence < 1:
            issues.append(
                "invalid_marker_occurrence"
            )

        if not source_id:
            issues.append(
                "empty_source_id"
            )

        if not node_id:
            issues.append(
                "empty_node_id"
            )

        if not marker:
            issues.append(
                "empty_marker_text"
            )

        if node_id_counts[node_id] > 1:
            issues.append(
                "duplicate_node_id"
            )

        if order_counts[
            (
                source_id,
                order,
            )
        ] > 1:
            issues.append(
                "duplicate_order_within_source"
            )

        if (
            row["node_type"]
            not in ALLOWED_NODE_TYPES
        ):
            issues.append(
                "invalid_node_type"
            )

        if (
            row["include_status"]
            not in ALLOWED_INCLUDE_STATUS
        ):
            issues.append(
                "invalid_include_status"
            )

        if (
            row["extraction_mode"]
            not in ALLOWED_EXTRACTION_MODES
        ):
            issues.append(
                "invalid_extraction_mode"
            )

        if (
            row["include_status"]
            == "excluded"
            and row["extraction_mode"]
            != "exclude"
        ):
            warnings.append(
                "excluded_node_should_use_"
                "extraction_mode_exclude"
            )

        if (
            row["include_status"]
            == "included"
            and row["extraction_mode"]
            == "exclude"
        ):
            issues.append(
                "included_node_cannot_use_"
                "extraction_mode_exclude"
            )

        if parent_id:
            parent = node_map.get(
                parent_id
            )

            if parent is None:
                issues.append(
                    "parent_not_found"
                )

            else:
                parent_level = (
                    parse_required_integer(
                        raw_value=parent["level"],
                        field_name="parent level",
                        node_id=parent_id,
                    )
                )

                parent_order = (
                    parsed_orders[parent_id]
                )

                if (
                    parent["source_id"]
                    != source_id
                ):
                    issues.append(
                        "parent_from_different_source"
                    )

                if parent_level != level - 1:
                    issues.append(
                        "parent_level_mismatch"
                    )

                if parent_order >= order:
                    issues.append(
                        "parent_not_before_child"
                    )

                if (
                    parent["include_status"]
                    == "excluded"
                    and row["include_status"]
                    == "included"
                ):
                    warnings.append(
                        "included_child_under_"
                        "excluded_parent"
                    )

                if (
                    row["node_type"]
                    == "recommendation"
                    and parent["node_type"]
                    != "question"
                ):
                    issues.append(
                        "recommendation_parent_"
                        "is_not_question"
                    )

        elif level != 1:
            issues.append(
                "non_root_node_missing_parent"
            )

        if (
            row["node_type"]
            == "section"
            and level != 1
        ):
            warnings.append(
                "section_expected_at_level_1"
            )

        if (
            row["node_type"]
            == "recommendation"
            and level < 2
        ):
            issues.append(
                "invalid_recommendation_level"
            )

        page_text = pages.get(
            source_id,
            {},
        ).get(
            pdf_page
        )

        marker_status = "not_checked"
        marker_count = 0
        marker_position = -1

        if page_text is None:
            issues.append(
                "pdf_page_not_found"
            )

            marker_status = (
                "page_not_found"
            )

        elif not marker:
            marker_status = (
                "empty_marker"
            )

        else:
            marker_count = (
                page_text.count(marker)
            )

            if marker_count == 0:
                issues.append(
                    "marker_not_found"
                )

                marker_status = (
                    "not_found"
                )

            elif (
                marker_occurrence
                > marker_count
            ):
                issues.append(
                    "marker_occurrence_"
                    "out_of_range"
                )

                marker_status = (
                    "occurrence_out_of_range"
                )

            else:
                marker_position = (
                    find_nth_occurrence(
                        text=page_text,
                        marker=marker,
                        occurrence=(
                            marker_occurrence
                        ),
                    )
                )

                if marker_position < 0:
                    issues.append(
                        "marker_occurrence_"
                        "position_not_found"
                    )

                    marker_status = (
                        "not_found"
                    )

                elif marker_count == 1:
                    marker_status = (
                        "found_unique"
                    )

                elif (
                    marker_occurrence_explicit
                ):
                    marker_status = (
                        "found_selected"
                    )

                else:
                    warnings.append(
                        "marker_found_multiple_"
                        "without_occurrence"
                    )

                    marker_status = (
                        "found_multiple"
                    )

        if source_id in boundaries:
            body_start, body_end = (
                boundaries[source_id]
            )

            if not (
                body_start
                <= pdf_page
                <= body_end
            ):
                if (
                    row["include_status"]
                    == "included"
                ):
                    issues.append(
                        "included_node_outside_"
                        "document_body"
                    )

                else:
                    warnings.append(
                        "excluded_node_outside_"
                        "document_body"
                    )

        else:
            issues.append(
                "document_boundary_not_found"
            )

        overall_status = (
            "error"
            if issues
            else "review"
            if warnings
            else "pass"
        )

        issue_text = "; ".join(
            [
                *issues,
                *warnings,
            ]
        )

        public_result = {
            "source_id": source_id,
            "node_id": node_id,
            "order": order,
            "node_type": row["node_type"],
            "level": level,
            "pdf_page": pdf_page,
            "marker_occurrence":
                marker_occurrence,
            "marker_status":
                marker_status,
            "marker_count":
                marker_count,
            "overall_status":
                overall_status,
            "issues":
                issue_text,
        }

        context = ""

        if (
            marker_position >= 0
            and page_text is not None
        ):
            context_start = max(
                0,
                marker_position - 60,
            )

            context_end = min(
                len(page_text),
                marker_position
                + len(marker)
                + 100,
            )

            context = page_text[
                context_start:context_end
            ].replace(
                "\n",
                " ",
            )

        local_result = {
            **public_result,
            "parent_id":
                parent_id,
            "marker_text":
                marker,
            "display_title":
                row["display_title"],
            "extraction_mode":
                row["extraction_mode"],
            "include_status":
                row["include_status"],
            "marker_position":
                marker_position,
            "marker_context":
                context,
            "notes":
                row["notes"],
        }

        public_rows.append(
            public_result
        )

        local_rows.append(
            local_result
        )

        if overall_status != "pass":
            print(
                f"{source_id} | "
                f"{node_id} | "
                f"{overall_status} | "
                f"marker={marker_status} | "
                f"selected_occurrence="
                f"{marker_occurrence} | "
                f"{issue_text}"
            )

    # Check whether selected marker positions follow the manually
    # configured order on each PDF page.
    grouped_page_rows: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in local_rows:
        grouped_page_rows[
            (
                str(row["source_id"]),
                int(row["pdf_page"]),
            )
        ].append(row)

    ordering_warning_pages: list[
        tuple[str, int]
    ] = []

    for (
        source_id,
        pdf_page,
    ), page_rows in grouped_page_rows.items():
        valid_rows = [
            row
            for row in page_rows
            if int(
                row["marker_position"]
            ) >= 0
        ]

        valid_rows.sort(
            key=lambda row: int(
                row["order"]
            )
        )

        positions = [
            int(
                row["marker_position"]
            )
            for row in valid_rows
        ]

        if positions != sorted(
            positions
        ):
            ordering_warning_pages.append(
                (
                    source_id,
                    pdf_page,
                )
            )

    write_csv(
        path=PUBLIC_OUTPUT_PATH,
        rows=public_rows,
    )

    write_csv(
        path=LOCAL_OUTPUT_PATH,
        rows=local_rows,
    )

    status_counts: dict[
        str,
        int,
    ] = defaultdict(int)

    marker_status_counts: dict[
        str,
        int,
    ] = defaultdict(int)

    for row in public_rows:
        status_counts[
            str(row["overall_status"])
        ] += 1

        marker_status_counts[
            str(row["marker_status"])
        ] += 1

    print("-" * 100)
    print(
        f"Rows validated: "
        f"{len(public_rows)}"
    )
    print(
        f"Status counts: "
        f"{dict(status_counts)}"
    )
    print(
        f"Marker status counts: "
        f"{dict(marker_status_counts)}"
    )
    print(
        f"Pages with possible marker-order "
        f"problems: "
        f"{len(ordering_warning_pages)}"
    )

    if ordering_warning_pages:
        print(
            "Possible marker-order pages:"
        )

        for (
            source_id,
            pdf_page,
        ) in ordering_warning_pages:
            print(
                f"  - {source_id} "
                f"page {pdf_page}"
            )

    print(
        f"Public result: "
        f"{PUBLIC_OUTPUT_PATH}"
    )
    print(
        f"Local details: "
        f"{LOCAL_OUTPUT_PATH}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()