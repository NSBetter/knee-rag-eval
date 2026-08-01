"""Validate human-reviewed manual table evidence for the MVP corpus.

The source evidence CSV is stored under data/processed/ and is not
committed to the public repository. This script writes a public audit
report containing metadata, text length, and text hash, but no guideline
text.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]

EVIDENCE_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "manual_tables"
    / "manual_table_evidence.csv"
)

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

PUBLIC_OUTPUT_PATH = (
    ROOT_DIR
    / "docs"
    / "manual_table_evidence_audit.csv"
)


REQUIRED_COLUMNS = [
    "record_id",
    "source_id",
    "node_id",
    "pdf_page",
    "table_id",
    "table_title",
    "record_type",
    "scope",
    "display_title",
    "text",
    "include_status",
    "extraction_mode",
    "review_status",
    "verification_notes",
]


EXPECTED_RECORD_TERMS = {
    "SRC001-T002-KNEE-SYMPTOMS": [
        "关节疼痛",
        "关节活动受限",
        "关节僵硬",
    ],
    "SRC001-T002-KNEE-SIGNS": [
        "关节压痛",
        "关节畸形",
        "关节肿大",
        "骨摩擦音（感）",
        "肌肉萎缩",
        "步态异常",
    ],
    "SRC001-T003-KNEE-DIAGNOSIS": [
        "近一个月内反复膝痛",
        "年龄≥50岁",
        "晨僵≤30分钟",
        "活动时有骨擦音（感）",
        "X线片",
        "满足第1条",
        "任意2条",
    ],
    "SRC001-T003-IMAGING-NOTES": [
        "首选X线检查",
        "MRI",
        "Kellgren-Lawrence",
        "超声",
    ],
}


EXPECTED_RECORD_IDS = set(
    EXPECTED_RECORD_TERMS
)


def read_csv(
    path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    """Read a UTF-8 CSV and normalize empty cell values."""

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        fieldnames = list(
            reader.fieldnames or []
        )

        rows = [
            {
                key: (value or "").strip()
                for key, value in row.items()
            }
            for row in reader
        ]

    return rows, fieldnames


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write public validation records."""

    if not rows:
        raise ValueError(
            "No audit rows were generated."
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


def text_sha256(
    text: str,
) -> str:
    """Return a stable SHA-256 hash for evidence text."""

    normalized = " ".join(
        text.split()
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def parse_page(
    raw_value: str,
    record_id: str,
    issues: list[str],
) -> int | None:
    """Parse a PDF page number without stopping the full audit."""

    try:
        page = int(raw_value)

    except ValueError:
        issues.append(
            "invalid_pdf_page"
        )
        return None

    if page < 1:
        issues.append(
            "invalid_pdf_page"
        )
        return None

    return page


def main() -> None:
    """Validate manual table evidence and create a public audit."""

    evidence_rows, evidence_columns = read_csv(
        EVIDENCE_PATH
    )

    if not evidence_rows:
        raise ValueError(
            "The manual table evidence CSV is empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in evidence_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required evidence columns: "
            f"{missing_columns}"
        )

    structure_rows, _ = read_csv(
        STRUCTURE_PATH
    )

    boundary_rows, _ = read_csv(
        BOUNDARY_PATH
    )

    structure_map: dict[
        str,
        dict[str, str],
    ] = {}

    duplicate_structure_nodes: set[str] = set()

    for row in structure_rows:
        node_id = row["node_id"]

        if node_id in structure_map:
            duplicate_structure_nodes.add(
                node_id
            )

        structure_map[node_id] = row

    if duplicate_structure_nodes:
        raise ValueError(
            "Duplicate node IDs in structure map: "
            f"{sorted(duplicate_structure_nodes)}"
        )

    boundaries: dict[
        str,
        tuple[int, int],
    ] = {}

    for row in boundary_rows:
        boundaries[row["source_id"]] = (
            int(row["body_start_page"]),
            int(row["body_end_page"]),
        )

    record_counts = Counter(
        row["record_id"]
        for row in evidence_rows
    )

    text_hash_counts = Counter(
        text_sha256(row["text"])
        for row in evidence_rows
        if row["text"]
    )

    actual_record_ids = {
        row["record_id"]
        for row in evidence_rows
    }

    missing_expected_records = (
        EXPECTED_RECORD_IDS
        - actual_record_ids
    )

    audit_rows: list[
        dict[str, Any]
    ] = []

    print(
        "\nManual table evidence validation"
    )
    print("=" * 100)

    for row in evidence_rows:
        issues: list[str] = []
        warnings: list[str] = []

        record_id = row["record_id"]
        source_id = row["source_id"]
        node_id = row["node_id"]
        text = row["text"]

        pdf_page = parse_page(
            raw_value=row["pdf_page"],
            record_id=record_id,
            issues=issues,
        )

        if not record_id:
            issues.append(
                "empty_record_id"
            )

        if record_counts[record_id] > 1:
            issues.append(
                "duplicate_record_id"
            )

        if source_id != "SRC001":
            issues.append(
                "unexpected_source_for_mvp_v1"
            )

        if not node_id:
            issues.append(
                "empty_node_id"
            )

        structure_node = structure_map.get(
            node_id
        )

        structure_node_status = (
            "not_found"
        )

        if structure_node is None:
            issues.append(
                "structure_node_not_found"
            )

        else:
            structure_node_status = "found"

            if (
                structure_node["source_id"]
                != source_id
            ):
                issues.append(
                    "structure_source_mismatch"
                )

            if (
                structure_node["include_status"]
                != "included"
            ):
                issues.append(
                    "structure_node_not_included"
                )

            if (
                structure_node["extraction_mode"]
                not in {
                    "manual_table",
                    "hybrid",
                }
            ):
                issues.append(
                    "structure_node_not_manual_or_hybrid"
                )

            try:
                structure_page = int(
                    structure_node["pdf_page"]
                )

                if (
                    pdf_page is not None
                    and structure_page
                    != pdf_page
                ):
                    warnings.append(
                        "evidence_page_differs_from_"
                        "structure_start_page"
                    )

            except ValueError:
                issues.append(
                    "invalid_structure_pdf_page"
                )

        if source_id not in boundaries:
            issues.append(
                "document_boundary_not_found"
            )

        elif pdf_page is not None:
            body_start, body_end = (
                boundaries[source_id]
            )

            if not (
                body_start
                <= pdf_page
                <= body_end
            ):
                issues.append(
                    "evidence_outside_document_body"
                )

        if row["include_status"] != "included":
            issues.append(
                "evidence_not_included"
            )

        if (
            row["extraction_mode"]
            != "manual_table"
        ):
            issues.append(
                "invalid_evidence_extraction_mode"
            )

        if (
            row["review_status"]
            != "human_verified"
        ):
            issues.append(
                "evidence_not_human_verified"
            )

        if (
            row["record_type"]
            != "manual_table_chunk"
        ):
            issues.append(
                "invalid_record_type"
            )

        if not row["table_id"]:
            issues.append(
                "empty_table_id"
            )

        if not row["table_title"]:
            issues.append(
                "empty_table_title"
            )

        if not row["display_title"]:
            issues.append(
                "empty_display_title"
            )

        if not row["scope"]:
            issues.append(
                "empty_scope"
            )

        if not text:
            issues.append(
                "empty_text"
            )

        elif len(text) < 40:
            warnings.append(
                "text_unusually_short"
            )

        if (
            text
            and text_hash_counts[
                text_sha256(text)
            ] > 1
        ):
            issues.append(
                "duplicate_evidence_text"
            )

        required_terms = (
            EXPECTED_RECORD_TERMS.get(
                record_id
            )
        )

        if required_terms is None:
            warnings.append(
                "record_not_in_mvp_v1_expectation"
            )

        else:
            missing_terms = [
                term
                for term in required_terms
                if term not in text
            ]

            if missing_terms:
                issues.append(
                    "missing_required_content:"
                    + "|".join(missing_terms)
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

        audit_row = {
            "record_id": record_id,
            "source_id": source_id,
            "node_id": node_id,
            "pdf_page": (
                pdf_page
                if pdf_page is not None
                else row["pdf_page"]
            ),
            "table_id": row["table_id"],
            "record_type": row["record_type"],
            "include_status": row["include_status"],
            "extraction_mode": row[
                "extraction_mode"
            ],
            "review_status": row[
                "review_status"
            ],
            "structure_node_status":
                structure_node_status,
            "text_length": len(text),
            "text_sha256": (
                text_sha256(text)
                if text
                else ""
            ),
            "validation_status":
                overall_status,
            "issues": issue_text,
        }

        audit_rows.append(
            audit_row
        )

        if overall_status != "pass":
            print(
                f"{record_id} | "
                f"{overall_status} | "
                f"{issue_text}"
            )

    if missing_expected_records:
        print(
            "Missing expected MVP records: "
            f"{sorted(missing_expected_records)}"
        )

    write_csv(
        path=PUBLIC_OUTPUT_PATH,
        rows=audit_rows,
    )

    status_counts = Counter(
        row["validation_status"]
        for row in audit_rows
    )

    print("-" * 100)
    print(
        f"Rows validated: "
        f"{len(audit_rows)}"
    )
    print(
        f"Status counts: "
        f"{dict(status_counts)}"
    )
    print(
        f"Record IDs unique: "
        f"{all(count == 1 for count in record_counts.values())}"
    )
    print(
        f"Missing expected records: "
        f"{len(missing_expected_records)}"
    )
    print(
        f"Public audit: "
        f"{PUBLIC_OUTPUT_PATH}"
    )
    print("=" * 100)

    has_errors = any(
        row["validation_status"] == "error"
        for row in audit_rows
    )

    if missing_expected_records:
        has_errors = True

    if has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
