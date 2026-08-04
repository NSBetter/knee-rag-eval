"""Validate retrieval benchmark labels against gold_v1_3."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "gold_corpus"
    / "gold_corpus_v1_3.csv"
)

BENCHMARK_PATH = (
    ROOT
    / "data"
    / "benchmark"
    / "retrieval_eval_v1.csv"
)

LOCAL_AUDIT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "reviews"
    / "retrieval_eval_v1_audit.csv"
)

PUBLIC_SUMMARY_PATH = (
    ROOT
    / "docs"
    / "retrieval_benchmark_v1_summary.csv"
)

REQUIRED_COLUMNS = [
    "query_id",
    "phase",
    "split",
    "topic",
    "query_type",
    "difficulty",
    "query",
    "answerability",
    "gold_chunk_ids",
    "supporting_chunk_ids",
    "expected_source_ids",
    "evidence_scope",
    "review_status",
    "reviewer_notes",
]

ALLOWED = {
    "phase": {"pilot", "expansion"},
    "split": {"dev", "test"},
    "difficulty": {"easy", "medium", "hard"},
    "answerability": {"answerable", "unanswerable"},
    "evidence_scope": {
        "single_chunk",
        "multi_chunk",
        "no_gold",
    },
    "review_status": {"draft", "verified"},
}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames or [])
        rows = [
            {
                key: (value or "").strip()
                for key, value in row.items()
            }
            for row in reader
        ]
    return rows, columns


def split_ids(value: str) -> list[str]:
    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("pilot", "all"),
        default="pilot",
        help="Validate verified pilot rows or all verified rows.",
    )
    args = parser.parse_args()

    corpus_rows, corpus_columns = read_csv(
        CORPUS_PATH
    )
    benchmark_rows, benchmark_columns = read_csv(
        BENCHMARK_PATH
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in benchmark_columns
    ]
    if missing_columns:
        raise ValueError(
            f"Benchmark is missing columns: {missing_columns}"
        )

    if "chunk_id" not in corpus_columns:
        raise ValueError(
            "Gold corpus has no chunk_id column."
        )

    chunk_to_source = {
        row["chunk_id"]: row["source_id"]
        for row in corpus_rows
    }

    query_counts = Counter(
        row["query_id"]
        for row in benchmark_rows
    )

    selected_rows = [
        row
        for row in benchmark_rows
        if (
            row["review_status"] == "verified"
            and (
                args.scope == "all"
                or row["phase"] == "pilot"
            )
        )
    ]

    audit_rows: list[dict[str, Any]] = []

    for row in selected_rows:
        issues: list[str] = []
        warnings: list[str] = []

        query_id = row["query_id"]

        if not query_id:
            issues.append("empty_query_id")
        elif query_counts[query_id] > 1:
            issues.append("duplicate_query_id")

        for field, allowed_values in ALLOWED.items():
            if row[field] not in allowed_values:
                issues.append(
                    f"invalid_{field}:{row[field]}"
                )

        if not row["topic"]:
            issues.append("empty_topic")

        if not row["query_type"]:
            issues.append("empty_query_type")

        if not row["query"]:
            issues.append("empty_query")
        elif len(row["query"]) < 6:
            warnings.append("query_unusually_short")

        gold_ids = split_ids(
            row["gold_chunk_ids"]
        )
        supporting_ids = split_ids(
            row["supporting_chunk_ids"]
        )
        expected_sources = set(
            split_ids(
                row["expected_source_ids"]
            )
        )

        all_labeled_ids = gold_ids + supporting_ids
        missing_chunk_ids = [
            chunk_id
            for chunk_id in all_labeled_ids
            if chunk_id not in chunk_to_source
        ]
        if missing_chunk_ids:
            issues.append(
                "unknown_chunk_ids:"
                + "|".join(missing_chunk_ids)
            )

        if len(gold_ids) != len(set(gold_ids)):
            issues.append("duplicate_gold_chunk_id")

        overlap = sorted(
            set(gold_ids) & set(supporting_ids)
        )
        if overlap:
            issues.append(
                "gold_supporting_overlap:"
                + "|".join(overlap)
            )

        if row["answerability"] == "answerable":
            if not gold_ids:
                issues.append(
                    "answerable_without_gold"
                )
            if row["evidence_scope"] == "no_gold":
                issues.append(
                    "answerable_with_no_gold_scope"
                )

        if row["answerability"] == "unanswerable":
            if gold_ids:
                issues.append(
                    "unanswerable_has_gold"
                )
            if row["evidence_scope"] != "no_gold":
                issues.append(
                    "unanswerable_scope_not_no_gold"
                )

        if (
            row["evidence_scope"] == "single_chunk"
            and len(gold_ids) != 1
        ):
            issues.append(
                "single_chunk_scope_requires_one_gold"
            )

        if (
            row["evidence_scope"] == "multi_chunk"
            and len(gold_ids) < 2
        ):
            issues.append(
                "multi_chunk_scope_requires_multiple_gold"
            )

        actual_sources = {
            chunk_to_source[chunk_id]
            for chunk_id in gold_ids
            if chunk_id in chunk_to_source
        }

        if (
            expected_sources
            and actual_sources != expected_sources
        ):
            issues.append(
                "expected_source_mismatch:"
                + "|".join(sorted(actual_sources))
            )

        status = (
            "error"
            if issues
            else "review"
            if warnings
            else "pass"
        )

        audit_rows.append(
            {
                "query_id": query_id,
                "phase": row["phase"],
                "split": row["split"],
                "topic": row["topic"],
                "query_type": row["query_type"],
                "difficulty": row["difficulty"],
                "answerability": row[
                    "answerability"
                ],
                "evidence_scope": row[
                    "evidence_scope"
                ],
                "gold_count": len(gold_ids),
                "gold_source_ids": "|".join(
                    sorted(actual_sources)
                ),
                "validation_status": status,
                "issues": "; ".join(
                    [*issues, *warnings]
                ),
                "query": row["query"],
                "gold_chunk_ids": row[
                    "gold_chunk_ids"
                ],
                "supporting_chunk_ids": row[
                    "supporting_chunk_ids"
                ],
                "reviewer_notes": row[
                    "reviewer_notes"
                ],
            }
        )

    if not audit_rows:
        raise ValueError(
            "No verified rows selected. "
            "Set review_status=verified after labeling."
        )

    write_csv(
        LOCAL_AUDIT_PATH,
        audit_rows,
    )

    status_counts = Counter(
        row["validation_status"]
        for row in audit_rows
    )
    topic_counts = Counter(
        row["topic"]
        for row in audit_rows
    )
    scope_counts = Counter(
        row["evidence_scope"]
        for row in audit_rows
    )
    source_counts = Counter(
        source
        for row in audit_rows
        for source in (
            row["gold_source_ids"].split("|")
            if row["gold_source_ids"]
            else []
        )
    )

    summary_rows = [
        {
            "benchmark_version":
                "retrieval_eval_v1",
            "validation_scope": args.scope,
            "verified_rows": len(audit_rows),
            "pass_count": status_counts[
                "pass"
            ],
            "review_count": status_counts[
                "review"
            ],
            "error_count": status_counts[
                "error"
            ],
            "single_chunk_count":
                scope_counts["single_chunk"],
            "multi_chunk_count":
                scope_counts["multi_chunk"],
            "no_gold_count":
                scope_counts["no_gold"],
            "src001_gold_queries":
                source_counts["SRC001"],
            "src003_gold_queries":
                source_counts["SRC003"],
            "topic_distribution": json_compact(
                topic_counts
            ),
        }
    ]

    write_csv(
        PUBLIC_SUMMARY_PATH,
        summary_rows,
    )

    print("\nRetrieval benchmark validation")
    print("=" * 88)
    print(
        f"Scope: {args.scope}; "
        f"verified rows: {len(audit_rows)}"
    )
    print(
        f"Status counts: "
        f"{dict(status_counts)}"
    )
    print(
        f"Evidence scopes: "
        f"{dict(scope_counts)}"
    )
    print(f"Local audit: {LOCAL_AUDIT_PATH}")
    print(
        f"Public summary: "
        f"{PUBLIC_SUMMARY_PATH}"
    )
    print("=" * 88)

    if status_counts["error"] > 0:
        raise SystemExit(1)


def json_compact(counter: Counter[str]) -> str:
    import json
    return json.dumps(
        dict(sorted(counter.items())),
        ensure_ascii=False,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    main()
