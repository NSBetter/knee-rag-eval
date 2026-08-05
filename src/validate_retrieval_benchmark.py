"""Strictly validate Retrieval Benchmark v1 against gold_v1_3."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/processed/gold_corpus/gold_corpus_v1_3.csv"
BENCHMARK = ROOT / "data/benchmark/retrieval_eval_v1.csv"
LOCAL_AUDIT = ROOT / "data/processed/reviews/retrieval_eval_v1_audit.csv"
PUBLIC_SUMMARY = ROOT / "docs/retrieval_benchmark_v1_summary.csv"

EXPECTED_PILOT_IDS = {f"RET-{i:03d}" for i in range(1, 13)}
REQUIRED_COLUMNS = [
    "query_id", "phase", "split", "topic", "query_type", "difficulty",
    "query", "answerability", "gold_chunk_ids", "supporting_chunk_ids",
    "expected_source_ids", "evidence_scope", "review_status",
    "reviewer_notes",
]
ALLOWED = {
    "phase": {"pilot", "expansion"},
    "split": {"dev", "test"},
    "difficulty": {"easy", "medium", "hard"},
    "answerability": {"answerable", "unanswerable"},
    "evidence_scope": {"single_chunk", "multi_chunk", "no_gold"},
    "review_status": {"draft", "verified"},
}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = [
            {k: (v or "").strip() for k, v in row.items()}
            for row in reader
        ]
    return rows, cols


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ids(value: str) -> list[str]:
    return [x.strip() for x in value.split("|") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("pilot", "all"), default="pilot")
    args = parser.parse_args()

    corpus_rows, corpus_cols = read_csv(CORPUS)
    rows, cols = read_csv(BENCHMARK)

    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"Missing benchmark columns: {missing}")
    if "chunk_id" not in corpus_cols:
        raise ValueError("Gold corpus has no chunk_id column.")

    chunk_source = {r["chunk_id"]: r["source_id"] for r in corpus_rows}
    query_counts = Counter(r["query_id"] for r in rows)

    selected = (
        [r for r in rows if r["phase"] == "pilot"]
        if args.scope == "pilot"
        else [r for r in rows if r["review_status"] == "verified"]
    )

    global_issues: list[str] = []
    selected_ids = {r["query_id"] for r in selected}

    if args.scope == "pilot":
        missing_ids = sorted(EXPECTED_PILOT_IDS - selected_ids)
        extra_ids = sorted(selected_ids - EXPECTED_PILOT_IDS)
        if len(selected) != 12:
            global_issues.append(f"pilot_row_count={len(selected)} expected=12")
        if missing_ids:
            global_issues.append("missing_pilot_ids:" + "|".join(missing_ids))
        if extra_ids:
            global_issues.append("unexpected_pilot_ids:" + "|".join(extra_ids))

    audit: list[dict[str, Any]] = []

    for row in selected:
        errors: list[str] = []
        warnings: list[str] = []

        for field, allowed in ALLOWED.items():
            if row[field] not in allowed:
                errors.append(f"invalid_{field}:{row[field]}")

        if query_counts[row["query_id"]] > 1:
            errors.append("duplicate_query_id")
        if row["review_status"] != "verified":
            errors.append("row_not_verified")
        if not row["query"]:
            errors.append("empty_query")
        elif len(row["query"]) < 6:
            warnings.append("query_unusually_short")
        if not row["topic"]:
            errors.append("empty_topic")
        if not row["query_type"]:
            errors.append("empty_query_type")

        gold = ids(row["gold_chunk_ids"])
        supporting = ids(row["supporting_chunk_ids"])
        expected_sources = set(ids(row["expected_source_ids"]))

        unknown = [x for x in gold + supporting if x not in chunk_source]
        if unknown:
            errors.append("unknown_chunk_ids:" + "|".join(unknown))
        if len(gold) != len(set(gold)):
            errors.append("duplicate_gold_chunk_id")
        overlap = sorted(set(gold) & set(supporting))
        if overlap:
            errors.append("gold_supporting_overlap:" + "|".join(overlap))

        if row["answerability"] == "answerable":
            if not gold:
                errors.append("answerable_without_gold")
            if row["evidence_scope"] == "no_gold":
                errors.append("answerable_with_no_gold_scope")
        else:
            if gold:
                errors.append("unanswerable_has_gold")
            if row["evidence_scope"] != "no_gold":
                errors.append("unanswerable_scope_not_no_gold")

        if row["evidence_scope"] == "single_chunk" and len(gold) != 1:
            errors.append("single_chunk_requires_one_gold")
        if row["evidence_scope"] == "multi_chunk" and len(gold) < 2:
            errors.append("multi_chunk_requires_multiple_gold")

        actual_sources = {
            chunk_source[x] for x in gold if x in chunk_source
        }
        if expected_sources and actual_sources != expected_sources:
            errors.append(
                "expected_source_mismatch:" + "|".join(sorted(actual_sources))
            )

        status = "error" if errors else "review" if warnings else "pass"
        audit.append({
            "query_id": row["query_id"],
            "phase": row["phase"],
            "split": row["split"],
            "topic": row["topic"],
            "query_type": row["query_type"],
            "difficulty": row["difficulty"],
            "answerability": row["answerability"],
            "evidence_scope": row["evidence_scope"],
            "gold_count": len(gold),
            "gold_source_ids": "|".join(sorted(actual_sources)),
            "review_status": row["review_status"],
            "validation_status": status,
            "issues": "; ".join(errors + warnings),
            "query": row["query"],
            "gold_chunk_ids": row["gold_chunk_ids"],
            "supporting_chunk_ids": row["supporting_chunk_ids"],
            "reviewer_notes": row["reviewer_notes"],
        })

    verified = [r for r in audit if r["review_status"] == "verified"]
    scope_counts = Counter(r["evidence_scope"] for r in verified)
    cross_source = sum(
        len(ids(r["gold_source_ids"])) >= 2 for r in verified
    )

    if args.scope == "pilot":
        if scope_counts["multi_chunk"] < 2:
            global_issues.append("pilot_requires_at_least_2_multi_chunk_rows")
        if scope_counts["no_gold"] < 1:
            global_issues.append("pilot_requires_at_least_1_no_gold_row")
        if cross_source < 1:
            global_issues.append("pilot_requires_at_least_1_cross_source_row")

    write_csv(LOCAL_AUDIT, audit)
    status_counts = Counter(r["validation_status"] for r in audit)
    benchmark_status = (
        "error"
        if status_counts["error"] or global_issues
        else "review"
        if status_counts["review"]
        else "pass"
    )

    topic_counts = Counter(r["topic"] for r in verified)
    write_csv(PUBLIC_SUMMARY, [{
        "benchmark_version": "retrieval_eval_v1",
        "validator_version": "v1.1",
        "validation_scope": args.scope,
        "expected_rows": 12 if args.scope == "pilot" else "",
        "selected_rows": len(audit),
        "verified_rows": len(verified),
        "pass_count": status_counts["pass"],
        "review_count": status_counts["review"],
        "error_count": status_counts["error"],
        "single_chunk_count": scope_counts["single_chunk"],
        "multi_chunk_count": scope_counts["multi_chunk"],
        "no_gold_count": scope_counts["no_gold"],
        "cross_source_count": cross_source,
        "topic_distribution": json.dumps(
            dict(sorted(topic_counts.items())),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "benchmark_status": benchmark_status,
        "global_issues": "; ".join(global_issues),
    }])

    print("\nRetrieval benchmark validation v1.1")
    print("=" * 88)
    print(
        f"Scope: {args.scope}; selected rows: {len(audit)}; "
        f"verified rows: {len(verified)}"
    )
    print(f"Status counts: {dict(status_counts)}")
    print(f"Evidence scopes: {dict(scope_counts)}")
    print(f"Cross-source rows: {cross_source}")
    if global_issues:
        print("Global issues: " + "; ".join(global_issues))
    print(f"Local audit: {LOCAL_AUDIT}")
    print(f"Public summary: {PUBLIC_SUMMARY}")
    print("=" * 88)

    if benchmark_status == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
