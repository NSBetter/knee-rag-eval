"""Build the human medical review queue for generation evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

GENERATION_PATH = (
    ROOT
    / "data"
    / "processed"
    / "generation_runs"
    / "deepseek_v4_flash_generation_v1.jsonl"
)

JUDGE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "evaluation_runs"
    / "deepseek_v4_flash_generation_v1_judge.jsonl"
)

SUMMARY_PATH = (
    ROOT
    / "data"
    / "processed"
    / "evaluation_runs"
    / "deepseek_v4_flash_generation_v1_summary.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "evaluation_runs"
    / "deepseek_v4_flash_generation_v1_manual_review.jsonl"
)


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["query_id"]] = row

    return rows


def read_summary() -> list[dict[str, str]]:
    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def main() -> None:
    generation = read_jsonl(GENERATION_PATH)
    judge = read_jsonl(JUDGE_PATH)

    review_rows = []

    for summary in read_summary():
        if summary["manual_review_required"] != "1":
            continue

        query_id = summary["query_id"]
        generation_row = generation[query_id]
        judge_row = judge[query_id]
        judge_result = judge_row.get("judge_result") or {}

        evidence = [
            {
                "rank": item["rank"],
                "chunk_id": item["chunk_id"],
                "source_id": item["source_id"],
                "title_path": item["title_path"],
                "text": item["text"],
            }
            for item in generation_row["evidence"]
        ]

        review_rows.append(
            {
                "query_id": query_id,
                "query": generation_row["query"],
                "answerability": generation_row["answerability"],
                "answer": generation_row["answer"],
                "evidence": evidence,
                "judge_result": judge_result,
                "review_reason": summary["review_reason"],
                "human_relevance_score": None,
                "human_groundedness_score": None,
                "human_completeness_score": None,
                "human_unsupported_claim": None,
                "human_verdict": None,
                "human_review_notes": None,
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for row in review_rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    print("Human medical review queue")
    print("=" * 60)
    print(f"Review cases: {len(review_rows)}")
    print(
        "Query IDs:",
        ", ".join(row["query_id"] for row in review_rows),
    )
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
