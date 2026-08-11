"""Summarize deterministic and LLM-judge generation evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

RULES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "evaluation_runs"
    / "deepseek_v4_flash_generation_v1_rules.csv"
)

JUDGE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "evaluation_runs"
    / "deepseek_v4_flash_generation_v1_judge.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "evaluation_runs"
    / "deepseek_v4_flash_generation_v1_summary.csv"
)


def read_rules() -> dict[str, dict[str, str]]:
    with RULES_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return {
            row["query_id"]: row
            for row in csv.DictReader(file)
        }


def read_judge() -> dict[str, dict[str, Any]]:
    rows = {}

    with JUDGE_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["query_id"]] = row

    return rows


def main() -> None:
    rules = read_rules()
    judge = read_judge()

    if set(rules) != set(judge):
        raise ValueError(
            "Rule and Judge query_id sets do not match."
        )

    output_rows = []

    for query_id in sorted(rules):
        rule = rules[query_id]
        judge_row = judge[query_id]

        result = judge_row.get("judge_result") or {}

        deterministic_pass = (
            rule["deterministic_pass"] == "1"
        )

        judge_success = (
            judge_row["judge_status"] == "success"
        )

        verdict = result.get("overall_verdict", "error")
        unsupported = bool(
            result.get("unsupported_claim", False)
        )

        scores = [
            result.get("relevance_score"),
            result.get("groundedness_score"),
            result.get("completeness_score"),
        ]

        manual_review_required = (
            not deterministic_pass
            or not judge_success
            or verdict != "pass"
            or unsupported
            or any(score != 2 for score in scores)
        )

        reasons = []

        if not deterministic_pass:
            reasons.append("deterministic_fail")
        if not judge_success:
            reasons.append("judge_error")
        if verdict == "fail":
            reasons.append("judge_fail")
        if unsupported:
            reasons.append("unsupported_claim")
        if any(score != 2 for score in scores):
            reasons.append("non_max_score")

        output_rows.append(
            {
                "query_id": query_id,
                "answerability": rule["answerability"],
                "deterministic_pass": int(
                    deterministic_pass
                ),
                "judge_status": judge_row["judge_status"],
                "relevance_score": result.get(
                    "relevance_score"
                ),
                "groundedness_score": result.get(
                    "groundedness_score"
                ),
                "completeness_score": result.get(
                    "completeness_score"
                ),
                "unsupported_claim": result.get(
                    "unsupported_claim"
                ),
                "unsupported_claim_severity": result.get(
                    "unsupported_claim_severity"
                ),
                "unanswerable_behavior": result.get(
                    "unanswerable_behavior"
                ),
                "judge_verdict": verdict,
                "manual_review_required": int(
                    manual_review_required
                ),
                "review_reason": "|".join(reasons),
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    successful = [
        row
        for row in output_rows
        if row["judge_status"] == "success"
    ]

    print("Generation evaluation summary")
    print("=" * 60)
    print(f"Queries: {len(output_rows)}")
    print(
        "Deterministic pass:",
        sum(row["deterministic_pass"] for row in output_rows),
    )
    print(
        "Judge pass:",
        sum(
            row["judge_verdict"] == "pass"
            for row in successful
        ),
    )
    print(
        "Unsupported claims:",
        sum(
            row["unsupported_claim"] is True
            for row in successful
        ),
    )
    print(
        "Manual review required:",
        sum(
            row["manual_review_required"]
            for row in output_rows
        ),
    )

    for field in (
        "relevance_score",
        "groundedness_score",
        "completeness_score",
    ):
        values = [
            row[field]
            for row in successful
            if isinstance(row[field], int)
        ]
        print(
            f"Mean {field}:",
            round(mean(values), 3),
        )

    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
