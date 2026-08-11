"""Deterministic rule evaluation for generation runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "generation_runs"
    / "deepseek_v4_flash_generation_v1.jsonl"
)

DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "evaluation_runs"
    / "deepseek_v4_flash_generation_v1_rules.csv"
)


INSUFFICIENT_PATTERNS = [
    r"证据不足",
    r"证据中未",
    r"现有证据.*不足",
    r"无法.*回答",
    r"无法明确",
    r"不能.*回答",
    r"未明确",
    r"未提及",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL: {path}, line {line_number}"
                ) from exc

    return rows


def detects_insufficient_evidence(answer: str) -> bool:
    return any(
        re.search(pattern, answer)
        for pattern in INSUFFICIENT_PATTERNS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    results = []

    for row in rows:
        answer = str(row.get("answer") or "").strip()
        evidence = row.get("evidence") or []
        top_k = int(row["top_k"])

        generation_success = (
            row.get("generation_status") == "success"
        )
        answer_nonempty = bool(answer)
        evidence_count_ok = len(evidence) == top_k

        insufficient_detected = (
            detects_insufficient_evidence(answer)
            if answer
            else False
        )

        answerability = row["answerability"]

        if answerability == "unanswerable":
            answerability_behavior_ok = insufficient_detected
        else:
            answerability_behavior_ok = True

        deterministic_pass = all(
            [
                generation_success,
                answer_nonempty,
                evidence_count_ok,
                answerability_behavior_ok,
            ]
        )

        results.append(
            {
                "query_id": row["query_id"],
                "answerability": answerability,
                "generation_success": int(generation_success),
                "answer_nonempty": int(answer_nonempty),
                "evidence_count": len(evidence),
                "top_k": top_k,
                "evidence_count_ok": int(evidence_count_ok),
                "insufficient_evidence_detected": int(
                    insufficient_detected
                ),
                "answerability_behavior_ok": int(
                    answerability_behavior_ok
                ),
                "deterministic_pass": int(deterministic_pass),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    passed = sum(
        row["deterministic_pass"]
        for row in results
    )

    unanswerable = [
        row
        for row in results
        if row["answerability"] == "unanswerable"
    ]

    print("Generation deterministic evaluation")
    print("=" * 60)
    print(f"Queries: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed}")
    print(f"Unanswerable queries: {len(unanswerable)}")

    if unanswerable:
        print(
            "Unanswerable behavior pass:",
            sum(
                row["answerability_behavior_ok"]
                for row in unanswerable
            ),
            "/",
            len(unanswerable),
        )

    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
