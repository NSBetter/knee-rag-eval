"""Compare LLM Judge results with human medical review labels."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REVIEW_PATH = (
    ROOT
    / "data"
    / "processed"
    / "evaluation_runs"
    / "deepseek_v4_flash_generation_v1_manual_review.jsonl"
)


FIELD_PAIRS = [
    ("relevance_score", "human_relevance_score"),
    ("groundedness_score", "human_groundedness_score"),
    ("completeness_score", "human_completeness_score"),
    ("unsupported_claim", "human_unsupported_claim"),
    ("overall_verdict", "human_verdict"),
]


def main() -> None:
    rows = []

    with REVIEW_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    total_comparisons = 0
    matched_comparisons = 0

    print("Judge-Human calibration")
    print("=" * 60)

    for row in rows:
        judge = row["judge_result"]
        query_id = row["query_id"]

        case_matches = 0

        print(f"\n{query_id}")

        for judge_field, human_field in FIELD_PAIRS:
            judge_value = judge.get(judge_field)
            human_value = row.get(human_field)

            match = judge_value == human_value

            total_comparisons += 1
            matched_comparisons += int(match)
            case_matches += int(match)

            print(
                f"  {judge_field}: "
                f"judge={judge_value!r} "
                f"human={human_value!r} "
                f"{'MATCH' if match else 'MISMATCH'}"
            )

        print(
            f"  field agreement: "
            f"{case_matches}/{len(FIELD_PAIRS)}"
        )

    print("\n" + "=" * 60)
    print(f"Calibration cases: {len(rows)}")
    print(
        "Field agreement:",
        f"{matched_comparisons}/{total_comparisons}",
    )


if __name__ == "__main__":
    main()
