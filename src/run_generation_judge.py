"""Run LLM-as-a-Judge evaluation for generated RAG answers."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
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

DEFAULT_RUBRIC = (
    ROOT
    / "configs"
    / "generation_judge_v1.json"
)

DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "evaluation_runs"
    / "deepseek_v4_flash_generation_v1_judge.jsonl"
)


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


def format_evidence(evidence: list[dict[str, Any]]) -> str:
    blocks = []

    for item in evidence:
        blocks.append(
            "\n".join(
                [
                    f"[Evidence {item['rank']}]",
                    f"source_id: {item['source_id']}",
                    f"title_path: {item['title_path']}",
                    str(item["text"]),
                ]
            )
        )

    return "\n\n".join(blocks)


def build_user_prompt(
    row: dict[str, Any],
    rubric: dict[str, Any],
) -> str:
    criteria = json.dumps(
        rubric["criteria"],
        ensure_ascii=False,
        indent=2,
    )

    groundedness_rules = json.dumps(
        rubric.get("groundedness_rules", []),
        ensure_ascii=False,
        indent=2,
    )

    required_output = json.dumps(
        rubric["required_output"],
        ensure_ascii=False,
        indent=2,
    )

    return (
        f"query_id: {row['query_id']}\n\n"
        f"问题：\n{row['query']}\n\n"
        f"answerability: {row['answerability']}\n\n"
        f"检索证据：\n{format_evidence(row['evidence'])}\n\n"
        f"待评回答：\n{row['answer']}\n\n"
        f"评分标准：\n{criteria}\n\n"
        f"Groundedness强制规则：\n{groundedness_rules}\n\n"
        f"不可回答策略：\n{rubric['unanswerable_policy']}\n\n"
        f"必须输出以下JSON结构：\n{required_output}\n\n"
        "只输出一个合法JSON对象。"
    )


def call_judge(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_base: str,
    api_key: str,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "response_format": {"type": "json_object"},
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"Judge API HTTP {exc.code}: {body}"
        ) from exc

    return str(
        result["choices"][0]["message"]["content"]
    ).strip()


def parse_judge_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    result = json.loads(cleaned)

    for field in (
        "query_id",
        "relevance_score",
        "groundedness_score",
        "completeness_score",
        "unsupported_claim",
        "unsupported_claim_severity",
        "unanswerable_behavior",
        "overall_verdict",
        "reason",
    ):
        if field not in result:
            raise ValueError(
                f"Judge output missing field: {field}"
            )

    for field in (
        "relevance_score",
        "groundedness_score",
        "completeness_score",
    ):
        if result[field] not in (0, 1, 2):
            raise ValueError(
                f"Invalid {field}: {result[field]}"
            )

    if result["unsupported_claim_severity"] not in (
        "none",
        "minor",
        "major",
    ):
        raise ValueError(
            "Invalid unsupported_claim_severity"
        )

    if result["unanswerable_behavior"] not in (
        "pass",
        "fail",
        "not_applicable",
    ):
        raise ValueError(
            "Invalid unanswerable_behavior"
        )

    if result["overall_verdict"] not in (
        "pass",
        "fail",
    ):
        raise ValueError(
            "Invalid overall_verdict"
        )

    return result



def call_and_parse_judge(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_base: str,
    api_key: str,
) -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None
    retry_prompt = user_prompt

    for attempt in range(2):
        raw = call_judge(
            system_prompt=system_prompt,
            user_prompt=retry_prompt,
            model=model,
            api_base=api_base,
            api_key=api_key,
        )

        try:
            return parse_judge_json(raw), raw
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

            if attempt == 0:
                retry_prompt = (
                    user_prompt
                    + "\n\n上一次输出不是合法JSON。"
                    + "请重新完成同一评测，只输出一个完整、合法的JSON对象，"
                    + "所有required fields必须位于同一个对象内部。"
                )

    raise ValueError(
        f"Judge output invalid after one retry: {last_error}"
    )

def apply_hard_fail_policy(
    result: dict[str, Any],
) -> dict[str, Any]:
    hard_fail = (
        result["groundedness_score"] == 0
        or result["relevance_score"] == 0
        or result["unsupported_claim_severity"] == "major"
        or result["unanswerable_behavior"] == "fail"
    )

    if hard_fail:
        result["overall_verdict"] = "fail"

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=DEFAULT_RUBRIC,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--query-id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)

    rubric = json.loads(
        args.rubric.read_text(encoding="utf-8")
    )

    if args.query_id:
        rows = [
            row
            for row in rows
            if row["query_id"] == args.query_id
        ]

    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        raise ValueError("No rows selected for judge run.")

    model = os.getenv("JUDGE_MODEL", "")
    api_base = os.getenv("JUDGE_API_BASE", "")
    api_key = os.getenv("JUDGE_API_KEY", "")

    if not args.dry_run:
        missing = [
            name
            for name, value in (
                ("JUDGE_MODEL", model),
                ("JUDGE_API_BASE", api_base),
                ("JUDGE_API_KEY", api_key),
            )
            if not value
        ]

        if missing:
            raise RuntimeError(
                "Missing judge environment variables: "
                + ", ".join(missing)
            )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success_count = 0
    error_count = 0

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            user_prompt = build_user_prompt(
                row,
                rubric,
            )

            judge_result = None
            judge_error = None
            raw_response = None

            if args.dry_run:
                status = "prompt_ready"
            else:
                try:
                    judge_result, raw_response = call_and_parse_judge(
                        system_prompt=rubric[
                            "system_prompt"
                        ],
                        user_prompt=user_prompt,
                        model=model,
                        api_base=api_base,
                        api_key=api_key,
                    )

                    if (
                        judge_result["query_id"]
                        != row["query_id"]
                    ):
                        raise ValueError(
                            "Judge query_id mismatch"
                        )

                    judge_result = apply_hard_fail_policy(
                        judge_result
                    )

                    status = "success"
                    success_count += 1

                except Exception as exc:
                    status = "error"
                    judge_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    error_count += 1

            output_row = {
                "judge_version": rubric[
                    "judge_version"
                ],
                "judge_model": (
                    None if args.dry_run else model
                ),
                "query_id": row["query_id"],
                "judge_status": status,
                "judge_result": judge_result,
                "judge_error": judge_error,
                "raw_response": raw_response,
            }

            file.write(
                json.dumps(
                    output_row,
                    ensure_ascii=False,
                )
                + "\n"
            )
            file.flush()

    print("Generation LLM judge")
    print("=" * 60)
    print(f"Queries: {len(rows)}")
    print(f"Judge version: {rubric['judge_version']}")
    print(
        "Mode:",
        "dry_run" if args.dry_run else "judge",
    )

    if not args.dry_run:
        print(f"Success: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Model: {model}")

    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
