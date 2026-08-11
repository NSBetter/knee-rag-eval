"""Run the generation stage for the medical RAG evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "generation_inputs"
    / "generation_input_v1.jsonl"
)

DEFAULT_PROMPT_PATH = (
    ROOT
    / "configs"
    / "generation_prompt_v1.json"
)

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "generation_runs"
    / "generation_prompt_preview_v1.jsonl"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

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


def load_prompt(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt config not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def format_evidence(
    evidence_rows: list[dict[str, Any]],
    template: str,
) -> str:
    blocks = []

    for evidence in evidence_rows:
        blocks.append(
            template.format(
                rank=evidence["rank"],
                source_id=evidence["source_id"],
                title_path=evidence["title_path"],
                text=evidence["text"],
            )
        )

    return "\n\n".join(blocks)


def build_prompt(
    row: dict[str, Any],
    prompt_config: dict[str, Any],
) -> dict[str, str]:
    evidence_text = format_evidence(
        row["evidence"],
        prompt_config["evidence_template"],
    )

    user_prompt = prompt_config["user_template"].format(
        query=row["query"],
        evidence=evidence_text,
    )

    return {
        "system_prompt": prompt_config["system_prompt"],
        "user_prompt": user_prompt,
    }



def generate_openai_compatible(
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
        "thinking": {"type": "disabled"},
        "temperature": 0,
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
            f"Generation API HTTP {exc.code}: {body}"
        ) from exc

    return str(
        result["choices"][0]["message"]["content"]
    ).strip()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts without calling a generation model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate only the first N queries.",
    )
    args = parser.parse_args()

    model = os.getenv("GENERATION_MODEL", "")
    api_base = os.getenv("GENERATION_API_BASE", "")
    api_key = os.getenv("GENERATION_API_KEY", "")

    if not args.dry_run:
        missing = [
            name
            for name, value in [
                ("GENERATION_MODEL", model),
                ("GENERATION_API_BASE", api_base),
                ("GENERATION_API_KEY", api_key),
            ]
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing generation environment variables: "
                + ", ".join(missing)
            )

    input_rows = read_jsonl(args.input)
    prompt_config = load_prompt(args.prompt)

    if args.limit is not None:
        input_rows = input_rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0

    with args.output.open("w", encoding="utf-8") as file:
        for row in input_rows:
            prompt = build_prompt(row, prompt_config)

            generation_error = None

            if args.dry_run:
                answer = None
                status = "prompt_ready"
                generator_model = None
            else:
                generator_model = model

                try:
                    answer = generate_openai_compatible(
                        system_prompt=prompt["system_prompt"],
                        user_prompt=prompt["user_prompt"],
                        model=model,
                        api_base=api_base,
                        api_key=api_key,
                    )
                    status = "success"
                    success_count += 1
                except Exception as exc:
                    answer = None
                    status = "error"
                    generation_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    error_count += 1

            output_row = {
                **row,
                "generator_model": generator_model,
                "prompt_version": prompt_config["prompt_version"],
                "system_prompt": prompt["system_prompt"],
                "user_prompt": prompt["user_prompt"],
                "answer": answer,
                "generation_status": status,
                "generation_error": generation_error,
            }

            file.write(
                json.dumps(output_row, ensure_ascii=False) + "\n"
            )
            file.flush()

    print("Generation run")
    print("=" * 60)
    print(f"Queries: {len(input_rows)}")
    if not args.dry_run:
        print(f"Success: {success_count}")
        print(f"Errors: {error_count}")
    print(f"Prompt version: {prompt_config['prompt_version']}")
    print(
        "Mode:",
        "dry_run" if args.dry_run else "model_generation",
    )
    if not args.dry_run:
        print(f"Model: {model}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
