"""Build generation inputs from frozen dense retrieval results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RETRIEVAL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "retrieval_runs"
    / "qwen3_embedding_0_6b_dense_v1_results.csv"
)

CORPUS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "gold_corpus"
    / "gold_corpus_v1_3.csv"
)

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "generation_inputs"
    / "generation_input_v1.jsonl"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    retrieval_rows = read_csv(RETRIEVAL_PATH)
    corpus_rows = read_csv(CORPUS_PATH)

    corpus_by_id: dict[str, dict[str, str]] = {}

    for row in corpus_rows:
        chunk_id = row["chunk_id"]
        if chunk_id in corpus_by_id:
            raise ValueError(f"Duplicate corpus chunk_id: {chunk_id}")
        corpus_by_id[chunk_id] = row

    retrieval_by_query: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in retrieval_rows:
        retrieval_by_query[row["query_id"]].append(row)

    output_rows: list[dict] = []

    for query_id in sorted(retrieval_by_query):
        rows = sorted(
            retrieval_by_query[query_id],
            key=lambda row: int(row["rank"]),
        )

        first = rows[0]
        selected = rows[: args.top_k]

        evidence = []

        for row in selected:
            chunk_id = row["retrieved_chunk_id"]

            if chunk_id not in corpus_by_id:
                raise ValueError(
                    f"Retrieved chunk not found in corpus: {chunk_id}"
                )

            chunk = corpus_by_id[chunk_id]

            evidence.append(
                {
                    "rank": int(row["rank"]),
                    "chunk_id": chunk_id,
                    "source_id": chunk["source_id"],
                    "title_path": chunk["title_path"],
                    "score": float(row["score"]),
                    "text": chunk["text"],
                    "text_sha256": chunk["text_sha256"],
                }
            )

        output_rows.append(
            {
                "run_id": f"generation_input_v1_top{args.top_k}",
                "query_id": query_id,
                "query": first["query"],
                "answerability": first["answerability"],
                "retrieval_run_id": first["run_id"],
                "top_k": args.top_k,
                "evidence": evidence,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        for row in output_rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    print("Generation input build")
    print("=" * 60)
    print(f"Queries: {len(output_rows)}")
    print(
        "Evidence rows:",
        sum(len(row["evidence"]) for row in output_rows),
    )
    print(f"Top-K: {args.top_k}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
