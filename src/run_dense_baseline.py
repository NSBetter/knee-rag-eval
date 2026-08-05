"""Run a local dense retrieval baseline with Qwen3-Embedding-0.6B.

The corpus is small, so retrieval uses exact cosine similarity over all
Gold Corpus chunks rather than an approximate vector database.

Usage:
    uv run python src/run_dense_baseline.py --scope pilot

Rebuild cached corpus embeddings:
    uv run python src/run_dense_baseline.py --scope pilot --rebuild
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np


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

CONFIG_PATH = (
    ROOT
    / "configs"
    / "retrieval_dense_qwen3_v1.json"
)

CACHE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "qwen3_embedding_0_6b_dense_v1"
)

LOCAL_RESULTS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "retrieval_runs"
    / "qwen3_embedding_0_6b_dense_v1_results.csv"
)

PUBLIC_METRICS_PATH = (
    ROOT
    / "docs"
    / "retrieval_dense_qwen3_v1_metrics.csv"
)

PUBLIC_QUERY_METRICS_PATH = (
    ROOT
    / "docs"
    / "retrieval_dense_qwen3_v1_query_metrics.csv"
)


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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")

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


def split_ids(value: str) -> list[str]:
    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


def mean(values: list[float]) -> float:
    return (
        sum(values) / len(values)
        if values
        else 0.0
    )


def corpus_text(row: dict[str, str]) -> str:
    retrieval_text = row.get(
        "retrieval_text",
        "",
    ).strip()

    if retrieval_text:
        return retrieval_text

    parts = [
        row.get("title_path", ""),
        row.get("display_title", ""),
        row.get("text", ""),
    ]

    return "\n".join(
        part
        for part in parts
        if part
    )


def corpus_signature(
    rows: list[dict[str, str]],
    texts: list[str],
    config: dict[str, Any],
) -> str:
    hasher = hashlib.sha256()

    hasher.update(
        config["model_name"].encode("utf-8")
    )
    hasher.update(
        str(config["max_seq_length"]).encode(
            "utf-8"
        )
    )

    for row, text in zip(rows, texts):
        hasher.update(
            row["chunk_id"].encode("utf-8")
        )
        hasher.update(b"\0")
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\0")

    return hasher.hexdigest()


def choose_device() -> str:
    import torch

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def encode_corpus(
    model: Any,
    texts: list[str],
    config: dict[str, Any],
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=int(config["batch_size"]),
        show_progress_bar=True,
        normalize_embeddings=bool(
            config["normalize_embeddings"]
        ),
        convert_to_numpy=True,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


def load_or_build_corpus_embeddings(
    model: Any,
    corpus_rows: list[dict[str, str]],
    texts: list[str],
    config: dict[str, Any],
    rebuild: bool,
) -> tuple[np.ndarray, str]:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_path = CACHE_DIR / "corpus_embeddings.npy"
    metadata_path = CACHE_DIR / "metadata.json"

    signature = corpus_signature(
        corpus_rows,
        texts,
        config,
    )

    cache_valid = False

    if (
        matrix_path.exists()
        and metadata_path.exists()
        and not rebuild
        and not bool(
            config.get(
                "force_rebuild_embeddings",
                False,
            )
        )
    ):
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        cache_valid = (
            metadata.get("signature") == signature
            and metadata.get("model_name")
            == config["model_name"]
            and metadata.get("chunk_count")
            == len(corpus_rows)
        )

    if cache_valid:
        matrix = np.load(
            matrix_path,
            allow_pickle=False,
        )
        cache_status = "loaded"
    else:
        matrix = encode_corpus(
            model,
            texts,
            config,
        )
        np.save(
            matrix_path,
            matrix,
            allow_pickle=False,
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "run_id": config["run_id"],
                    "model_name":
                        config["model_name"],
                    "corpus_version":
                        config["corpus_version"],
                    "chunk_count":
                        len(corpus_rows),
                    "embedding_dimension":
                        int(matrix.shape[1]),
                    "max_seq_length":
                        config["max_seq_length"],
                    "normalized":
                        config[
                            "normalize_embeddings"
                        ],
                    "signature": signature,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        cache_status = "built"

    if matrix.shape[0] != len(corpus_rows):
        raise ValueError(
            "Embedding row count does not match "
            "the Gold Corpus."
        )

    return matrix, cache_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("pilot", "all"),
        default="pilot",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recompute cached corpus embeddings.",
    )
    args = parser.parse_args()

    try:
        from sentence_transformers import (
            SentenceTransformer,
        )
    except ImportError as error:
        raise SystemExit(
            "sentence-transformers is not installed. "
            "Run: uv add sentence-transformers "
            "transformers numpy"
        ) from error

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    corpus_rows, corpus_columns = read_csv(
        CORPUS_PATH
    )
    benchmark_rows, benchmark_columns = read_csv(
        BENCHMARK_PATH
    )

    required_corpus = {
        "chunk_id",
        "source_id",
        "title_path",
        "display_title",
        "text",
    }
    missing_corpus = sorted(
        required_corpus - set(corpus_columns)
    )
    if missing_corpus:
        raise ValueError(
            f"Corpus is missing columns: "
            f"{missing_corpus}"
        )

    required_benchmark = {
        "query_id",
        "phase",
        "query",
        "answerability",
        "gold_chunk_ids",
        "supporting_chunk_ids",
        "review_status",
    }
    missing_benchmark = sorted(
        required_benchmark
        - set(benchmark_columns)
    )
    if missing_benchmark:
        raise ValueError(
            f"Benchmark is missing columns: "
            f"{missing_benchmark}"
        )

    chunk_ids = [
        row["chunk_id"]
        for row in corpus_rows
    ]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Gold Corpus contains duplicate chunk IDs."
        )

    selected = [
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

    if not selected:
        raise ValueError(
            "No verified benchmark rows selected."
        )

    device = choose_device()

    model = SentenceTransformer(
        config["model_name"],
        device=device,
    )
    model.max_seq_length = int(
        config["max_seq_length"]
    )

    texts = [
        corpus_text(row)
        for row in corpus_rows
    ]

    corpus_embeddings, cache_status = (
        load_or_build_corpus_embeddings(
            model,
            corpus_rows,
            texts,
            config,
            rebuild=args.rebuild,
        )
    )

    queries = [
        row["query"]
        for row in selected
    ]

    query_embeddings = model.encode(
        queries,
        prompt_name=config[
            "query_prompt_name"
        ],
        batch_size=int(config["batch_size"]),
        show_progress_bar=True,
        normalize_embeddings=bool(
            config["normalize_embeddings"]
        ),
        convert_to_numpy=True,
    )
    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32,
    )

    similarities = (
        query_embeddings
        @ corpus_embeddings.T
    )

    top_k = int(config["top_k"])

    local_rows: list[dict[str, Any]] = []
    query_metric_rows: list[
        dict[str, Any]
    ] = []

    hit_at_1: list[float] = []
    hit_at_3: list[float] = []
    hit_at_5: list[float] = []
    recall_at_5: list[float] = []
    mrr_at_10: list[float] = []

    relevant_hit_at_1: list[float] = []
    relevant_hit_at_3: list[float] = []
    relevant_hit_at_5: list[float] = []

    answerable_count = 0
    unanswerable_count = 0

    for query_index, benchmark in enumerate(
        selected
    ):
        scores = similarities[query_index]

        ranking = sorted(
            range(len(corpus_rows)),
            key=lambda index: (
                -float(scores[index]),
                corpus_rows[index][
                    "chunk_id"
                ],
            ),
        )[:top_k]

        ranked_ids = [
            corpus_rows[index]["chunk_id"]
            for index in ranking
        ]

        gold_ids = set(
            split_ids(
                benchmark[
                    "gold_chunk_ids"
                ]
            )
        )
        support_ids = set(
            split_ids(
                benchmark[
                    "supporting_chunk_ids"
                ]
            )
        )
        relevant_ids = gold_ids | support_ids

        first_gold_rank: int | str = ""
        first_relevant_rank: int | str = ""

        relevant_ranks = [
            rank
            for rank, chunk_id
            in enumerate(
                ranked_ids,
                start=1,
            )
            if chunk_id in relevant_ids
        ]
        if relevant_ranks:
            first_relevant_rank = min(
                relevant_ranks
            )

        if (
            benchmark["answerability"]
            == "answerable"
        ):
            answerable_count += 1

            gold_ranks = [
                rank
                for rank, chunk_id
                in enumerate(
                    ranked_ids,
                    start=1,
                )
                if chunk_id in gold_ids
            ]

            if gold_ranks:
                first_gold_rank = min(
                    gold_ranks
                )

            hit_at_1.append(
                float(
                    bool(
                        gold_ids
                        & set(ranked_ids[:1])
                    )
                )
            )
            hit_at_3.append(
                float(
                    bool(
                        gold_ids
                        & set(ranked_ids[:3])
                    )
                )
            )
            hit_at_5.append(
                float(
                    bool(
                        gold_ids
                        & set(ranked_ids[:5])
                    )
                )
            )

            recall_at_5.append(
                len(
                    gold_ids
                    & set(ranked_ids[:5])
                )
                / len(gold_ids)
            )

            mrr_at_10.append(
                1.0 / first_gold_rank
                if first_gold_rank
                else 0.0
            )

            relevant_hit_at_1.append(
                float(
                    bool(
                        relevant_ids
                        & set(ranked_ids[:1])
                    )
                )
            )
            relevant_hit_at_3.append(
                float(
                    bool(
                        relevant_ids
                        & set(ranked_ids[:3])
                    )
                )
            )
            relevant_hit_at_5.append(
                float(
                    bool(
                        relevant_ids
                        & set(ranked_ids[:5])
                    )
                )
            )

        else:
            unanswerable_count += 1

        query_metric_rows.append(
            {
                "run_id": config["run_id"],
                "scope": args.scope,
                "query_id": benchmark[
                    "query_id"
                ],
                "answerability":
                    benchmark[
                        "answerability"
                    ],
                "gold_count":
                    len(gold_ids),
                "support_count":
                    len(support_ids),
                "first_gold_rank":
                    first_gold_rank,
                "first_relevant_rank":
                    first_relevant_rank,
                "hit_at_1": (
                    ""
                    if not gold_ids
                    else int(
                        bool(
                            gold_ids
                            & set(
                                ranked_ids[:1]
                            )
                        )
                    )
                ),
                "hit_at_3": (
                    ""
                    if not gold_ids
                    else int(
                        bool(
                            gold_ids
                            & set(
                                ranked_ids[:3]
                            )
                        )
                    )
                ),
                "hit_at_5": (
                    ""
                    if not gold_ids
                    else int(
                        bool(
                            gold_ids
                            & set(
                                ranked_ids[:5]
                            )
                        )
                    )
                ),
                "recall_at_5": (
                    ""
                    if not gold_ids
                    else round(
                        len(
                            gold_ids
                            & set(
                                ranked_ids[:5]
                            )
                        )
                        / len(gold_ids),
                        6,
                    )
                ),
                "relevant_hit_at_5": (
                    ""
                    if not relevant_ids
                    else int(
                        bool(
                            relevant_ids
                            & set(
                                ranked_ids[:5]
                            )
                        )
                    )
                ),
                "top1_score": round(
                    float(
                        scores[ranking[0]]
                    ),
                    6,
                ),
                "top1_chunk_id":
                    ranked_ids[0],
            }
        )

        for rank, corpus_index in enumerate(
            ranking,
            start=1,
        ):
            corpus = corpus_rows[
                corpus_index
            ]

            local_rows.append(
                {
                    "run_id":
                        config["run_id"],
                    "scope": args.scope,
                    "query_id":
                        benchmark[
                            "query_id"
                        ],
                    "query":
                        benchmark["query"],
                    "answerability":
                        benchmark[
                            "answerability"
                        ],
                    "gold_chunk_ids":
                        benchmark[
                            "gold_chunk_ids"
                        ],
                    "supporting_chunk_ids":
                        benchmark[
                            "supporting_chunk_ids"
                        ],
                    "rank": rank,
                    "retrieved_chunk_id":
                        corpus["chunk_id"],
                    "retrieved_source_id":
                        corpus["source_id"],
                    "retrieved_title_path":
                        corpus["title_path"],
                    "score": round(
                        float(
                            scores[
                                corpus_index
                            ]
                        ),
                        6,
                    ),
                    "is_gold": int(
                        corpus["chunk_id"]
                        in gold_ids
                    ),
                    "is_support": int(
                        corpus["chunk_id"]
                        in support_ids
                    ),
                }
            )

    metrics_rows = [
        {
            "run_id":
                config["run_id"],
            "scope": args.scope,
            "corpus_version":
                config["corpus_version"],
            "benchmark_version":
                config[
                    "benchmark_version"
                ],
            "retriever":
                "dense_cosine_exact",
            "model_name":
                config["model_name"],
            "query_prompt_name":
                config[
                    "query_prompt_name"
                ],
            "device": device,
            "embedding_dimension":
                int(
                    corpus_embeddings.shape[
                        1
                    ]
                ),
            "max_seq_length":
                config["max_seq_length"],
            "evaluated_answerable_queries":
                answerable_count,
            "reported_unanswerable_queries":
                unanswerable_count,
            "hit_at_1": round(
                mean(hit_at_1),
                6,
            ),
            "hit_at_3": round(
                mean(hit_at_3),
                6,
            ),
            "hit_at_5": round(
                mean(hit_at_5),
                6,
            ),
            "recall_at_5": round(
                mean(recall_at_5),
                6,
            ),
            "mrr_at_10": round(
                mean(mrr_at_10),
                6,
            ),
            "relevant_hit_at_1": round(
                mean(
                    relevant_hit_at_1
                ),
                6,
            ),
            "relevant_hit_at_3": round(
                mean(
                    relevant_hit_at_3
                ),
                6,
            ),
            "relevant_hit_at_5": round(
                mean(
                    relevant_hit_at_5
                ),
                6,
            ),
            "corpus_embedding_cache":
                cache_status,
            "python_version":
                platform.python_version(),
        }
    ]

    write_csv(
        LOCAL_RESULTS_PATH,
        local_rows,
    )
    write_csv(
        PUBLIC_METRICS_PATH,
        metrics_rows,
    )
    write_csv(
        PUBLIC_QUERY_METRICS_PATH,
        query_metric_rows,
    )

    result = metrics_rows[0]

    print(
        "\nDense retrieval baseline"
    )
    print("=" * 88)
    print(
        f"Model: "
        f"{config['model_name']}"
    )
    print(
        f"Device: {device}; "
        f"embedding cache: "
        f"{cache_status}; "
        f"dimension: "
        f"{result['embedding_dimension']}"
    )
    print(
        f"Scope: {args.scope}; "
        f"answerable: "
        f"{answerable_count}; "
        f"unanswerable: "
        f"{unanswerable_count}"
    )
    print(
        "Strict Gold metrics: "
        f"Hit@1="
        f"{result['hit_at_1']} "
        f"Hit@3="
        f"{result['hit_at_3']} "
        f"Hit@5="
        f"{result['hit_at_5']} "
        f"Recall@5="
        f"{result['recall_at_5']} "
        f"MRR@10="
        f"{result['mrr_at_10']}"
    )
    print(
        "Gold-or-support metrics: "
        f"RelevantHit@1="
        f"{result['relevant_hit_at_1']} "
        f"RelevantHit@3="
        f"{result['relevant_hit_at_3']} "
        f"RelevantHit@5="
        f"{result['relevant_hit_at_5']}"
    )
    print(
        f"Local results: "
        f"{LOCAL_RESULTS_PATH}"
    )
    print(
        f"Public metrics: "
        f"{PUBLIC_METRICS_PATH}"
    )
    print(
        f"Public query metrics: "
        f"{PUBLIC_QUERY_METRICS_PATH}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
