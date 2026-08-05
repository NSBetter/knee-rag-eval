"""Compare BM25 and dense retrieval query-level metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

BM25_PATH = (
    ROOT
    / "docs"
    / "retrieval_bm25_v1_query_metrics.csv"
)

DENSE_PATH = (
    ROOT
    / "docs"
    / "retrieval_dense_qwen3_v1_query_metrics.csv"
)

OUTPUT_PATH = (
    ROOT
    / "docs"
    / "retrieval_bm25_vs_dense_v1.csv"
)


def read_indexed(
    path: Path,
) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return {
            row["query_id"]: {
                key: (value or "").strip()
                for key, value in row.items()
            }
            for row in csv.DictReader(file)
        }


def rank_value(
    value: str,
) -> int:
    return int(value) if value else 999


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
    bm25 = read_indexed(BM25_PATH)
    dense = read_indexed(DENSE_PATH)

    query_ids = sorted(
        set(bm25) | set(dense)
    )

    rows: list[dict[str, Any]] = []

    for query_id in query_ids:
        lexical = bm25.get(
            query_id,
            {},
        )
        semantic = dense.get(
            query_id,
            {},
        )

        bm25_rank = lexical.get(
            "first_gold_rank",
            "",
        )
        dense_rank = semantic.get(
            "first_gold_rank",
            "",
        )

        bm25_numeric = rank_value(
            bm25_rank
        )
        dense_numeric = rank_value(
            dense_rank
        )

        if dense_numeric < bm25_numeric:
            winner = "dense"
        elif bm25_numeric < dense_numeric:
            winner = "bm25"
        else:
            winner = "tie"

        rows.append(
            {
                "query_id": query_id,
                "answerability":
                    semantic.get(
                        "answerability",
                        lexical.get(
                            "answerability",
                            "",
                        ),
                    ),
                "bm25_first_gold_rank":
                    bm25_rank,
                "dense_first_gold_rank":
                    dense_rank,
                "rank_delta_dense_minus_bm25":
                    (
                        ""
                        if (
                            bm25_numeric == 999
                            or dense_numeric == 999
                        )
                        else dense_numeric
                        - bm25_numeric
                    ),
                "winner": winner,
                "bm25_hit_at_5":
                    lexical.get(
                        "hit_at_5",
                        "",
                    ),
                "dense_hit_at_5":
                    semantic.get(
                        "hit_at_5",
                        "",
                    ),
                "dense_relevant_hit_at_5":
                    semantic.get(
                        "relevant_hit_at_5",
                        "",
                    ),
                "bm25_top1_chunk_id":
                    lexical.get(
                        "top1_chunk_id",
                        "",
                    ),
                "dense_top1_chunk_id":
                    semantic.get(
                        "top1_chunk_id",
                        "",
                    ),
            }
        )

    write_csv(
        OUTPUT_PATH,
        rows,
    )

    winner_counts: dict[str, int] = {}

    for row in rows:
        winner = str(row["winner"])
        winner_counts[winner] = (
            winner_counts.get(
                winner,
                0,
            )
            + 1
        )

    print(
        "\nBM25 vs dense comparison"
    )
    print("=" * 88)
    print(
        f"Query rows: {len(rows)}"
    )
    print(
        f"First-Gold rank winners: "
        f"{winner_counts}"
    )
    print(
        f"Comparison CSV: "
        f"{OUTPUT_PATH}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
