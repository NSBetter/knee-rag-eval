"""Prepare local files for retrieval benchmark authoring.

Reads the frozen Gold Corpus and writes a local chunk catalog containing
text for manual evidence labeling. It also installs the public benchmark
template when the file does not yet exist.

Usage:
    uv run python src/prepare_retrieval_benchmark.py

Optional:
    uv run python src/prepare_retrieval_benchmark.py --force-template
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "gold_corpus"
    / "gold_corpus_v1_3.csv"
)

CATALOG_PATH = (
    ROOT
    / "data"
    / "processed"
    / "reviews"
    / "gold_chunk_catalog_v1_3.csv"
)

TEMPLATE_SOURCE = (
    ROOT
    / "configs"
    / "retrieval_eval_v1_template.csv"
)

BENCHMARK_PATH = (
    ROOT
    / "data"
    / "benchmark"
    / "retrieval_eval_v1.csv"
)

REQUIRED_CORPUS_COLUMNS = {
    "chunk_id",
    "source_id",
    "node_id",
    "content_type",
    "title_path",
    "display_title",
    "pdf_page_start",
    "pdf_page_end",
    "text",
}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-template",
        action="store_true",
        help="Overwrite retrieval_eval_v1.csv from the template.",
    )
    args = parser.parse_args()

    rows, columns = read_csv(CORPUS_PATH)

    missing = sorted(
        REQUIRED_CORPUS_COLUMNS - set(columns)
    )
    if missing:
        raise ValueError(
            f"Gold corpus is missing columns: {missing}"
        )

    if not rows:
        raise ValueError("Gold corpus is empty.")

    chunk_ids = [row["chunk_id"] for row in rows]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Gold corpus contains duplicate chunk_id values."
        )

    catalog_rows: list[dict[str, str | int]] = []

    for row in sorted(
        rows,
        key=lambda item: (
            item["source_id"],
            item["title_path"],
            item["chunk_id"],
        ),
    ):
        text = row["text"]
        catalog_rows.append(
            {
                "chunk_id": row["chunk_id"],
                "source_id": row["source_id"],
                "node_id": row["node_id"],
                "content_type": row["content_type"],
                "title_path": row["title_path"],
                "display_title": row["display_title"],
                "pdf_page_start": row["pdf_page_start"],
                "pdf_page_end": row["pdf_page_end"],
                "char_count": row.get(
                    "char_count",
                    str(len(text)),
                ),
                "text": text,
                "authoring_notes": "",
            }
        )

    CATALOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CATALOG_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                catalog_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(catalog_rows)

    BENCHMARK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    template_action = "kept_existing"

    if args.force_template or not BENCHMARK_PATH.exists():
        if not TEMPLATE_SOURCE.exists():
            raise FileNotFoundError(
                f"Template not found: {TEMPLATE_SOURCE}"
            )
        shutil.copyfile(
            TEMPLATE_SOURCE,
            BENCHMARK_PATH,
        )
        template_action = "created"

    print("\nRetrieval benchmark preparation")
    print("=" * 88)
    print(f"Gold chunks: {len(catalog_rows)}")
    print(f"Local chunk catalog: {CATALOG_PATH}")
    print(
        f"Benchmark template: {BENCHMARK_PATH} "
        f"({template_action})"
    )
    print("-" * 88)
    print(
        "Next: complete RET-001 through RET-012, "
        "then run validate_retrieval_benchmark.py."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
