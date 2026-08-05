"""Run a dependency-free BM25 lexical retrieval baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/processed/gold_corpus/gold_corpus_v1_3.csv"
BENCHMARK = ROOT / "data/benchmark/retrieval_eval_v1.csv"
CONFIG = ROOT / "configs/retrieval_bm25_v1.json"
LOCAL_RESULTS = (
    ROOT / "data/processed/retrieval_runs/bm25_char_bigram_v1_results.csv"
)
PUBLIC_METRICS = ROOT / "docs/retrieval_bm25_v1_metrics.csv"
PUBLIC_QUERY_METRICS = ROOT / "docs/retrieval_bm25_v1_query_metrics.csv"


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = [
            {k: (v or "").strip() for k, v in row.items()}
            for row in reader
        ]
    return rows, cols


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ids(value: str) -> list[str]:
    return [x.strip() for x in value.split("|") if x.strip()]


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in re.finditer(
        r"[\u4e00-\u9fff]+|[a-z0-9]+(?:[._/-][a-z0-9]+)*",
        text,
    ):
        segment = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            chars = list(segment)
            tokens.extend(chars)
            tokens.extend(
                chars[i] + chars[i + 1]
                for i in range(len(chars) - 1)
            )
        else:
            tokens.append(segment)
    return tokens


class BM25:
    def __init__(
        self,
        documents: list[list[str]],
        k1: float,
        b: float,
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.lengths = [len(doc) for doc in documents]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.term_frequencies = [Counter(doc) for doc in documents]

        document_frequency: Counter[str] = Counter()
        for doc in documents:
            document_frequency.update(set(doc))

        n = len(documents)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def score(self, query: list[str]) -> list[float]:
        scores = [0.0] * len(self.documents)
        for term, query_frequency in Counter(query).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, frequencies in enumerate(self.term_frequencies):
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                denominator = (
                    tf
                    + self.k1
                    * (
                        1 - self.b
                        + self.b
                        * self.lengths[i]
                        / self.average_length
                    )
                )
                scores[i] += (
                    idf * tf * (self.k1 + 1) / denominator
                    * query_frequency
                )
        return scores


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("pilot", "all"), default="pilot")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    corpus, corpus_cols = read_csv(CORPUS)
    benchmark, benchmark_cols = read_csv(BENCHMARK)

    for col in ("chunk_id", "source_id", "title_path", "display_title", "text"):
        if col not in corpus_cols:
            raise ValueError(f"Corpus missing column: {col}")
    for col in (
        "query_id", "phase", "query", "answerability",
        "gold_chunk_ids", "review_status",
    ):
        if col not in benchmark_cols:
            raise ValueError(f"Benchmark missing column: {col}")

    title_repeat = int(config["title_repeat"])
    document_tokens = []
    for row in corpus:
        title = " ".join([row["title_path"], row["display_title"]])
        document_tokens.append(
            tokenize(" ".join([title] * title_repeat + [row["text"]]))
        )

    index = BM25(
        document_tokens,
        float(config["k1"]),
        float(config["b"]),
    )

    selected = [
        row for row in benchmark
        if row["review_status"] == "verified"
        and (args.scope == "all" or row["phase"] == "pilot")
    ]
    if not selected:
        raise ValueError("No verified benchmark rows selected.")

    local_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    h1: list[float] = []
    h3: list[float] = []
    h5: list[float] = []
    r5: list[float] = []
    mrr10: list[float] = []
    answerable = 0
    unanswerable = 0
    top_k = int(config["top_k"])

    for query_row in selected:
        scores = index.score(tokenize(query_row["query"]))
        ranking = sorted(
            range(len(corpus)),
            key=lambda i: (-scores[i], corpus[i]["chunk_id"]),
        )[:top_k]
        ranked_ids = [corpus[i]["chunk_id"] for i in ranking]
        gold = set(ids(query_row["gold_chunk_ids"]))

        first_gold_rank: int | str = ""

        if query_row["answerability"] == "answerable":
            answerable += 1
            gold_ranks = [
                rank for rank, chunk_id in enumerate(ranked_ids, start=1)
                if chunk_id in gold
            ]
            first_gold_rank = min(gold_ranks) if gold_ranks else ""
            h1.append(float(bool(gold & set(ranked_ids[:1]))))
            h3.append(float(bool(gold & set(ranked_ids[:3]))))
            h5.append(float(bool(gold & set(ranked_ids[:5]))))
            r5.append(len(gold & set(ranked_ids[:5])) / len(gold))
            mrr10.append(1.0 / first_gold_rank if first_gold_rank else 0.0)
        else:
            unanswerable += 1

        query_rows.append({
            "run_id": config["run_id"],
            "scope": args.scope,
            "query_id": query_row["query_id"],
            "answerability": query_row["answerability"],
            "gold_count": len(gold),
            "first_gold_rank": first_gold_rank,
            "hit_at_1": "" if not gold else int(bool(gold & set(ranked_ids[:1]))),
            "hit_at_3": "" if not gold else int(bool(gold & set(ranked_ids[:3]))),
            "hit_at_5": "" if not gold else int(bool(gold & set(ranked_ids[:5]))),
            "recall_at_5": (
                "" if not gold else round(len(gold & set(ranked_ids[:5])) / len(gold), 6)
            ),
            "top1_score": round(scores[ranking[0]], 6),
            "top1_chunk_id": ranked_ids[0],
        })

        for rank, corpus_index in enumerate(ranking, start=1):
            chunk = corpus[corpus_index]
            local_rows.append({
                "run_id": config["run_id"],
                "scope": args.scope,
                "query_id": query_row["query_id"],
                "query": query_row["query"],
                "answerability": query_row["answerability"],
                "gold_chunk_ids": query_row["gold_chunk_ids"],
                "rank": rank,
                "retrieved_chunk_id": chunk["chunk_id"],
                "retrieved_source_id": chunk["source_id"],
                "retrieved_title_path": chunk["title_path"],
                "score": round(scores[corpus_index], 6),
                "is_gold": int(chunk["chunk_id"] in gold),
            })

    metrics = [{
        "run_id": config["run_id"],
        "scope": args.scope,
        "corpus_version": config["corpus_version"],
        "benchmark_version": config["benchmark_version"],
        "retriever": "BM25",
        "tokenization": config["tokenization"],
        "k1": config["k1"],
        "b": config["b"],
        "title_repeat": config["title_repeat"],
        "evaluated_answerable_queries": answerable,
        "reported_unanswerable_queries": unanswerable,
        "hit_at_1": round(average(h1), 6),
        "hit_at_3": round(average(h3), 6),
        "hit_at_5": round(average(h5), 6),
        "recall_at_5": round(average(r5), 6),
        "mrr_at_10": round(average(mrr10), 6),
    }]

    write_csv(LOCAL_RESULTS, local_rows)
    write_csv(PUBLIC_METRICS, metrics)
    write_csv(PUBLIC_QUERY_METRICS, query_rows)

    result = metrics[0]
    print("\nBM25 retrieval baseline")
    print("=" * 88)
    print(
        f"Scope: {args.scope}; answerable: {answerable}; "
        f"unanswerable: {unanswerable}"
    )
    print(
        f"Hit@1={result['hit_at_1']} "
        f"Hit@3={result['hit_at_3']} "
        f"Hit@5={result['hit_at_5']} "
        f"Recall@5={result['recall_at_5']} "
        f"MRR@10={result['mrr_at_10']}"
    )
    print(f"Local results: {LOCAL_RESULTS}")
    print(f"Public metrics: {PUBLIC_METRICS}")
    print(f"Public query metrics: {PUBLIC_QUERY_METRICS}")
    print("=" * 88)


if __name__ == "__main__":
    main()
