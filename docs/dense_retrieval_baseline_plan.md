# Dense Retrieval Baseline

## Goal

Compare a local semantic retriever with the audited BM25 baseline using
the same Gold Corpus and Retrieval Benchmark.

## Fixed inputs

- Corpus: `gold_v1_3`
- Benchmark: `retrieval_eval_v1`
- Scope for the first run: `pilot`
- Main metrics: Hit@1, Hit@3, Hit@5, Recall@5, MRR@10
- Auxiliary metric: Gold-or-support RelevantHit@K

## Model

`Qwen/Qwen3-Embedding-0.6B`

The first experiment uses the model's predefined `query` prompt for user
queries and no prompt for documents. Corpus and query embeddings are
normalized, so matrix multiplication computes cosine similarity.

## Retrieval

The corpus has only dozens of chunks. Exact scoring over every vector is
therefore used instead of FAISS, HNSW, or a vector database.

## Interpretation

The dense model should be compared query by query with BM25. RET-004 is
a key semantic-mismatch case because the user says "片子" while the Gold
evidence uses terms such as "影像学检查", "X线", and "MRI".

Do not change Gold labels merely because a new model ranks a different
chunk highly.

Due to Hugging Face network timeouts, the official model weights were
downloaded from the ModelScope mirror and loaded through a local
symbolic link. The canonical model ID remains
Qwen/Qwen3-Embedding-0.6B.