# RUNBOOK

本文件只记录已经实际验证通过的运行方法。

## Gold Corpus v1.3

运行命令：

    python src/build_gold_corpus.py

当前验证结果：

- Total chunks: 74
- SRC001: 45
- SRC003: 29
- Warnings: 23

构建依赖本地受限文件：

data/processed/private/gold_corpus_recommendation_overrides_v1_3.json

该文件不进入Git；缺失必需Override时构建应直接失败。

## Retrieval

Frozen benchmark: retrieval_eval_v1_pilot_frozen

MVP default retriever: Qwen3-Embedding-0.6B Dense Retrieval

BM25保留为词法基线；BM25 + Dense RRF保留为Hybrid消融实验。
