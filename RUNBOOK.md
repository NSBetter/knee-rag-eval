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

## Generation + Automated Evaluation MVP v1

### 1. 构建 Generation Input

    python src/build_generation_inputs.py

当前验证范围：

- Frozen Pilot: 12 queries
- Retrieval: Qwen3 Dense
- Top-K: 5
- Evidence rows: 60

### 2. Generation Dry Run

    python src/run_generation.py --dry-run --limit 1

用于验证 Generation Input、Evidence 格式化和 Prompt 构建，不调用模型。

### 3. DeepSeek V4 Flash Generation

运行前需在本地环境中设置：

- `GENERATION_API_BASE`
- `GENERATION_API_KEY`
- `GENERATION_MODEL`

当前验证模型：

`deepseek-v4-flash`

完整 Pilot：

    python src/run_generation.py \
      --output data/processed/generation_runs/deepseek_v4_flash_generation_v1.jsonl

验证结果：

- Queries: 12
- Success: 12
- Errors: 0

### 4. Deterministic Rules

    python src/evaluate_generation_rules.py

验证结果：

- Queries: 12
- Passed: 12
- Failed: 0

### 5. LLM-as-a-Judge

运行前需在本地环境中设置：

- `JUDGE_API_BASE`
- `JUDGE_API_KEY`
- `JUDGE_MODEL`

当前验证模型：

`deepseek-v4-pro`

完整 Pilot：

    python src/run_generation_judge.py \
      --output data/processed/evaluation_runs/deepseek_v4_flash_generation_v1_judge.jsonl

验证结果：

- Queries: 12
- Success: 12
- Errors: 0

### 6. Evaluation Summary

    python src/summarize_generation_evaluation.py

当前结果：

- Deterministic pass: 12 / 12
- Judge pass: 12 / 12
- Unsupported claims: 1
- Manual review required: 2

### 7. Human Review Queue

    python src/build_manual_review_queue.py

当前自动筛出：

- RET-002
- RET-012

### 8. Judge-Human Calibration

人工医学复核标签写回 Review Artifact 后运行：

    python src/evaluate_judge_calibration.py

当前首轮 Calibration：

- Cases: 2
- Field agreement: 10 / 10

该样本量不足以证明 Judge 已具有充分泛化可靠性。

