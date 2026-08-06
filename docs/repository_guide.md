# Repository Guide

本文档用于快速了解仓库中各目录和主要文件的作用。

> 原则：`README.md` 负责说明“项目做了什么、结果如何”；本文件负责说明“代码和文件放在哪里、分别有什么作用”。

---

## 根目录

| 文件 | 作用 | 建议公开 |
|---|---|---|
| `README.md` | 项目概览、主要成果、实验结果、当前状态和运行方式 | 是 |
| `DECISIONS.md` | 记录数据源、切块、表格处理、Benchmark、检索器选型等关键工程决策 | 是 |
| `pyproject.toml` | Python 项目配置与依赖声明 | 是 |
| `uv.lock` | 锁定依赖版本，保证环境可重复 | 是 |
| `.gitignore` | 排除原始指南、模型权重、Embedding、完整语料和本地结果 | 是 |

---

## `configs/`

用于保存可重复实验所需的配置。

| 文件或模式 | 作用 |
|---|---|
| `gold_corpus_v1*.json` | 不同版本 Gold Corpus 的构建规则 |
| `gold_corpus_cleanup_v1*.json` | 语料清洗和人工修正规则；提交前需确认不包含大段受限原文 |
| `retrieval_bm25_v1.json` | BM25 分词、权重、参数和 Top-K 设置 |
| `retrieval_dense_qwen3_v1.json` | Dense 模型、本地路径、批量大小、最大长度、向量归一化和 Top-K |
| `retrieval_rrf_v1.json` | BM25 与 Dense 的 RRF 融合参数 |
| `retrieval_benchmark_freeze_v1.json` | Benchmark 冻结版本、数量约束和输出路径 |

---

## `src/`

项目主要脚本。

### 语料与审计

| 文件 | 作用 |
|---|---|
| `build_gold_corpus.py` | 按配置构建 Gold Corpus，并生成 CSV、JSONL 和诊断结果 |
| `audit_project_artifacts.py` | 检查公开文件、本地受限文件和工程产物是否符合治理规则 |

### Retrieval Benchmark

| 文件 | 作用 |
|---|---|
| `prepare_retrieval_benchmark.py` | 生成或整理 Retrieval Benchmark 初始模板 |
| `validate_retrieval_benchmark.py` | 检查字段、Gold / Support ID、证据范围和 Pilot 覆盖要求 |
| `freeze_retrieval_benchmark.py` | 复制最终 Benchmark，记录 SHA-256 和冻结清单 |

### 检索器

| 文件 | 作用 |
|---|---|
| `run_bm25_baseline.py` | 运行字符级 BM25，全量评分并计算检索指标 |
| `run_dense_baseline.py` | 使用 Qwen3 Embedding 运行本地 Dense Retrieval |
| `run_rrf_hybrid.py` | 使用 Reciprocal Rank Fusion 融合 BM25 与 Dense 排名 |

### 比较与分析

| 文件 | 作用 |
|---|---|
| `compare_retrieval_baselines.py` | 比较 BM25 与 Dense 的问题级排名 |
| `compare_retrieval_three_way.py` | 比较 BM25、Dense 与 RRF 的问题级结果 |

---

## `data/benchmark/`

Retrieval Benchmark 及其冻结版本。

| 文件或目录 | 作用 | 建议公开 |
|---|---|---|
| `retrieval_eval_v1.csv` | 当前开发用 Benchmark，包含 Query、Gold、Support 和审核状态 | 视内容而定；当前问题为人工合成，可公开 |
| `frozen/retrieval_eval_v1_pilot_frozen.csv` | 冻结后的 Pilot Benchmark | 是 |
| `frozen/retrieval_eval_v1_pilot_manifest.json` | Benchmark 与 Gold Corpus 的哈希和冻结元数据 | 是 |
| `retrieval_eval_v1_pre_audit.csv` | 标签审计前的历史快照 | 可本地保留，非必需公开 |
| `*_before_*` | 临时备份 | 不建议公开 |

---

## `data/processed/`

本地生成的处理结果。大部分不应公开。

| 子目录 | 作用 | 建议公开 |
|---|---|---|
| `gold_corpus/` | 完整 Gold Corpus 正文、review 文件和 JSONL | 否 |
| `embeddings/` | 本地语料向量缓存 | 否 |
| `retrieval_runs/` | 每个问题的完整 Top-K、Query 和详细检索结果 | 通常否 |
| `reviews/` | Benchmark 与语料人工审核输出 | 视是否包含受限正文而定 |

---

## `docs/`

公开展示的工程证据、摘要和指标。

### 语料工程

| 文件或模式 | 作用 |
|---|---|
| `engineering_artifact_inventory.md` | 工程产物清单和公开边界 |
| `gold_corpus_build_summary.csv` | 初始 Gold Corpus 构建摘要 |
| `gold_corpus_v1_1_issue_audit.csv` | v1.1 问题审计 |
| `gold_corpus_v1_2_build_summary.csv` | v1.2 构建摘要 |
| `gold_corpus_v1_2_diagnostics.csv` | v1.2 诊断结果 |
| `gold_corpus_v1_2_review_summary.csv` | v1.2 人工审核摘要 |
| `gold_corpus_v1_3_build_summary.csv` | 最终 v1.3 构建摘要 |
| `gold_corpus_v1_3_diagnostics.csv` | 最终 v1.3 诊断结果 |
| `boundary_chunk_build_summary.csv` | 文档边界和 Chunk 构建摘要 |

### Benchmark

| 文件 | 作用 |
|---|---|
| `retrieval_benchmark_plan.md` | Retrieval Benchmark 设计原则 |
| `retrieval_benchmark_v1_summary.csv` | Benchmark 验证摘要 |
| `retrieval_benchmark_v1_frozen_manifest.csv` | 冻结版本的公开清单 |

### BM25

| 文件 | 作用 |
|---|---|
| `retrieval_bm25_v1_metrics.csv` | BM25 总体指标 |
| `retrieval_bm25_v1_query_metrics.csv` | BM25 问题级指标 |

### Dense

| 文件 | 作用 |
|---|---|
| `dense_retrieval_baseline_plan.md` | Dense 基线设计说明 |
| `retrieval_dense_qwen3_v1_metrics.csv` | Dense 总体指标 |
| `retrieval_dense_qwen3_v1_query_metrics.csv` | Dense 问题级指标 |

### Hybrid 与比较

| 文件 | 作用 |
|---|---|
| `retrieval_benchmark_freeze_and_rrf_plan.md` | Benchmark 冻结和 RRF 实验计划 |
| `retrieval_bm25_vs_dense_v1.csv` | BM25 与 Dense 的逐题比较 |
| `retrieval_rrf_v1_metrics.csv` | RRF 总体指标 |
| `retrieval_rrf_v1_query_metrics.csv` | RRF 问题级指标 |
| `retrieval_bm25_dense_rrf_v1.csv` | BM25、Dense、RRF 三方逐题比较 |

---

## `models/`

本地模型权重或软链接，例如：

```text
models/Qwen3-Embedding-0.6B
```

该目录不应提交 GitHub。公开仓库只记录规范模型 ID、加载方式和实验配置。

---

## 文件命名规则

| 形式 | 含义 |
|---|---|
| `v1_3` | 主版本 1.3，文件名中使用下划线避免路径歧义 |
| `metrics.csv` | 某次运行的总体指标 |
| `query_metrics.csv` | 每个 Query 的排名和指标 |
| `results.csv` | 完整 Top-K 结果，通常只保存在本地 |
| `summary.csv` | 不含完整正文的公开摘要 |
| `diagnostics.csv` | 构建或验证异常统计 |
| `manifest.json/csv` | 版本、哈希、数量和冻结信息 |

---

## 推荐阅读路径

### 招聘者 / HR

1. `README.md`
2. `docs/retrieval_bm25_dense_rrf_v1.csv`
3. `DECISIONS.md` 中的最新检索器选型决策

### 技术面试官

1. `README.md`
2. `DECISIONS.md`
3. `src/build_gold_corpus.py`
4. `src/validate_retrieval_benchmark.py`
5. `src/run_bm25_baseline.py`
6. `src/run_dense_baseline.py`
7. `src/run_rrf_hybrid.py`
8. `docs/retrieval_*_metrics.csv`

### 医学或评测方向面试官

1. `README.md`
2. `DECISIONS.md`
3. `docs/gold_corpus_v1_3_diagnostics.csv`
4. `data/benchmark/frozen/retrieval_eval_v1_pilot_frozen.csv`
5. `docs/retrieval_bm25_dense_rrf_v1.csv`

---

## 维护建议

- `README.md` 只保留最重要的项目故事、核心结果和当前状态；
- 详细文件说明放在本文件，不要让 README 变成文件清单；
- 新增脚本或公开指标后同步更新本文件；
- 临时备份、完整语料和模型文件不要加入目录说明的公开推荐路径；
- 冻结后的 Benchmark 不直接覆盖，发现标签问题时创建新版本。
