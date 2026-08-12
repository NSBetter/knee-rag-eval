# 基于临床指南的膝痛与膝骨关节炎 RAG 问答及医学安全评测

**Guideline-Grounded RAG QA and Medical Safety Evaluation for Knee Pain and Knee Osteoarthritis**

> 基于中文临床指南与专家共识，构建可审计的医学知识语料、Retrieval Benchmark、RAG 回答生成和自动化评测流程，用于研究医学 RAG 中的证据召回、回答完整性、Groundedness 与 Unsupported Claim。

> 当前已完成 **Gold Corpus、Retrieval Benchmark、Retriever 选型，以及 Generation + Automated Evaluation MVP v1**。下一阶段将扩展 Benchmark，并使用未参与开发的问题验证 Evaluator v1 的泛化能力。

本项目仅用于研究和工程演示，不提供医疗建议，也不替代医生诊疗。

---

## 项目概览

当前已经建立如下端到端流程：

```text
临床指南 / 专家共识
        ↓
Gold Corpus
        ↓
Retrieval Benchmark
        ↓
Qwen3 Dense Retrieval
        ↓
Generation Input
        ↓
DeepSeek V4 Flash
        ↓
Deterministic Rules
        +
DeepSeek V4 Pro LLM Judge
        ↓
Manual Review / Human Calibration
```

项目重点解决：

1. 如何将临床指南转化为可检索、可追溯的医学 Evidence；
2. 如何评估 Retriever 是否召回回答问题所必需的证据；
3. 如何约束生成模型仅基于 Retrieval Evidence 回答；
4. 如何结合规则、LLM-as-a-Judge 和人工复核进行自动化评测。

---

## 当前数据与 Benchmark

### Gold Corpus

当前冻结版本：

```text
gold_v1_3
```

正式索引来源：

* `SRC001`：中国骨关节炎诊疗指南（2024）
* `SRC003`：中国膝骨关节炎临床药物治疗专家共识

Corpus 保留稳定的：

* `chunk_id`
* `source_id`
* `title_path`
* `text_sha256`

关键表格、推荐意见和证据边界经过人工核验。

### Retrieval Pilot

当前完成验证并用于 MVP 开发的 Pilot Benchmark：

| 类型   | 数量 |
| ---- | -: |
| 总问题  | 12 |
| 可回答  | 11 |
| 不可回答 |  1 |
| 单证据  |  8 |
| 多证据  |  3 |
| 跨来源  |  1 |

每个问题人工标注 `Gold Chunk`、`Support Chunk`、预期来源和 Evidence Scope，并通过冻结文件与 SHA-256 保持版本稳定。

---

## Retrieval 实验

| Retriever        |     Hit@1 |      Hit@3 |      Hit@5 |  Recall@5 |    MRR@10 |
| ---------------- | --------: | ---------: | ---------: | --------: | --------: |
| BM25             |     72.7% |      81.8% |      81.8% |     74.2% |     78.6% |
| Qwen3 Dense      | **81.8%** | **100.0%** | **100.0%** | **97.0%** | **89.4%** |
| BM25 + Dense RRF |     72.7% |      81.8% |      90.9% |     87.9% |     80.6% |

当前 MVP 默认 Retriever：

```text
Qwen/Qwen3-Embedding-0.6B
```

主要结论：

* Dense 对口语表达与指南术语之间的语义差异处理更好；
* BM25 保留较好的关键词可解释性；
* 等权 RRF 在当前 Pilot 中没有超过 Dense；
* Hybrid Retrieval 并不天然优于最强单一 Retriever。

> Pilot 规模较小，以上指标主要用于验证 Pipeline 和分析失败模式，不代表生产环境性能。

---

## Generation + Automated Evaluation MVP v1

Retrieval 完成后，项目进一步实现：

```text
Dense Top-5 Evidence
→ Generation Input
→ Evidence-grounded Answer
→ Deterministic Rules
→ LLM-as-a-Judge
→ Manual Review
→ Human Calibration
```

### Generator

当前 Baseline：

```text
deepseek-v4-flash
```

Generation Prompt 要求模型仅依据提供的 Evidence 回答，并在证据不足时明确说明。

Generation Input 不包含：

```text
gold_chunk_ids
supporting_chunk_ids
is_gold
is_support
```

Gold / Support 标签只在生成完成后用于 Evaluation，避免 Label Leakage。

### Automated Evaluation

当前 Judge：

```text
deepseek-v4-pro
```

核心评测维度：

* Relevance
* Groundedness
* Completeness
* Unsupported Claim
* Unanswerable Behavior

同时使用 Deterministic Rules 检查生成失败、空回答、Evidence 数量和不可回答场景等可程序化问题。

---

## MVP v1 结果

冻结的 12 条 Pilot 完成端到端测试：

| 指标                      |      结果 |
| ----------------------- | ------: |
| Generation Success      | 12 / 12 |
| Deterministic Rule Pass | 12 / 12 |
| Judge Success           | 12 / 12 |
| Judge Overall Pass      | 12 / 12 |
| Unsupported Claim       |       1 |
| Manual Review           |       2 |
| Mean Relevance          |   2.000 |
| Mean Groundedness       |   1.917 |
| Mean Completeness       |   1.917 |

2 个边界案例经过人工复核后，Judge 与 Human 在 5 个核心字段上的结果为：

```text
10 / 10 field agreement
```

该结果仅用于 MVP 校准，**不能解释为 Judge 准确率 100%**，因为人工复核样本目前只有 2 个。

因此当前已经冻结 Evaluator v1，不再继续针对相同 12 条 Pilot 调整 Judge。

---

## 数据治理

由于临床指南版权和使用边界，公开仓库不提供：

* 原始指南 PDF；
* 完整指南正文；
* 本地 Gold Corpus 全文；
* 受限来源的人工 Override 文本；
* Embedding Cache 与模型权重；
* 含完整医学 Evidence 的本地运行结果；
* API Key。

公开仓库主要保留：

* Corpus / Retrieval / Generation / Evaluation 代码；
* 配置文件；
* Benchmark 结构；
* 公开安全的实验指标；
* 工程决策和版本记录。

---

## 我的主要工作

* 临床指南筛选、医学证据审计与数据边界定义；
* Gold Corpus 与 Chunk 规则设计；
* Retrieval Benchmark 和 Gold / Support Evidence 标注；
* BM25、Dense、RRF Retrieval Baseline 构建与错误分析；
* Generation I/O、Evidence-grounded Prompt 与批量生成流程设计；
* Deterministic Rules 与 LLM-as-a-Judge 评测体系设计；
* Unsupported Claim、Unanswerable Behavior 与人工复核规则设计；
* Judge-Human Calibration；
* Benchmark Freeze、SHA-256 校验和 Public / Private Data Governance。

开发过程中使用 AI 辅助完成部分代码实现；医学证据边界、Benchmark 设计、评测规则、人工标注和结果审计由人工完成。

---

## 当前状态

### 已完成

* [x] 临床指南审计与 Corpus Engineering
* [x] Gold Corpus v1.3
* [x] Retrieval Pilot Benchmark
* [x] BM25 / Qwen3 Dense / RRF 对比
* [x] 默认 Retriever 选型
* [x] Generation Pipeline
* [x] Deterministic Evaluation
* [x] LLM-as-a-Judge
* [x] Manual Review Queue
* [x] Judge-Human Calibration
* [x] Generation + Automated Evaluation MVP v1

### 下一阶段

* [ ] 扩展 Benchmark
* [ ] 完成新增问题的医学审核和 Gold Evidence 标注
* [ ] 完善 Dev / Test Split
* [ ] 使用未见问题运行冻结的 Evaluator v1
* [ ] 收集新的 Judge-Human Disagreement
* [ ] 根据错误分析决定是否开发 Evaluator v2

---

## 项目文档

* [`PROJECT_STATE.md`](PROJECT_STATE.md)：当前项目状态
* [`DECISIONS.md`](DECISIONS.md)：关键工程决策
* [`RUNBOOK.md`](RUNBOOK.md)：经过验证的运行流程
* [`docs/repository_guide.md`](docs/repository_guide.md)：仓库结构
* [`docs/generation_eval_mvp_v1_summary.md`](docs/generation_eval_mvp_v1_summary.md)：Generation + Evaluation MVP v1 阶段总结

完整运行命令以 [`RUNBOOK.md`](RUNBOOK.md) 为准。

---

## 当前结论

项目目前已经完成两个连续的 MVP 闭环：

```text
Corpus
→ Retrieval Benchmark
→ Retriever Evaluation
→ Retriever Selection
```

以及：

```text
Retrieval Evidence
→ Generation
→ Automated Evaluation
→ Human Calibration
```

下一阶段的重点不是继续优化当前 12 条 Pilot 的结果，而是使用新的 Benchmark 验证当前 Evaluator 是否能够泛化。

---

## 免责声明

本项目仅用于技术研究、工程展示和医学 AI 评测方法探索，不构成医疗建议，不用于真实患者的诊断或治疗决策。
