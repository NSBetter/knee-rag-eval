# Knee RAG Eval — Project State

**Last updated:** 2026-08-11

## 1. Current phase

Retrieval evaluation is complete and committed.

Generation + automated evaluation MVP v1 is now functionally complete on the
12-query frozen Pilot Benchmark.

Completed pipeline:

Retrieval Benchmark
→ Qwen3 Dense Retrieval
→ Generation input construction
→ DeepSeek V4 Flash answer generation
→ Deterministic rule evaluation
→ DeepSeek V4 Pro LLM-as-a-Judge
→ Human medical calibration

Current Pilot results:

- Generation: 12 / 12 successful
- Deterministic evaluation: 12 / 12 passed
- LLM Judge: 12 / 12 successful
- Judge pass: 12 / 12
- Unsupported claims detected: 1
- Manual-review cases: 2
- Human calibration cases: 2
- Judge-Human field agreement: 10 / 10

The 2-case calibration set is too small to establish general Judge reliability.
Do not continue tuning the Judge against the current 12-query Pilot. Freeze the
current evaluation MVP as v1 and expand the Benchmark before further evaluator
iteration.

---

## 2. Repository state

- Branch: `main`
- Latest known commit:
  - `7e72236 docs: update project overview and repository guide`
- Working tree: NOT clean
- There are modified and untracked retrieval benchmark / freeze artifacts that still
  require audit before the next development phase.

---

## 3. Frozen corpus

MVP corpus version:

`gold_v1_3`

Included sources:

- SRC001 — 《中国骨关节炎诊疗指南（2024版）》
- SRC003 — 《中国膝骨关节炎临床药物治疗专家共识》

Other sources remain excluded from the MVP retrieval index.

---

## 4. Frozen retrieval benchmark

Benchmark version:

`retrieval_eval_v1_pilot_frozen`

Current Pilot:

- 12 total queries
- 11 answerable
- 1 unanswerable
- single-evidence and multi-evidence cases included
- cross-source evidence included

After freeze, Query / Gold / Support labels must not be modified based on retrieval
performance.

If a factual annotation error is discovered, create a new benchmark version and rerun
all retrieval baselines.

---

## 5. Retrieval baselines

Evaluated retrievers:

1. BM25
2. Qwen3 Dense Retrieval
3. BM25 + Dense RRF

Frozen Pilot results:

| Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.7273 | 0.8182 | 0.8182 | 0.7424 | 0.7857 |
| Qwen3 Dense | 0.8182 | 1.0000 | 1.0000 | 0.9697 | 0.8939 |
| BM25 + Dense RRF | 0.7273 | 0.8182 | 0.9091 | 0.8788 | 0.8061 |

---

## 6. Current default retriever

MVP default retriever:

`Qwen3-Embedding-0.6B` Dense Retrieval

BM25 is retained as the lexical baseline.

Equal-weight RRF is retained only as a hybrid ablation result and is not the MVP
default retriever.

Hybrid retrieval should not be re-tuned on the current 12-query Pilot to avoid
overfitting.

---

## 7. Known retrieval limitation

RET-006 remains an important multi-evidence failure-analysis case.

Its third Gold chunk does not enter the Top 10 candidates of either BM25 or Dense,
so equal-weight RRF cannot recover that missing evidence.

This limitation should be retained as an error-analysis example rather than tuned
away on the frozen Pilot.

---

## 8. Current development gate

Generation + automated evaluation MVP v1 has completed its first end-to-end
Pilot run.

Before implementing evaluator v2 or changing the current Judge rubric:

1. document and freeze the current Generation/Evaluation MVP v1;
2. preserve the current 12-query Pilot as a development and smoke-test set;
3. expand the evaluation Benchmark with additional question types, difficulty,
   evidence scopes, and unanswerable cases;
4. run the frozen evaluator v1 on the expanded Benchmark;
5. use new error analysis and human medical calibration to justify v2 changes.

This gate is intended to reduce overfitting to the current 12-query Pilot.

---

## 9. Next planned development phase

Benchmark expansion and evaluator generalization validation.

The next stage should prioritize adding new evaluation cases rather than
continuing to optimize the current Judge on the existing Pilot.

Planned sequence:

1. freeze the current Generation + Evaluation MVP as v1;
2. define the expanded Benchmark composition;
3. add and medically review new evaluation questions;
4. run the unchanged v1 pipeline on the expanded Benchmark;
5. collect Judge-Human disagreements and other failure cases;
6. perform error analysis;
7. introduce evaluator v2 only when new evidence supports a change;
8. later compare model, prompt, retriever, and evaluator versions through
   regression runs.
