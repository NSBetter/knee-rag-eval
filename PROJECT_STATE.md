# Knee RAG Eval — Project State

**Last updated:** 2026-08-11

## 1. Current phase

Retrieval evaluation phase is functionally complete.

The project is preparing to transition from:

Retrieval Benchmark
→ Retrieval baseline comparison
→ Default retriever selection

to:

Generation pipeline
→ Automated answer evaluation
→ Version regression evaluation

Do not start generation/evaluation development until the current retrieval artifacts
have been audited and committed.

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

Before starting the generation pipeline:

1. audit the current uncommitted retrieval-related files;
2. determine which files are valid final artifacts;
3. remove or archive temporary artifacts if needed;
4. commit the completed Retrieval stage;
5. create and verify `RUNBOOK.md`.

Only after this gate is complete should development move to generation and automated
answer evaluation.

---

## 9. Next planned development phase

Generation + automated evaluation MVP.

Planned high-level sequence:

1. define generation input/output contract;
2. run Dense retrieval for benchmark questions;
3. send retrieved evidence to the generation model;
4. save model answer and full run metadata;
5. add deterministic / rule-based evaluation;
6. add LLM-as-a-Judge evaluation;
7. calibrate automated evaluation against manual medical review;
8. compare model / prompt versions and generate regression reports.

This section is a roadmap only. Development should proceed one verified step at a time.

---

## 10. Source of truth

For future development conversations:

- `PROJECT_STATE.md` = current project status
- `DECISIONS.md` = accepted design decisions and their rationale
- `RUNBOOK.md` = verified commands only
- Git history = implementation history
- generated experiment artifacts = empirical evidence

Chat history should not override these repository records.
