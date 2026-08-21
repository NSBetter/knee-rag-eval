# 阶段0：冻结现状与限定学习范围

## 1. Frozen Pilot 定位

当前12题是已参与Retriever、Generation和Judge开发的Frozen Pilot / Smoke Test Set，用于验证数据契约、端到端链路、失败分析和回归检查。它不是独立测试集，不能用于证明系统或Judge对未见医学问题的泛化能力。

## 2. 当前项目已经完成什么

- 将正式索引范围限定为 `SRC001` 和 `SRC003`，完成并冻结 `gold_v1_3` Gold Corpus。
- 完成12题Retrieval Pilot的医学标注、验证、快照和SHA-256清单。
- 完成BM25、Qwen3 Dense和BM25 + Dense RRF基线比较，选择Qwen3 Dense作为MVP默认Retriever。
- 建立Dense Top-5到Generation Input的数据管道，并排除Gold/Support标签以防止评测泄漏。
- 使用`generation_prompt_v1`和DeepSeek V4 Flash完成12题Generation运行。
- 完成Deterministic Rules、`generation_judge_v1`、自动人工复核队列和首轮Judge-Human Calibration。
- 完成第一次Retrieval、Generation、Automated Evaluation与Human Calibration端到端闭环。

## 3. 当前项目不做什么

- 不作为临床产品，不用于真实患者诊断、处方、治疗决策或临床部署。
- 不把12题结果解释为Retriever、Generator或Judge的生产性能和泛化能力。
- 不宣称Judge准确率为100%；现有人工校准只有2个案例、10个字段比较。
- 不把Judge pass等同于回答完全正确；当前结果仍包含Unsupported Claim和Completeness边界案例。
- 不在阶段0完成无RAG对照、模型横评、完整医学安全评测或正式Demo实现。
- 不自动把`learning_lab/`中的观察写入README、PROJECT_STATE或正式项目结论。

## 4. 暂停事项

在正式Demo设计完成并通过进入条件前：

- 暂不扩题。
- 暂不重写Judge或继续针对当前12题调优Rubric。
- 暂不迁移、替换或重构主项目框架。

## 5. 冻结资产和保护边界

### 冻结或受保护资产

- Corpus基线：`gold_v1_3`及其已核验构建输入、结构映射、人工表格证据和受限Override。
- Benchmark快照：`data/benchmark/frozen/retrieval_eval_v1_pilot_frozen.csv`。
- 冻结清单：`data/benchmark/frozen/retrieval_eval_v1_pilot_manifest.json`和公开Manifest。
- 题目标签：RET-001至RET-012的Query、Answerability、Gold、Support、Source和Evidence Scope。
- Retrieval v1配置、运行结果、公开指标和Retriever选型结论。
- Generation/Evaluation v1的I/O contract、Prompt、Judge Rubric、Generation输出、规则结果、Judge结果、汇总与人工复核结果。

### 保护规则

- 未经明确授权，不原位修改、覆盖、删除或重新生成上述资产。
- 发现事实性标注错误时，记录问题并创建新版本；不得根据当前模型排名或分数反向修改冻结标签。
- 新实验使用新版本号或新`run_id`，其输入、配置和输出默认保存在`learning_lab/`。
- 原始指南、完整Corpus正文、Embedding和含完整Evidence的结果继续遵守公开/本地数据治理边界。
- 学习实验失败、临时结果或未完成人工核验的结论不得进入正式基线。

## 6. 待解决问题

阶段0之后的学习应先回答：

1. 模型评测与RAG系统评测如何分层？
2. Benchmark由哪些部分构成？
3. OpenCompass各组件分别负责什么？
4. LLM Judge的可靠性如何验证？

这些问题应通过概念梳理和`learning_lab/`中的最小复现实验回答，不直接改造当前主Pipeline。

## 7. 进入正式Demo设计的条件

只有同时满足以下条件，才能进入正式Demo设计：

- 已形成模型评测层、Retrieval层、Generation层、RAG系统层和医学安全层的清晰边界。
- 已明确Benchmark的样本、输入、参考答案或Gold Evidence、元数据、切分、指标和人工复核组成。
- 已说明OpenCompass中Dataset、Template/Prompt、Retriever或Inferencer、Evaluator、Summarizer等组件的职责，以及哪些适用于本项目。
- 已形成LLM Judge可靠性验证方案，至少包括人工Gold、抽样策略、一致性指标、分歧分析和版本冻结原则。
- 已明确Demo目标、目标受众、最小任务、输入输出、评测指标、数据边界和非目标。
- 已确认Demo方案不会覆盖Frozen Pilot、v1配置和已有结果，并经用户明确批准进入下一阶段。

