# Generation + Automated Evaluation MVP v1 阶段总结

## 1. 这一阶段从哪里开始？

这一阶段可以从“Generation I/O contract v1”开始计算。

但需要区分两个概念：

- 从 Generation I/O contract 开始，是整个“Generation + Automated Evaluation”阶段的起点；
- 真正的“自动评测”则从 Deterministic Rules 开始；
- 真正使用另一个 LLM 对回答进行语义评测，则从 LLM-as-a-Judge 开始。

整个数据流可以概括为：

Retrieval
→ Generation Input
→ Prompt
→ Generator
→ Model Answer
→ Deterministic Rules
→ LLM-as-a-Judge
→ Human Medical Calibration

在进入这一阶段之前，Retrieval 已经完成：

- Benchmark 冻结；
- BM25、Dense、RRF 对比完成；
- Qwen3-Embedding-0.6B Dense Retrieval 被选为 MVP 默认检索器；
- 12 条 Pilot Benchmark 已被冻结，不再根据模型表现修改 Gold。

因此 Generation 阶段是在一个固定 Retrieval 基线上继续向下开发的。


## 2. Generation I/O contract v1：先规定模块之间传什么数据

新增：

configs/generation_io_v1.json

这一阶段首先没有直接写 LLM 调用，而是定义 Generation 模块的数据契约。

核心思路是：

上一阶段的 Retrieval 会输出：

query_id
→ rank
→ retrieved_chunk_id
→ score

但 Generation 真正需要的是：

query
→ Top-K 证据正文
→ Prompt
→ 模型回答

因此必须明确 Retrieval 和 Generation 之间如何传数据。

Generation Input 中包含：

- run_id
- query_id
- query
- answerability
- retrieval_run_id
- top_k
- evidence

每条 evidence 包含：

- rank
- chunk_id
- source_id
- title_path
- score
- text
- text_sha256

同时明确禁止 Generation Input 包含：

- gold_chunk_ids
- supporting_chunk_ids
- is_gold
- is_support

这是非常重要的设计。

原因是这些字段属于“评测答案”。

如果生成模型在回答问题之前已经知道哪个 Chunk 是 Gold Evidence，那么相当于把考试答案提前告诉模型，最后得到的 Generation 结果就失去了评测意义。

这一原则叫 Gold Label Leakage，也就是评测标签泄漏。


## 3. build_generation_inputs.py：把 Retrieval 结果真正变成 Generation 输入

新增：

src/build_generation_inputs.py

Dense Retrieval 的结果文件只有 retrieved_chunk_id，没有完整正文。

因此这个脚本完成了一次 Join：

Dense Retrieval Results
+
Gold Corpus
↓
retrieved_chunk_id == chunk_id
↓
取回真实 Evidence Text

然后把同一个问题的 Top-5 检索结果聚合起来。

原来 Retrieval 文件的数据类似：

RET-001 + rank 1 + chunk A

RET-001 + rank 2 + chunk B

RET-001 + rank 3 + chunk C

……

经过聚合后变成：

RET-001
→ query
→ Evidence 1
→ Evidence 2
→ Evidence 3
→ Evidence 4
→ Evidence 5

最终输出：

data/processed/generation_inputs/generation_input_v1.jsonl

结果：

- 12 条 Query
- 每条 Top-5 Evidence
- 共 60 条 Evidence

这一步仍然没有调用任何 LLM。

它只是建立 Retrieval 和 Generation 之间的数据管道。


## 4. generation_prompt_v1：固定模型回答规则

新增：

configs/generation_prompt_v1.json

Prompt 被单独保存成配置，而没有直接硬编码在生成脚本里。

主要要求包括：

- 只能依据提供的 Evidence 回答；
- 不使用 Evidence 之外的信息；
- Evidence 不足时明确说明证据不足；
- 不捏造诊断、治疗、剂量或指南内容；
- 回答保持准确、简洁；
- 优先回答与问题直接相关的信息。

为什么 Prompt 要有版本号？

因为 Prompt 本身也是一个实验变量。

以后如果有：

generation_prompt_v1
generation_prompt_v2

就可以在同一个 Benchmark 上比较：

Prompt v1 的结果
vs
Prompt v2 的结果

而不是不断修改一个 Prompt，却不知道结果变化到底来自哪里。


## 5. run_generation.py dry-run：先验证模型无关部分

新增：

src/run_generation.py

第一版没有马上连接 DeepSeek API，而是先实现 dry-run。

dry-run 做的是：

Generation Input
→ Evidence 格式化
→ System Prompt
→ User Prompt
→ 保存 Prompt Preview

但：

不调用模型
不产生真实 Answer

这样可以把两个问题分开：

第一层：
数据有没有正确进入 Prompt？

第二层：
模型 API 能不能正常生成？

如果一开始同时做这两件事，一旦失败，就很难判断问题究竟出在数据处理还是 API。


## 6. 建立模型 Backend：让 Generator 与具体模型解耦

随后给 run_generation.py 增加 OpenAI-compatible HTTP Backend。

核心思想是：

Generation Pipeline 本身不关心模型供应商。

它只需要：

system_prompt
+
user_prompt
↓
generate()
↓
answer

具体模型通过环境变量指定：

GENERATION_API_BASE
GENERATION_API_KEY
GENERATION_MODEL

这样模型调用层和 Generation 数据管道保持分离。

以后理论上可以把：

DeepSeek V4 Flash

替换成：

其他 OpenAI-compatible Model

而不需要重新设计整个 Generation Pipeline。


## 7. DeepSeek V4 Flash：建立第一版 Generator Baseline

最终 Generator 使用：

DeepSeek V4 Flash

第一版 Generation Baseline 使用 non-thinking 模式。

原因不是 thinking 一定不好，而是 baseline 最重要的是：

固定实验条件。

以后如果比较：

V4 Flash non-thinking
vs
V4 Flash thinking

应该把它作为新的实验变量，而不是依赖供应商默认行为。


## 8. RET-001 Smoke Test：第一次真实生成

在批量调用 12 个问题之前，只使用 RET-001 做真实 API 测试。

Smoke Test 的目的不是评价模型总体能力。

而是确认完整链路：

Generation Input
→ Prompt
→ DeepSeek API
→ Answer
→ JSONL Output

第一次调用出现了：

401 Authentication Required

通过单独访问 DeepSeek /models API，最终确认：

- API Base 正确；
- API Key 正确；
- deepseek-v4-flash 可用；
- deepseek-v4-pro 可用。

之后 RET-001 成功生成回答。

这一步说明整个真实 Generation 链路已经打通。


## 9. Batch Fault Tolerance：避免一条失败毁掉整批实验

在开始 12 条批量 Generation 前，对 run_generation.py 增加了单条容错。

原来的风险是：

如果第 6 条 API 调用失败，
整个 Python 程序可能中断，
前面的结果也可能无法形成完整 Run。

改进后：

每个 Query 独立执行。

每条结果保存：

generation_status = success / error

以及：

generation_error

而且每完成一条就立即写入文件。

因此：

一个 Query 失败
≠
整个 Evaluation Run 失败

这是批处理系统非常重要的工程属性。


## 10. 第一版完整 Generation Run

最终对 Frozen Pilot 的 12 条问题全部生成。

结果：

Queries: 12
Success: 12
Errors: 0

固定条件为：

Retriever:
Qwen3-Embedding-0.6B Dense Retrieval

Retrieval depth:
Top-5

Prompt:
generation_prompt_v1

Generator:
DeepSeek V4 Flash

因此这 12 条结果可以视为一个固定实验 Run。

后面如果改变模型、Prompt 或 Retriever，就可以和这个 Run 做版本比较。


## 11. Deterministic Rules：自动评测正式开始

新增：

src/evaluate_generation_rules.py

从这里开始，才真正进入“自动评测系统”。

第一层没有让 LLM 打分，而是先做确定性规则检查。

检查内容包括：

1. Generation 是否成功；
2. Answer 是否为空；
3. Evidence 数量是否等于 Top-K；
4. 对 unanswerable Query，回答是否明确表达证据不足。

结果：

Queries: 12
Passed: 12
Failed: 0

Unanswerable Query:
RET-012

其基本拒答行为也通过规则检查。

为什么需要 Rules？

因为一些问题根本不需要另一个 LLM 判断。

例如：

API 有没有成功？

Answer 是不是空字符串？

Evidence 是不是 5 条？

这些问题用普通程序判断：

更便宜
更稳定
更可重复

因此自动评测不应该全部交给 LLM。


## 12. 为什么 Rules 不能代替医学语义评测？

RET-012 是一个典型例子。

问题：

膝关节置换有哪些不同类型？

当前 Evidence 只足以支持：

人工全膝关节置换

Evidence 并不能完整回答“有哪些不同类型”。

Generator 正确意识到 Evidence 不足，但回答中又写出了：

单髁置换
髌股关节置换

这些医学概念本身可能是真实存在的。

但是：

它们没有出现在当前 Evidence 中。

对于严格 RAG 来说，这仍然属于：

Unsupported Information

普通字符串 Rules 很难可靠判断：

某个医学实体究竟有没有被 Evidence 支持。

因此需要第二层：

LLM-as-a-Judge。


## 13. generation_judge_v1：建立语义评测标准

新增：

configs/generation_judge_v1.json

Judge 主要评价：

### Relevance

模型有没有真正回答问题。

0：
基本没回答。

1：
部分回答或明显偏题。

2：
直接、聚焦地回答问题。


### Groundedness

回答中的医学陈述有没有 Evidence 支持。

0：
存在重要幻觉、冲突或核心证据外信息。

1：
主体有证据，但混入少量 unsupported information。

2：
关键医学内容都受到 Evidence 支持。


### Completeness

Evidence 已经提供的重要信息有没有被回答覆盖。

0：
严重遗漏。

1：
存在重要遗漏。

2：
主要信息基本完整。


### Unsupported Claim

这是 Boolean：

true / false

只要模型加入 Evidence 没有支持的医学事实、实体、数字、疗法、药物等，就属于 Unsupported Claim。


### Unanswerable Behavior

对于 Benchmark 标记为 unanswerable 的问题：

模型是否正确表达：

现有 Evidence 不足以完整回答。

而不是偷偷使用模型自己的医学知识补全答案。


## 14. DeepSeek V4 Pro：作为 Judge

Generation 使用：

DeepSeek V4 Flash

Judge 使用：

DeepSeek V4 Pro

这样至少避免让完全相同的模型实例直接给自己的回答打分。

Judge 被要求输出结构化 JSON，包括：

- relevance_score
- groundedness_score
- completeness_score
- unsupported_claim
- unsupported_claim_severity
- unanswerable_behavior
- overall_verdict
- reason
- unsupported_items


## 15. RET-012 暴露了第一个 Judge 缺陷

第一次 Judge RET-012 时得到：

groundedness = 2
unsupported_claim = false

也就是说：

Judge 没有识别 Generator 添加的：

单髁置换
髌股关节置换

一开始怀疑：

Judge rubric 不够严格。

于是增加 groundedness rules。

但重新运行后结果仍然没有变化。


## 16. 一个非常典型的工程 Bug：配置写了，但运行时没使用

进一步检查发现：

generation_judge_v1.json

虽然增加了：

groundedness_rules

但是：

run_generation_judge.py

并没有把这个字段放进实际发送给模型的 Prompt。

因此出现了：

“配置文件已经写了规则”

但：

“LLM 根本没有看到这些规则”

这说明一个很重要的问题：

Configuration Exists
≠
Configuration Is Used At Runtime

修复后，groundedness_rules 真正进入 Judge Prompt。


## 17. Claim-level Groundedness：从整体印象改为逐事实检查

即使增加规则后，non-thinking Judge 仍然没有识别 RET-012 的问题。

于是进一步要求 Judge：

不要先凭整体印象打分。

而是先检查 Answer 中每一个：

- 医学实体
- 分类
- 疗法
- 药物
- 剂量
- 数字
- 事实性细节

并输出：

unsupported_items

理论上：

如果 unsupported_items 不为空，

则：

unsupported_claim = true

并且 groundedness 不应为 2。


## 18. Thinking Judge：解决语义判断不足

随后把 DeepSeek V4 Pro Judge 改为 Thinking Mode。

这次 Judge 成功识别：

groundedness_score = 1

unsupported_claim = true

unsupported_items:

- 单髁置换
- 髌股关节置换

说明问题已经从：

“Judge 看不出证据外信息”

变成：

“Judge 看出来了，但 JSON 输出格式坏了”。


## 19. JSON 输出错误：语义正确 ≠ 工程结果可用

Thinking Judge 第一次返回了正确的医学判断，但 JSON 出现：

多余的大括号。

因此程序报：

JSONDecodeError

这说明自动评测系统还必须区分：

Semantic Evaluation Error

和：

Structured Output Error

模型可能：

判断是对的

但：

输出文件仍然无法被程序解析。


## 20. Judge JSON Retry：建立结构化输出容错

为 Judge Runner 增加一次自动重试。

第一次：

Judge Output
→ JSON Parse

如果失败：

再请求一次，
明确要求只输出一个合法 JSON。

第二次如果仍失败：

才记录：

judge_status = error

最终 RET-012：

Success: 1
Errors: 0

这使 Judge Runner 具备了基本的批量容错能力。


## 21. 12 条完整 LLM Judge

随后对 12 条 Generation Answer 全部运行 DeepSeek V4 Pro Judge。

结果：

Queries: 12
Success: 12
Errors: 0

至此完成：

Generator
→ Rules
→ LLM Judge

三层自动化流程。


## 22. 自动评测汇总

新增：

src/summarize_generation_evaluation.py

这个脚本把：

Deterministic Rules

和：

LLM Judge

按照 query_id 合并。

最终得到：

Queries: 12

Deterministic pass:
12

Judge pass:
12

Unsupported claims:
1

Manual review required:
2

Mean relevance_score:
2.000

Mean groundedness_score:
1.917

Mean completeness_score:
1.917

这里出现一个很重要的现象：

Judge Pass = 12

并不等于：

所有回答都是完美的。

因为 RET-012 虽然：

groundedness = 1
unsupported_claim = true

但 unsupported severity 只是 minor，
整体仍可以判为 pass。

因此：

Pass Rate

只是一个汇总指标，

不能代替：

Groundedness
Completeness
Unsupported Claim

等细粒度指标。


## 23. Human Review Queue：自动评测负责筛问题

新增：

src/build_manual_review_queue.py

自动评测并没有要求医生重新人工阅读全部 12 条。

而是根据以下情况筛选：

- Rule Fail
- Judge Error
- Judge Fail
- Unsupported Claim
- 任一语义分数低于 2

最后自动筛出：

2 条

分别是：

RET-002
RET-012

这就是 Human-in-the-loop 的核心价值：

机器先大规模筛选，
医生只集中处理边界案例。


## 24. RET-002 人工复核

人工评价：

Relevance = 2

Groundedness = 2

Completeness = 1

Unsupported Claim = false

Verdict = pass

原因：

模型回答直接回答问题，
所有医学陈述都有 Evidence 支持，

但：

遗漏了“关节畸形”这一 Evidence 中存在的常见体征。

因此问题不是 hallucination，

而是：

Incomplete Answer。


## 25. RET-012 人工复核

人工评价：

Relevance = 2

Groundedness = 1

Completeness = 2

Unsupported Claim = true

Severity = minor

Unanswerable Behavior = pass

Verdict = pass

Unsupported Items：

- 单髁置换
- 髌股关节置换

人工认为：

模型正确识别了 Evidence 不足，

但不应该主动加入 Evidence 没有提供的具体置换类型。

因此：

拒答行为正确，

但：

存在轻微 Groundedness 问题。


## 26. Human Calibration：Judge 不是 Gold

自动筛出的两条问题随后由人工医学复核。

人工标签被写回结构化 artifact。

然后使用：

src/evaluate_judge_calibration.py

比较：

LLM Judge
vs
Human Medical Review

比较字段：

- Relevance
- Groundedness
- Completeness
- Unsupported Claim
- Overall Verdict


## 27. 当前 Judge-Human Calibration 结果

RET-002：

5 / 5 字段一致。

RET-012：

5 / 5 字段一致。

最终：

Calibration cases: 2

Field agreement:

10 / 10

这意味着：

当前 LLM Judge 在这两个边界案例上的判断与人工医学复核一致。


## 28. 但为什么不能说“Judge 已经很可靠”？

因为：

Calibration Case 只有 2 条。

10 / 10 一致只能说明：

这两个案例一致。

不能说明：

换成 50 条、100 条新的医学问题，
Judge 仍然保持同样可靠。

因此当前项目应该描述为：

“完成了首轮 Human Calibration”

而不是：

“已经证明 LLM Judge 高度可靠”。


## 29. 当前 Automated Evaluation MVP v1 的完整结构

现在整个项目已经形成：

Frozen Benchmark
↓
Qwen3 Dense Retrieval
↓
Top-5 Evidence
↓
Generation Input Builder
↓
generation_prompt_v1
↓
DeepSeek V4 Flash Generator
↓
Generation Run
↓
Deterministic Rules
↓
DeepSeek V4 Pro Thinking Judge
↓
Evaluation Summary
↓
Automatic Manual Review Queue
↓
Human Medical Review
↓
Judge-Human Calibration

这已经是一个真正意义上的：

Automated LLM/RAG Evaluation MVP。


## 30. 当前系统解决了什么问题？

如果以后修改：

Retriever
Prompt
Generator Model
Judge
Generation 参数

理论上都可以重新运行同一条 Pipeline。

然后得到：

- Retrieval 指标
- Generation Success Rate
- Rule Pass Rate
- Relevance
- Groundedness
- Completeness
- Unsupported Claim
- Unanswerable Behavior
- Human Review Queue
- Judge-Human Agreement

因此这个项目已经从：

“人工看几个模型回答”

升级到了：

“固定 Benchmark + 自动生成 + 自动评分 + 专家复核”的工程流程。


## 31. 当前 MVP 最大的限制

目前最重要的限制不是代码功能，

而是：

Benchmark 太小。

目前只有：

12 条 Pilot Query

Human Calibration：

2 条

这意味着当前系统已经：

跑通

但还没有：

充分验证泛化能力。


## 32. 为什么现在不应该继续调这 12 条？

RET-012 已经被反复用于：

- Judge Smoke Test
- Groundedness Rule 调整
- Claim-level Rule 调整
- Thinking Judge 测试
- JSON Retry 测试

继续针对这 12 条修改 Judge，

会逐渐让 Evaluator 特别适合当前 Pilot。

这就是：

Evaluator Overfitting。


## 33. 下一阶段更合理的方向

当前建议：

先冻结 Automated Evaluation MVP v1。

当前 12 条继续作为：

Development / Smoke Test Set

下一阶段优先：

扩展 Benchmark。

新增问题应该覆盖更多：

- Question Type
- Difficulty
- Evidence Scope
- Single Evidence
- Multi Evidence
- Cross-source Evidence
- Answerable
- Unanswerable
- 容易出现 Unsupported Claim 的问题
- 容易发生 Evidence Incompleteness 的问题

然后：

保持 Evaluator v1 不修改，

直接在这些新问题上运行。


## 34. 为什么应该“先增加问题，再做 v2”？

正确顺序是：

Freeze v1
↓
Add New Benchmark Cases
↓
Run unchanged v1
↓
Collect New Failures
↓
Human Review
↓
Error Analysis
↓
Develop v2

这样 v2 的修改依据来自：

新的真实失败案例

而不是：

为了让原来的 12 条分数更漂亮。


## 35. 这一阶段最值得理解的几个工程概念

### Data Contract

先规定模块之间传什么数据，
再写下游代码。


### Data Leakage

Gold Label 不能进入 Generation。


### Prompt Versioning

Prompt 是实验变量，
必须版本化。


### Smoke Test

先用 1 条真实样本测试完整链路，
再批量运行。


### Batch Fault Tolerance

一个样本失败，
不能让整个 Run 丢失。


### Deterministic Evaluation

能用程序确定判断的事情，
不要浪费 LLM。


### LLM-as-a-Judge

用另一个模型做复杂语义评测，
但它本身也可能犯错。


### Groundedness

医学上正确不等于 RAG 中有证据支持。


### Claim-level Evaluation

先拆出具体事实，
再判断是否有 Evidence。


### Structured Output Validation

模型判断正确，
不代表 JSON 一定正确。


### Human-in-the-loop

自动系统筛选高风险案例，
专家负责边界判断。


### Judge Calibration

Judge 的输出不是 Gold，
必须和人工医学判断比较。


### Evaluator Overfitting

评测器本身也会对小型开发集过拟合。


### Regression Evaluation

冻结 v1 后，
未来版本都应该和固定基线比较。


## 36. 当前阶段的准确结论

截至 MVP v1：

Retrieval 已完成并冻结。

Generation 已完整跑通。

Deterministic Evaluation 已跑通。

LLM-as-a-Judge 已跑通。

Human Review Queue 已跑通。

首轮 Human Calibration 已完成。

Judge-Human 在首批 2 个边界案例上：

10 / 10 字段一致。

因此可以认为：

Generation + Automated Evaluation MVP v1
已经完成第一次端到端闭环。

但：

Benchmark 和 Calibration Set 仍然很小。

下一阶段不应该继续针对当前 Pilot 优化评分器，

而应该：

扩展 Benchmark，
验证 Evaluator v1 的泛化能力，
再基于新的 Error Analysis 开发 v2。

