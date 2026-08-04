# Retrieval Benchmark v1

## 目标

在建立Embedding和向量索引前，先固定一套可以计算检索指标的查询—证据基准。

本阶段回答的问题是：

> 给定一个真实用户问题，检索器能否把支持答案的Gold Corpus知识块排在Top-K结果中？

本阶段不评估最终生成答案质量。生成正确性、忠实性和医学安全评测将在检索基线稳定后进行。

## 数据源

- 语料版本：`gold_v1_3`
- 纳入来源：`SRC001`、`SRC003`
- 正式基准：`data/benchmark/retrieval_eval_v1.csv`
- 本地知识块目录：`data/processed/reviews/gold_chunk_catalog_v1_3.csv`

`gold_chunk_catalog_v1_3.csv`包含指南正文，只用于本地标注，不上传GitHub。

## 标注流程

### 第一轮：12条pilot

先完成`RET-001`至`RET-012`：

- 3条临床表现或诊断；
- 2条影像或诊断边界；
- 2条非药物治疗；
- 3条药物或条件性推荐；
- 1条多证据问题；
- 1条知识库不可回答问题。

pilot的目的是确认：

- 问题字段是否好用；
- Gold chunk粒度是否合理；
- 单块和多块证据能否区分；
- 检索代码能否计算指标。

### 第二轮：扩展至36条

pilot跑通后，再完成`RET-013`至`RET-036`，并保留`dev/test`划分。

- `dev`：可用于选择Embedding、Top-K、混合检索权重；
- `test`：冻结后只用于最终结果，不反复调参。

## 字段定义

| 字段 | 含义 |
|---|---|
| `query_id` | 唯一问题编号 |
| `phase` | `pilot`或`expansion` |
| `split` | `dev`或`test` |
| `topic` | 医学主题 |
| `query_type` | 查询类型 |
| `difficulty` | `easy`、`medium`、`hard` |
| `query` | 普通用户自然语言问题 |
| `answerability` | `answerable`或`unanswerable` |
| `gold_chunk_ids` | 必需证据块，多个ID用`|`分隔 |
| `supporting_chunk_ids` | 有帮助但非必需的块，可留空 |
| `expected_source_ids` | 预期来源，多个用`|`分隔 |
| `evidence_scope` | `single_chunk`、`multi_chunk`、`no_gold` |
| `review_status` | `draft`或`verified` |
| `reviewer_notes` | 标注说明 |

## Gold证据规则

### 必需证据与辅助证据

`gold_chunk_ids`只放：

> 缺少这些块，就无法完整、准确回答问题的知识块。

`supporting_chunk_ids`放：

> 能够补充背景，但不是回答问题所必需的知识块。

不要因为两个块主题相近，就都标为Gold。

### 单块问题

一个知识块已经完整覆盖：

- 问题主体；
- 适用条件；
- 关键限制；
- 推荐内容。

填写：

```text
evidence_scope = single_chunk
gold_chunk_ids = 一个chunk_id
```

### 多块问题

需要两个或更多块才能完整回答，例如：

- 临床表现＋诊断规则；
- 基础治疗＋某类药物推荐；
- SRC001总体原则＋SRC003具体药物条件。

填写：

```text
evidence_scope = multi_chunk
gold_chunk_ids = ID1|ID2
```

### 不可回答问题

问题与膝骨关节炎相关，但`gold_v1_3`没有足够证据支持明确回答。

填写：

```text
answerability = unanswerable
evidence_scope = no_gold
gold_chunk_ids留空
```

不要用常识或指南外知识补Gold证据。

## 问题编写规则

1. 使用普通用户自然语言，不复制指南标题或原句；
2. 不在问题中泄露答案关键词组合；
3. 保留真实表达差异，如口语、简称、条件描述；
4. 问题必须能从Gold Corpus支持，除非明确标为`unanswerable`；
5. 不把两个互不相关的问题强行放在一行；
6. 多证据问题必须确实需要多个证据块；
7. `test`问题不要依据后续检索结果反复改写。

## 第一阶段检索指标

完成pilot后计算：

- `Hit@1`：Top 1是否命中任一Gold块；
- `Hit@3`；
- `Hit@5`；
- `Recall@5`：所有必需Gold块中有多少进入Top 5；
- `MRR@10`：第一个Gold块的倒数排名。

不可回答问题不参与上述Gold命中指标，后续单独评估检索分数阈值和错误证据召回。

## Pilot通过标准

12条pilot全部满足：

- `review_status=verified`；
- 所有answerable问题存在有效Gold chunk；
- 所有Gold chunk ID存在于`gold_v1_3`；
- unanswerable问题没有Gold chunk；
- 至少2条`multi_chunk`；
- 至少1条跨SRC001和SRC003的问题；
- 无重复或近乎重复问题；
- 验证脚本返回0个error。

通过后才开始BM25与Dense Embedding基线。
