# 工程文件清单与Git记录检查

本清单对应`gold_v1_3`语料冻结时点。它区分：

- **公开工程资产**：应由Git跟踪并推送到GitHub；
- **本地受限资产**：必须存在于本地，但应被`.gitignore`排除；
- **实验记录**：用于展示错误发现和优化过程，建议公开保存不含指南正文的版本。

> Git跟踪的是文件版本，不会自动记录“操作步骤”。一个步骤只有在执行`git add`、`git commit`后才进入本地历史；只有执行`git push`后才出现在GitHub。

## 一、公开仓库核心文件

### 项目治理

| 路径 | 作用 | 要求 |
|---|---|---|
| `.gitignore` | 隔离PDF、处理后正文、密钥和运行输出 | 必须跟踪 |
| `README.md` | 项目说明和复现入口 | 必须跟踪 |
| `PROJECT_CHARTER.md` | 项目范围、用户和交付目标 | 必须跟踪 |
| `DECISIONS.md` | 关键工程决策与实验结论 | 必须跟踪 |
| `docs/engineering_artifact_inventory.md` | 当前工程资产清单 | 必须跟踪 |

### 文档治理与验证

| 路径 | 作用 | 要求 |
|---|---|---|
| `docs/page_content_policy.csv` | 表格、图、异常页面策略 | 必须跟踪 |
| `docs/document_boundaries.csv` | 正文起止边界 | 必须跟踪 |
| `docs/document_boundary_validation.csv` | 边界验证结果 | 必须跟踪 |
| `docs/index_source_policy.csv` | MVP来源纳入策略 | 必须跟踪 |
| `docs/document_structure_map.csv` | 人工核验结构图 | 必须跟踪 |
| `docs/document_structure_validation.csv` | 结构定位验证结果 | 必须跟踪 |
| `docs/manual_table_evidence_audit.csv` | 不含正文的人工表格审核摘要 | 必须跟踪 |

### Gold Corpus构建

| 路径 | 作用 | 要求 |
|---|---|---|
| `configs/gold_corpus_v1_3.json` | 最终构建参数 | 必须跟踪 |
| `configs/gold_corpus_cleanup_v1_3.json` | 显式版面清理和推荐上下文规则 | 必须跟踪 |
| `src/build_gold_corpus.py` | 最终双策略语料构建器 | 必须跟踪 |
| `docs/gold_corpus_v1_2_review_summary.csv` | 优化前人工抽检摘要 | 建议跟踪 |
| `docs/gold_corpus_v1_3_build_summary.csv` | 最终语料统计 | 必须跟踪 |
| `docs/gold_corpus_v1_3_diagnostics.csv` | 最终构建诊断 | 必须跟踪 |

### 验证脚本

| 路径 | 作用 | 要求 |
|---|---|---|
| `src/validate_document_boundaries.py` | 正文边界验证 | 必须跟踪 |
| `src/validate_structure_map.py` | 结构表和marker验证 | 必须跟踪 |
| `src/validate_manual_table_evidence.py` | 人工表格证据验证 | 必须跟踪 |
| `src/audit_project_artifacts.py` | 文件、Git跟踪和远端同步审计 | 必须跟踪 |

## 二、建议保留的失败实验

以下文件证明项目进行了基线比较和错误归因，不建议删除：

| 路径 | 作用 |
|---|---|
| `configs/chunking_baseline.json` | 页内字符窗口基线 |
| `configs/chunking_section_aware.json` | 通用正则章节实验 |
| `configs/chunking_boundary_aware.json` | 正文边界感知实验 |
| `src/build_chunks.py` | 基线构建器 |
| `src/build_section_chunks.py` | 失败的章节正则构建器 |
| `src/build_boundary_chunks.py` | 边界感知v3实验 |
| `docs/chunk_build_summary.csv` | 基线统计 |
| `docs/section_chunk_build_summary.csv` | 章节实验统计 |
| `docs/boundary_chunk_build_summary.csv` | 边界实验统计 |
| `src/audit_tables.py` | 表格检测实验 |
| `src/compare_table_extractors.py` | 多种表格解析方法对照 |

这些结果应只公开统计和代码，不公开指南正文或含正文的抽检样本。

## 三、本地必须保留但不上传GitHub

| 路径或模式 | 作用 |
|---|---|
| `data/corpus/*.pdf` | 原始指南 |
| `data/processed/pages/*_pages.jsonl` | 页级提取正文 |
| `data/processed/full_text/*.txt` | 全文预览 |
| `data/processed/manual_tables/manual_table_evidence.csv` | 人工核验表格正式数据 |
| `data/processed/manual_tables/review/*.xlsx` | 人工审核底稿 |
| `data/processed/gold_corpus/gold_corpus_v1_3.jsonl` | 最终本地语料 |
| `data/processed/gold_corpus/gold_corpus_v1_3.csv` | 最终本地语料表 |
| `data/processed/reviews/gold_corpus_v1_3_review.csv` | 含正文的最终抽检记录 |
| `.env`及API密钥文件 | 模型和Embedding凭证 |

这些文件应满足：

```bash
git check-ignore -v <文件路径>
```

能够显示对应的`.gitignore`规则。

## 四、如何判断一个步骤是否有Git记录

### 查看尚未提交的修改

```bash
git status
```

- `Untracked files`：还没有执行`git add`；
- `Changes not staged`：文件已跟踪，但修改尚未暂存；
- `Changes to be committed`：已`git add`，尚未`git commit`；
- `working tree clean`：当前工作区没有未提交修改。

### 查看某个文件是否被Git跟踪

```bash
git ls-files --error-unmatch DECISIONS.md
```

成功输出文件路径表示已跟踪；报错表示未跟踪。

### 查看某个文件的提交历史

```bash
git log --oneline --follow -- DECISIONS.md
```

查看某次提交中的变化：

```bash
git show <commit_id> -- DECISIONS.md
```

### 查看整个项目的步骤记录

```bash
git log --oneline --decorate --graph --all -30
```

查看每次提交涉及哪些文件：

```bash
git log --reverse --name-status --format="COMMIT %h %ad %s" --date=short
```

## 五、如何判断是否已经上传GitHub

本地`commit`不等于上传GitHub，必须检查远端同步。

先更新远端信息：

```bash
git fetch origin
```

查看当前分支和上游：

```bash
git branch -vv
git status -sb
```

查看本地是否有尚未推送的提交：

```bash
git log --oneline @{upstream}..HEAD
```

- 没有输出：本地没有比远端更新的提交；
- 有输出：这些commit尚未`git push`。

查看远端是否有本地尚未拉取的提交：

```bash
git log --oneline HEAD..@{upstream}
```

查看准确的落后/领先数量：

```bash
git rev-list --left-right --count @{upstream}...HEAD
```

输出格式：

```text
远端独有提交数    本地独有提交数
```

例如：

```text
0    3
```

表示本地有3个commit尚未推送。

## 六、推荐的阶段性提交

完成当前治理文件后：

```bash
git add \
  DECISIONS.md \
  docs/engineering_artifact_inventory.md \
  src/audit_project_artifacts.py \
  configs/gold_corpus_v1_3.json \
  configs/gold_corpus_cleanup_v1_3.json \
  src/build_gold_corpus.py \
  docs/gold_corpus_v1_2_review_summary.csv \
  docs/gold_corpus_v1_3_build_summary.csv \
  docs/gold_corpus_v1_3_diagnostics.csv
```

检查暂存内容：

```bash
git status
git diff --cached --stat
```

提交：

```bash
git commit -m "feat: freeze validated gold corpus v1.3"
```

推送：

```bash
git push
```

最后确认：

```bash
git status -sb
git log --oneline @{upstream}..HEAD
```
