# Repository Working Rules

## 开始工作前

每次工作前先阅读以下文件，并以仓库中的最新内容为准：

- `README.md`
- `PROJECT_STATE.md`
- `DECISIONS.md`
- `RUNBOOK.md`
- `docs/STAGE_0_SCOPE.md`

如文档之间存在冲突，先核对 Git 状态和实际资产，再明确指出冲突；不要自行假定旧状态仍然成立。

## 当前阶段边界

- 当前 12 题是 `Frozen Pilot / Smoke Test Set`，用于开发验证、冒烟测试、失败分析和回归检查，不是独立泛化测试集。
- 未经用户明确授权，不修改冻结 Benchmark、冻结 Manifest、RET-001 至 RET-012 的标签，以及已有 Retrieval、Generation、Judge、人工复核或评测结果。
- 正式 Demo 设计完成前，不扩题、不重写 Judge、不迁移主项目框架。
- 学习和最小复现实验默认只放在 `learning_lab/`；实验输入、输出和配置不得写入正式项目路径。

## 工作方式

- 每次只完成一个边界清楚、可独立验证的任务。
- 只运行与本次改动直接相关的必要验证，不顺带运行完整 Pipeline、模型 API 或无关实验。
- 不擅自清理、覆盖、提交、回退或重置已有 Git 改动；发现非本次任务产生的变更时应保留并避开。
- 不原位覆盖冻结的 v1 资产。确需演进时，先获得明确授权，并使用新版本号或新 `run_id`。

