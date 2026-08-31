# AI-DataFlux 4.0 Bootstrap Acceptance Ledger

## 1. 基线与用途

| 项目 | 值 |
|---|---|
| Bootstrap commit | `406b72894d504a60bdc1ffec003c3ef60d6a6550` |
| Bootstrap tree | `cc6c9fffb7f929bd111c0e2c4fe1783b0306df1c` |
| 冻结分支 | `codex/4.0-bootstrap` |
| 活动研发线 | `codex/4.0-integration` |
| 当前 Ledger 状态 | H1 本地可验收子项已更新；外部证据与 H2～H4 仍有 `UNREVIEWED` |

本 Ledger 追踪 Bootstrap 中每个主要模块如何进入 AI-DataFlux 4.0。Bootstrap 的本地测试通过只说明它可以作为 extraction/baseline 起点，不构成模块验收。

## 2. 状态定义

| 状态 | 含义 |
|---|---|
| `UNREVIEWED` | 尚未按 4.0 目标合同完成评审和证据验证 |
| `ACCEPTED_AS_IS` | Bootstrap 实现无需代码修改，已按 4.0 合同验证并接受 |
| `REVISED_AND_ACCEPTED` | Bootstrap 实现经过修订，修订版本已验证并接受 |
| `REPLACED` | Bootstrap 实现已被新的 4.0 实现替换，旧路径不再参与运行 |
| `DELETED` | Bootstrap 中不属于 4.0 的实现已删除，且引用和文档已清理 |

状态变更规则：

1. 禁止只修改 `status`。每次离开 `UNREVIEWED` 时，必须同时填写完整 `accepted_at_commit` 和 `evidence`。
2. `accepted_at_commit` 使用完整 commit SHA；该 commit 必须包含被接受的代码和对应测试。
3. `evidence` 必须指向可复核的测试命令、CI run、Validation Report 或专项验收记录，不能只写“已检查”。
4. 一个表格行包含多个模块时，只有这些模块全部满足同一条件才能整体变更状态；否则必须拆行。
5. 后续改动使既有证据失效时，该行必须退回 `UNREVIEWED` 或更新到新的接受 commit 和证据。

## 3. Acceptance Ledger

| module | stage | status | acceptance criteria | accepted_at_commit | evidence |
|---|---|---|---|---|---|
| `src/config/`、`config-example.yaml` | H1.0 | `REVISED_AND_ACCEPTED` | 只接受 canonical v4 schema；覆盖 `schema_version`、datasource、retry/writeback、jobs/workspace、gateway routes、server/token；旧配置明确拒绝 | `d4447c5dee86906bbfc4a2720ca5ed85b3fda234` | [H1 local validation](./H1_LOCAL_VALIDATION.md) |
| 3.x config migration/compatibility paths，包括 `docs/MIGRATION_3_2.md`、`web/src/pages/configMigration.ts` 及相关 backend branches | H1.0 | `DELETED` | 旧 key alias、自动迁移、兼容分支和双轨运行时已删除；无运行时引用残留 | `d4447c5dee86906bbfc4a2720ca5ed85b3fda234` | [H1 local validation](./H1_LOCAL_VALIDATION.md) |
| version metadata、`README.md`、root entrypoints | H1.0 | `REVISED_AND_ACCEPTED` | 活动研发线统一报告 `4.0.0-dev`；Bootstrap 的 `3.2.0-dev` 历史身份不被改写 | `d4447c5dee86906bbfc4a2720ca5ed85b3fda234` | [H1 local validation](./H1_LOCAL_VALIDATION.md) |
| `src/core/processor.py` retry/source failure | H1.1 | `REVISED_AND_ACCEPTED` | source/content/model/system failure 分类、total-attempt retry、reload、nonretryable 与 record/job failure 边界符合执行合同 | `cf29b4e013244598e5b1dbc7789cfb0433e24a34` | [H1 local validation](./H1_LOCAL_VALIDATION.md) |
| `src/core/processor.py` PreparedResult/writeback flow | H1.2 | `REVISED_AND_ACCEPTED` | PreparedResult、reconciliation、persisted 计数、unresolved write 和 Job 终态优先级符合执行合同 | `eb9ebb5d32471c9b54503b57dec23b8418ce200c` | [H1 local validation](./H1_LOCAL_VALIDATION.md) |
| `src/data/contracts.py`、`src/data/base.py` | H1.2 | `REVISED_AND_ACCEPTED` | `WritebackReceipt` 逐记录、可校验、冲突 fail closed；只有 `COMMITTED` 增加 persisted | `eb9ebb5d32471c9b54503b57dec23b8418ce200c` | [H1 local validation](./H1_LOCAL_VALIDATION.md) |
| `src/data/mysql.py` | H1.3 | `UNREVIEWED` | 在 disposable real MySQL 上验证事务、`FOUND_ROWS`、rowcount、rollback、commit unknown 和 readback 合同 | — | 本地 mock 合同通过；真实 MySQL integration 本轮 `SKIPPED` |
| `src/data/postgresql.py` | H1.3 | `UNREVIEWED` | 在 disposable real PostgreSQL 上验证事务、rowcount、rollback、commit unknown 和 readback 合同 | — | 本地 mock 合同通过；真实 PostgreSQL integration 本轮 `SKIPPED` |
| `src/data/sqlite.py` | H1.3 | `REVISED_AND_ACCEPTED` | 使用真实 temp database 验证事务、rowcount、rollback、相同值、missing ID、commit 与 readback 合同 | `f6cba67fd4c8cd20f53a84ca5c49965f59188b22` | [H1 local validation](./H1_LOCAL_VALIDATION.md) |
| `src/data/excel.py` 及 CSV 路径 | H1.4 | `REVISED_AND_ACCEPTED` | 原子写回、保留未返回输出列、磁盘 readback、reconciliation、receipt 与失败语义通过真实文件合同测试 | `b890799a9485318085217c901223da563730f9ef` | [H1 local validation](./H1_LOCAL_VALIDATION.md) |
| `src/data/feishu/bitable.py`、`src/data/feishu/sheet.py` | H1.4 | `UNREVIEWED` | 分页、分块、限流、逐记录 receipt、确认不明和真实授权环境 UAT 均有证据 | — | mock contract 通过；没有测试应用/测试表，真实 UAT 未执行 |
| `src/jobs/models.py`、`src/jobs/io.py` | H2.1 / H2.3 | `UNREVIEWED` | Repository schema 独立版本化；原子文件语义、PreparedResult 引用/hash、终态优先级和损坏处理符合合同 | — | — |
| `src/jobs/repository.py` | H2.1 / H2.2 | `UNREVIEWED` | ownership/mutation 锁分离；revision CAS 在 mutation lock 内；并发创建和状态变更无丢失更新 | — | — |
| `src/core/job_tracker.py` | H2.1 / H2.3 | `UNREVIEWED` | state/event/checkpoint 一致；`PENDING_COMMIT` 恢复点、orphan、hash 损坏和 reconciliation 可恢复 | — | — |
| `src/core/job_runner.py`、`src/jobs/worker.py` | H2.2 / H2.3 / H2.4 | `UNREVIEWED` | claim/lease、stale recovery、cancel/resume、进程中断和重复执行边界通过故障注入 | — | — |
| `src/jobs/scheduler.py`、`src/core/scheduler.py` | H2.5 | `UNREVIEWED` | FIFO/资源压力/并发 admission 可预测，无饥饿或超出已批准单机边界 | — | — |
| `src/gateway/service.py` | H3.1 / H3.2 / H3.4 / H3.5 | `UNREVIEWED` | strict/auto/fallback、Chat/Responses 协议、SSE 生命周期和 error envelope 通过协议矩阵 | — | — |
| `src/gateway/dispatcher.py`、`src/gateway/resolver.py` 及 affinity state | H3.1 / H3.3 | `UNREVIEWED` | capability 匹配、fallback group 边界、`previous_response_id` affinity、TTL/容量/并发和过期语义可验证 | — | — |
| `src/control/server.py`、`src/control/job_service.py`、`src/control/process_manager.py` | H4.2 | `UNREVIEWED` | local-only、token auth、path containment、ETag/revision、Job/command、进程生命周期和日志合同通过 | — | — |
| `cli.py`、`main.py`、`gateway.py` | H4.1 / H4.2 | `UNREVIEWED` | v4 命令、机器可读 JSON、exit code、help、config 拒绝和 Control/Gateway 启动合同一致 | — | — |
| `web/` | H4.3 | `UNREVIEWED` | canonical v4 表单/API 类型、关键状态、错误、响应式和实际 Control workflow 通过；无 3.x migration UI | — | — |
| `.github/workflows/`、`.github/scripts/`、requirements 与 packaging specs | H4.4 / H4.5 | `UNREVIEWED` | exact-head CI、coverage/audit、真实 DB contracts、各平台构建及产物实际执行均有证据 | — | — |
| `tests/` | H1～H4 | `UNREVIEWED` | 测试覆盖目标合同而非仅复刻 Bootstrap 行为；外部 mock、真实合同和 UAT 分层清楚 | — | — |
| `docs/` | H4.1～H4.6 | `UNREVIEWED` | 文档只描述已实现合同；版本、config、API、runtime、build、release 和 rollback 与 exact commit 一致 | — | — |
| `legacy/` 及其他废弃 3.x 路径 | H1.0 / H4.1 | `UNREVIEWED` | 明确保留为非运行参考或删除；不得被 4.0 runtime、打包或用户流程隐式依赖 | — | — |

## 4. Promotion 阻断检查

Promotion commit 固定后，必须完成以下检查：

1. 本文件中 `UNREVIEWED` 的计数为 0。
2. 每个 `accepted_at_commit` 都是完整 SHA 且可解析。
3. 对每个接受 SHA 执行：

~~~bash
git merge-base --is-ancestor <accepted_at_commit> <promotion_commit>
~~~

4. 每条 evidence 存在，并能证明对应 acceptance criteria。
5. 若状态为 `REPLACED` 或 `DELETED`，验证旧模块不再被 import、打包、配置、文档或用户流程引用。
6. 对 Promotion commit 重跑全量 release gates；Bootstrap Validation Report 不能替代 Promotion 验证。

任何一项失败都阻断 Promotion、tag、Release 和部署。
