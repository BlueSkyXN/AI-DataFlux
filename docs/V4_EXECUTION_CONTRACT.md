# AI-DataFlux 4.0 执行合同

## 1. 文档状态

| 项目 | 值 |
|---|---|
| 状态 | 已批准的目标合同，等待 H1.0～H4 实现与逐项验收 |
| Bootstrap commit | `406b72894d504a60bdc1ffec003c3ef60d6a6550` |
| Bootstrap tree | `cc6c9fffb7f929bd111c0e2c4fe1783b0306df1c` |
| 冻结分支 | `codex/4.0-bootstrap` |
| 活动研发线 | `codex/4.0-integration` |
| 原型源码版本 | `3.2.0-dev` |
| 目标开发版本 | `4.0.0-dev` |

本文固定 AI-DataFlux 4.0 的实施边界和验收规则。它描述的是**批准后的目标合同**，不是对 Bootstrap 当前实现的完成声明。Bootstrap 只是可追溯的研发起点；任何模块只有在 [BOOTSTRAP_ACCEPTANCE.md](./BOOTSTRAP_ACCEPTANCE.md) 中完成验收并附证据后，才能进入 Promotion 候选。

本文中的“必须”“不得”是阻断性要求；“可以”是允许的实现空间。

## 2. 版本与产品边界

1. 4.0 是从 `3.0.0` 正式版进入的新 major development line，不再把当前原型包装成 3.2 minor upgrade。
2. H1.0 的第一个机械切片必须把活动研发线的版本元数据统一为 `4.0.0-dev`。Bootstrap commit 保持 `3.2.0-dev`，不得改写。
3. 4.0 只接受 canonical v4 config：
   - 不兼容 3.x 配置；
   - 不提供配置迁移器；
   - 不保留新旧 schema 双轨运行时；
   - 不为旧内部结构增加 adapter 或 compatibility shim。
4. 可以保留已经验证且符合 4.0 合同的实现，但必须通过 Acceptance Ledger 明确标记为接受、修订、替换或删除。
5. Job Runtime 的正式支持边界为：
   - 单机；
   - 本地文件系统；
   - 单个 active supervisor；
   - `state.json` 是恢复事实源；
   - Repository 格式独立版本化。

## 3. Supervisor 与 Repository 锁合同

Supervisor ownership 与 Repository mutation 必须使用不同的锁，禁止用一把生命周期排他锁同时承担两种职责。

~~~text
state_dir/
├── .supervisor.lock
│   └── active supervisor 生命周期内持续持有
│       只用于阻止第二个 active supervisor
│
├── .repository.lock
│   └── Repository 元数据、Job 创建等原子读改写期间短暂持有
│
└── jobs/<job_id>/.lock
    └── 可用于单 Job state、command、claim、lease 和 receipt 的短时变更
~~~

阻断性要求：

1. `.supervisor.lock` 不得作为 Control、CLI 或 Repository mutation 的写锁；否则 Supervisor 运行期间其他进程将无法合法提交 Job 或 command。
2. 每次 mutation 必须在对应 mutation lock 内完成“读取当前 revision → 校验预期 revision → 生成新状态 → 原子替换文件”。
3. Revision CAS 必须在 mutation lock 内执行。禁止把“先读 revision、释放同步边界、再写文件”当成跨进程 CAS。
4. Repository 级不变量和 Job 创建使用 `.repository.lock`；若实现 per-job lock，单 Job 更新使用 `jobs/<job_id>/.lock`，不得破坏 Repository 级不变量。
5. 状态文件写入必须继续采用临时文件、flush/fsync 和 `os.replace` 等本地文件系统原子替换语义。

H2.1/H2.2 必须用并发 mutation、revision 冲突、第二 Supervisor 拒绝、Control 与 Supervisor 同时写入等测试证明该合同。

## 4. PreparedResult 与恢复提交点

模型调用成功不等于业务提交成功。每条记录在外部写回前必须先形成可校验的 `PreparedResult`。

正式顺序：

~~~text
模型结果产生
  ↓
原子写入 PreparedResult blob
  ├── 写临时文件
  ├── flush + fsync
  └── os.replace
  ↓
在 state.json 中原子提交
  ├── record_state = PENDING_COMMIT
  ├── prepared_result_ref
  └── prepared_result_hash
  ↓
恢复提交点成立
  ↓
执行外部写回并记录逐记录 WritebackReceipt
~~~

阻断性要求：

1. `state.json` 中已经原子提交的 `PENDING_COMMIT + prepared_result_ref + prepared_result_hash` 才是恢复提交点。
2. PreparedResult blob 是内容载体；`state.json` 中的引用决定该 blob 是否有效、是否必须恢复。
3. 若进程在 blob 写入后、`state.json` 更新前崩溃，该 blob 是未引用 orphan，可以在恢复或 prune 中删除，不得据此跳过模型调用。
4. 若进程在 `state.json` 更新后崩溃，恢复必须校验 hash 并重放已引用 PreparedResult，不得再次调用模型。
5. hash 缺失、不匹配、引用越界或 blob 无法读取必须 fail closed，进入明确的恢复/失败状态，不得猜测结果。
6. 外部写回结果不明确时，记录进入 `INDETERMINATE`，随后进入 `RECONCILING`；不得把确认不明直接当成可重试失败。

H1.2 与 H2.3 必须覆盖两个文件之间所有崩溃窗口、hash 损坏、orphan 清理和 PreparedResult 重放。

## 5. H1.0 canonical v4 schema

H1.0 必须在 H1、H2、H3 行为开发前冻结最小 canonical v4 schema。至少包含以下逻辑域：

| 逻辑域 | 最小合同 |
|---|---|
| `schema_version` | 必填，只接受 v4；未知版本明确拒绝 |
| `datasource` | 数据源类型、连接/文件定位、读取与写回字段合同 |
| `retry/writeback` | 模型重试、写回重试、reconciliation budget 和不可重试分类 |
| `jobs/workspace` | Repository/state 路径、workspace roots、Supervisor 与调度参数 |
| `gateway routes` | model route、capability、strict/auto/fallback group、affinity 参数 |
| `server/token` | Control/Gateway 监听边界、认证 token 和管理接口保护 |

H1.0 同时完成：

1. 将活动研发线的 source metadata 统一为 `4.0.0-dev`；
2. 删除旧 key alias、旧配置探测、自动升级和双 schema 分支；
3. 固定后端 schema、默认值、拒绝条件和配置 hash 语义；
4. 添加 canonical v4 config 的正反例测试。

H4.1 只做产品化收口：CLI 展示、`config-example.yaml`、Control 编辑、GUI 表单和完整文档。H4.1 不得重新改变已经被 H1～H3 依赖的核心 schema。

## 6. Record 写回与 Job 终态

### 6.1 逐记录提交语义

每条记录的业务持久化只能由有效 `WritebackReceipt` 证明：

~~~text
PENDING_COMMIT
├── 明确成功且 receipt 有效 → COMMITTED
├── 明确永久失败          → FAILED
└── 结果无法确认          → INDETERMINATE → RECONCILING
                                             ├── 已确认成功 → COMMITTED
                                             ├── 已确认失败 → FAILED
                                             └── 预算耗尽   → UNRESOLVED_WRITE
~~~

`persisted` 计数只能由 `COMMITTED` 增加。模型返回成功、PreparedResult 已落盘、写请求已发送、HTTP/DB 调用未抛异常，都不能单独增加 `persisted`。

### 6.2 Job 终态优先级

多个条件同时存在时，必须按以下固定优先级选择唯一 Job 终态：

1. `CANCELLED`：取消请求已接受并完成取消收敛。
2. `FAILED`：发生 Job 级致命失败，Job 无法正常完成扫描或运行。
3. `COMPLETED_WITH_UNRESOLVED_WRITES`：至少存在一个 `UNRESOLVED_WRITE`；可以同时存在普通 record failure。
4. `COMPLETED_WITH_ERRORS`：不存在 unresolved write，但至少存在一个 record `FAILED`。
5. `COMPLETED`：所有发现记录均进入正常成功或合同明确允许的 skipped 终态。

`Record FAILED != Job FAILED`。普通记录的永久失败不得直接把整个 Job 标记为 `FAILED`。`FAILED` 只用于 Repository 损坏、数据源无法初始化、配置/合同违规、Worker 无法继续、必需资源不可用等 Job 级致命错误。

## 7. Gateway 路由与流合同

1. 请求显式指定 model 时采用 strict routing；目标不可用或不兼容时明确失败，不得静默换 model。
2. `auto` 只能在 capability 匹配的已配置 route 中选择。
3. `fallback_group` 只能使用配置中明确声明的组和顺序，不得退化成任意全局 fallback。
4. `previous_response_id` 链必须保持后端亲和。H3.3 必须固定 affinity key、TTL、容量、并发和过期/丢失后的明确失败语义。
5. SSE/Responses 流开始向客户端发出数据后不得切换后端；中途失败必须按协议结束并返回可识别错误。
6. Chat Completions 与 Responses 的 event、usage、error envelope 和取消语义分别做协议矩阵验证。

## 8. Hardening 切片

每个切片只承担一个主要行为目标，必须带定向测试、Acceptance Ledger 更新和可复核证据。创建外部 PR、push 或合并仍需单独授权。

~~~text
H1.0 Canonical v4 foundation
└── version metadata + minimal canonical v4 schema

H1 Runtime Correctness
├── H1.1 retry + source failure
├── H1.2 WritebackReceipt + persisted
├── H1.3 MySQL/PostgreSQL/SQLite contracts
└── H1.4 Excel/CSV/Feishu writeback

H2 Durable Job Runtime
├── H2.1 repository state + revision
├── H2.2 supervisor lock + claim + lease
├── H2.3 checkpoint + recovery
├── H2.4 cancel/resume
└── H2.5 resource scheduling

H3 Gateway
├── H3.1 model strict/auto/fallback
├── H3.2 Responses protocol
├── H3.3 affinity
├── H3.4 SSE lifecycle
└── H3.5 error envelope

H4 Product and Release
├── H4.1 v4 config + CLI productization
├── H4.2 Control API
├── H4.3 GUI
├── H4.4 CI/database contracts
├── H4.5 binary smoke
└── H4.6 UAT/release evidence
~~~

## 9. Acceptance 与 Promotion

Promotion 前必须同时满足：

1. [BOOTSTRAP_ACCEPTANCE.md](./BOOTSTRAP_ACCEPTANCE.md) 不存在 `UNREVIEWED`；
2. 每项状态只能是 `ACCEPTED_AS_IS`、`REVISED_AND_ACCEPTED`、`REPLACED` 或 `DELETED`；
3. 每项 `accepted_at_commit` 必须是 Promotion commit 的祖先，可用 `git merge-base --is-ancestor <accepted_at_commit> <promotion_commit>` 验证；
4. 每项 `evidence` 指向的测试、报告或 CI 证据真实存在，并对应同一 commit 或其可证明的后继；
5. 固定 Promotion commit 后重跑全量 CI、coverage、dependency audit、真实 MySQL/PostgreSQL 合同、各平台二进制实际执行和业务 UAT；
6. 本地 source smoke、mock Playwright、打包成功、CI green、Release、部署、在线回读和业务 UAT 分别记录，不得互相替代；
7. 程序版本回退与持久化数据/Repository schema 回退分别设计和验证。

## 10. M0 交付边界

M0 只负责固定 Bootstrap、建立可恢复副本、重跑本地门禁和建立治理产物，不实现 H1～H4 的代码修复。

M0 已授权的本地动作：

- 只读检查；
- 创建本地分支与本地 commit；
- 创建工作树外 Git Bundle；
- 本地依赖安装与验证；
- 生成本地 Validation Report；
- 生成并提交本执行合同与 Acceptance Ledger。

M0 未授权的动作：

- `push`；
- 创建、修改、关闭或合并 GitHub PR，包括 PR #23；
- tag、Release、发布或部署；
- 真实外部 provider、数据库、Feishu 或其他业务写入；
- 将 Bootstrap 直接宣称为正式验收或发布候选。
