# AI-DataFlux 4.0 H2 本地验证记录

## 当前进度（2026-09-07，H2.2～H2.5 更新）

以下 H2.1 是历史切片快照；当前工作区已经继续实现 H2.2～H2.5。全量和跨层验证统一见 [V4_LOCAL_VALIDATION.md](./V4_LOCAL_VALIDATION.md)。没有接受 SHA、远端 CI 或真实外部 UAT，不能将本地通过改写为已发布。

| 切片 | 当前实现与证据 |
|---|---|
| H2.2 ownership/claim/lease | 独立 Supervisor 生命周期锁；Control/CLI mutation 不被 ownership 阻塞；跨进程只有一个 claim 成功；失去或过期 lease 的 Worker 拒绝 state/checkpoint/blob/event 写入并取消 runner |
| H2.3 state/checkpoint/recovery | state 格式提升到 2，内嵌完整 checkpoint 及 command receipt；状态与计数同一快照提交；shards 是诊断副本，不作为恢复事实；不自动迁移 schema 1 |
| H2.3 故障窗口 | blob 前失败、blob 后 state 提交失败、state 已提交但 event 失败、诊断 shard 损坏、prepared hash 不符、未引用 orphan、prepared 父目录 symlink 越界均有回归；失败时 tracker 内存回到已提交状态 |
| H2.4 cancel/resume | command 先写文件，状态和 receipt 同一提交点；未确认 command 可重放；已接受取消优先于完成/失败；Supervisor shutdown 留 interrupted 而不是用户 cancelled；最近一次成功 resume 的 execution hash 在自动恢复时继续生效 |
| H2.4 延迟过期 lease | Supervisor 持续检查待恢复 Job；启动时 lease 尚新、稍后过期的旧任务也会被发现，不会永久停在 running |
| H2.5 资源准入 | FIFO、pressure 时不准入、协同下调与恢复、psutil 缺失降级，以及 25 个排队任务在压力解除后按序排空且 active 不超过上限的测试通过 |

Worker 退出时关闭数据源；Feishu 提供原事件循环上的 async close，避免后台线程代关 aiohttp session。状态公开 JSON 不包含 checkpoints 或 command receipts，避免把内部记录输入直接暴露为进度 API。

本轮没有在用户实际 Repository 上执行迁移、prune 或恢复。新的 state schema 2 必须使用新目录；旧 schema 1 会被明确拒绝，不得手工只改版本号。POSIX/Windows 的完整实机 crash matrix 和断电级目录 fsync 保证仍不能从 macOS 本地单元测试推导。

## H2.1 Repository state + revision（2026-09-07，历史快照）

结论：**H2.1 工作区实现与本地回归通过，尚未形成接受 commit；不代表整个 H2 或 4.0 已验收。**

- 分支：`codex/4.0-integration`。
- 基线 HEAD：`3d377e1011a29c8d07d59730973e1e8a79ef2469`。
- 本轮代码、测试与文档尚未提交；以上 SHA 是改动前基线，不是接受 SHA。
- 依据：`V4_EXECUTION_CONTRACT.md` 的 Repository 独立版本、短时 mutation 锁、锁内 revision CAS 和 H2.1 切片。
- 本机环境：macOS、Python 3.11.9；没有调用真实数据库、Feishu 或 AI provider。

### 实现与测试追踪

| 目标 | 实现 | 验证 |
|---|---|---|
| Repository 版本独立且明确拒绝不支持的格式 | request/state 仍使用整数 schema 1，与配置 schema 4 独立；拒绝缺失、未知和错误类型的版本 | 两种文件的版本正反例；拒绝后文件未被改写 |
| state 不被旧快照或身份漂移覆盖 | 严格非负整数 revision；`save_state` 默认使用快照 revision；核对不可变 request 字段 | stale snapshot、非法 revision、跨目录 identity、updater 修改元数据的拒绝测试 |
| 读改写发生在同一锁内 | Repository 根级操作系统 mutation 锁；所有现有文件写入入口共用该短时锁 | 两个独立 spawn 进程的 CAS 竞争；30 次并发增量没有丢失更新 |
| Job 创建不暴露半成品 | 同目录 staging 完成后原子 rename；失败清理本次 staging | 创建期间列表不可见；写入失败无正式目录；并发重复创建只有一个成功 |
| 不按年龄抢占活锁 | POSIX flock / Windows msvcrt；锁文件不删除 | 老时间戳下获取锁超时；持有进程被终止后另一进程立即重新获取 |
| 写入失败保留原 state | 沿用 atomic JSON helper | 注入 replace 失败后旧快照不变，临时文件清理，后续写入成功 |

当前选择单个根级短时 mutation 锁，而非新增 per-job 锁体系。锁不覆盖外部模型/数据源调用或 Supervisor 生命周期；updater 必须保持纯状态变换。该策略会串行化 Repository 文件变更，本轮没有进行吞吐量基准测试。

### 本次验证

| 命令 | 结果 |
|---|---|
| `python3 -m pytest tests/jobs/test_repository.py tests/jobs/test_repository_hardening.py -q` | `49 passed`，其中新增 34 个用例 |
| `python3 -m pytest tests/ -q -m 'not integration'` | `577 passed, 1 skipped, 11 deselected` |
| `env -u DATAFLUX_DB_INTEGRATION python3 -m pytest tests/ -q -m integration -rs` | `9 passed, 2 skipped, 578 deselected`；真实 MySQL/PostgreSQL 未启用 |
| `ruff check src/ tests/ cli.py main.py gateway.py .github/scripts` | PASS |
| `black --check src/ tests/ cli.py main.py gateway.py .github/scripts` | PASS，107 files unchanged |
| `mypy src/ --ignore-missing-imports` | PASS，64 source files；保留未注解函数体默认不检查的提示 |
| `mypy src/jobs/ --ignore-missing-imports --platform win32` | PASS，仅静态平台分支检查，不是 Windows 实机执行 |
| `git diff --check` | PASS |

首次测试运行在故障注入用例的清理阶段卡住：子进程在 `multiprocessing.Event.wait()` 中被终止后，父进程再次 `set()` 同一 Event 可能卡在同步原语。中断该轮后，测试改为子进程限时 sleep、父进程终止并 join，不再操作被终止进程等待的 Event；正式定向和全量复跑通过。生产锁实现未因这一测试问题弱化。

### 尚未验收的边界

- **不支持新旧锁实现混跑。** 更新运行代码前，必须停止使用同一 Repository 的旧版进程；不迁移、不删除现有合法 Job JSON。本轮没有启动或停止用户的运行中服务。
- Supervisor ownership、claim/lease 完整合同属于 H2.2，尚未实现或接受；本轮对 lease 写入口只统一了 mutation 锁。
- state/checkpoint/PreparedResult 全面 crash matrix、orphan 管理和终态优先级属于 H2.3；仅完成本表列出的 H2.1 故障注入。
- 硬中断创建后可能留下 `.creating-*`，不会当成 Job 加载；自动清理留待恢复切片。目录 fsync 仍沿用现有 best-effort helper，未声明断电恢复保证。
- 未在 Windows/Linux 实机运行新锁和并发用例；未重跑 coverage、前端、依赖审计或二进制 smoke；没有远端 exact-head CI、PR、Release、部署或业务 UAT。
- Acceptance Ledger 对应行涉及 H2.2/H2.3 且缺少接受 SHA，仍为 `UNREVIEWED`。提交后必须用真实完整 SHA 和对应验证证据更新，不能用基线 HEAD 代替。
