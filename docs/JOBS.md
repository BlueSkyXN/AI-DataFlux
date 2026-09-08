# Job / Worker 持久化与状态契约

4.0 延续文件型 durable Job Repository、资源调度器和 Worker。本页描述持久化真相和对外 schema，不把内部类的存在等同于 Release、部署或业务 UAT。

## 目录布局

Job Repository 默认位于 `runtime.workspace.state_dir`，每个 Job 的 `job_id` 必须是 canonical UUID4：

```text
.dataflux/jobs/
├── .supervisor.lock        # 单 active supervisor 的生命周期锁
├── .repository.lock        # 短时 mutation 锁，文件保留不代表仍被持有
└── <job_id>/
    ├── request.json
    ├── state.json
    ├── events.jsonl
    ├── lease.json          # Worker 持有 lease 时存在
    ├── prepared/
    ├── shards/
    └── commands/
        ├── <command_id>.json
        └── <command_id>.receipt.json
```

`request.json` 和 `state.json` 使用临时文件 + flush + fsync + `os.replace`。`events.jsonl` 为 append-only；只容忍最后一条不完整记录，中间损坏是硬错误。

H2.1 创建 Job 时先在同一 Repository 下的 `.creating-<job_id>-<随机后缀>/` 写完整初始文件，再原子 rename 为正式 UUID 目录。列表不会暴露半成品；普通异常清理本次临时目录，进程硬中断留下的临时目录不作为 Job 加载，不在本切片自动清理。目录 fsync 沿用现有平台 best-effort helper，不将这一点表述为跨平台断电持久性已验收。

Repository 格式独立于产品版本和 YAML 配置版本：request/lease/command/receipt 的 `schema_version` 为整数 `1`，state 为 `2`，shard/record 为 `2`，不是 YAML 的 `4`。读取时拒绝缺失或未知版本；不自动补版本、不迁移旧数据。state schema 1 不具备完整恢复事实，明确拒绝加载；使用新目录启动 schema 2，不要覆盖或就地改写旧 Repository。

## Job request

```json
{
  "schema_version": 1,
  "job_id": "<uuid4>",
  "mode": "background",
  "config_path": "/resolved/workspace/config.yaml",
  "config_sha256": "<64 hex chars>",
  "created_at": 0.0,
  "options": {"max_in_flight": 100}
}
```

`options` 禁止 `password`、`secret`、`token`、`api_key` 及相同后缀的秘密型键。请求不复制原始 YAML；`config_sha256` 是包含默认值和已解析 routing profile 的 canonical execution hash，不是原始 YAML bytes hash。注释或排版变化不阻断恢复，运行语义变化会阻断。

## Shard checkpoint

Runner 每扫描一个 datasource cursor page，会先在 `state.json.checkpoints` 原子提交记录及计数，再发出 AI 请求。`shards/scan-<sequence>.json` 只作诊断副本，不用于重启恢复。记录状态为 `pending`、`in_flight`、`pending_commit`、`persisted`、`unresolved_write` 或 `failed`。每条记录同时保存：

- datasource record ID 与当次输入快照；
- datasource-owned opaque cursor；
- 实际 AI request `attempt`；
- 按 `api_error`、`content_error`、`system_error`、`source_error` 分开的 durable `retry_counts`；
- PreparedResult 的 `prepared_ref/prepared_hash/commit_id`、提交与 reconciliation 尝试次数或最后错误代码。

`retry_counts` 在 Worker 崩溃、stale lease takeover 和显式 resume 后继续使用，不能因新建进程而重新获得 retry budget。PreparedResult blob 先写入，随后 `state.json` 中的引用/hash/record status 与计数一起原子提交；只有该提交点成立才能恢复结果。未被 state 引用的 blob 只是 orphan；诊断 shard 或事件不能把它提升为已提交结果。恢复校验通过后先 reconciliation，只有 datasource receipt 确认后才能转为 `persisted`。

## Job state

`status` 可用值：

- 非终态：`queued`、`running`、`cancelling`、`interrupted`
- 需人工处理、可显式 resume：`blocked`
- 终态：`completed`、`completed_with_errors`、`completed_with_unresolved_writes`、`failed`、`cancelled`

```json
{
  "schema_version": 2,
  "revision": 3,
  "job_id": "<uuid4>",
  "mode": "background",
  "config_path": "/resolved/workspace/config.yaml",
  "config_sha256": "<sha256>",
  "status": "running",
  "created_at": 0.0,
  "updated_at": 0.0,
  "started_at": 0.0,
  "finished_at": null,
  "last_error": null,
  "counts": {
    "discovered": 1000,
    "pending": 600,
    "in_flight": 20,
    "ai_complete": 380,
    "persisted": 370,
    "unresolved_writes": 0,
    "failed": 10,
    "cancelled": 0,
    "retries": 12
  },
  "resource": {
    "effective_max_in_flight": 20,
    "pressure": false,
    "control_status": "normal"
  }
}
```

`revision` 用于乐观并发控制。`persisted` 是已成功写回的数量，不应用 `ai_complete` 代替它声明业务完成。磁盘 state 额外包含 `checkpoints` 和 `command_receipts`，上述公开 API/CLI JSON 不暴露记录输入或内部恢复载荷。

### H2.1 状态变更边界

- 当前实现用 Repository 根目录的 `.repository.lock` 串行化短时文件变更，包括创建、state、event、lease、shard、PreparedResult、command/receipt 和显式 prune；尚未引入可选 per-job 锁。
- POSIX 使用 `flock`，Windows 使用 `msvcrt.locking`。锁由内核在句柄关闭或进程退出后释放，不因文件年龄过大抢占；锁文件不在解锁时删除。
- mutation 锁仅覆盖文件读改写，不覆盖模型调用、数据源网络请求或 Supervisor 生命周期。updater 必须是纯状态变换，不能递归调用 Repository 写接口。H2.2 用独立 `.supervisor.lock` 持有 Supervisor 生命周期；第二个 Supervisor 在启动时明确失败，Control/CLI 仍可使用 mutation 锁提交 Job 和 command。
- `update_state()` 在锁内读取 revision、检查预期版本、执行 updater，并原子写入 `revision + 1`。`save_state()` 未显式传 `expected_revision` 时使用快照自身的 revision，旧快照不能覆盖新状态。
- revision 必须是非负整数。updater 不能修改 revision 或 request 固定的 `job_id/mode/config_path/config_sha256/created_at`；读取也会核对 request、state 和目录身份。
- 升级锁实现前必须停止使用同一个 Repository 的旧版进程；不支持新旧锁实现混跑。该要求不需要迁移或删除现有合法 Job JSON。

### claim、取消与恢复

- 文件和飞书任务首次扫描前将 `source_identity` 写入 state；公开 Job JSON 不暴露该内部字段。文件/Sheet 的摘要覆盖有序输入，不包含独立输出列的值，因此自身写回不改变输入身份。恢复时插行、重排或输入变化将进入 `blocked`；有旧 checkpoint 却缺少身份元数据时不猜测、不自动迁移。
- Bitable 扫描、checkpoint、PreparedResult 和写回统一使用原生 `record_id`；整数只用于当前内存快照的分片位置。队列游标携带已消费条数，第二页仍有数据时也必须前进。
- CSV/Excel 恢复先核对已有输出的输入身份和已确认结果，再以输出文件重建工作表。已确认输出丢失或被修改会阻止恢复，不会跳过旧记录后把它们覆盖为空。
- 文件任务在扫描/恢复前取得规范化输出路径的排他锁，直到关闭任务池才释放。不同 Repository 中的同目标任务也不能同时写入；冲突明确失败/阻止启动，不让两个旧 DataFrame 相互覆盖。锁文件保留不代表锁仍被持有。
- Sheet 没有原生稳定行 ID；当前恢复要求输入顺序及内容保持一致。运行期间也应保持输入结构不变，不宣称对并发人工插行提供事务隔离。
- claim 在 mutation 锁内检查 `queued` 和现有 lease，再持久化 lease/running 状态。Worker 使用绑定 owner 的 Repository 写入口；lease 失效或被接管后拒绝修改 state、checkpoint、blob 或 event，并取消旧 runner。
- Supervisor 正常关闭或任务被中断，保留 `interrupted` 供恢复，不伪装成用户取消。用户已接受取消的状态保留 `cancelling`，恢复时完成取消收敛。
- command 文件先落盘，状态与 command receipt 在同一个 state 快照提交；中断留下的未确认 command 可以重放，不能仅凭诊断 receipt 文件跳过执行。
- 显式 resume 接受当前 execution hash；此后运行及自动恢复只接受最近一次成功 resume 的 hash，不接受任意历史 hash。配置发生进一步变化时进入 `blocked`。
- orphan blob 清理接口默认只预览；显式确认后只删除 state 未引用的文件。PreparedResult 父目录为 symlink 时拒绝读写或清理，防止恢复引用越界。

本地验证与剩余边界见 [H2_LOCAL_VALIDATION.md](./H2_LOCAL_VALIDATION.md)，不代表 Windows/Linux 实机验收或正式发布完成。

## Event schema

`events.jsonl` 每行是一个 JSON object：

```json
{
  "seq": 12,
  "ts": 0.0,
  "type": "status_changed",
  "job_id": "<uuid4>",
  "record_id": null,
  "attempt": null,
  "payload": {"status": "running", "revision": 3}
}
```

- `seq` 在单 Job 内单调递增，是 SSE 恢复游标。
- `type` 是可扩展字符串；除 Job/Worker 生命周期事件外，还会产生 `shard_checkpointed`、`record_in_flight`、`record_retry_scheduled`、`record_ai_complete`、`record_persisted` 和 `record_failed`。
- `payload` 必须保持非秘密；不允许记录配置正文、API Key 或完整 Bearer token。

SSE 输出使用：

```text
id: 12
data: {"seq":12,"ts":...,"type":"status_changed",...}

```

终态且没有新 event 时流结束。重连应传入上次成功接收的 `seq`，而不是按时间戳猜测。

## Worker 与恢复

- FIFO 调度，`max_active_jobs: auto` 默认上限为 `min(4, max(1, cpu_count // 2))`。
- CPU、内存比例或剩余内存触发 pressure 时，只会合作式下调已运行 Job 的 target concurrency，不抢占或强杀任务。
- Worker 持有 `lease.json`，默认心跳 5 秒，30 秒无心跳视为 stale。
- `running/cancelling/interrupted` 且 lease 失效时，仅在 config 存在且 hash 一致时重回 `queued`；否则进入 `blocked`。
- cancel 是合作式的。`cancelling` 不等于已取消，要以最终 `cancelled/failed` 状态为准。
- 显式 resume 只接受 `blocked`、`interrupted`、`failed`。resume command 会记录操作者接受的当前 `accepted_config_sha256` 并重新入队，因此它与要求原始 hash 不变的自动恢复规则不同。

## 交付边界

当前源码已注册以下版本化 Control 路由：

- `POST /api/v1/jobs`
- `GET /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`
- `POST /api/v1/jobs/{job_id}/cancel`
- `POST /api/v1/jobs/{job_id}/resume`
- `GET /api/v1/jobs/{job_id}/events?after_seq=&limit=`
- `GET /api/v1/jobs/{job_id}/events/stream?after_seq=`

路由统一使用 Bearer token、结构化 error envelope 和 request ID。分页只返回 `seq > after_seq`，`limit` 范围为 `1..1000`，并返回 `next_seq`；SSE 同时接受 `after_seq` 与 `Last-Event-ID`，取较大的合法整数。完整 HTTP 契约见 [CONTROL_API.md](./CONTROL_API.md)。

这仍只是实现证据。要声明“用户可用”，还需要同时具备并验证：

1. Control REST/SSE 的认证、状态码、分页和断线恢复通过测试。
2. CLI JSON 命令的 stdout、stderr、JSON Lines 和退出码通过测试。
3. Control Server 启动/关闭管理 JobService 循环的行为通过 lifespan 验证。
4. GUI 的 workspace 选择、提交、列表、cancel/resume 和 event stream 通过浏览器流程验证。
5. 端到端验证覆盖 submit → run → event → terminal state → datasource writeback。

当前 `worker`、`config validate`、`job submit/list/status/cancel/resume/events/prune` 已在 argparse 中暴露，帮助面可读；但 Control 路由、CLI 和 GUI 的存在不能替代上述自动化验证与业务 UAT。未满足上述条件时，只能声明内部 Job/Worker、Control 路由、CLI surface 或 GUI 页面已实现，不能声明完整用户流程已完成。
