# Job / Worker 持久化与状态契约

4.0 延续文件型 durable Job Repository、资源调度器和 Worker。本页描述持久化真相和对外 schema，不把内部类的存在等同于 Release、部署或业务 UAT。

## 目录布局

Job Repository 默认位于 `runtime.workspace.state_dir`，每个 Job 的 `job_id` 必须是 canonical UUID4：

```text
.dataflux/jobs/<job_id>/
├── request.json
├── state.json
├── events.jsonl
├── lease.json              # Worker 持有 lease 时存在
├── shards/
└── commands/
    ├── <command_id>.json
    └── <command_id>.receipt.json
```

`request.json` 和 `state.json` 使用临时文件 + flush + fsync + `os.replace`。`events.jsonl` 为 append-only；只容忍最后一条不完整记录，中间损坏是硬错误。

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

Runner 每扫描一个 datasource cursor page，会先创建独立的 `shards/scan-<sequence>.json`，再发出 AI 请求。记录状态为 `pending`、`in_flight`、`ai_complete`、`persisted` 或 `failed`。每条记录同时保存：

- datasource record ID 与当次输入快照；
- datasource-owned opaque cursor；
- 实际 AI request `attempt`；
- 按 `api_error`、`content_error`、`system_error` 分开的 durable `retry_counts`；
- 待写回 AI 结果或最后错误代码。

`retry_counts` 在 Worker 崩溃、stale lease takeover 和显式 resume 后继续使用，不能因新建进程而重新获得 retry budget。`ai_complete` 结果必须先从 checkpoint replay，只有 datasource receipt 确认后才能转为 `persisted`。

## Job state

`status` 可用值：

- 非终态：`queued`、`running`、`cancelling`、`interrupted`
- 需人工处理、可显式 resume：`blocked`
- 终态：`completed`、`completed_with_errors`、`failed`、`cancelled`

```json
{
  "schema_version": 1,
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

`revision` 用于乐观并发控制。`persisted` 是已成功写回的数量，不应用 `ai_complete` 代替它声明业务完成。

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
