# Control API 与 CLI 自动化契约

4.0 本地 Control Server 提供 workspace 浏览、带版本保护的配置读写、durable Job 管理和可恢复的 event stream。本页描述 `4.0.0-dev` 当前源码契约；兼容页面仍使用的旧版 `/api/config`、`/api/status`、进程控制和日志 WebSocket 不能与 `/api/v1/*` 混为同一稳定接口。

## 启动与认证

```bash
DATAFLUX_TOKEN=<token> python cli.py gui --no-browser
```

- `GET /health` 不需要认证，返回 `{"status":"ok","version":"..."}`。
- 所有 `/api/*` 请求都需要 `Authorization: Bearer <token>`。
- token 解析顺序为 `DATAFLUX_TOKEN` → `runtime.auth.token` → loopback 启动时生成临时 token。
- 绑定非 loopback 地址时必须显式设置 `DATAFLUX_TOKEN` 或 `runtime.auth.token`。
- `POST`、`PUT`、`PATCH`、`DELETE` 请求必须使用 `Content-Type: application/json`，否则返回 HTTP 415。

Control Server 会为请求采用调用方提供的 `X-Request-ID`，未提供时生成 UUID。正常下游响应包含 `X-Request-ID`；版本化错误正文同时包含 `request_id`：

```json
{
  "error": {
    "code": "revision_conflict",
    "message": "expected revision ..., found ...",
    "details": null,
    "request_id": "<uuid>"
  }
}
```

常见错误：

| HTTP | `error.code` | 含义 |
|---:|---|---|
| 401 | `unauthorized` | Bearer token 缺失或错误 |
| 403 | `path_forbidden` / `config_forbidden` | workspace 越界或目标不是允许的 YAML 配置 |
| 404 | `api_not_found` / `path_not_found` / `config_not_found` / `job_not_found` | API 路由或资源不存在 |
| 409 | `revision_conflict` / `job_conflict` | 乐观并发冲突或 Job 状态不允许该动作 |
| 415 | `unsupported_media_type` | 写请求不是 JSON |
| 422 | `validation_error` / `job_invalid` | 请求 schema 或 Job 配置无效 |
| 428 | `precondition_required` | 配置写入缺少 `If-Match` |
| 503 | `job_service_unavailable` | JobService 无法初始化 |

## Workspace

客户端不向 API 传任意绝对路径，而是使用 `root_id + relative_path`。服务端按 `runtime.workspace.roots` 解析，并拒绝绝对路径、`..` 和 symlink 越界。

### `GET /api/v1/workspace/roots`

```json
{
  "roots": [
    {"id": "primary", "path": "/allowed/workspace"}
  ]
}
```

### `GET /api/v1/workspace/entries`

查询参数：

- `root_id`：必填；
- `relative_path`：默认 `.`，必须指向目录。

```json
{
  "root_id": "primary",
  "relative_path": "configs",
  "entries": [
    {
      "name": "batch.yaml",
      "relative_path": "configs/batch.yaml",
      "type": "file"
    }
  ]
}
```

`type` 只会是 `directory` 或 `file`。越界或无法解析的目录项不会返回。

## 配置 API

### `GET /api/v1/config`

必填查询参数为 `root_id` 和 `relative_path`。成功响应同时在正文返回 `revision`，并在 `ETag` header 中返回带引号的同一 SHA-256：

```json
{
  "root_id": "primary",
  "relative_path": "configs/batch.yaml",
  "content": "schema_version: 4\n...",
  "revision": "<sha256>"
}
```

### `PUT /api/v1/config`

写入采用乐观并发控制。客户端必须把上次读取的 `ETag` 原样放入 `If-Match`：

```http
PUT /api/v1/config
Authorization: Bearer <token>
Content-Type: application/json
If-Match: "<sha256>"
```

```json
{
  "root_id": "primary",
  "relative_path": "configs/batch.yaml",
  "content": "schema_version: 4\n..."
}
```

- 缺少 `If-Match`：HTTP 428 `precondition_required`；
- revision 已变化：HTTP 409 `revision_conflict`；
- `If-Match: *`：接受当前 revision 或目标尚不存在的情况；
- 成功后正文和 `ETag` 返回新 revision；
- 写入使用临时文件、flush/fsync 和 `os.replace`，不会先截断目标文件。

### `POST /api/v1/config/validate`

该接口只校验候选 YAML，不写文件：

```json
{
  "content": "schema_version: 4\n...",
  "root_id": "primary",
  "relative_path": "configs/batch.yaml"
}
```

`root_id` 与 `relative_path` 必须同时提供或同时省略。提供路径时，该路径用于解析相对配置引用。业务校验失败仍返回结构化结果：

```json
{
  "valid": false,
  "errors": ["..."],
  "warnings": []
}
```

## Job API

Job state、counts、resource 和 event 的字段定义见 [JOBS.md](./JOBS.md)。

| 方法 | 路径 | 成功响应 |
|---|---|---|
| `POST` | `/api/v1/jobs` | HTTP 201 + 新 Job state |
| `GET` | `/api/v1/jobs` | `{jobs, resource}`，Job 按 `created_at` 倒序 |
| `GET` | `/api/v1/jobs/{job_id}` | 单个 Job state |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | 更新后的 Job state |
| `POST` | `/api/v1/jobs/{job_id}/resume` | 更新后的 Job state |

提交请求：

```json
{
  "root_id": "primary",
  "relative_path": "configs/batch.yaml",
  "options": {"max_in_flight": 50}
}
```

`options` 当前只允许可选的 `max_in_flight`，必须是 `1..10000` 的严格整数；字符串、布尔值、越界值和未知 options key 返回 HTTP 422，不会创建 Job。

Job config 必须是 workspace 内存在且通过严格校验的 `.yaml` 或 `.yml` 文件。Job request 持久化配置路径与提交时 SHA-256，不复制 YAML 正文。

cancel 是合作式动作：

- `queued` 直接转为 `cancelled`；
- `running` 或 `cancelling` 转为/保持 `cancelling`，等待 Worker 收敛；
- 其他状态返回 HTTP 409 `job_conflict`。

resume 仅接受 `blocked`、`interrupted`、`failed`。服务端重新读取当前配置并计算 SHA-256，把 `accepted_config_sha256` 写入 resume command，然后将 Job 重新入队。也就是说，操作者通过 resume 明确接受当前配置版本，而不是要求它必须等于原始 hash。

## Feishu 连接测试

`POST /api/v1/feishu/test-connection` 使用当前请求提供的 `app_id`、`app_secret` 获取临时 tenant token，仅用于验证连接，不写配置：

```json
{
  "app_id": "cli_xxx",
  "app_secret": "<secret>"
}
```

接口受统一 Bearer token 和 JSON Content-Type 保护。响应中的 `success` 表示连接是否成功；失败原因在 `message` 中返回。

## Event 分页与 SSE

### `GET /api/v1/jobs/{job_id}/events`

- `after_seq`：默认 `0`，必须 `>= 0`；只返回 `seq > after_seq` 的事件；
- `limit`：默认 `100`，范围 `1..1000`；
- `next_seq`：最后一个返回事件的 `seq`；没有新事件时等于传入的 `after_seq`。

```json
{
  "events": [{"seq": 12, "type": "status_changed", "payload": {}}],
  "next_seq": 12
}
```

### `GET /api/v1/jobs/{job_id}/events/stream`

响应类型为 `text/event-stream`：

```text
id: 12
data: {"seq":12,"type":"status_changed",...}

```

恢复规则：

1. 可用 query `after_seq=<last-seq>`；
2. 也可发送 `Last-Event-ID: <last-seq>`；
3. 两者同时存在时，服务端采用较大的合法整数；
4. 终态或 `blocked` 且没有新事件时结束流。

浏览器内使用 fetch streaming，因为原生 `EventSource` 不能设置 Bearer header。

## CLI 自动化边界

4.0 CLI 暴露 `worker`、`config validate` 和 `job` 命令。Job 命令不带 `--server` 时直接操作本地 Repository；带 `--server` 时通过 Control API 执行，token 读取顺序为 `DATAFLUX_TOKEN` → 本地 `runtime.auth.token`。

Supervisor ownership 与 Repository mutation 分离：同一 Repository 的第二个 active Supervisor 在启动时明确失败，但不阻止 Control/CLI 提交 Job 或 command。cancel/resume 的并发 revision 或 lease 冲突返回 409；状态与 command receipt 以同一个 state 快照提交。公开 Job JSON 的 state schema 为 2，不暴露内部 checkpoints/command receipts。详见 [JOBS.md](./JOBS.md)。

```bash
# 严格配置校验
python cli.py config validate -c config.yaml --json

# 单独运行后台 Worker
python cli.py worker -c config.yaml --json

# 另一个终端启动 Control/GUI，不争抢同一 Repository 的 Supervisor ownership
python cli.py gui -c config.yaml --no-worker --no-browser

# 本地 Repository
python cli.py job submit -c config.yaml --json
python cli.py job list -c config.yaml --json
python cli.py job status <job-id> -c config.yaml --json
python cli.py job cancel <job-id> -c config.yaml --json
python cli.py job resume <job-id> -c config.yaml --json
python cli.py job events <job-id> -c config.yaml --after-seq 0 --limit 100 --follow --json

# 远程 Control API；远程 submit 的 -c 必须是 workspace-relative path
DATAFLUX_TOKEN=<token> python cli.py job submit \
  -c configs/batch.yaml --root-id primary \
  --server http://127.0.0.1:8790 --json
DATAFLUX_TOKEN=<token> python cli.py job list \
  -c config.yaml --server http://127.0.0.1:8790 --json

# 只预览清理；--confirm 才执行删除。本命令没有远程 API。
python cli.py job prune -c config.yaml --older-than 7d --json
python cli.py job prune -c config.yaml --older-than 7d --confirm --json
```

稳定退出码：

| 退出码 | 含义 |
|---:|---|
| 0 | 成功 |
| 1 | 本地运行时错误 |
| 2 | 配置错误 |
| 3 | Process/Job 执行失败 |
| 4 | Control API 连接或 HTTP 错误 |
| 5 | Job 不存在或状态冲突 |

`--json` 模式要求 stdout 只写机器可解析 JSON；诊断文本写 stderr。event follow 模式按行输出一个 JSON object（JSON Lines），不把整个长流缓存成数组。

`job events --follow --server ...` 使用 SSE；远程非 follow 模式使用分页 REST。`job prune` 只支持本地 Repository，指定 `--server` 会以冲突类退出码拒绝。清理规则与持久化删除边界见 [JOBS.md](./JOBS.md)。

## 验证边界

路由存在只证明实现已接线。发布前仍需分别验证：

1. token 缺失、错误和正确路径；
2. workspace traversal/symlink 越界；
3. ETag 首次写入、并发冲突和重读；
4. submit → list/get → event → terminal state；
5. cancel 与 resume 的允许/拒绝状态；
6. SSE 断线后以 `Last-Event-ID` 无重无漏恢复；
7. CLI JSON stdout/stderr/exit code；
8. datasource 实际写回与 `counts.persisted/failed` 一致。

实现、CI、打包、Release、部署和业务 UAT 的证据层见 [RELEASE_GATES.md](./RELEASE_GATES.md)。
