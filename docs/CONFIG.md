# AI-DataFlux 4.0 配置合同

AI-DataFlux 4.0 只接受 canonical v4 `RootConfig`。配置加载由 Pydantic v2 完成，所有模型统一 `extra="forbid"`；不存在旧 key alias、自动迁移、warning fallback 或双轨 loader。

## 1. 根合同

```yaml
schema_version: 4
runtime: {}
job: {}      # optional
gateway: {}  # optional
control: {}  # optional
```

- `schema_version` 必须是整数 `4`。缺失、字符串 `"4"`、3、5、浮点数或布尔值都拒绝。
- `runtime` 必填。
- `job`、`gateway`、`control` 至少存在一个。
- job-only、gateway-only、control-only 和组合配置都合法。
- `process`、`token`、本地 `job submit` 要求 `job`；`gateway` 要求 `gateway`；`gui` 要求 `control`。
- `global`、顶层 `datasource`、`mysql`、`columns_to_extract`、`models`、`channels`、`server` 等 3.x key 全部作为 unknown key 拒绝。

配置校验：

```bash
python3 cli.py config validate --config config.yaml
python3 cli.py process --config config.yaml --validate
```

## 2. `runtime`

```yaml
runtime:
  log:
    level: info                 # debug | info | warning | error
    format: text                # text | json
    output: console             # console | file
    file_path: ./logs/ai_dataflux.log
  auth:
    token: ""
  workspace:
    roots:
      project: .
    state_dir: ./.dataflux/jobs
  scheduler:
    max_active_jobs: auto       # auto 或正整数
    cpu_high_watermark: 85
    memory_high_watermark: 80
    min_free_memory_mb: 512
    sample_interval_seconds: 2
  token_estimation:
    mode: io                    # in | out | io
    sample_size: -1             # -1 或正整数
    encoding: o200k_base
```

`DATAFLUX_TOKEN` 优先于 `runtime.auth.token`。完整 token 不进入 validation error、日志或 Job evidence。绑定非 loopback 地址且两处都没有 token 时启动失败。

`runtime.workspace.roots` 是名称到目录的非空映射；Config API 和 Job submit 使用 `root_id + relative_path`。解析后越过 root 的绝对路径、`..` 或 symlink 会拒绝。`state_dir` 必须落在至少一个 workspace root 内。

## 3. `job`

```yaml
job:
  gateway_url: http://127.0.0.1:8787
  datasource: {}
  columns:
    extract: []
    write: {}
  model_selection:
    mode: auto
  prompt: {}
  validation: {}
  routing: {}
  concurrency: {}
  retry: {}
  writeback: {}
```

### 3.1 Datasource discriminator

`job.datasource.type` 决定唯一允许的字段集合：

| `type` | 必需字段 | 可选字段与默认值 |
|---|---|---|
| `excel` | `input_path`, `output_path` | `engine=auto`, `reader=auto`, `writer=auto`, `require_all_input_fields=true` |
| `csv` | `input_path`, `output_path` | `engine=auto`, `require_all_input_fields=true` |
| `sqlite` | `db_path`, `table_name` | `require_all_input_fields=true` |
| `mysql` | `host`, `user`, `password`, `database`, `table_name` | `port=3306`, `pool_size=10`, `require_all_input_fields=true` |
| `postgresql` | MySQL 同类字段 | `port=5432`, `schema_name=public`, `pool_size=10`, `require_all_input_fields=true` |
| `feishu_bitable` | `app_id`, `app_secret`, `app_token`, `table_id` | `max_retries=3`, `qps_limit=0`, `require_all_input_fields=true` |
| `feishu_sheet` | `app_id`, `app_secret`, `spreadsheet_token`, `sheet_id` | `max_retries=3`, `qps_limit=0`, `require_all_input_fields=true` |

示例：

```yaml
job:
  datasource:
    type: csv
    input_path: ./data/input.csv
    output_path: ./data/output.csv
    engine: pandas
    require_all_input_fields: true
```

Discriminator 是严格小写值；`Excel`、未知 type、把 `table_name` 放进 Excel 或把 `input_path` 放进 MySQL 都会失败。

### 3.2 Columns、model selection、prompt

```yaml
job:
  columns:
    extract: [question, context]
    write:
      answer: ai_answer
      category: ai_category
  model_selection:
    mode: auto
  prompt:
    required_fields: [answer, category]
    use_json_schema: true
    temperature: 0.3
    temperature_override: true
    system_prompt: "只返回 JSON"
    template: "处理记录：{record_json}"
  validation:
    enabled: true
    field_rules:
      category: [technical, business]
```

- `columns.extract` 是无重复的非空字符串列表。
- `columns.write` 是非空 alias → datasource column 映射。
- `prompt` 不接受 `model`；模型选择只在 `model_selection`。
- `mode=auto` 不接受 `route_id/group`。
- `mode=strict` 必须提供 `route_id`。
- `mode=fallback_group` 必须提供 `group`。

### 3.3 Routing profile

```yaml
job:
  routing:
    enabled: true
    field: category
    subtasks:
      - match: technical
        profile: .config/rules/technical.yaml
```

Profile 是独立局部 YAML，不是 `RootConfig`，不带 `schema_version`：

```yaml
prompt:
  temperature: 0.1
  template: "技术问题：{record_json}"
validation:
  field_rules:
    category: [technical]
```

Profile 顶层只允许 `prompt`、`validation`，且只覆盖声明的字段。Datasource、columns、routing、gateway 或任何其他顶层字段都会拒绝。主配置加载时会解析全部 profile；缺失或非法 profile 在启动组件前失败。

### 3.4 Concurrency、retry、writeback

```yaml
job:
  concurrency:
    batch_size: 100
    max_in_flight: 100
    save_interval: 300
    shard_size: 10000
    min_shard_size: 1000
    max_shard_size: 50000
    max_connections: 1000
    max_connections_per_host: 0
  retry:
    task_max_attempts:
      api_error: 4
      content_error: 2
      system_error: 3
      source_error: 3
    model_max_attempts: 3
    api_pause_duration_seconds: 2
    api_error_trigger_window_seconds: 2
  writeback:
    commit_max_attempts: 3
    reconciliation_max_attempts: 3
    backoff_initial_seconds: 1
    backoff_max_seconds: 30
```

所有 `*_max_attempts` 都包含首次执行。值为 1 表示不重试。Writeback 退避为 1、2、4……秒并封顶 `backoff_max_seconds`。

## 4. `gateway`

```yaml
gateway:
  listen:
    host: 127.0.0.1
    port: 8787
    workers: 1
  connection_pool:
    max_connections: 1000
    max_connections_per_host: 1000
  channels: {}
  routes: []
  fallback_groups: {}
  retry:
    max_attempts_per_request: 3
  affinity:
    ttl_seconds: 3600
    max_entries: 10000
```

H3 前 `workers` 只允许整数 1。

Channel 示例：

```yaml
gateway:
  channels:
    openai:
      base_url: https://api.openai.com
      endpoints:
        chat_completions: /v1/chat/completions
        responses: /v1/responses
      timeout_seconds: 300
      proxy: ""
      ssl_verify: true
      ip_pool: []
```

Route 示例：

```yaml
gateway:
  routes:
    - id: model-1
      display_name: Model 1
      aliases: [gpt-primary]
      upstream_model: gpt-4-turbo
      channel_id: openai
      api_key: "..."
      capabilities: [chat_completions, responses, stream, json_schema]
      weight: 10
      safe_rps: 5
      timeout_seconds: 300
      temperature: 0.3
  fallback_groups:
    standard: [model-1]
```

- Route ID、display name、upstream model 和显式 alias 在所有 route 间不得产生歧义。
- `channel_id` 必须存在。
- `chat_completions` / `responses` capability 必须有对应 channel endpoint。
- `fallback_groups` 是有序 route ID 列表；未知 ID、空列表和重复 ID 都失败。
- H1 只冻结 schema 并接入当前 Gateway；strict/fallback/affinity 完整语义属于 H3 验收。

## 5. `control`

```yaml
control:
  listen:
    host: 127.0.0.1
    port: 8790
```

Control 与 Gateway 共享 `runtime.auth.token`，没有第二套 token 字段。

## 6. Hash、ETag 与秘密

两个 SHA-256 有不同用途：

1. **Execution config hash**：对包含 Pydantic 默认值和已解析 routing profile 的 canonical JSON 计算，用于 Job request 和恢复判断。注释或 YAML 排版变化不改变它。
2. **Config API ETag**：对原始 YAML bytes 计算，用于 `If-Match` 乐观并发。注释、换行和排版变化会改变它。

Validation error 使用 `path [pydantic_code]: message`。格式化时不包含输入值，因此错误不会回显完整 password、token、API key 或 Feishu app secret。

## 7. 完整示例

仓库根目录的 [`config-example.yaml`](../config-example.yaml) 是可校验的 combined 配置，包含 `runtime + job + gateway + control`。复制后按实际使用组件删除不需要的 optional section，并替换 datasource 路径和测试凭据。
