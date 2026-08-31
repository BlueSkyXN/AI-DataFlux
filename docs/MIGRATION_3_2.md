# AI-DataFlux 3.2 配置迁移

3.2 是开发中版本，尚未完成 GitHub Release。本页记录当前 3.2 严格配置契约；它不代表已部署或已通过业务 UAT。

## 迁移原则

1. 备份现有 `config.yaml`，但不要将其中的 Key、密码或 Feishu 凭据提交到 Git。
2. 以当前 `config-example.yaml` 为骨架手工迁移。程序不会自动改写旧配置。
3. 运行 `python cli.py process --config config.yaml --validate`。未知、已移除或类型错误的字段是启动阻断项，不再默默忽略。
4. 验证通过只表示本地 schema/语义检查通过；数据源、上游模型、Feishu 和业务写回仍需独立验证。

## 必须迁移的字段

### Channel endpoint

`channels.<id>.api_path` 已移除。改为显式 endpoint 映射：

```yaml
channels:
  "openai":
    base_url: "https://api.example.com"
    endpoints:
      chat_completions: "/v1/chat/completions"
      responses: "/v1/responses"
```

只配置实际存在的 endpoint。网关不会在 Chat Completions 与 Responses 之间模拟高级语义。

### Model capabilities

`models[*].capabilities` 是必填非空字符串列表。可用值仅为：

- `chat_completions`
- `responses`
- `stream`
- `multimodal`
- `tools`
- `n`
- `logprobs`
- `json_schema`
- `previous_response_id`

```yaml
models:
  - id: "model-1"
    name: "model-1"
    model: "provider-model-id"
    channel_id: "openai"
    api_key: "your_api_key"
    capabilities:
      - chat_completions
      - stream
      - tools
      - json_schema
```

`supports_json_schema`、`supports_advanced_params`、channel-level capabilities、布尔 capability 映射和 capability 别名都不是 3.2 契约。

### 工作区与 Job 状态

3.2 不接受任意绝对路径作为 Job 输入。配置可访问根目录，对外只传 `root_id + relative_path`：

```yaml
workspace:
  roots:
    project: "."
    data: "./data"
  state_dir: "./.dataflux/jobs"
```

- root 名称只能包含字母、数字、`_` 和 `-`。
- `relative_path` 不能是绝对路径，解析后不能越出所选 root，包括 symlink 绕过。
- `workspace.state_dir` 必须位于至少一个 workspace root 内。
- Job 持久化记录不保存配置正文或密钥，只保存 config 路径与 SHA-256。

### 调度与并发

```yaml
datasource:
  concurrency:
    batch_size: 100
    max_in_flight: 100

scheduler:
  max_active_jobs: auto
  cpu_high_watermark: 85
  memory_high_watermark: 80
  min_free_memory_mb: 512
  sample_interval_seconds: 2
```

`batch_size` 是数据源读写批次大小；`max_in_flight` 是单 Job 期望 AI 并发上限，资源调度器可在运行时下调，不会超过该值。

### 服务器与统一 token

```yaml
server:
  host: "127.0.0.1"
  control_port: 8790
  gateway_port: 8787
  token: ""
```

token 优先级是 `DATAFLUX_TOKEN` 环境变量、`server.token`、loopback 上的随机临时 token。绑定非 loopback 地址时不允许自动生成，必须显式配置 token。不要在文档、日志或 Job event 中输出完整 token。

### 其他移除项

- `token_estimation.tiktoken_model` 已移除，使用 `token_estimation.encoding`。
- `datasource.app_token/table_id/spreadsheet_token/sheet_id` 的 Feishu 旧路径已移除，必须放在 `feishu` 节。
- `datasource.concurrency.max_workers/retry_times/backoff_factor` 不再作为兼容键。

## 迁移验证清单

```bash
python cli.py process --config config.yaml --validate
pytest tests/test_config.py -v
```

若用到 Gateway，还需独立核对 `/v1/models` 与 `/admin/capabilities`；若用到 Job/Worker，需核对 Job 状态、event 序列和最终写回结果。
