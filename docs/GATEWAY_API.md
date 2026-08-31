# Gateway Chat Completions / Responses 契约

Gateway 对外提供 OpenAI-compatible Chat Completions 和 Responses 代理。它根据明确 capability 选择模型，不在两种 API 之间伪造高级语义。

## Endpoint

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat Completions 透明代理 |
| `POST` | `/v1/responses` | Responses 透明代理 |
| `GET` | `/v1/models` | OpenAI-compatible 模型列表 |
| `GET` | `/admin/models` | 模型运行统计 |
| `GET` | `/admin/health` | 健康状态 |
| `GET` | `/admin/capabilities` | 每个模型的有效 capability 与 endpoint |

`/v1/*` 和 `/admin/*` 使用统一 incoming Bearer token 检查。token 来源和非 loopback 约束见 [CONFIG.md](./CONFIG.md#2-runtime)。

## 最小请求校验与透传

- Chat 只硬性校验 `model` 和 `messages`。
- Responses 只硬性校验 `model` 和 `input`。
- 其余已知或未知字段、嵌套字段和显式 `null` 保持不变，只将对外 model alias 替换为所选上游物理 model ID。
- 非流式 JSON 响应不重建，应保留多 `choices`、`tool_calls`、`logprobs`、usage 扩展和 Responses `output`。
- SSE 以上游字节块原样转发，不注入 keepalive、不补 `[DONE]`、不改写 event 名称。

## Canonical capability

route 的 `capabilities` 必须是非空列表，且只能使用：

| Capability | 触发条件 |
|---|---|
| `chat_completions` | 请求 `/v1/chat/completions` |
| `responses` | 请求 `/v1/responses` |
| `stream` | `stream: true` |
| `multimodal` | 输入包含 image/audio/video 等非纯文本 part |
| `tools` | 使用 `tools` / `tool_choice` |
| `n` | `n > 1` |
| `logprobs` | 请求 logprobs/top_logprobs |
| `json_schema` | 严格 `json_schema` 输出；`json_object` 不等同于该能力 |
| `previous_response_id` | Responses 请求使用 `previous_response_id` |

指定 model 暂时不可用或缺少能力时，网关可在其他满足全部 capability 的模型中选择；不允许为了可用性丢弃请求特性。

## Channel 配置

```yaml
schema_version: 4
runtime: {}
gateway:
  channels:
    openai:
      base_url: https://api.example.com
      endpoints:
        chat_completions: /v1/chat/completions
        responses: /v1/responses
      timeout_seconds: 300
      proxy: ""
      ssl_verify: true
      ip_pool: []
  routes:
    - id: route-a
      display_name: Route A
      aliases: [model-a]
      upstream_model: upstream-a
      channel_id: openai
      api_key: "..."
      capabilities: [chat_completions, responses]
      weight: 1
      safe_rps: 10
      timeout_seconds: 300
      temperature: 0.7
```

`endpoints` 是 channel 的唯一 API path 来源。不支持 `api_path`、`responses_api_path` 或 channel capability。route capability 必须与 channel 实际配置的 endpoint 相容。

## Failover 边界

failover 只能发生在尚未向客户端开始响应时，且仅限：

- 连接建立失败；
- HTTP `429`；
- 明确可重试的 `500/502/503/504`。

非重试型 `4xx`、成功响应头之后的 body/JSON 错误、或已向客户端发送任何 SSE 数据后的中断，都不能切换模型。

没有任何模型具备所需能力时，返回 OpenAI error object：HTTP `400`、`type=invalid_request_error`、`code=unsupported_capability`、`param=model`。存在合适模型但当前全部不可用时返回 HTTP `503` 和 `code=model_unavailable`。

## 验证边界

Gateway 单元测试与 OpenAPI 路由存在不等于真实上游已验证。发布前至少需要：

1. 本地单元/契约测试通过。
2. 使用不含生产数据的测试凭据完成 Chat 与 Responses 的流式/非流式 smoke。
3. 对 tools、multimodal、json_schema、`previous_response_id` 等能力逐项验证，不以纯文本 `200` 替代。
