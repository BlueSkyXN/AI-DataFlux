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
- 其余已知或未知字段、嵌套字段和显式 `null` 保持不变，只将对外 model alias 替换为所选上游物理 model ID。网关专用字段 `fallback_group` 用于选择路由组，转发前移除。
- 非流式 JSON 响应不重建，应保留多 `choices`、`tool_calls`、`logprobs`、usage 扩展和 Responses `output`。
- 正常 SSE 以上游字节块原样转发，不注入 keepalive、不补 `[DONE]`、不改写正常 event 名称；流中失败或缺少终止事件时发出明确协议错误并关闭上游，不切换后端。

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

显式 model 使用 strict routing：暂时不可用、被禁用或缺少能力时明确失败，不能切换到其他模型。`auto` 只从能力匹配的可用 route 中加权选择。使用 `{"model":"auto","fallback_group":"primary",...}` 时只按照已配置组的 route 顺序尝试，不能逃逸到组外；同时指定显式 model 和 group 会被拒绝。

批处理 `job.model_selection` 直接接入此协议：strict 发送 `route_id`，auto 发送 `auto`，fallback_group 发送 `auto` 与 `group`。规则 profile 不覆盖模型选择。

### Responses affinity

- key 是上游 Responses `id`，value 是具体 route ID（包含 channel、上游模型与 credential 的选择）；非流式响应及 SSE 的 `response` 对象会建立映射。
- 映射保存在单 Gateway 进程内，无 await 的缓存更新使同一事件循环内的查找/插入不可交错；重启不恢复映射。
- `gateway.affinity.ttl_seconds` 默认 3600，按插入时刻固定过期；`max_entries` 默认 10000，超出容量时淘汰最早插入的条目。
- `previous_response_id` 必须命中映射。缺失、过期、被淘汰或 ID 跨 route 冲突时返回 409 `response_affinity_lost`；显式 model/group 与已绑定 route 冲突时返回 409 `response_affinity_conflict`。不能随机选择其他后端。
- 后端返回跨 route 重复 ID 时标记该 key 为歧义并拒绝该响应；不会覆盖成新的归属。当前运行入口限制 `workers=1`，不声明多进程共享 affinity。

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

尝试上限由 `gateway.retry.max_attempts_per_request` 控制（默认 3），且不重复尝试本请求已经失败的 route。strict 和 affinity 请求最多使用原 route；只有 auto 或显式 group 可以切换。

非重试型 `4xx`、成功响应头之后的 body/JSON 错误、或已向客户端发送任何 SSE 数据后的中断，都不能切换模型。

Chat 流错误发送 `data: {"error":{...}}`，Responses 流错误发送 `event: error` 与 `type=error`、`sequence_number`；错误码为 `upstream_stream_error`。客户端取消不伪造成功或补终止事件，连接在生成器退出或未消费流的显式 close 中释放。SSE 不完整帧的缓冲上限为 1 MiB。

请求 schema 校验失败返回 400 OpenAI error object，不回显请求输入；非流式上游错误保留其扩展字段，并补齐标准 `message/type/param/code` 键。

没有任何模型具备所需能力时，返回 OpenAI error object：HTTP `400`、`type=invalid_request_error`、`code=unsupported_capability`、`param=model`。存在合适模型但当前全部不可用时返回 HTTP `503` 和 `code=model_unavailable`。

## 验证边界

Gateway 单元测试与 OpenAPI 路由存在不等于真实上游已验证。发布前至少需要：

1. 本地单元/契约测试通过。
2. 使用不含生产数据的测试凭据完成 Chat 与 Responses 的流式/非流式 smoke。
3. 对 tools、multimodal、json_schema、`previous_response_id` 等能力逐项验证，不以纯文本 `200` 替代。
