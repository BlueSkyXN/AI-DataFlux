# 飞书数据源指南 (Native Async)

AI-DataFlux 内置了高性能的**原生飞书客户端**，直接使用 `aiohttp` 实现异步 I/O，不依赖任何第三方 SDK。该设计参考了 [XTF](https://github.com/BlueSkyXN/XTF) 的高并发架构，专为大批量数据处理优化。

## 核心特性

- **原生异步**：基于 `aiohttp` 和 `asyncio`，支持高并发读写。
- **自动流控**：内置智能 Rate Limiter，自动处理 HTTP 429 和业务层频控。
- **批量并发**：自动将大批量写入请求拆分为并发块（Chunk），利用 Semaphore 控制并发度。
- **自动重试**：网络超时、5xx 错误、Token 过期自动指数退避重试。
- **容错分片**：如果 Bitable 或 Sheet 请求因 Payload 过大失败，自动二分重试。

## 配置指南

### 1. 基础配置 (`config.yaml`)

将飞书应用的 AppID 和 AppSecret 配置在 `feishu` 节中，然后在 `datasource` 中引用。

```yaml
# 飞书应用凭证
feishu:
  app_id: "cli_a1b2c3d4e5f6"
  app_secret: "your_app_secret_here"
  max_retries: 3        # 最大重试次数 (默认 3)
  qps_limit: 10         # QPS 限制 (默认 0=不限制，建议 5-10)

# 数据源配置 (多维表格)
datasource:
  type: feishu_bitable
  app_token: "bascnXXXXXXXXXXXXXXX"   # 多维表格 App Token
  table_id: "tblXXXXXXXXXXXX"         # 数据表 ID

  # 并发控制
  concurrency:
    batch_size: 100
```

### 2. 电子表格配置

```yaml
datasource:
  type: feishu_sheet
  spreadsheet_token: "shtcnXXXXXXXXXXXXXXX" # 电子表格 Token
  sheet_id: "a1b2c3"                        # 工作表 ID (不是名称)
```

## 权限要求

请确保你的飞书自建应用已开通以下权限：

- **多维表格**:
  - `bitable:app:read` (查看多维表格内容)
  - `bitable:record:read` (查看多维表格记录)
  - `bitable:record:create` (新增多维表格记录)
  - `bitable:record:update` (更新多维表格记录)
- **电子表格**:
  - `sheets:spreadsheet:read` (查看电子表格内容)
  - `sheets:spreadsheet:write` (修改电子表格内容)

## 架构设计

`FeishuClient` (`src/data/feishu/client.py`) 是核心组件：

1. **Token 管理**:
   - 内存缓存 `tenant_access_token`。
   - 提前 5 分钟自动刷新。
   - 遇到 Token 失效错误 (99991661) 立即强制刷新。

2. **并发策略**:
   - `bitable_batch_create/update`: 将大列表切分为 `1000` 条的 Chunk。
   - 使用 `asyncio.gather` 并发提交 Chunk。
   - 使用 `asyncio.Semaphore(5)` 限制同时进行的 HTTP 请求数，防止瞬间拥塞。

3. **错误处理**:
   - `FeishuRateLimitError`: 遇到 429 或 99991400 时，自动等待 `Retry-After` 秒数。
   - `FeishuAPIError`: 遇到 5xx 或网络错误，指数退避重试 (1s, 2s, 4s...)。

## 测试连接

控制面板提供了测试 API：

```bash
curl -X POST http://127.0.0.1:8790/api/feishu/test_connection \
  -H "Content-Type: application/json" \
  -d '{"app_id": "cli_xxx", "app_secret": "xxx"}'
```

