# 飞书数据源指南 (Native Async)

AI-DataFlux 内置了高性能的**原生飞书客户端**，直接使用 `aiohttp` 实现异步 I/O，不依赖任何第三方 SDK。该设计参考了 [XTF](https://github.com/BlueSkyXN/XTF) 的高并发架构，专为大批量数据处理优化。

## 核心特性

- **原生异步**：基于 `aiohttp` 和 `asyncio`，支持高并发读写。
- **自动流控**：内置智能 Rate Limiter，自动处理 HTTP 429 和业务层频控。
- **批量并发**：
    - **多维表格 (Bitable)**: 写入操作自动合并为批量请求（单次上限 1000 条），利用 Semaphore 控制并发度。
    - **电子表格 (Sheet)**: 读取和写入支持自动分块（二分法），解决 "TooLarge" 错误。
- **自动重试**：网络超时、5xx 错误、Token 过期自动指数退避重试。
- **连接复用**: 全局单例 `ClientSession`，复用 TCP 连接。

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
  sheet_id: "Wk1"                           # 工作表 ID (不是名称)

  # 并发控制
  concurrency:
    batch_size: 20                    # Sheet 并发数建议较小，避免写冲突
```

### 3. 字段映射配置

```yaml
columns_to_extract:
  - "问题描述"
  - "上下文信息"

columns_to_write:
  "ai_analysis": "AI分析结果"      # AI 输出字段 -> 飞书字段
  "category": "分类标签"
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

## Web GUI 连接测试

启动控制面板后，可在 "配置" 页面测试飞书连接：

1.  启动 GUI: `python cli.py gui`
2.  进入配置编辑器，填写 `feishu` 节配置。
3.  点击 "测试连接" 按钮。
    *   后端调用 `/api/feishu/test_connection` 接口。
    *   自动尝试获取 `tenant_access_token` 验证 App ID/Secret 有效性。

## 常见问题

**Q: 为什么写入速度比读取慢？**
A: 飞书 Sheet API 写入存在较严格的频率限制，且需要串行锁以保证数据一致性。多维表格 (Bitable) 针对批量写入做了优化，推荐优先使用 Bitable。

**Q: 遇到 "TooLargeRequest" 错误怎么办？**
A: 客户端内置了自动二分重试机制。如果仍然报错，请尝试减小 `batch_size`。

**Q: 如何获取 app_token 和 table_id？**
A: 打开飞书多维表格，URL格式如下：
`https://{tenant}.feishu.cn/base/{app_token}?table={table_id}&...`
