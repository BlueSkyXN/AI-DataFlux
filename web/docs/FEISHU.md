# AI-DataFlux 飞书数据源集成文档

本文档详细介绍 AI-DataFlux 对飞书电子表格（Sheet）和飞书多维表格（Bitable）两种云端数据源的支持。

> 设计原则：参考 [XTF](https://github.com/BlueSkyXN/XTF) 的经验，在 AI-DataFlux 中直接实现一套原生异步飞书客户端，不使用共享 SDK，不做跨项目耦合。

---

## 核心特性

- **原生异步**：基于 `aiohttp` 实现的高性能异步客户端，支持高并发处理。
- **自动鉴权**：内置 `tenant_access_token` 管理，自动刷新，无需人工干预。
- **智能限流**：自动处理飞书 API 的频控限制（429 错误），支持指数退避重试。
- **会话自愈**：`ClientSession` 绑定当前事件循环，检测到 loop 切换时自动重建，避免跨 loop 复用异常。
- **大容量支持**：
  - **多维表格**：自动处理分页（Page Token），支持大批量数据的分片写入（自动拆分为 1000 条/批）。
  - **电子表格**：自动处理 "TooLarge" 错误，通过递归二分法拆解大范围读写请求。
- **断点续传**：支持任务状态持久化，中断后可从上次进度继续。

---

## 支持的数据源类型

| `datasource.type` | 对应飞书产品 | 读取方式 | 写回方式 | `task_id` 体系 | 依赖 |
|---|---|---|---|---|---|
| `feishu_bitable` | 飞书多维表格 | 快照拉取（search API 分页） | batch_update（1000 条/次，过大自动二分） | 连续整数 ↔ record_id 映射 | `aiohttp` |
| `feishu_sheet` | 飞书电子表格 | 快照拉取（values API 范围读取，过大自动行二分） | 范围写入（连续行合并，过大自动二分） | 数据行索引 ↔ 实际行号映射 | `aiohttp` |

---

## 快速开始

### 1. 前置条件

1. **创建飞书自建应用**：前往 [飞书开放平台](https://open.feishu.cn/) 创建自建应用
2. **获取凭据**：记录 `App ID` 和 `App Secret`
3. **开通权限**：
   - 多维表格：`bitable:app:readonly`（读取）、`bitable:app`（读写）
   - 电子表格：`sheets:spreadsheet:readonly`（读取）、`sheets:spreadsheet`（读写）
4. **添加应用**：将自建应用添加到目标文档的协作者中
5. **发布版本**：在“版本管理与发布”页创建版本并发布（这一步很重要，否则无法调用 API）。

### 2. 安装依赖

```bash
pip install aiohttp>=3.9.0
```

> `aiohttp` 已包含在 `requirements.txt` 和 `requirements-optional.txt` 中。

### 3. 配置示例

#### 飞书多维表格

适用于结构化数据处理，类似数据库表。

**获取 ID 信息**：
打开多维表格，URL 格式如下：
`https://{domain}.feishu.cn/base/{app_token}?table={table_id}&...`

- **app_token**: URL 中 `base/` 后面的一串字符
- **table_id**: URL 参数 `table=` 后面的一串字符

```yaml
datasource:
  type: feishu_bitable
  # 要求所有提取字段非空才处理（可选，默认 true）
  require_all_input_fields: true

feishu:
  app_id: "cli_xxxxxxxxxxxxx"
  app_secret: "xxxxxxxxxxxxxxxxxxxxxxxx"
  app_token: "bascxxxxxxxxxxxxx"    # 从多维表格 URL 中获取
  table_id: "tblxxxxxxxxxxxxx"      # 数据表 ID
  max_retries: 3      # API 重试次数
  qps_limit: 5        # API 限流 (每秒请求数)

columns_to_extract:
  - "问题"
  - "上下文"

columns_to_write:
  answer: "AI回答"
  category: "分类"
```

兼容旧配置时，`app_token/table_id/spreadsheet_token/sheet_id` 也可放在 `datasource` 节。读取规则是：只有 `feishu` 节对应键缺失（`null/undefined`）时才回退 `datasource`；若 `feishu` 键显式为空字符串，视为主动清空，不回退旧值。

#### 飞书电子表格

适用于传统 Excel 风格的表格处理。

**获取 ID 信息**：
打开电子表格，URL 格式如下：
`https://{domain}.feishu.cn/sheets/{spreadsheet_token}?sheet={sheet_id}`

- **spreadsheet_token**: URL 中 `sheets/` 后面的一串字符
- **sheet_id**: URL 参数 `sheet=` 后面的一串字符（通常是 `0` 或一串哈希值）
- `sheet_id` 在配置中可写为 `"0"` 或 `0`（YAML 数值会自动转为字符串）。

**注意**：飞书电子表格第一行必须是表头，数据从第二行开始。

```yaml
datasource:
  type: feishu_sheet
  require_all_input_fields: true

feishu:
  app_id: "cli_xxxxxxxxxxxxx"
  app_secret: "xxxxxxxxxxxxxxxxxxxxxxxx"
  spreadsheet_token: "shtcnxxxxxxxxxxxxx"  # 从电子表格 URL 中获取
  sheet_id: "0"                            # 工作表 ID 或名称
  max_retries: 3
  qps_limit: 5

columns_to_extract:
  - "Question"
  - "Context"

columns_to_write:
  answer: "Answer"
  category: "Category"
```

---

## 最佳实践

1. **并发控制**：
   飞书 API 有较严格的频率限制（通常 5-50 QPS 不等，取决于应用等级）。建议在 `config.yaml` 中设置合理的 `qps_limit`（推荐 5-10），并配合 `concurrency.batch_size` 控制并发度。

   ```yaml
   concurrency:
     batch_size: 20  # 适当降低批大小
   ```

2. **写入优化**：
   - **多维表格**：引擎会自动将结果积攒到一定数量（上限 1000）后批量调用 `batch_update`，减少 API 调用次数。
   - **电子表格**：引擎会尝试合并连续行的写入操作。如果数据量巨大，建议优先使用多维表格。

3. **错误处理**：
   如果遇到 `429 Too Many Requests`，引擎会自动等待并重试。如果重试多次仍失败，该批次任务会被标记为失败，你可以稍后重新运行程序，它会自动跳过已成功的记录（基于检查输出列是否非空）。

4. **安全建议**：
   不要将包含 `app_secret` 的配置文件提交到版本控制系统。可以使用环境变量注入（需修改代码支持）或在本地维护 `config.yaml`。

---

## 常见问题 (FAQ)

**Q: 为什么提示 "No permission" (Code 99991401/99991403)?**
A: 请检查：
1. 飞书开放平台后台是否已开通对应权限（Bitable 或 Sheets 的 read/write）。
2. 应用是否已经**发布版本**（权限变更后必须重新发布才生效）。
3. 如果是多维表格，确认机器人在该文档的协作者列表中（通常企业自建应用默认有权限，但部分文档设置了高级权限管控）。

**Q: 支持处理多大的文件？**
A:
- **多维表格**：理论上支持飞书上限（通常 5万-10万行）。引擎采用流式/分片处理，内存占用可控。
- **电子表格**：初始化时会读取整个工作表快照。对于超大表格（>5万行），建议增加机器内存或拆分 Sheet。

**Q: 程序中断了怎么办？**
A: 直接重新运行即可。AI-DataFlux 会读取数据源的最新状态，自动跳过输出列已有内容的行，只处理剩余任务。

---

## 架构设计

### 模块结构

```
src/data/feishu/
├── __init__.py      # 模块入口
├── client.py        # 原生异步 HTTP 客户端
├── bitable.py       # 多维表格 TaskPool
├── sheet.py         # 电子表格 TaskPool
```

### 核心组件

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FeishuClient                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ Token 管理        │  │ 速率控制          │  │ 重试策略          │  │
│  │ - 自动获取        │  │ - QPS 限制        │  │ - 指数退避        │  │
│  │ - 提前 5 分刷新   │  │ - 最小间隔        │  │ - 429 限流处理    │  │
│  │ - 失效后重新获取  │  │                   │  │ - Token 过期重试  │  │
│  └──────────────────┘  └──────────────────┘  └───────────────────┘  │
│                                                                      │
│  ┌──────────────────────────┐  ┌──────────────────────────────────┐  │
│  │ Bitable API              │  │ Sheet API                         │  │
│  │ - list_records (分页)     │  │ - read_range (范围读取)           │  │
│  │ - batch_update (500/次)   │  │ - write_range (范围写入)          │  │
│  │ - batch_create (500/次)   │  │ - get_meta (工作表信息)           │  │
│  │ - list_fields (字段列表)  │  │ - get_info (基本信息)             │  │
│  └──────────────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
         │                                │
         ▼                                ▼
┌────────────────────┐    ┌────────────────────────────────────────┐
│ FeishuBitableTask  │    │ FeishuSheetTaskPool                    │
│ Pool               │    │                                        │
│                    │    │                                        │
│ 快照: [records]    │    │ 快照: header[] + data_rows[][]         │
│ 映射: task_id ↔    │    │ 映射: col_name → col_index             │
│       record_id    │    │ task_id = 数据行索引（0-based）         │
│                    │    │ 实际行号 = task_id + 2                  │
└────────────────────┘    └────────────────────────────────────────┘
```

---

## 云端表格 vs 本地文件的关键差异

### 读取阶段

| 要点 | 本地文件 | 云端表格（飞书） |
|---|---|---|
| 读取速度 | 微秒级，一次性读入内存 | 分页拉取，每页一次 HTTP 请求 |
| 一致性 | 文件不会被外部修改 | 拉取期间数据可能被其他人修改 |
| **AI-DataFlux 策略** | 直接读文件 | **快照读取**：一次性拉完缓存到内存 |

### ID 体系

| 要点 | 本地文件 | 云端表格（飞书） |
|---|---|---|
| 多维表格 | 行号连续 | record_id 是字符串（recXXX） |
| 电子表格 | 行号连续 | 行号可能因插入/删除而偏移 |
| **AI-DataFlux 策略** | 行号/主键 | **ID 映射表**：连续 task_id ↔ record_id |

### 写回阶段

| 要点 | 本地文件 | 云端表格（飞书） |
|---|---|---|
| 写入速度 | 内存操作，极快 | HTTP 请求，几十到几百毫秒 |
| 批量限制 | 无 | Bitable 1000条/次（create/update），500条/次（delete），Sheet 有频控 |
| 并发写入 | 随意 | Sheet 单文档必须串行 |
| 原子性 | 最终统一保存 | 每次 API 调用立即生效 |
| 部分失败 | 不存在 | 可能写了一半断开 |
| **AI-DataFlux 策略** | 内存更新+定时落盘 | **分批写入**+**错误日志**+**重试** |

### 错误处理

| 错误场景 | 处理方式 |
|---|---|
| 429 限流 | 读 `x-ogw-ratelimit-reset` 响应头，等待后重试 |
| 业务层频控 (99991400) | 指数退避重试 |
| 网络超时 | 指数退避重试（1s, 2s, 4s, ...） |
| Token 失效 | 自动刷新后重试当前请求 |
| 请求/响应过大 (90221/90227) | 自动二分：行数减半重试，栈式迭代直到成功（参考 XTF） |
| record_id 不存在 | 跳过并记日志 |
| 权限不足 | 抛出 `FeishuPermissionError`，终止任务 |
| 飞书服务端 500 | 指数退避重试，连续失败则抛出异常 |

---

## 开发者入口

### 工厂函数

```python
from src.data.factory import create_task_pool

# 创建多维表格任务池
pool = create_task_pool(
    config={
        "datasource": {"type": "feishu_bitable"},
        "feishu": {
            "app_id": "cli_xxx",
            "app_secret": "xxx",
            "app_token": "bascXXX",
            "table_id": "tblXXX",
        },
    },
    columns_to_extract=["问题", "上下文"],
    columns_to_write={"answer": "AI回答"},
)
```

### 处理流程

飞书数据源完全兼容 AI-DataFlux 的标准处理流程：

```
create_task_pool() → get_id_boundaries() → initialize_shard()
→ get_task_batch() → [AI处理] → update_task_results() → close()
```

唯一区别是首次调用 `get_total_task_count()` 或 `get_id_boundaries()` 时会触发快照拉取（一次性从飞书拉取全部数据到内存），后续操作均在内存中完成。

---

## 注意事项

1. **快照一致性**：快照在拉取时是一个近似的快照，不是严格一致性快照。拉取期间其他用户的修改可能导致部分不一致。
2. **大数据量**：飞书多维表格单表最大支持约 50,000 条记录，电子表格最大支持约 200,000 行。超过此限制请分表处理。
3. **Token 安全**：`app_secret` 是敏感信息，请勿提交到代码仓库。建议使用环境变量或加密配置文件管理。
4. **频控策略**：飞书 API 有频控限制（通常 100 QPS/应用），建议设置 `qps_limit` 以避免触发 429。
5. **网络依赖**：飞书数据源需要网络访问 `open.feishu.cn`，请确保网络可达。
