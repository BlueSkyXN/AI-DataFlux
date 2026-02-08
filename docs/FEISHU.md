# AI-DataFlux 飞书数据源集成文档

本文档详细介绍 AI-DataFlux 对飞书电子表格（Sheet）和飞书多维表格（Bitable）两种云端数据源的支持。

> 设计原则：参考 [XTF](https://github.com/BlueSkyXN/XTF) 的经验，在 AI-DataFlux 中直接实现一套原生异步飞书客户端，不使用共享 SDK，不做跨项目耦合。

---

## 支持的数据源类型

| `datasource.type` | 对应飞书产品 | 读取方式 | 写回方式 | `task_id` 体系 | 依赖 |
|---|---|---|---|---|---|
| `feishu_bitable` | 飞书多维表格 | 快照拉取（search API 分页） | batch_update（500 条/次） | 连续整数 ↔ record_id 映射 | `aiohttp` |
| `feishu_sheet` | 飞书电子表格 | 快照拉取（values API 范围读取） | 单元格写入（串行） | 数据行索引 ↔ 实际行号映射 | `aiohttp` |

---

## 快速开始

### 1. 前置条件

1. **创建飞书自建应用**：前往 [飞书开放平台](https://open.feishu.cn/) 创建自建应用
2. **获取凭据**：记录 `App ID` 和 `App Secret`
3. **开通权限**：
   - 多维表格：`bitable:app:readonly`（读取）、`bitable:app`（读写）
   - 电子表格：`sheets:spreadsheet:readonly`（读取）、`sheets:spreadsheet`（读写）
4. **添加应用**：将自建应用添加到目标文档的协作者中

### 2. 安装依赖

```bash
pip install aiohttp>=3.9.0
```

> `aiohttp` 已包含在 `requirements.txt` 和 `requirements-optional.txt` 中。

### 3. 配置示例

#### 飞书多维表格

```yaml
datasource:
  type: feishu_bitable
  require_all_input_fields: true

feishu:
  app_id: "cli_xxxxxxxxxxxxx"
  app_secret: "xxxxxxxxxxxxxxxxxxxxxxxx"
  app_token: "bascxxxxxxxxxxxxx"    # 从多维表格 URL 中获取
  table_id: "tblxxxxxxxxxxxxx"      # 数据表 ID
  max_retries: 3
  qps_limit: 5

columns_to_extract:
  - "问题"
  - "上下文"

columns_to_write:
  answer: "AI回答"
  category: "分类"
```

#### 飞书电子表格

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
  - "问题"
  - "上下文"

columns_to_write:
  answer: "AI回答"
  category: "分类"
```

---

## 架构设计

### 模块结构

```
src/data/feishu/
├── __init__.py      # 模块入口
├── client.py        # 原生异步 HTTP 客户端
├── bitable.py       # 多维表格 TaskPool
└── sheet.py         # 电子表格 TaskPool
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
| 批量限制 | 无 | Bitable 500条/次，Sheet 有频控 |
| 并发写入 | 随意 | Sheet 单文档必须串行 |
| 原子性 | 最终统一保存 | 每次 API 调用立即生效 |
| 部分失败 | 不存在 | 可能写了一半断开 |
| **AI-DataFlux 策略** | 内存更新+定时落盘 | **分批写入**+**错误日志**+**重试** |

### 错误处理

| 错误场景 | 处理方式 |
|---|---|
| 429 限流 | 读 `x-ogw-ratelimit-reset` 响应头，等待后重试 |
| 网络超时 | 指数退避重试（1s, 2s, 4s, ...） |
| Token 失效 | 自动刷新后重试当前请求 |
| record_id 不存在 | 跳过并记日志 |
| 权限不足 | 抛出 `FeishuPermissionError`，终止任务 |
| 飞书服务端 500 | 指数退避重试，连续失败则抛出异常 |

---

## 配置参考

### feishu 配置节

| 字段 | 类型 | 必需 | 默认值 | 说明 |
|---|---|---|---|---|
| `app_id` | string | ✅ | - | 飞书自建应用 App ID |
| `app_secret` | string | ✅ | - | 飞书自建应用 App Secret |
| `app_token` | string | 多维表格 ✅ | - | 多维表格 App Token（URL 中获取） |
| `table_id` | string | 多维表格 ✅ | - | 多维表格数据表 ID |
| `spreadsheet_token` | string | 电子表格 ✅ | - | 电子表格 Token（URL 中获取） |
| `sheet_id` | string | 电子表格 ✅ | - | 工作表 ID 或名称 |
| `max_retries` | int | ❌ | 3 | API 最大重试次数 |
| `qps_limit` | float | ❌ | 0 | 每秒最大请求数，0 表示不限制 |

### 如何获取 Token

- **多维表格 App Token**：打开多维表格，URL 中 `https://xxx.feishu.cn/base/{app_token}` 的 `{app_token}` 部分
- **数据表 ID**：在多维表格中，点击数据表名称旁的 `...` → 复制链接，URL 中的 `table={table_id}` 部分
- **电子表格 Token**：打开电子表格，URL 中 `https://xxx.feishu.cn/sheets/{spreadsheet_token}` 的 `{spreadsheet_token}` 部分
- **工作表 ID**：通常为 `"0"`（第一个工作表），也可使用工作表名称

---

## 代码入口

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
