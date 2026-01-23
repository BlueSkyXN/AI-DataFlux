# AI-DataFlux 设计文档

本文档详细记录项目结构、模块功能、核心类与函数，方便开发者理解和维护代码。

---

## 目录

- [项目结构](#项目结构)
- [入口文件](#入口文件)
- [核心模块 (src/core)](#核心模块-srccore)
  - [processor - 核心协调器](#processor---核心协调器)
  - [clients - API 客户端](#clients---api-客户端)
  - [content - 内容处理](#content---内容处理)
  - [state - 状态管理](#state---状态管理)
  - [retry - 重试策略](#retry---重试策略)
  - [scheduler - 分片调度](#scheduler---分片调度)
  - [validator - 验证器](#validator---验证器)
  - [token_estimator - Token 估算](#token_estimator---token-估算)
- [数据源层 (src/data)](#数据源层-srcdata)
- [网关层 (src/gateway)](#网关层-srcgateway)
- [系统架构图](#系统架构图)
- [数据流向](#数据流向)

---

## 项目结构

```
AI-DataFlux/
├── main.py              # 数据处理入口
├── gateway.py           # API 网关入口
├── cli.py               # 统一命令行入口
├── config.yaml          # 配置文件
├── config-example.yaml  # 配置示例
├── requirements.txt     # 依赖列表
├── docs/
│   ├── DESIGN.md           # 本设计文档
│   └── architecture_diagram.md # 架构图文档
├── src/
│   ├── __init__.py      # 包初始化
│   ├── config/          # 配置管理模块
│   │   ├── __init__.py
│   │   └── settings.py  # 配置加载与日志初始化
│   ├── models/          # 数据模型定义
│   │   ├── __init__.py
│   │   ├── errors.py    # 错误类型与异常
│   │   └── task.py      # 任务元数据
│   ├── core/            # 核心处理逻辑 (组件化架构)
│   │   ├── __init__.py
│   │   ├── clients/     # AI API 客户端层
│   │   │   ├── base.py
│   │   │   └── flux_client.py
│   │   ├── content/     # 内容处理层
│   │   │   └── processor.py
│   │   ├── retry/       # 重试策略层
│   │   │   └── strategy.py
│   │   ├── state/       # 状态管理层
│   │   │   └── manager.py
│   │   ├── processor.py # 核心协调器 (UniversalAIProcessor)
│   │   ├── scheduler.py # 分片任务调度器
│   │   ├── validator.py # JSON 字段验证器
│   │   └── token_estimator.py # Token 估算器
│   ├── data/            # 数据源层
│   │   ├── __init__.py
│   │   ├── base.py      # 任务池抽象基类
│   │   ├── excel.py     # Excel/CSV 任务池
│   │   ├── mysql.py     # MySQL 任务池
│   │   ├── postgresql.py # PostgreSQL 任务池
│   │   ├── sqlite.py    # SQLite 任务池
│   │   ├── factory.py   # 任务池工厂
│   │   └── engines/     # 可插拔数据引擎 (Pandas/Polars)
│   └── gateway/         # API 网关
│       ├── __init__.py
│       ├── app.py       # FastAPI 应用
│       ├── service.py   # 核心服务逻辑
│       ├── dispatcher.py # 模型调度器
│       ├── limiter.py   # 限流组件
│       ├── session.py   # HTTP 连接池管理
│       └── resolver.py  # 自定义 DNS 解析器
└── legacy/              # 旧版代码备份
```

---

## 入口文件

### `main.py` - 数据处理入口

**功能**: 批量 AI 数据处理的主入口，负责启动 `UniversalAIProcessor`。

**命令行参数**:
- `--config, -c`: 配置文件路径
- `--validate`: 仅验证配置

### `gateway.py` - API 网关入口

**功能**: 启动 OpenAI 兼容的 API 网关服务。

**命令行参数**:
- `--port`: 监听端口
- `--config`: 配置文件路径

### `cli.py` - 统一命令行工具

**功能**: 提供 `process`, `gateway`, `check`, `token` 等子命令。

---

## 核心模块 (src/core)

采用组件化架构，`UniversalAIProcessor` 作为协调者，各个功能模块职责单一。

### processor - 核心协调器

**文件**: `src/core/processor.py`
**类**: `UniversalAIProcessor`

**功能**:
整个处理流程的指挥官。它不直接执行具体逻辑，而是编排以下组件完成工作流：
1. **数据加载**: 调用 `ShardedTaskManager` 加载分片。
2. **任务分发**: 将任务提交到 `aiohttp` 会话循环。
3. **流程控制**: 调用 `TaskStateManager` 管理并发，调用 `ContentProcessor` 生成提示词，调用 `FluxAIClient` 发送请求，调用 `RetryStrategy` 处理错误。
4. **结果回写**: 调用 `BaseTaskPool` 批量保存结果。

**核心方法**:
- `process_shard_async_continuous()`: 实现了连续任务流模式 (Continuous Task Flow)，动态补充任务，保持恒定并发度。

### clients - API 客户端

**文件**: `src/core/clients/flux_client.py`
**类**: `FluxAIClient` (继承自 `BaseAIClient`)

**功能**:
- 封装 HTTP 请求细节。
- 处理 OpenAI 兼容的 API 格式。
- 统一的超时处理 (`aiohttp.ClientTimeout`)。
- 错误封装 (将 HTTP 状态码转换为异常)。

### content - 内容处理

**文件**: `src/core/content/processor.py`
**类**: `ContentProcessor`

**功能**:
- **Prompt 生成**: 将行数据渲染到模板 (`create_prompt`)。
- **响应解析**: 从 AI 返回的文本中提取 JSON (`parse_response`)。
  - 支持 Markdown 代码块提取 (```json ... ```)。
  - 支持正则智能查找 JSON 对象。
  - 自动修复常见 JSON 格式错误。
- **验证**: 调用 `JsonValidator` 验证字段值。
- **Schema**: 构建 JSON Schema (`build_schema`) 用于结构化输出。

### state - 状态管理

**文件**: `src/core/state/manager.py`
**类**: `TaskStateManager`

**功能**:
- **并发控制**: 维护 `_tasks_in_progress` 集合，防止同一任务被重复处理。
- **元数据管理**: 维护 `TaskMetadata` (重试次数、错误历史)，与业务数据分离。
- **自动清理**: 提供 `cleanup_expired()` 清理过期的元数据，防止内存泄漏。
- **线程安全**: 所有操作由锁保护。

### retry - 重试策略

**文件**: `src/core/retry/strategy.py`
**类**: `RetryStrategy`

**功能**:
- **决策引擎**: `decide(error_type, metadata)` 返回 `RetryDecision`。
- **分类处理**:
  - `API_ERROR`: 触发熔断 (Pause)，要求重载数据 (Reload Data)。
  - `CONTENT_ERROR`: 直接重试，不暂停，不重载数据 (因为输入没变，可能是 AI 随机性)。
  - `SYSTEM_ERROR`: 重试并重载数据。
- **熔断机制**: 管理 API 错误触发的全局暂停 (`api_pause_duration`)。

### scheduler - 分片调度

**文件**: `src/core/scheduler.py`
**类**: `ShardedTaskManager`

**功能**:
- **分片计算**: 根据总量和内存动态计算分片大小 (`calculate_optimal_shard_size`)。
- **进度追踪**: 跟踪处理进度、成功率、速率 (EMA)。
- **内存监控**: 监控进程内存，超过阈值 (85%) 触发 GC。

### validator - 验证器

**文件**: `src/core/validator.py`
**类**: `JsonValidator`

**功能**:
- 根据配置 (`validation.field_rules`) 验证提取出的 JSON 字段值是否在枚举范围内。

### token_estimator - Token 估算

**文件**: `src/core/token_estimator.py`
**类**: `TokenEstimator`

**功能**:
- 使用 `tiktoken` 估算任务的 Token 消耗。
- **Input Token**: 采样未处理数据 -> 渲染 Prompt -> 计算 Token。
- **Output Token**: 采样已处理数据 -> 序列化 JSON -> 计算 Token。
- 支持 `in`, `out`, `io` 三种模式。

---

## 数据源层 (src/data)

所有数据源均继承自 `BaseTaskPool` 抽象基类。

### PostgreSQL 任务池

**文件**: `src/data/postgresql.py`
**类**: `PostgreSQLTaskPool`

**特点**:
- **连接池**: 使用 `psycopg2.pool.ThreadedConnectionPool` 管理连接。
- **批量更新**: 使用 `psycopg2.extras.execute_batch` 进行高性能批量写入。
- **事务管理**: 自动管理事务提交与回滚。
- **状态**: ✅ 已完整实现。

### SQLite 任务池

**文件**: `src/data/sqlite.py`
**类**: `SQLiteTaskPool`

**特点**:
- **线程模型**: 使用 `threading.local` 实现线程隔离的连接管理 (SQLite 连接不能跨线程)。
- **WAL 模式**: 自动开启 WAL (Write-Ahead Logging) 模式提升并发性能。
- **状态**: ✅ 已完整实现。

### MySQL 任务池

**文件**: `src/data/mysql.py`
**类**: `MySQLTaskPool`

**特点**:
- 使用 `mysql.connector.pooling` 管理连接池。
- 实现了基本的 CRUD 和批量更新。

### Excel/CSV 任务池

**文件**: `src/data/excel.py`
**类**: `ExcelTaskPool`

**特点**:
- **多引擎支持**: 通过 `src/data/engines` 支持 Pandas 和 Polars。
- **CSV 支持**: 自动识别 `.csv` 后缀，底层复用 Excel 逻辑但使用 CSV 读写器。
- **向量化优化**: 使用 DataFrame 向量化操作过滤未处理索引，比逐行遍历快 50-100 倍。
- **编码修复**: 自动处理 Unicode 编码错误，支持清空问题单元格。

---

## 网关层 (src/gateway)

提供 OpenAI 兼容的 API 接口，负责请求路由和负载均衡。

- **`service.py`**: `FluxApiService` 核心服务，管理模型池。
- **`dispatcher.py`**: `ModelDispatcher` 实现加权随机选择和错误退避算法。
- **`limiter.py`**: `TokenBucket` 令牌桶限流算法。
- **`session.py`**: `SessionPool` 实现基于 (Verify, Proxy) 键的连接复用。
- **`resolver.py`**: `RoundRobinResolver` 实现自定义 IP 池轮询解析。

---

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI-DataFlux 批处理引擎                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐        ┌──────────────────┐        ┌─────────────────┐  │
│  │   CLI 入口    │        │  配置管理层       │        │   日志系统       │  │
│  │  (cli.py)    │───────▶│ (settings.py)    │◀───────│  (logging)      │  │
│  └──────┬───────┘        └──────────────────┘        └─────────────────┘  │
│         │                                                                  │
│         ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                       核心处理引擎                                     │ │
│  │                 (UniversalAIProcessor)                                │ │
│  │                                                                        │ │
│  │  • 状态管理 (TaskStateManager)         • API 通信 (FluxAIClient)      │ │
│  │  • 内容处理 (ContentProcessor)         • 策略决策 (RetryStrategy)     │ │
│  │  • 分片管理 (ShardedTaskManager)       • 验证器 (JsonValidator)       │ │
│  └─────────────────────────────┬──────────────────────────────────────────┘ │
│                                │                                           │
│                                ▼                                           │
│         ┌──────────────────────────────────────────────┐                  │
│         │          数据源抽象层 (BaseTaskPool)          │                  │
│         └──────────────────┬───────────────────────────┘                  │
│                            │                                               │
│         ┌──────────────────┴───────────────────────┐                      │
│         ▼                                           ▼                      │
│  ┌─────────────────┐                      ┌─────────────────┐             │
│  │  数据库类型数据源 │                      │  文件类型数据源  │             │
│  │ (MySQL/PG/Lite) │                      │  (Excel/CSV)    │             │
│  └─────────────────┘                      └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 数据流向

### 1. 初始化阶段
1. `main.py` 加载配置。
2. `create_task_pool` 工厂创建具体数据源 (如 `PostgreSQLTaskPool`)。
3. `UniversalAIProcessor` 初始化各组件 (`ContentProcessor`, `RetryStrategy` 等)。
4. `ShardedTaskManager` 计算分片边界。

### 2. 处理循环 (Process Loop)
1. **加载分片**: 调度器加载一批任务 ID 到内存队列。
2. **获取任务**: 处理器从队列获取一批任务 (`task_pool.get_task_batch`)。
3. **状态标记**: `TaskStateManager.try_start_task(id)` 锁定任务。
4. **生成请求**: `ContentProcessor.create_prompt(data)`。
5. **API 调用**: `FluxAIClient.call()` 发送请求。
6. **解析结果**: `ContentProcessor.parse_response()` 提取 JSON。
7. **错误处理**:
   - 若失败，`RetryStrategy.decide()` 决定是否重试。
   - 若需重试，`task_pool.add_task_to_front()` 放回队列。
8. **结果缓存**: 成功结果暂存至缓冲区。

### 3. 结果回写
1. **批量更新**: `task_pool.update_task_results()` 将缓冲区结果一次性写入数据源。
   - Excel: 更新 DataFrame 内存并定期 flush 到磁盘。
   - DB: 执行 SQL `UPDATE` 或 `execute_batch`。
2. **状态清理**: `TaskStateManager.complete_task()` 移除任务状态。

---

*文档版本: 2.3 | 最后更新: 2026-01-23*
