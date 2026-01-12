# AI-DataFlux 设计文档

本文档详细记录项目结构、模块功能、核心类与函数，方便开发者理解和维护代码。

---

## 目录

- [项目结构](#项目结构)
- [入口文件](#入口文件)
- [核心模块 (src/)](#核心模块-src)
  - [config - 配置管理](#config---配置管理)
  - [models - 数据模型](#models---数据模型)
  - [core - 核心处理逻辑](#core---核心处理逻辑)
  - [data - 数据源层](#data---数据源层)
  - [gateway - API 网关](#gateway---api-网关)
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
│   └── DESIGN.md        # 本设计文档
├── src/
│   ├── __init__.py      # 包初始化，版本信息
│   ├── config/          # 配置管理模块
│   │   ├── __init__.py
│   │   └── settings.py  # 配置加载与日志初始化
│   ├── models/          # 数据模型定义
│   │   ├── __init__.py
│   │   ├── errors.py    # 错误类型与异常
│   │   └── task.py      # 任务元数据
│   ├── core/            # 核心处理逻辑
│   │   ├── __init__.py
│   │   ├── processor.py # AI 处理引擎
│   │   ├── scheduler.py # 分片任务调度器
│   │   └── validator.py # JSON 字段验证器
│   ├── data/            # 数据源层
│   │   ├── __init__.py
│   │   ├── base.py      # 任务池抽象基类
│   │   ├── excel.py     # Excel 任务池
│   │   ├── mysql.py     # MySQL 任务池
│   │   ├── factory.py   # 任务池工厂
│   │   └── engines/     # 可插拔数据引擎
│   │       ├── __init__.py
│   │       ├── base.py          # 引擎抽象基类
│   │       ├── pandas_engine.py # Pandas 实现
│   │       └── polars_engine.py # Polars 预留接口
│   └── gateway/         # API 网关
│       ├── __init__.py
│       ├── app.py       # FastAPI 应用
│       ├── service.py   # 核心服务逻辑
│       ├── dispatcher.py # 模型调度器
│       ├── limiter.py   # 限流组件
│       ├── session.py   # HTTP 连接池管理
│       ├── resolver.py  # 自定义 DNS 解析器 (IP 池轮询)
│       └── schemas.py   # Pydantic 数据模型
└── legacy/              # 旧版代码 (保留备份)
```

---

## 入口文件

### `main.py` - 数据处理入口

**功能**: 批量 AI 数据处理的主入口，从数据源读取任务，调用 API 网关，将结果写回。

**主要函数**:

| 函数 | 描述 |
|------|------|
| `main()` | 主入口，解析命令行参数，初始化处理器并运行 |

**命令行参数**:
- `--config, -c`: 配置文件路径 (默认: `config.yaml`)
- `--validate`: 仅验证配置文件，不执行处理

**使用示例**:
```bash
python main.py --config config.yaml
python main.py -c my_config.yaml --validate
```

---

### `gateway.py` - API 网关入口

**功能**: 启动 OpenAI 兼容的 API 网关服务。

**使用示例**:
```bash
python gateway.py --config config.yaml --port 8787
```

---

## 核心模块 (src/)

### config - 配置管理

#### `settings.py`

**功能**: 配置文件加载、日志系统初始化、配置合并工具。

**常量**:

| 常量 | 描述 |
|------|------|
| `DEFAULT_CONFIG` | 默认配置字典，包含全局设置、数据源配置、并发参数等 |

**函数**:

| 函数 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `load_config(config_path)` | `str \| Path` | `dict[str, Any]` | 加载 YAML 配置文件 |
| `init_logging(log_config)` | `dict \| None` | `None` | 初始化日志系统，支持控制台/文件输出，text/json 格式 |
| `get_nested(config, *keys, default)` | `dict, *str, Any` | `Any` | 安全获取嵌套配置值 |
| `merge_config(base, override)` | `dict, dict` | `dict` | 深度合并配置字典 |

---

### models - 数据模型

#### `errors.py`

**功能**: 定义错误类型枚举和异常类层次结构。

**枚举类**:

| 类 | 值 | 描述 |
|----|-----|------|
| `ErrorType.API` | `"api_error"` | API 调用错误 (超时、HTTP 错误) |
| `ErrorType.CONTENT` | `"content_error"` | 内容错误 (JSON 解析失败、验证失败) |
| `ErrorType.SYSTEM` | `"system_error"` | 系统错误 (内部异常) |

**异常类层次**:

```
AIDataFluxError (基类)
├── ConfigError        # 配置错误
├── DataSourceError    # 数据源错误
├── APIError           # API 调用错误
└── ContentError       # 内容处理错误
    └── ValidationError  # 验证错误
```

---

#### `task.py`

**功能**: 任务元数据管理，独立于业务数据，用于跟踪重试状态。

**数据类**:

| 类 | 描述 |
|----|------|
| `ErrorRecord` | 错误记录，包含时间戳、错误类型、消息 |
| `TaskMetadata` | 任务元数据，管理重试计数和错误历史 |

**TaskMetadata 方法**:

| 方法 | 描述 |
|------|------|
| `increment_retry(error_type)` | 递增指定错误类型的重试计数 |
| `get_retry_count(error_type)` | 获取指定错误类型的重试次数 |
| `add_error(error_type, message)` | 添加错误记录到历史 |
| `reset_retry_count(error_type)` | 重置指定错误类型的重试计数 |
| `reset_all()` | 重置所有状态 |

**属性**:

| 属性 | 描述 |
|------|------|
| `total_retries` | 所有错误类型的总重试次数 |
| `has_errors` | 是否有错误记录 |
| `last_error` | 最近一次错误记录 |

---

### core - 核心处理逻辑

#### `processor.py` - UniversalAIProcessor

**功能**: 通用 AI 数据处理器，编排整个处理工作流。

**类**: `UniversalAIProcessor`

**初始化参数**:
- `config_path: str` - 配置文件路径

**核心属性**:

| 属性 | 类型 | 描述 |
|------|------|------|
| `config` | `dict` | 配置字典 |
| `flux_api_url` | `str` | Flux API 端点 URL |
| `task_pool` | `BaseTaskPool` | 数据源任务池 |
| `task_manager` | `ShardedTaskManager` | 分片任务管理器 |
| `validator` | `JsonValidator` | JSON 验证器 |

**主要方法**:

| 方法 | 描述 |
|------|------|
| `run()` | 同步入口，启动处理引擎 |
| `process_shard_async_continuous()` | 连续任务流模式的异步处理 |
| `mark_task_in_progress(record_id)` | 标记任务为处理中 |
| `mark_task_completed(record_id)` | 标记任务处理完成 |
| `get_task_metadata(task_id)` | 获取或创建任务元数据 |
| `create_prompt(record_data)` | 创建 AI 提示词 |
| `extract_json_from_response(content)` | 从 AI 响应中提取 JSON |
| `build_json_schema()` | 构建 JSON Schema |
| `call_ai_api_async(session, prompt)` | 异步调用 AI API |
| `process_one_record_async(session, record_id, row_data)` | 处理单条记录 |

**处理流程**:

```
1. 加载配置 → 初始化组件
2. 分片加载任务
3. 并发调用 AI API (连续任务流模式)
4. 处理结果并写回数据源
5. 监控内存，定期清理
```

---

#### `scheduler.py` - ShardedTaskManager

**功能**: 分片任务调度管理器，处理数据分片、进度跟踪、内存监控。

**类**: `ShardedTaskManager`

**初始化参数**:

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `task_pool` | `BaseTaskPool` | - | 数据源任务池 |
| `optimal_shard_size` | `int` | `10000` | 理想分片大小 |
| `min_shard_size` | `int` | `1000` | 最小分片大小 |
| `max_shard_size` | `int` | `50000` | 最大分片大小 |
| `max_retry_counts` | `dict` | - | 各错误类型最大重试次数 |

**主要方法**:

| 方法 | 描述 |
|------|------|
| `initialize()` | 初始化分片，获取数据边界，创建分片边界列表 |
| `load_next_shard()` | 加载下一个分片 |
| `calculate_optimal_shard_size(total_range)` | 动态计算最优分片大小 |
| `update_processing_metrics(batch_success_count, batch_processing_time)` | 更新处理指标 (EMA) |
| `monitor_memory_usage()` | 监控内存使用，高内存时触发 GC |
| `finalize()` | 完成处理，输出统计信息 |

**属性**:

| 属性 | 描述 |
|------|------|
| `has_more_shards` | 是否还有更多分片 |
| `progress_percent` | 当前进度百分比 |

---

#### `validator.py` - JsonValidator

**功能**: 根据配置规则验证 AI 响应中的字段值。

**类**: `JsonValidator`

**方法**:

| 方法 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `configure(validation_config)` | `dict \| None` | `None` | 从配置加载验证规则 |
| `validate(data)` | `dict` | `tuple[bool, list[str]]` | 验证数据，返回 (是否通过, 错误列表) |
| `get_rules_summary()` | - | `dict[str, int]` | 获取规则摘要 |

**属性**:

| 属性 | 类型 | 描述 |
|------|------|------|
| `enabled` | `bool` | 是否启用验证 |
| `field_rules` | `dict[str, list]` | 字段验证规则 {字段: 允许值列表} |

---

### data - 数据源层

#### `base.py` - BaseTaskPool

**功能**: 数据源任务池抽象基类，定义统一接口。

**类**: `BaseTaskPool` (ABC)

**初始化参数**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `columns_to_extract` | `list[str]` | 需要提取的列名列表 |
| `columns_to_write` | `dict[str, str]` | 写回映射 {别名: 实际列名} |
| `require_all_input_fields` | `bool` | 是否要求所有输入字段都非空 |

**抽象方法** (子类必须实现):

| 方法 | 描述 |
|------|------|
| `get_total_task_count()` | 获取未处理任务总数 |
| `get_id_boundaries()` | 获取任务 ID 边界 |
| `initialize_shard(shard_id, min_id, max_id)` | 初始化分片 |
| `get_task_batch(batch_size)` | 从内存队列获取一批任务 |
| `update_task_results(results)` | 批量写回任务结果 |
| `reload_task_data(task_id)` | 重新加载任务原始输入数据 |
| `close()` | 关闭资源 |

**具体方法**:

| 方法 | 描述 |
|------|------|
| `add_task_to_front(task_id, record_dict)` | 将任务放回队列头部 (重试) |
| `add_task_to_back(task_id, record_dict)` | 将任务放到队列尾部 |
| `has_tasks()` | 检查内存队列是否有任务 |
| `get_remaining_count()` | 获取剩余任务数量 |
| `clear_tasks()` | 清空内存队列 |

---

#### `excel.py` - ExcelTaskPool

**功能**: Excel 数据源任务池实现，基于 DataFrame 引擎抽象。

**类**: `ExcelTaskPool(BaseTaskPool)`

**初始化参数**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `input_path` | `str \| Path` | 输入 Excel 文件路径 |
| `output_path` | `str \| Path` | 输出 Excel 文件路径 |
| `save_interval` | `int` | 自动保存间隔 (秒) |
| `engine_type` | `str` | DataFrame 引擎类型 ("pandas" \| "polars") |

**内部方法**:

| 方法 | 描述 |
|------|------|
| `_validate_and_prepare_columns()` | 验证和准备列 |
| `_filter_unprocessed_indices(min_idx, max_idx)` | 向量化过滤未处理索引 |
| `_save_excel()` | 保存 Excel 文件，处理编码问题 |
| `_clear_problematic_cells(df)` | 清空有编码问题的单元格 |

---

#### `mysql.py` - MySQLTaskPool

**功能**: MySQL 数据源任务池实现，支持连接池管理。

**类**: `MySQLConnectionPoolManager` (单例模式)

| 方法 | 描述 |
|------|------|
| `get_pool(config, pool_name, pool_size)` | 获取连接池实例 |
| `close_pool()` | 关闭连接池 |

**类**: `MySQLTaskPool(BaseTaskPool)`

**初始化参数**:

| 参数 | 类型 | 描述 |
|------|------|------|
| `connection_config` | `dict` | 数据库连接配置 |
| `table_name` | `str` | 目标表名 |
| `pool_size` | `int` | 连接池大小 |

**内部方法**:

| 方法 | 描述 |
|------|------|
| `_get_connection()` | 从连接池获取连接 |
| `execute_with_connection(callback, is_write)` | 使用连接执行回调，自动管理事务 |
| `_build_unprocessed_condition()` | 构建未处理任务的 WHERE 条件 |

---

#### `factory.py`

**功能**: 根据配置创建对应的任务池实例。

**函数**:

| 函数 | 描述 |
|------|------|
| `create_task_pool(config, columns_to_extract, columns_to_write)` | 任务池工厂函数 |

**常量**:

| 常量 | 描述 |
|------|------|
| `MYSQL_AVAILABLE` | MySQL 连接器是否可用 |
| `EXCEL_ENABLED` | Excel 支持是否可用 (pandas + openpyxl) |

---

#### `engines/__init__.py` - 引擎工厂

**功能**: 引擎工厂函数和库可用性检测。

**常量**:

| 常量 | 描述 |
|------|------|
| `POLARS_AVAILABLE` | Polars 是否可用 |
| `FASTEXCEL_AVAILABLE` | FastExcel (Calamine) 是否可用 |
| `XLSXWRITER_AVAILABLE` | xlsxwriter 是否可用 |
| `NUMPY_AVAILABLE` | NumPy 是否可用 |

**类型定义**:

| 类型 | 可选值 | 描述 |
|------|--------|------|
| `EngineType` | `"pandas"`, `"polars"`, `"auto"` | 引擎类型 |
| `ReaderType` | `"openpyxl"`, `"calamine"`, `"auto"` | Excel 读取器类型 |
| `WriterType` | `"openpyxl"`, `"xlsxwriter"`, `"auto"` | Excel 写入器类型 |

**函数**:

| 函数 | 描述 |
|------|------|
| `get_engine(engine_type, excel_reader, excel_writer)` | 获取引擎实例，支持自动回退 |
| `get_available_libraries()` | 获取所有库的可用状态 |
| `_resolve_engine(engine_type)` | 解析实际使用的引擎 |
| `_resolve_reader(reader_type)` | 解析实际使用的读取器 |
| `_resolve_writer(writer_type)` | 解析实际使用的写入器 |

**自动回退机制**:

| 配置值 | 优先选择 | 回退选择 |
|--------|---------|---------|
| `engine: auto` | polars | pandas |
| `excel_reader: auto` | calamine | openpyxl |
| `excel_writer: auto` | xlsxwriter | openpyxl |

---

#### `engines/base.py` - BaseEngine

**功能**: DataFrame 引擎抽象基类，定义统一接口。

**类**: `BaseEngine` (ABC)

**抽象方法分类**:

**文件 I/O**:
| 方法 | 描述 |
|------|------|
| `read_excel(path, sheet_name)` | 读取 Excel 文件 |
| `write_excel(df, path, sheet_name)` | 写入 Excel 文件 |
| `read_csv(path)` | 读取 CSV 文件 |
| `write_csv(df, path)` | 写入 CSV 文件 |

**行操作**:
| 方法 | 描述 |
|------|------|
| `get_row(df, idx)` | 获取指定行数据 |
| `get_rows_by_indices(df, indices)` | 批量获取多行数据 |
| `set_value(df, idx, column, value)` | 设置单元格值 |
| `set_values_batch(df, updates)` | 批量设置多个单元格值 |

**列操作**:
| 方法 | 描述 |
|------|------|
| `get_column_names(df)` | 获取所有列名 |
| `has_column(df, column)` | 检查列是否存在 |
| `add_column(df, column, default_value)` | 添加新列 |

**过滤与查询**:
| 方法 | 描述 |
|------|------|
| `filter_indices(df, column, condition, value)` | 根据条件过滤行 |
| `filter_indices_vectorized(df, input_columns, output_columns, require_all_inputs)` | 向量化过滤未处理行 |

**值操作**:
| 方法 | 描述 |
|------|------|
| `is_empty(value)` | 判断值是否为空 |
| `is_empty_vectorized(series)` | 向量化判断空值 |
| `to_string(value)` | 将值转换为字符串 |

**信息查询**:
| 方法 | 描述 |
|------|------|
| `row_count(df)` | 获取行数 |
| `get_index_range(df)` | 获取索引范围 |
| `get_indices(df)` | 获取所有索引 |

**迭代器**:
| 方法 | 描述 |
|------|------|
| `iter_rows(df, columns)` | 迭代所有行 |

**DataFrame 操作**:
| 方法 | 描述 |
|------|------|
| `slice_by_index_range(df, min_idx, max_idx)` | 按索引范围切片 |
| `copy(df)` | 创建 DataFrame 副本 |

---

#### `engines/pandas_engine.py` - PandasEngine

**功能**: 基于 pandas 的默认实现，支持高性能读写器。

**特点**:
- 成熟稳定，生态丰富
- 支持 calamine (fastexcel) 高速读取 (10x+)
- 支持 xlsxwriter 高速写入 (2-5x)
- 支持 numpy 向量化加速
- 适合中小规模数据 (< 100万行)

**初始化参数**:

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `excel_reader` | `str` | `"openpyxl"` | Excel 读取器 (openpyxl \| calamine) |
| `excel_writer` | `str` | `"openpyxl"` | Excel 写入器 (openpyxl \| xlsxwriter) |

---

#### `engines/polars_engine.py` - PolarsEngine

**功能**: 基于 Polars 的高性能实现（已完整实现）。

**特点**:
- 多线程并行处理
- 惰性求值 (LazyFrame)
- 内存效率高 (-30% ~ -50%)
- 原生支持 fastexcel 和 xlsxwriter
- 适合大规模数据 (> 100万行)

**初始化参数**:

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `excel_reader` | `str` | `"calamine"` | Excel 读取器 |
| `excel_writer` | `str` | `"xlsxwriter"` | Excel 写入器 |

**性能对比**:

| 操作 | pandas+openpyxl | polars+calamine+xlsxwriter | 提升 |
|------|-----------------|---------------------------|------|
| 读取 100万行 | ~60s | ~5s | **12x** |
| 过滤 100万行 | ~10s | ~0.5s | **20x** |
| 写入 100万行 | ~30s | ~10s | **3x** |

---

### gateway - API 网关

#### `app.py`

**功能**: FastAPI 应用创建和路由定义。

**函数**:

| 函数 | 描述 |
|------|------|
| `create_app(config_path)` | 创建 FastAPI 应用 |
| `run_server(config_path, host, port, workers, reload)` | 运行服务器 |
| `main()` | 命令行入口 |

**API 端点**:

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/chat/completions` | POST | 聊天补全 (OpenAI 兼容) |
| `/v1/models` | GET | 可用模型列表 |
| `/admin/models` | GET | 模型详细状态 |
| `/admin/health` | GET | 健康检查 |
| `/` | GET | 根路径 |

---

#### `service.py` - FluxApiService

**功能**: Flux API 核心服务，管理模型池和请求处理。

**类**: `FluxApiService`

**初始化参数**:
- `config_path: str` - 配置文件路径

**核心属性**:

| 属性 | 类型 | 描述 |
|------|------|------|
| `models` | `list[ModelConfig]` | 模型配置列表 |
| `dispatcher` | `ModelDispatcher` | 模型调度器 |
| `rate_limiter` | `ModelRateLimiter` | 限流器 |
| `session_pool` | `SessionPool` | HTTP 连接池 |

**主要方法**:

| 方法 | 描述 |
|------|------|
| `startup()` | 启动服务 (异步初始化) |
| `shutdown()` | 关闭服务 |
| `resolve_model_id(model_name_or_id)` | 解析模型名称/ID |
| `get_available_model(requested_model_name, exclude_models)` | 获取可用模型 (加权随机) |
| `chat_completion(request)` | 处理聊天补全请求 (支持流式/非流式) |
| `get_health_status()` | 获取健康状态 |
| `get_models_info()` | 获取模型信息 |

---

#### `dispatcher.py`

**功能**: 模型调度器，实现错误退避和加权随机选择。

**类**: `ModelConfig`

存储单个模型的配置信息。

| 属性 | 类型 | 描述 |
|------|------|------|
| `id` | `str` | 模型 ID |
| `name` | `str` | 模型名称 |
| `model` | `str` | 实际模型标识符 |
| `channel_id` | `str` | 所属通道 ID |
| `api_key` | `str` | API 密钥 |
| `weight` | `int` | 调度权重 |
| `safe_rps` | `float` | 每秒安全请求数 |
| `api_url` | `str` | 完整 API URL |

**类**: `ModelDispatcher`

| 方法 | 描述 |
|------|------|
| `is_model_available(model_id)` | 判断模型是否可用 |
| `mark_model_success(model_id)` | 标记模型调用成功 |
| `mark_model_failed(model_id, error_type)` | 标记模型调用失败，触发退避 |
| `select_model(exclude_model_ids)` | 加权随机选择可用模型 |
| `get_available_models(exclude_model_ids)` | 获取所有可用模型 ID |
| `update_model_metrics(model_id, response_time, success)` | 更新模型性能指标 |
| `get_model_success_rate(model_id)` | 获取模型成功率 |
| `get_all_model_stats()` | 获取所有模型统计信息 |

**退避算法**:
- 1-3 次失败: `fail_count * 2` 秒
- 4+ 次失败: 指数退避，最长 60 秒

---

#### `limiter.py`

**功能**: 限流组件，包含读写锁和令牌桶。

**类**: `RWLock`

读写锁，允许多读单写。

| 方法 | 描述 |
|------|------|
| `read_acquire()` / `read_release()` | 获取/释放读锁 |
| `write_acquire()` / `write_release()` | 获取/释放写锁 |
| `read_lock()` | 返回读锁上下文管理器 |
| `write_lock()` | 返回写锁上下文管理器 |

**类**: `TokenBucket`

令牌桶限流器。

| 属性 | 描述 |
|------|------|
| `capacity` | 桶容量 (最大令牌数) |
| `refill_rate` | 每秒补充的令牌数 |

| 方法 | 描述 |
|------|------|
| `consume(tokens)` | 尝试消耗指定数量的令牌 |
| `get_tokens()` | 获取当前令牌数 |

**类**: `ModelRateLimiter`

模型限流管理器，为每个模型维护独立的令牌桶。

| 方法 | 描述 |
|------|------|
| `configure(models_config)` | 从模型配置中配置限流器 |
| `can_process(model_id)` | 检查是否可处理请求 (不消耗) |
| `acquire(model_id)` | 尝试获取请求许可 (消耗令牌) |
| `get_status(model_id)` | 获取限流器状态 |

---

#### `session.py` - SessionPool

**功能**: HTTP Session 连接池管理，按 (ssl_verify, proxy) 组合复用连接。

**类**: `SessionPool`

**初始化参数**:

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `max_connections` | `int` | `1000` | 总最大连接数 |
| `max_connections_per_host` | `int` | `1000` | 每个主机最大连接数 |
| `resolver` | `AbstractResolver \| None` | `None` | 自定义 DNS 解析器 |

| 方法 | 描述 |
|------|------|
| `get_or_create(ssl_verify, proxy)` | 获取或创建 ClientSession |
| `close_all()` | 关闭所有 Session |
| `get_stats()` | 获取连接池统计信息 |

**DNS 缓存行为**:
- 无自定义解析器: 启用 DNS 缓存 (TTL=10s)
- 有自定义解析器: 禁用 DNS 缓存，确保每次新连接触发轮询

---

#### `resolver.py` - IP 池轮询解析器

**功能**: 自定义 DNS 解析器，支持按通道配置多 IP 地址，实现均匀负载分配和故障回退。

**使用场景**:
- 目标 API 有多个服务器 IP，希望均匀分配请求
- 需要在应用层控制 IP 选择策略
- DNS 轮询不满足需求（如需要更均匀的分布）

**类**: `RoundRobinResolver`

实现 `aiohttp.abc.AbstractResolver` 接口，对配置了 IP 池的域名进行轮询解析。

**初始化参数**:

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `ip_pools` | `dict[str, list[str]] \| None` | `None` | 域名到 IP 列表的映射 |
| `default_port` | `int` | `443` | 默认端口号 |

**方法**:

| 方法 | 描述 |
|------|------|
| `resolve(host, port, family)` | 解析域名，返回轮询排序的 IP 列表 |
| `close()` | 关闭解析器 |
| `get_stats()` | 获取解析器统计信息 |

**轮询行为**:
```
IP 列表: [ip1, ip2, ip3]

第 1 次解析: [ip1, ip2, ip3]  # 从 ip1 开始
第 2 次解析: [ip2, ip3, ip1]  # 从 ip2 开始
第 3 次解析: [ip3, ip1, ip2]  # 从 ip3 开始
第 4 次解析: [ip1, ip2, ip3]  # 循环回到 ip1
```

**故障回退**:
- aiohttp 会按返回的 IP 列表顺序尝试连接
- 如果第一个 IP 连接失败，自动尝试下一个
- 无需额外配置，由 aiohttp 内置机制处理

**函数**: `build_ip_pools_from_channels(channels)`

从通道配置自动构建 IP 池映射。

| 参数 | 类型 | 描述 |
|------|------|------|
| `channels` | `dict[str, Any]` | 通道配置字典 |

**返回值**: `dict[str, list[str]]` - 域名到 IP 列表的映射

**注意事项**:
- 配置了 `proxy` 的通道会忽略 `ip_pool`（代理模式下无效）
- 只有合法的 IPv4/IPv6 地址会被添加到池中
- 多个通道指向同一域名时，IP 列表会合并

---

#### `schemas.py`

**功能**: Pydantic 模型定义，用于 API 请求/响应验证。

**请求模型**:

| 模型 | 描述 |
|------|------|
| `ChatMessage` | 聊天消息 (role, content, name) |
| `ResponseFormat` | 响应格式定义 |
| `ChatCompletionRequest` | 聊天补全请求体 |

**响应模型**:

| 模型 | 描述 |
|------|------|
| `ChatCompletionResponseChoice` | 响应选项 |
| `ChatCompletionResponseUsage` | Token 使用情况 |
| `ChatCompletionResponse` | 聊天补全响应体 |
| `ModelInfo` | 模型信息 |
| `ModelsResponse` | /admin/models 响应体 |
| `HealthResponse` | /admin/health 响应体 |

---

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户/客户端                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   main.py     │      │  gateway.py   │      │   外部调用    │
│ (数据处理)     │      │  (API 网关)   │      │ (OpenAI 兼容) │
└───────┬───────┘      └───────┬───────┘      └───────┬───────┘
        │                      │                      │
        │                      └──────────┬───────────┘
        │                                 │
        ▼                                 ▼
┌───────────────────────┐      ┌─────────────────────────────────────┐
│      core/            │      │          gateway/                   │
│  ┌─────────────────┐  │      │  ┌─────────────┐  ┌─────────────┐  │
│  │ processor.py    │  │─────▶│  │ service.py  │  │ dispatcher  │  │
│  │ (处理引擎)       │  │      │  │ (服务核心)   │◀─│ (模型调度)   │  │
│  └─────────────────┘  │      │  └──────┬──────┘  └─────────────┘  │
│  ┌─────────────────┐  │      │         │         ┌─────────────┐  │
│  │ scheduler.py    │  │      │         │         │ limiter.py  │  │
│  │ (分片调度)       │  │      │         │         │ (限流组件)   │  │
│  └─────────────────┘  │      │         │         └─────────────┘  │
│  ┌─────────────────┐  │      │         │         ┌─────────────┐  │
│  │ validator.py    │  │      │         │         │ session.py  │  │
│  │ (JSON 验证)     │  │      │         │         │ (连接池)    │  │
│  └─────────────────┘  │      │         │         └─────────────┘  │
└───────────┬───────────┘      └─────────┼─────────────────────────┘
            │                            │
            ▼                            ▼
┌───────────────────────┐      ┌─────────────────────────────────────┐
│      data/            │      │          外部 AI API                 │
│  ┌─────────────────┐  │      │  ┌─────────────┐  ┌─────────────┐  │
│  │ excel.py        │  │      │  │ OpenAI      │  │ Claude      │  │
│  │ (Excel 任务池)  │  │      │  │ API         │  │ API         │  │
│  └─────────────────┘  │      │  └─────────────┘  └─────────────┘  │
│  ┌─────────────────┐  │      │  ┌─────────────┐  ┌─────────────┐  │
│  │ mysql.py        │  │      │  │ 其他         │  │ ...         │  │
│  │ (MySQL 任务池)  │  │      │  │ 兼容 API     │  │             │  │
│  └─────────────────┘  │      │  └─────────────┘  └─────────────┘  │
│  ┌─────────────────┐  │      └─────────────────────────────────────┘
│  │ engines/        │  │
│  │ (DataFrame引擎) │  │
│  └─────────────────┘  │
└───────────────────────┘
```

---

## 数据流向

### 批处理流程 (main.py)

```
1. 加载配置 (config/settings.py)
       ↓
2. 初始化任务池 (data/factory.py)
       ↓
3. 初始化分片管理器 (core/scheduler.py)
       ↓
4. 计算分片边界
       ↓
   ┌──▶ 5. 加载下一分片 ───────┐
   │          ↓                │
   │   6. 获取任务批次          │ 循环
   │          ↓                │
   │   7. 并发调用 API 网关     │
   │          ↓                │
   │   8. 解析响应/验证         │
   │          ↓                │
   │   9. 写回结果             │
   │          ↓                │
   └── 10. 检查更多分片? ◀─────┘
              ↓
      11. 输出统计/清理资源
```

### API 网关流程 (gateway.py)

```
1. 接收请求 (/v1/chat/completions)
       ↓
2. 解析模型名称 (resolve_model_id)
       ↓
3. 选择可用模型 (加权随机 + 限流检查)
       ↓
4. 获取 HTTP Session (session_pool)
       ↓
5. 调用外部 API
       ↓
   ┌────────────────────────┐
   │ 成功?                  │
   ├──────┬─────────────────┤
   │ 是   │ 否              │
   │      ↓                 │
   │  标记失败，触发退避     │
   │      ↓                 │
   │  尝试其他模型           │
   │  (最多 3 次)           │
   └──────┴─────────────────┘
       ↓
6. 返回响应 (流式/非流式)
       ↓
7. 更新模型指标
```

---

## 配置示例

详见 `config-example.yaml`，主要配置项:

| 配置块 | 描述 |
|--------|------|
| `global` | 全局配置 (日志、API URL) |
| `gateway` | 网关配置 (上游连接池并发上限) |
| `datasource` | 数据源配置 (类型、引擎、并发、重试) |
| `datasource.engine` | DataFrame 引擎 (auto/pandas/polars) |
| `datasource.excel_reader` | Excel 读取器 (auto/openpyxl/calamine) |
| `datasource.excel_writer` | Excel 写入器 (auto/openpyxl/xlsxwriter) |
| `mysql` | MySQL 连接配置 |
| `excel` | Excel 文件路径配置 |
| `columns_to_extract` | 输入列列表 |
| `columns_to_write` | 输出列映射 |
| `prompt` | AI 提示词配置 |
| `validation` | 字段验证规则 |
| `models` | 模型配置列表 |
| `channels` | 通道配置 (API 端点、IP 池) |

### IP 池配置示例

为通道配置多 IP 地址，实现均匀负载分配和故障回退：

```yaml
channels:
  "1":
    name: "openai-api"
    base_url: "https://api.openai.com"
    api_path: "/v1/chat/completions"
    timeout: 300
    ssl_verify: true
    # IP 池配置 - 均匀分配请求到多个 IP
    ip_pool:
      - "104.18.6.192"
      - "104.18.7.192"
      - "172.64.155.188"
```

**工作原理**:
1. 启动时从通道配置提取 IP 池，创建 `RoundRobinResolver`
2. 每次新建连接时，解析器返回轮询排序的 IP 列表
3. aiohttp 按列表顺序尝试连接，第一个失败自动尝试下一个
4. 连接复用不受影响，只在新建连接时触发轮询

**注意事项**:
- 配置了 `proxy` 的通道，`ip_pool` 会被忽略
- IP 必须是有效的 IPv4 或 IPv6 地址
- 域名的 TLS/SNI 验证不受影响（仍使用 `base_url` 中的域名）

### 高性能引擎配置示例

```yaml
datasource:
  type: excel
  
  # 引擎选择 (推荐 auto)
  engine: auto        # auto | pandas | polars
  excel_reader: auto  # auto | openpyxl | calamine  
  excel_writer: auto  # auto | openpyxl | xlsxwriter
```

---

*文档版本: 2.2 | 最后更新: 2026-01*
