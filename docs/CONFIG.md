# AI-DataFlux 配置文件说明

本文档详细说明 AI-DataFlux 配置文件（`config.yaml`）的所有配置项，包括结构、用途、代码位置和实际影响。

---

## 目录

- [配置文件结构](#配置文件结构)
- [配置加载模式](#配置加载模式) ⭐ 新增
- [全局配置 (global)](#全局配置-global)
- [网关配置 (gateway)](#网关配置-gateway)
- [数据源配置 (datasource)](#数据源配置-datasource)
- [数据源特定配置](#数据源特定配置)
- [字段配置](#字段配置)
- [验证配置 (validation)](#验证配置-validation)
- [模型配置 (models)](#模型配置-models)
- [通道配置 (channels)](#通道配置-channels)
- [提示词配置 (prompt)](#提示词配置-prompt)
- [规则路由 (routing)](#规则路由-routing)
- [Token 估算配置 (token_estimation)](#token-估算配置-token_estimation)
- [配置加载机制](#配置加载机制)

---

## 配置文件结构

```yaml
global:                      # 全局配置（日志、API 端点）
gateway:                     # API 网关配置
datasource:                  # 数据源类型和引擎配置
  concurrency:               # 并发和性能配置
    retry_limits:            # 重试限制配置
mysql/postgresql/sqlite/excel/csv:  # 数据源特定配置
columns_to_extract:          # 输入字段列表
columns_to_write:            # 输出字段映射
validation:                  # 字段验证规则
  field_rules:               # 枚举验证规则
models:                      # AI 模型配置（数组）
channels:                    # API 通道配置（字典）
prompt:                      # 提示词配置
  required_fields:           # 必需输出字段
token_estimation:            # Token 估算配置
```

**配置加载位置**：
- `src/config/settings.py:load_config()` - 加载 YAML 文件
- `src/config/settings.py:DEFAULT_CONFIG` - 默认配置定义
- `src/config/settings.py:merge_config()` - 深度合并用户配置和默认配置

**加载流程**：
1. 读取 YAML 文件
2. 与 `DEFAULT_CONFIG` 深度合并
3. 初始化日志系统
4. 传递给各组件使用

---

## 配置加载模式

系统使用单一配置文件模式，通过规则路由实现复杂场景的配置管理。

### 标准模式

**说明**：使用单一配置文件，包含所有配置项。

```bash
python cli.py process --config config.yaml
```

**配置结构**：
```yaml
# config.yaml（完整配置）
global:
  flux_api_url: http://127.0.0.1:8787
  log:
    level: info

datasource:
  type: excel
  # ...

models:
  - name: gpt-4
    # ... (多个模型实例)

channels:
  "1":
    api_key: "xxx"
    # ... (多个通道)

excel:
  input_path: "data.xlsx"
  output_path: "result.xlsx"

columns_to_extract: [...]
columns_to_write: {...}
prompt:
  template: "..."
```

### 规则路由模式

**适用场景**：单个数据文件包含多种数据类型（如不同部门、产品线），需要不同的 prompt 和 validation 配置。

```bash
python cli.py process --config config.yaml
```

**配置结构**：
```
project/
├── config.yaml                 # 主配置（包含路由规则）
└── .config/
    └── rules/                  # 规则配置目录
        ├── type_a.yaml         # 类型 A 的差异配置
        ├── type_b.yaml         # 类型 B 的差异配置
        └── ...
```

**主配置示例**：
```yaml
# config.yaml
global:
  flux_api_url: http://127.0.0.1:8787

datasource:
  type: excel

excel:
  input_path: "data.xlsx"
  output_path: "result.xlsx"

columns_to_extract:
  - "category"      # 路由字段（字段名可自定义）
  - "description"

columns_to_write:
  result: "result"

# 规则路由配置
routing:
  enabled: true
  field: "category"               # 用于路由的字段名（任意字段皆可）
  subtasks:
    - match: "type_a"
      profile: ".config/rules/type_a.yaml"
    - match: "type_b"
      profile: ".config/rules/type_b.yaml"

# 默认配置（路由未匹配时使用）
validation:
  enabled: true
  field_rules:
    result: ["0", "1", "2"]

prompt:
  required_fields: ["result"]
  template: |
    默认模板...
    {record_json}
```

**子配置文件示例**（只允许 `prompt` 和 `validation`）：
```yaml
# .config/rules/type_a.yaml
validation:
  enabled: true
  field_rules:
    result: ["0", "1", "2", "3", "4", "5"]  # type_a 专属标签

prompt:
  template: |
    type_a 专用模板...
    {record_json}
```

**合并顺序**：`DEFAULT_CONFIG → 主配置 → 子配置（仅 prompt/validation）`

**应用场景**：
- 单文件多业务单元场景
- 不同业务单元需要不同的分类标签
- 不同业务单元需要不同的提示词

**详细说明**：参见 [规则路由配置指南](./ROUTING.md)

---

## 全局配置 (global)

### 概述

全局配置包含日志系统和 API 端点设置，影响整个系统的运行。

### 配置项

#### `global.log.level`

- **类型**：字符串
- **默认值**：`"info"`
- **可选值**：`debug`, `info`, `warning`, `error`
- **说明**：日志输出级别
- **代码位置**：
  - 读取：`src/config/settings.py:init_logging()`
  - 使用：`logging.basicConfig(level=...)`
- **影响**：
  - `debug`：输出所有日志，包括 API 请求详情、DataFrame 操作细节
  - `info`：输出常规信息，推荐用于生产环境
  - `warning`/`error`：仅输出警告或错误

#### `global.log.format`

- **类型**：字符串
- **默认值**：`"text"`
- **可选值**：`text`, `json`
- **说明**：日志格式
- **代码位置**：`src/config/settings.py:init_logging()`
- **影响**：
  - `text`：人类可读格式，适合开发调试
  - `json`：JSON 格式，适合日志收集系统（如 ELK）

#### `global.log.output`

- **类型**：字符串
- **默认值**：`"console"`
- **可选值**：`console`, `file`
- **说明**：日志输出目标
- **代码位置**：`src/config/settings.py:init_logging()`
- **影响**：
  - `console`：输出到标准输出
  - `file`：输出到文件（路径由 `file_path` 指定）

#### `global.log.file_path`

- **类型**：字符串
- **默认值**：`"./logs/ai_dataflux.log"`
- **说明**：日志文件路径（仅当 `output=file` 时有效）
- **代码位置**：`src/config/settings.py:init_logging()`
- **影响**：指定日志文件的保存位置

#### `global.flux_api_url`

- **类型**：字符串
- **默认值**：`"http://127.0.0.1:8787"`
- **说明**：Flux API 网关端点 URL，会自动补全 `/v1/chat/completions`
- **代码位置**：
  - 读取：`src/core/processor.py:58-60`
  - 使用：`src/core/clients/flux_client.py:21-24`
- **影响**：数据处理引擎调用 API 网关的地址
- **相关文件**：
  - `src/core/processor.py` - 验证必需性
  - `src/core/clients/flux_client.py` - 自动补全路径逻辑

---

## 网关配置 (gateway)

### 概述

控制 API 网关到上游 AI 服务的连接池行为。**仅影响 Gateway 组件**，不影响数据处理引擎。

### 配置项

#### `gateway.max_connections`

- **类型**：整数
- **默认值**：`1000`
- **说明**：网关到上游 AI API 的总并发连接数上限
- **代码位置**：
  - 读取：`src/gateway/service.py:102`
  - 使用：`src/gateway/session.py:67`
- **影响**：传递给 `aiohttp.TCPConnector(limit=max_connections)`
- **调优建议**：
  - 高并发场景：增大此值（如 2000-5000）
  - 低内存环境：减小此值

#### `gateway.max_connections_per_host`

- **类型**：整数
- **默认值**：`1000`
- **说明**：网关对单个上游主机的最大并发连接数
- **代码位置**：
  - 读取：`src/gateway/service.py:103-104`
  - 使用：`src/gateway/session.py:68`
- **影响**：传递给 `aiohttp.TCPConnector(limit_per_host=...)`
- **相关文件**：
  - `src/gateway/service.py` - 读取配置
  - `src/gateway/session.py` - SessionPool 创建 Connector

---

## 数据源配置 (datasource)

### 概述

数据源配置控制数据读写方式、并发行为、性能引擎选择。

### 基础配置

#### `datasource.type`

- **类型**：字符串
- **默认值**：`"excel"`
- **可选值**：`excel`, `csv`, `mysql`, `postgresql`, `sqlite`
- **说明**：数据源类型
- **代码位置**：
  - 读取：`src/data/factory.py:69`
  - 路由：`src/data/factory.py:82-134`
- **影响**：决定使用哪个 TaskPool 实现
- **相关文件**：
  - `src/data/factory.py` - 工厂方法创建对应任务池
  - `src/data/excel.py` - ExcelTaskPool（excel/csv 共用）
  - `src/data/mysql.py` - MySQLTaskPool
  - `src/data/postgresql.py` - PostgreSQLTaskPool
  - `src/data/sqlite.py` - SQLiteTaskPool

#### `datasource.engine`

- **类型**：字符串
- **默认值**：`"pandas"`（DEFAULT_CONFIG）
- **推荐值**：`"auto"`
- **可选值**：`auto`, `pandas`, `polars`
- **说明**：DataFrame 引擎类型
- **代码位置**：
  - 读取：`src/data/factory.py:76`
  - 解析：`src/data/engines/__init__.py:_resolve_engine()`
  - 使用：`src/data/excel.py:72-76`
- **影响**：
  - `auto`：优先 Polars（如已安装），否则回退 Pandas
  - `pandas`：使用 Pandas（稳定，兼容性好）
  - `polars`：使用 Polars（高性能，多线程，100万+ 行优化）
- **性能对比**（100万行）：
  - Pandas：数据过滤 ~10s
  - Polars：数据过滤 ~0.5s（**20x 提升**）
- **相关文件**：
  - `src/data/engines/__init__.py` - 引擎选择逻辑，库可用性检测
  - `src/data/engines/pandas_engine.py` - Pandas 实现
  - `src/data/engines/polars_engine.py` - Polars 实现

#### `datasource.excel_reader`

- **类型**：字符串
- **默认值**：`"auto"`（配置示例）/ 无默认值（DEFAULT_CONFIG）
- **可选值**：`auto`, `openpyxl`, `calamine`
- **说明**：Excel 读取器类型
- **代码位置**：
  - 读取：`src/data/factory.py:77`
  - 使用：`src/data/engines/pandas_engine.py:63-90`
- **影响**：
  - `auto`：优先 calamine（如已安装 fastexcel），否则 openpyxl
  - `openpyxl`：纯 Python 实现，兼容性好
  - `calamine`：Rust 实现，**10x 读取速度提升**
- **性能对比**（100万行 Excel）：
  - openpyxl：~60s
  - calamine：~5s
- **依赖**：calamine 需要 `pip install fastexcel`
- **相关文件**：
  - `src/data/engines/pandas_engine.py:_resolve_excel_reader()` - 读取器选择逻辑

#### `datasource.excel_writer`

- **类型**：字符串
- **默认值**：`"auto"`
- **可选值**：`auto`, `openpyxl`, `xlsxwriter`
- **说明**：Excel 写入器类型
- **代码位置**：
  - 读取：`src/data/factory.py:78`
  - 使用：`src/data/engines/pandas_engine.py:92-119`
- **影响**：
  - `auto`：优先 xlsxwriter（如已安装），否则 openpyxl
  - `openpyxl`：默认实现
  - `xlsxwriter`：**3x 写入速度提升**
- **性能对比**（100万行 Excel）：
  - openpyxl：~30s
  - xlsxwriter：~10s
- **依赖**：xlsxwriter 需要 `pip install xlsxwriter`
- **相关文件**：
  - `src/data/engines/pandas_engine.py:_resolve_excel_writer()` - 写入器选择逻辑

#### `datasource.require_all_input_fields`

- **类型**：布尔值
- **默认值**：`true`
- **说明**：输入字段的非空检查策略
- **代码位置**：
  - 读取：`src/data/factory.py:72`
  - 传递：`src/data/base.py:37,49`
  - 使用：`src/data/engines/pandas_engine.py:310-324`
- **影响**：控制向量化过滤时的输入字段验证逻辑
  - `true`：**所有**输入字段必须非空（AND 逻辑）
    ```python
    input_valid_mask = True
    for col in input_columns:
        input_valid_mask &= ~is_empty(df[col])
    ```
  - `false`：**至少一个**输入字段非空即可（OR 逻辑）
    ```python
    input_valid_mask = False
    for col in input_columns:
        input_valid_mask |= ~is_empty(df[col])
    ```
- **使用场景**：
  - `true`：严格模式，确保所有上下文信息完整
  - `false`：宽松模式，允许部分缺失（如 question 或 context 任一存在即可）
- **相关文件**：
  - `src/data/engines/pandas_engine.py:filter_indices_vectorized()` - Pandas 实现
  - `src/data/engines/polars_engine.py:filter_indices_vectorized()` - Polars 实现

### 并发配置 (datasource.concurrency)

#### `datasource.concurrency.batch_size`

- **类型**：整数
- **默认值**：`100`
- **说明**：批处理大小，同时也是最大并发任务数
- **代码位置**：
  - 读取：`src/core/processor.py:67`
  - 使用：`src/core/processor.py:201-203`
- **影响**：控制同时处理的任务数量上限
  ```python
  space_available = batch_size - len(active_tasks)
  ```
- **数据库连接池影响**：
  - MySQL/PostgreSQL 连接池默认大小 = `max(5, batch_size // 10)`
  - 代码：`src/data/factory.py:173, 261`
- **调优建议**：
  - 高性能服务器：500-1000
  - 普通服务器：100-300
  - 内存受限：50-100
- **相关文件**：
  - `src/core/processor.py` - 并发控制逻辑

#### `datasource.concurrency.save_interval`

- **类型**：整数（秒）
- **默认值**：`300`
- **说明**：Excel/CSV 文件自动保存间隔
- **代码位置**：
  - 读取：`src/data/factory.py:216, 342`
  - 使用：`src/data/excel.py:288-299`
- **影响**：每隔 N 秒自动保存 DataFrame 到磁盘
  ```python
  if current_time - self.last_save_time >= self.save_interval:
      self._save_excel()
  ```
- **调优建议**：
  - 频繁保存：60-120s（数据安全，性能略低）
  - 正常保存：300s（默认，平衡性能和安全）
  - 延迟保存：600-900s（性能优先，风险略高）
- **注意**：仅对 Excel/CSV 数据源有效，数据库数据源实时写入

#### `datasource.concurrency.shard_size`

- **类型**：整数
- **默认值**：`10000`
- **说明**：默认分片大小（理想值）
- **代码位置**：
  - 读取：`src/core/processor.py:95`
  - 传递：`src/core/scheduler.py:54`
  - 使用：`src/core/scheduler.py:145-149`
- **影响**：计算最终分片大小的参考值
  ```python
  calculated_size = min(memory_limit, time_limit, shard_size)
  ```
- **相关文件**：
  - `src/core/scheduler.py:calculate_optimal_shard_size()` - 动态计算逻辑

#### `datasource.concurrency.min_shard_size`

- **类型**：整数
- **默认值**：`1000`
- **说明**：最小分片大小下限
- **代码位置**：
  - 读取：`src/core/processor.py:96`
  - 使用：`src/core/scheduler.py:149`
- **影响**：防止分片过小
  ```python
  shard_size = max(min_shard_size, calculated_size)
  ```

#### `datasource.concurrency.max_shard_size`

- **类型**：整数
- **默认值**：`50000`
- **说明**：最大分片大小上限
- **代码位置**：
  - 读取：`src/core/processor.py:97`
  - 使用：`src/core/scheduler.py:149`
- **影响**：防止分片过大导致内存溢出
  ```python
  shard_size = min(calculated_size, max_shard_size)
  ```

#### `datasource.concurrency.api_pause_duration`

- **类型**：浮点数（秒）
- **默认值**：`2.0`
- **说明**：API 错误触发后的全局暂停时长
- **代码位置**：
  - 读取：`src/core/processor.py:71`
  - 传递：`src/core/retry/strategy.py:43`
  - 使用：`src/core/retry/strategy.py:81`
- **影响**：API 错误触发熔断时的实际暂停秒数
  ```python
  return RetryDecision(
      action=RetryAction.PAUSE_THEN_RETRY,
      pause_duration=api_pause_duration  # ← 实际睡眠时长
  )
  ```
- **执行流程**（`src/core/processor.py:241-256`）：
  ```python
  if decision.action == RetryAction.PAUSE_THEN_RETRY:
      await asyncio.sleep(decision.pause_duration)  # ← 真实暂停
      self.retry_strategy.last_pause_end_time = time.time()
  ```
- **调优建议**：
  - API 频繁限流：增大（5-10s）
  - API 稳定：减小（1-2s）
- **相关文件**：
  - `src/core/retry/strategy.py` - 熔断决策逻辑
  - `src/core/processor.py` - 执行暂停

#### `datasource.concurrency.api_error_trigger_window`

- **类型**：浮点数（秒）
- **默认值**：`2.0`
- **说明**：API 错误触发熔断的时间窗口
- **代码位置**：
  - 读取：`src/core/processor.py:72`
  - 传递：`src/core/retry/strategy.py:44`
  - 使用：`src/core/retry/strategy.py:78`
- **影响**：判断是否需要触发新的暂停
  ```python
  if (current_time - last_pause_end_time) > api_error_trigger_window:
      # 触发新的暂停
  else:
      # 在窗口期内，不触发额外暂停
  ```
- **工作机制**：
  1. API 错误发生 → 触发暂停 N 秒
  2. 暂停结束 → 记录 `last_pause_end_time`
  3. 窗口期内的新错误 → 不触发额外暂停
  4. 窗口期过后的错误 → 触发新暂停
- **调优建议**：
  - 错误集中爆发：增大窗口（5-10s）避免重复暂停
  - 错误分散：减小窗口（1-2s）快速响应
- **相关文件**：
  - `src/core/retry/strategy.py:decide()` - 熔断判断逻辑

#### `datasource.concurrency.max_connections`

- **类型**：整数
- **默认值**：`1000`
- **说明**：数据处理引擎到 API 网关的总并发连接数上限
- **代码位置**：
  - 读取：`src/core/processor.py:68`
  - 使用：`src/core/processor.py:153`
- **影响**：传递给数据引擎的 `aiohttp.TCPConnector(limit=...)`
  ```python
  connector = aiohttp.TCPConnector(
      limit=self.max_connections,
      limit_per_host=self.max_connections_per_host
  )
  ```
- **与 gateway.max_connections 的区别**：
  - `gateway.max_connections`：网关到上游 AI API 的连接数
  - `datasource.concurrency.max_connections`：数据引擎到网关的连接数
- **调优建议**：通常与 `batch_size` 保持一致或略大

#### `datasource.concurrency.max_connections_per_host`

- **类型**：整数
- **默认值**：`0`（无限制）
- **说明**：数据处理引擎对单个主机（网关）的最大并发连接数
- **代码位置**：
  - 读取：`src/core/processor.py:69`
  - 使用：`src/core/processor.py:154`
- **影响**：传递给 `aiohttp.TCPConnector(limit_per_host=...)`
- **推荐值**：
  - 单网关：`0`（无限制）
  - 多网关：设置具体值

### 重试限制 (datasource.concurrency.retry_limits)

#### `retry_limits.api_error`

- **类型**：整数
- **默认值**：`3`
- **说明**：API 错误（超时、HTTP 错误）的最大重试次数
- **代码位置**：
  - 读取：`src/core/processor.py:77`
  - 传递：`src/core/retry/strategy.py:42`
  - 使用：`src/core/retry/strategy.py:67-82`
- **影响**：
  - 每次 API 错误后判断：`retry_count < max_retries[API_ERROR]`
  - 超过限制 → 标记失败，写入错误信息到所有输出列
- **特殊行为**：
  - **触发全局暂停**（如在时间窗口内）
  - **重载数据**：从数据源重新加载原始数据
- **相关文件**：
  - `src/core/retry/strategy.py:decide()` - 重试决策

#### `retry_limits.content_error`

- **类型**：整数
- **默认值**：`1`
- **说明**：内容错误（JSON 解析失败）的最大重试次数
- **代码位置**：
  - 读取：`src/core/processor.py:78`
  - 使用：`src/core/retry/strategy.py:84-88`
- **影响**：JSON 解析失败后的重试次数
- **特殊行为**：
  - **不触发暂停**（直接重试）
  - **不重载数据**（输入数据未变，可能是 AI 随机性导致）
- **调优建议**：
  - 稳定模型：1 次即可
  - 不稳定模型：2-3 次

#### `retry_limits.system_error`

- **类型**：整数
- **默认值**：`2`
- **说明**：系统错误（内部异常）的最大重试次数
- **代码位置**：
  - 读取：`src/core/processor.py:79`
  - 使用：`src/core/retry/strategy.py:90-94`
- **影响**：内部异常的重试次数
- **特殊行为**：
  - **不触发暂停**
  - **重载数据**（可能是数据处理问题）

---

## 数据源特定配置

### MySQL 配置 (mysql)

**生效条件**：`datasource.type: mysql`

#### 必需字段

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | 字符串 | - | MySQL 服务器地址 |
| `port` | 整数 | `3306` | MySQL 端口 |
| `user` | 字符串 | - | 用户名 |
| `password` | 字符串 | - | 密码 |
| `database` | 字符串 | - | 数据库名 |
| `table_name` | 字符串 | - | 表名 |

#### 可选字段

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pool_size` | 整数 | `max(5, batch_size // 10)` | 连接池大小 |

**代码位置**：
- 读取：`src/data/factory.py:152-158`
- 使用：`src/data/mysql.py:MySQLTaskPool`
- 连接池：`src/data/mysql.py:MySQLConnectionPoolManager`（单例模式）

**相关文件**：
- `src/data/mysql.py` - MySQL 任务池实现

### PostgreSQL 配置 (postgresql)

**生效条件**：`datasource.type: postgresql`

#### 必需字段

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | 字符串 | - | PostgreSQL 服务器地址 |
| `port` | 整数 | `5432` | PostgreSQL 端口 |
| `user` | 字符串 | - | 用户名 |
| `password` | 字符串 | - | 密码 |
| `database` | 字符串 | - | 数据库名 |
| `table_name` | 字符串 | - | 表名 |

#### 可选字段

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `schema_name` | 字符串 | `"public"` | Schema 名称 |
| `pool_size` | 整数 | `max(5, batch_size // 10)` | 连接池大小 |

**代码位置**：
- 读取：`src/data/factory.py:239-245`
- 使用：`src/data/postgresql.py:PostgreSQLTaskPool`
- 连接池：`psycopg2.pool.ThreadedConnectionPool`

**特殊特性**：
- 使用 `psycopg2.extras.execute_batch()` 批量更新
- 事务自动管理

**相关文件**：
- `src/data/postgresql.py` - PostgreSQL 任务池实现

### SQLite 配置 (sqlite)

**生效条件**：`datasource.type: sqlite`

#### 必需字段

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `db_path` | 字符串 | - | SQLite 数据库文件路径 |
| `table_name` | 字符串 | - | 表名 |

**代码位置**：
- 读取：`src/data/factory.py:278-284`
- 使用：`src/data/sqlite.py:SQLiteTaskPool`

**特殊特性**：
- 使用 `threading.local` 实现线程隔离连接（SQLite 不支持跨线程连接）
- 自动启用 WAL 模式提升并发性能
- 无需额外依赖（Python 标准库）

**相关文件**：
- `src/data/sqlite.py` - SQLite 任务池实现

### Excel 配置 (excel)

**生效条件**：`datasource.type: excel`

#### 必需字段

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_path` | 字符串 | - | 输入 Excel 文件路径 |

#### 可选字段

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_path` | 字符串 | `input_path` | 输出 Excel 文件路径 |

**代码位置**：
- 读取：`src/data/factory.py:197-209`
- 使用：`src/data/excel.py:ExcelTaskPool`

**相关文件**：
- `src/data/excel.py` - Excel 任务池实现（同时支持 CSV）

### CSV 配置 (csv)

**生效条件**：`datasource.type: csv`

#### 必需字段

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_path` | 字符串 | - | 输入 CSV 文件路径 |

#### 可选字段

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_path` | 字符串 | `input_path` | 输出 CSV 文件路径 |

**代码位置**：
- 读取：`src/data/factory.py:322-334`
- 使用：`src/data/excel.py:ExcelTaskPool`（复用）

**实现说明**：
- CSV 复用 `ExcelTaskPool` 实现
- 自动检测 `.csv` 后缀并调用 `engine.read_csv()` / `write_csv()`

**相关文件**：
- `src/data/excel.py` - CSV 自动检测逻辑（第 86-98 行）

---

## 字段配置

### columns_to_extract

- **类型**：字符串数组
- **说明**：从数据源提取的输入字段列表
- **代码位置**：
  - 读取：`src/core/processor.py:82`
  - 传递：`src/data/factory.py:50`
  - 使用：`src/data/base.py:47`
- **影响**：
  - 验证数据源中是否存在这些列
  - 用于过滤未处理的行（根据 `require_all_input_fields`）
  - 渲染到 Prompt 的 `{record_json}`
- **示例**：
  ```yaml
  columns_to_extract:
    - "question"
    - "context"
  ```
  渲染结果：
  ```json
  {"question": "...", "context": "..."}
  ```
- **相关文件**：
  - `src/core/content/processor.py:create_prompt()` - 渲染逻辑

### columns_to_write

- **类型**：字典（别名 → 实际列名）
- **说明**：AI 输出字段到数据源列的映射
- **代码位置**：
  - 读取：`src/core/processor.py:83`
  - 传递：`src/data/factory.py:51`
  - 使用：`src/data/base.py:48`
- **影响**：
  - 验证数据源中是否存在目标列
  - AI 返回的 JSON 使用别名（键名）
  - 写回数据源时使用实际列名
- **示例**：
  ```yaml
  columns_to_write:
    answer: "ai_answer"         # AI 返回 {"answer": "..."} → 写入 ai_answer 列
    category: "ai_category"     # AI 返回 {"category": "..."} → 写入 ai_category 列
  ```
- **相关文件**：
  - `src/data/excel.py:_validate_and_prepare_columns()` - 列验证
  - `src/core/content/processor.py:parse_response()` - 别名使用

---

## 验证配置 (validation)

### validation.enabled

- **类型**：布尔值
- **默认值**：`false`（DEFAULT_CONFIG 未包含）
- **说明**：是否启用字段验证
- **代码位置**：
  - 读取：`src/core/processor.py:117-122`
  - 使用：`src/core/validator.py:JsonValidator`
- **影响**：
  - `true`：启用字段枚举验证
  - `false`：跳过验证

### validation.field_rules

- **类型**：字典（字段名 → 允许值列表）
- **说明**：字段枚举验证规则
- **代码位置**：
  - 读取：`src/core/processor.py:119`
  - 使用：`src/core/validator.py:validate()`
- **影响**：验证 AI 返回的字段值是否在允许列表中
- **示例**：
  ```yaml
  validation:
    enabled: true
    field_rules:
      category:
        - "technical"
        - "business"
        - "general"
      sentiment:
        - "positive"
        - "neutral"
        - "negative"
  ```
  如果 AI 返回 `{"category": "other"}`，验证失败 → 抛出 `ValidationError`

**相关文件**：
- `src/core/validator.py` - 验证逻辑
- `src/core/content/processor.py:parse_response()` - 调用验证

---

## 模型配置 (models)

### 概述

模型配置是一个数组，每个元素定义一个 AI 模型的访问信息和行为参数。

### 配置项

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `id` | 整数 | ✓ | 模型唯一 ID |
| `name` | 字符串 | ✓ | 模型显示名称 |
| `model` | 字符串 | ✓ | 实际模型标识符（如 `gpt-4-turbo`） |
| `channel_id` | 字符串 | ✓ | 所属通道 ID |
| `api_key` | 字符串 | ✓ | API 密钥 |
| `timeout` | 整数 | ✓ | 超时时间（秒） |
| `weight` | 整数 | ✓ | 调度权重（加权随机算法） |
| `temperature` | 浮点数 | ✓ | 模型温度参数（0-1） |
| `safe_rps` | 整数 | ✓ | 每秒安全请求数（令牌桶容量 = safe_rps × 2） |
| `supports_json_schema` | 布尔值 | ✓ | 是否支持 JSON Schema |
| `supports_advanced_params` | 布尔值 | ✓ | 是否支持高级参数（presence_penalty 等） |

**代码位置**：
- 读取：`src/gateway/service.py:107-146`
- 调度：`src/gateway/dispatcher.py:ModelDispatcher`
- 限流：`src/gateway/limiter.py:ModelRateLimiter`

### 核心机制

#### 加权随机选择

**代码位置**：`src/gateway/service.py:207-228`

```python
# 过滤可用模型
available_models = [
    m for m in models
    if m.weight > 0
    and m not in excluded
    and dispatcher.is_available(m)
    and rate_limiter.can_process(m)
]

# 加权随机选择
weights = [m.weight for m in available_models]
chosen = random.choices(available_models, weights=weights, k=1)[0]
```

**影响**：
- `weight=10` 的模型被选中概率是 `weight=5` 的 **2 倍**
- `weight=0` 的模型不会被选中

#### 令牌桶限流

**代码位置**：`src/gateway/limiter.py:TokenBucket`

```python
capacity = safe_rps * 2
rate = safe_rps / 秒
```

**影响**：
- `safe_rps=5`：每秒生成 5 个令牌，容量 10
- 突发流量可消耗容量，平均速率不超过 safe_rps

#### 故障转移

**代码位置**：`src/gateway/service.py:237-248`

```python
for attempt in range(3):  # 最多尝试 3 个模型
    try:
        model = get_available_model(excluded)
        result = await call_model(model)
        return result
    except Exception:
        excluded.add(model)
        dispatcher.mark_model_failed(model)
```

**影响**：首选模型失败 → 自动切换其他模型

**相关文件**：
- `src/gateway/service.py` - 模型管理和选择
- `src/gateway/dispatcher.py` - 调度和退避逻辑
- `src/gateway/limiter.py` - 限流实现

---

## 通道配置 (channels)

### 概述

通道配置是一个字典，键为通道 ID（字符串），值为通道配置对象。

### 配置项

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `name` | 字符串 | ✓ | - | 通道名称 |
| `base_url` | 字符串 | ✓ | - | API 基础 URL |
| `api_path` | 字符串 | ✓ | - | API 路径（如 `/v1/chat/completions`） |
| `timeout` | 整数 | ✓ | - | 超时时间（秒） |
| `proxy` | 字符串 | - | `""` | 代理地址（如 `http://127.0.0.1:7890`） |
| `ssl_verify` | 布尔值 | - | `true` | 是否验证 SSL 证书 |
| `ip_pool` | 字符串数组 | - | - | IP 池（轮询 DNS） |

**代码位置**：
- 读取：`src/gateway/service.py:147-167`
- 使用：`src/gateway/service.py:277-326`

### 特殊配置说明

#### `ip_pool` - IP 池轮询

- **类型**：字符串数组
- **说明**：多 IP 负载均衡列表
- **代码位置**：
  - 读取：`src/gateway/service.py:161-166`
  - 使用：`src/gateway/resolver.py:RoundRobinResolver`
- **影响**：
  - 配置后使用自定义 DNS 解析器
  - 每次新连接轮询下一个 IP
  - 禁用 DNS 缓存（确保轮询）
- **示例**：
  ```yaml
  channels:
    "1":
      ip_pool:
        - "1.2.3.4"
        - "1.2.3.5"
        - "1.2.3.6"
  ```
- **注意**：配置 `proxy` 时 `ip_pool` 被忽略

#### `ssl_verify` - SSL 证书验证

- **类型**：布尔值
- **默认值**：`true`
- **说明**：是否验证 SSL 证书
- **使用场景**：
  - `true`：生产环境（推荐）
  - `false`：开发环境、Mac 证书问题临时解决

**相关文件**：
- `src/gateway/session.py` - SessionPool 使用 ssl_verify
- `src/gateway/resolver.py` - RoundRobinResolver 实现

---

## 提示词配置 (prompt)

### 配置项

#### `prompt.required_fields`

- **类型**：字符串数组
- **说明**：AI 必须返回的字段列表
- **代码位置**：
  - 读取：`src/core/processor.py:127`
  - 使用：`src/core/content/processor.py:parse_response()`
- **影响**：
  - 验证 AI 返回的 JSON 是否包含这些字段
  - 缺失字段 → 抛出 `ContentError`
  - 用于构建 JSON Schema（如启用）
- **示例**：
  ```yaml
  prompt:
    required_fields:
      - "answer"
      - "category"
  ```

#### `prompt.use_json_schema`

- **类型**：布尔值
- **默认值**：`false`
- **说明**：是否启用 JSON Schema 输出约束
- **代码位置**：
  - 读取：`src/core/processor.py:128`
  - 使用：`src/core/content/processor.py:build_schema()`
- **影响**：
  - `true`：构建 JSON Schema 并发送给 AI
  - 在 API 请求中添加 `response_format: {type: "json_object"}`
- **相关文件**：
  - `src/core/content/processor.py:build_schema()` - Schema 构建
  - `src/core/clients/flux_client.py:48-49` - API 请求添加 response_format

#### `prompt.temperature`

- **类型**：浮点数（0-1）
- **默认值**：`0.7`
- **说明**：模型温度参数
- **代码位置**：
  - 读取：`src/core/processor.py:129`
  - 使用：`src/core/processor.py:192-194`
- **影响**：传递给 AI 模型的 temperature 参数
- **推荐值**：
  - 分类任务：0.1-0.3（低温度，高确定性）
  - 创作任务：0.7-0.9（高温度，高创造性）

#### `prompt.system_prompt`

- **类型**：字符串（多行）
- **说明**：系统提示词
- **代码位置**：
  - 读取：`src/core/processor.py:130`
  - 使用：`src/core/content/processor.py:create_prompt()`
- **影响**：构建 AI 请求的 system message

#### `prompt.template`

- **类型**：字符串（多行）
- **说明**：用户提示词模板，包含 `{record_json}` 占位符
- **代码位置**：
  - 读取：`src/core/processor.py:131`
  - 使用：`src/core/content/processor.py:create_prompt()`
- **影响**：
  - `{record_json}` 会被替换为数据记录的 JSON 表示
  - 构建 AI 请求的 user message
- **示例**：
  ```yaml
  prompt:
    template: |
      请分析以下数据：
      {record_json}

      请返回 JSON 格式结果。
  ```

**相关文件**：
- `src/core/content/processor.py` - Prompt 渲染和解析

---

## 规则路由 (routing)

> **仅适用于单文件处理模式**（`cli.py process --config ...`）。
> 该路由**不拆分任务**，只在每条记录处理时按字段值切换 **prompt/validation**。
> 未命中规则时**使用主配置**。

### 配置结构

```yaml
routing:
  enabled: true
  field: "category"                # 用于路由的字段名（任意字段皆可）
  subtasks:
    - match: "type_a"
      profile: ".config/rules/type_a.yaml"
    - match: "type_b"
      profile: ".config/rules/type_b.yaml"
```

### 子配置文件要求（profile）
- **只能包含** `prompt` 与 `validation` 两个顶层键
- 仅写差异字段，其他字段继承主配置

示例：
```yaml
prompt:
  template: "..."
  required_fields: ["result"]
validation:
  enabled: true
  field_rules:
    result: ["0", "1", "2"]
```

### 行为规则
- `routing.enabled: true` 才启用
- `routing.field` 为分流字段（字段名可自定义，如 `category`、`type`、`department` 等）
- `routing.subtasks` 为规则列表，**精确匹配** `match` 值
- 每条规则必须包含 `match` 与 `profile`
- **容错处理**：
  - 路由字段不存在于记录中 → 使用主配置（不报错）
  - 路由字段值没有匹配规则 → 使用主配置
- profile 文件中出现其他键会直接报错

### 示例配置

在 `config-example.yaml` 中查看被注释的 routing 配置示例。子配置文件需放置在 `.config/rules/` 目录下。

### 代码位置
- 解析与缓存：`src/core/processor.py:_init_routing_contexts()`
- 规则匹配：`src/core/processor.py:_get_routing_context()`
- 应用逻辑：`src/core/processor.py:_process_one_record()`

---

## Token 估算配置 (token_estimation)

### 配置项

#### `token_estimation.mode`

- **类型**：字符串
- **默认值**：`"io"`
- **可选值**：`in`, `out`, `io`
- **说明**：Token 估算模式
- **代码位置**：
  - 读取：`src/core/token_estimator.py:TokenEstimator.__init__()`
  - 使用：`src/core/token_estimator.py:estimate()`
- **影响**：
  - `in`：仅估算输入 Token（system + user prompt）
  - `out`：仅估算输出 Token（AI 返回的 JSON）
  - `io`：估算输入 + 输出 Token
- **相关文件**：
  - `src/core/token_estimator.py` - Token 估算实现

#### `token_estimation.sample_size`

- **类型**：整数
- **默认值**：`-1`
- **说明**：采样数量
- **影响**：
  - `-1`：全量计算（忽略处理状态，计算所有行）
  - `> 0`：随机采样指定数量的行
- **代码位置**：
  - 读取：`src/core/token_estimator.py:TokenEstimator.__init__()`
  - 使用：`src/data/base.py:sample_unprocessed_rows()`

#### `token_estimation.encoding`

- **类型**：字符串
- **默认值**：`"o200k_base"`
- **说明**：Tiktoken 编码器名称
- **可选值**：`o200k_base`, `cl100k_base`, `p50k_base` 等
- **代码位置**：
  - 读取：`src/core/token_estimator.py:TokenEstimator.__init__()`
  - 使用：`tiktoken.get_encoding(encoding)`
- **影响**：使用不同的 BPE 编码器计算 Token 数

**相关文件**：
- `src/core/token_estimator.py` - Token 估算核心逻辑
- `src/data/base.py` - 采样方法

---

## 配置加载机制

### 加载流程

```
1. main.py / gateway.py
   ↓
2. load_config(config_path)
   ↓
3. 读取 YAML 文件 → user_config
   ↓
4. merge_config(DEFAULT_CONFIG, user_config)
   ↓
5. 初始化日志 (init_logging)
   ↓
6. 传递给各组件
```

### 深度合并逻辑

**代码位置**：`src/config/settings.py:merge_config()`

```python
def merge_config(base: dict, override: dict) -> dict:
    """
    深度合并两个配置字典

    - 嵌套字典会递归合并
    - override 中的值覆盖 base
    - base 中的默认值会保留（如 override 未提供）
    """
```

**示例**：

```yaml
# DEFAULT_CONFIG
datasource:
  concurrency:
    batch_size: 100
    retry_limits:
      api_error: 3

# user_config
datasource:
  concurrency:
    batch_size: 200
```

**合并结果**：
```yaml
datasource:
  concurrency:
    batch_size: 200          # 用户覆盖
    retry_limits:
      api_error: 3           # 保留默认值
```

### 配置验证

**位置**：各组件初始化时验证

| 组件 | 验证内容 | 代码位置 |
|------|---------|---------|
| UniversalAIProcessor | flux_api_url 必需 | `src/core/processor.py:59-60` |
| MySQLTaskPool | host/user/password/database/table_name 必需 | `src/data/factory.py:155-158` |
| PostgreSQLTaskPool | host/user/password/database/table_name 必需 | `src/data/factory.py:243-245` |
| SQLiteTaskPool | db_path/table_name 必需 | `src/data/factory.py:279-284` |
| ExcelTaskPool | input_path 必需且存在 | `src/data/excel.py:62-66` |

---

## 配置文件示例

完整的配置文件示例请参考项目根目录的 `config-example.yaml`。

---

## 常见配置场景

### 高性能配置（大数据集）

```yaml
datasource:
  engine: auto                 # 优先 Polars
  excel_reader: auto           # 优先 calamine
  excel_writer: auto           # 优先 xlsxwriter
  concurrency:
    batch_size: 500            # 高并发
    max_connections: 2000
    max_shard_size: 100000     # 大分片
```

### 低内存配置

```yaml
datasource:
  concurrency:
    batch_size: 50             # 低并发
    max_connections: 100
    max_shard_size: 5000       # 小分片
```

### API 限流严格场景

```yaml
datasource:
  concurrency:
    batch_size: 20             # 低并发
    api_pause_duration: 10.0   # 长暂停
    api_error_trigger_window: 5.0

models:
  - safe_rps: 2                # 低 RPS
    weight: 10
```

---

## 配置优先级

1. **用户配置** (`config.yaml`) - 最高优先级
2. **DEFAULT_CONFIG** (`src/config/settings.py`) - 默认值
3. **代码硬编码** - 部分参数的备用默认值

---

*文档版本: 1.0 | 最后更新: 2026-01-23*
