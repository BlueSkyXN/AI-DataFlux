# AI-DataFlux

AI-DataFlux 是一个高性能、可扩展的通用AI处理引擎，专为批量AI任务处理设计。支持从多种数据源读取任务，使用多种AI模型进行并行处理，并将结果写回数据源。

## 核心特性

- **全能数据源支持**：完整支持 **Excel (xlsx/xls)**、**CSV**、**MySQL**、**PostgreSQL** 和 **SQLite**。
- **可插拔数据引擎**：支持 Pandas 和 Polars 双引擎，可通过配置切换。
- **高性能读写器**：支持 Calamine (10x 读取) 和 xlsxwriter (3x 写入) 加速。
- **自动引擎选择**：`engine: auto` 自动选择最快的可用引擎和读写器。
- **智能模型调度**：基于加权负载均衡的多模型调度系统，支持自动故障切换。
- **高并发处理**：优化的异步架构实现高吞吐量任务处理。
- **连续任务流**：采用连续任务流模式，比传统批处理模式更高效。
- **API错误自适应**：全局暂停机制替代传统重试，更优雅地处理API限流。
- **分类重试机制**：按错误类型（API/内容/系统）独立配置重试次数。
- **灵活的配置系统**：通过YAML文件实现全方位配置。
- **字段值验证**：支持对返回结果的字段值进行枚举验证。
- **JSON Schema支持**：通过Schema约束AI输出格式，提高结果一致性。
- **读写锁优化**：使用读写锁分离，提高多线程下的并发性能。
- **Session连接池**：HTTP连接复用，减少连接建立开销。
- **向量化数据过滤**：DataFrame操作使用向量化方法，大数据集性能提升50-100x。
- **内存使用监控**：自动监控内存使用，在高内存使用时触发垃圾回收。
- **可视化进度**：实时显示处理进度和统计信息。
- **组件化架构**：清晰的 `src/core/` 组件化设计（Content, State, Retry, Clients），易于扩展和维护。
- **规则路由**：支持按记录字段值动态选择不同的 prompt 和 validation 配置，实现单文件多业务场景处理。

## 快速开始

### 环境要求

- **Python 3.10+**（支持 3.10, 3.11, 3.12, 3.13, 3.14）

> **平台兼容性说明**：
> - **Windows ARM/ARM64 当前不支持，CI 不覆盖**
> - 已支持平台：Linux x64/ARM64、Windows x64、macOS x64/ARM64
> - 部分高性能库在个别平台可能不兼容，程序会自动检测并回退到 Pandas 引擎

### 安装依赖

```bash
# 基础依赖 (必需)
pip install pyyaml aiohttp pandas openpyxl psutil pydantic fastapi uvicorn

# 数据库支持 (按需选择)
pip install mysql-connector-python  # MySQL
pip install psycopg2-binary         # PostgreSQL
# SQLite 为 Python 标准库，无需安装

# 高性能引擎 (推荐，可显著提升性能)
pip install numpy polars python-calamine fastexcel xlsxwriter
```

**性能对比**（处理100万行数据）:

| 操作 | pandas+openpyxl | polars+calamine+xlsxwriter | 提升 |
| --- | --- | --- | --- |
| Excel 读取 | ~60s | ~5s | **12x** |
| 数据过滤 | ~10s | ~0.5s | **20x** |
| Excel 写入 | ~30s | ~10s | **3x** |

### 配置文件

根据提供的`config-example.yaml`创建您的配置文件：

```bash
cp config-example.yaml config.yaml
```

然后编辑`config.yaml`，设置您的API密钥、数据源信息等配置。

### 使用统一命令行 (推荐)

```bash
# 查看帮助
python cli.py --help

# 检查依赖库状态
python cli.py check

# 查看版本
python cli.py version

# 启动 Web GUI 控制面板
python cli.py gui

# 启动 API 网关
python cli.py gateway --port 8787

# 运行数据处理
python cli.py process --config config.yaml

# 仅验证配置
python cli.py process --config config.yaml --validate

# 估算 Token 用量 (默认输入+输出, mode=io)
python cli.py token --config config.yaml

# 估算 Token 用量 (仅输出，需要 output 文件有已处理结果)
python cli.py token --config config.yaml --mode out

# 估算 Token 用量 (输入+输出)
python cli.py token --config config.yaml --mode io
```

### Web GUI 控制面板

AI-DataFlux 提供了一个本地 Web GUI 控制面板，方便管理 Gateway 和 Process 服务。

```bash
# 启动控制面板（自动打开浏览器）
python cli.py gui

# 指定端口
python cli.py gui --port 8080

# 不自动打开浏览器
python cli.py gui --no-browser
```

控制面板功能：

- **Dashboard**：启动/停止 Gateway 和 Process，查看运行状态和进度
  - 实时状态监控（PID、端口、模型数、运行时间等）
  - 进度条显示任务处理进度
  - 支持外部进程检测（External标签）
- **配置编辑**：在线编辑 YAML 配置文件，支持自动备份和语法验证
- **日志查看**：实时查看 Gateway 和 Process 的日志输出
  - 左右分栏同时显示 Gateway 和 Process 日志
  - WebSocket 自动重连
  - 支持日志复制和清空
- **多语言支持**：界面支持中文/英文切换
- **工作目录显示**：实时显示当前工作目录

控制面板默认运行在 `http://127.0.0.1:8790`，仅监听本地地址。

**详细文档**：[Web GUI 控制面板指南](./docs/GUI.md)

### 规则路由

规则路由允许在处理单个数据文件时，根据记录中某个字段的值动态选择不同的 `prompt` 和 `validation` 配置。

#### 使用方式

```bash
# 在 config.yaml 中启用 routing 配置
python cli.py process --config config.yaml

# 验证配置（会显示路由规则数量）
python cli.py process --config config.yaml --validate
```

#### 配置结构

```yaml
# 在主配置文件中添加 routing 节点
routing:
  enabled: true
  field: "category"                # 用于路由的字段名（任意字段皆可）
  subtasks:
    - match: "type_a"              # 字段值
      profile: ".config/rules/type_a.yaml"  # 差异配置文件
    - match: "type_b"
      profile: ".config/rules/type_b.yaml"
```

**核心功能**：
- **按记录路由**：每条记录根据字段值使用不同配置
- **差异配置**：子配置只需定义与主配置不同的 `prompt` 和 `validation`
- **优雅降级**：字段不存在或无匹配规则时使用默认配置
- **向后兼容**：不启用路由时配置文件格式完全兼容旧版本

**详细文档**：[规则路由配置指南](./docs/ROUTING.md)

### Token 估算说明

- 输入 token：系统提示词 + 用户提示词模板渲染后的完整文本（`{record_json}` 来自 `columns_to_extract`），system 与 user 内容用换行拼接后直接用 tiktoken 计数，不包含 chat 消息格式的固定开销。
- 输出 token：按 `columns_to_write` 重组 JSON（别名为键）并序列化后，用 tiktoken 计数。
- 默认 mode 为 `io`，可用 `--mode out` 或 `--mode in` 覆盖（命令行会覆盖配置文件中的 `token_estimation.mode`）。
- 当 `token_estimation.sample_size: -1` 时，会对所有行进行全量计算（忽略处理状态），并输出进度日志。
- 默认编码器为 `o200k_base`。

### 启动 API 网关

首先启动API网关（确保始终在后台运行）：

```bash
# 使用统一入口
python cli.py gateway --port 8787

# 或使用独立入口
python gateway.py --config config.yaml --port 8787

# 后台运行
nohup python cli.py gateway --port 8787 > gateway.log 2>&1 &
```

### 运行数据处理引擎

确保API网关已启动，然后运行数据处理引擎：

```bash
# 使用统一入口
python cli.py process --config config.yaml

# 或使用独立入口
python main.py --config config.yaml

# 仅验证配置文件
python cli.py process --validate
```

您也可以使用`screen`或`tmux`等工具在后台运行这两个组件。

## 详细配置说明

配置文件分为以下几个主要部分：

### 全局配置

```yaml
global:
  log:
    level: "info"        # 日志级别: debug, info, warning, error
    format: "text"       # 日志格式: text, json
    output: "console"    # 输出目标: console, file
    file_path: "./logs/ai_dataflux.log"  # 日志文件路径
  flux_api_url: "http://127.0.0.1:8787/v1/chat/completions"  # Flux API端点URL
```

### 网关配置

```yaml
gateway:
  max_connections: 1000           # 网关到上游的总并发连接上限
  max_connections_per_host: 1000  # 单个上游主机的并发连接上限
```

仅影响 API 网关自身的上游连接池，不影响数据处理引擎的 `datasource.concurrency` 并发配置。

### 数据源配置

```yaml
datasource:
  type: excel    # 数据源类型: mysql, postgresql, sqlite, excel, csv

  # === 高性能引擎配置 ===
  # 引擎类型: auto (自动选择) | pandas (默认) | polars (高性能)
  # auto: 优先使用 polars (如已安装)，否则回退到 pandas
  engine: auto

  # Excel 读取器: auto | openpyxl (默认) | calamine (高性能)
  # calamine: 基于 Rust 的读取器，速度提升 10-50x，需安装 fastexcel
  excel_reader: auto

  # Excel 写入器: auto | openpyxl (默认) | xlsxwriter (高性能)
  # xlsxwriter: 写入速度提升 2-5x
  excel_writer: auto

  require_all_input_fields: true  # 输入字段检查: true=全部非空才处理, false=至少一个非空即可
  concurrency:   # 并发配置
    batch_size: 100         # 批处理大小（也用作最大并发任务数）
    save_interval: 300      # Excel保存间隔（秒）
    shard_size: 10000       # 默认分片大小
    min_shard_size: 1000    # 最小分片大小
    max_shard_size: 50000   # 最大分片大小
    api_pause_duration: 2.0         # API错误时全局暂停秒数
    api_error_trigger_window: 2.0   # 多少秒内的API错误才会触发暂停
    max_connections: 1000           # aiohttp的最大并发连接数
    max_connections_per_host: 0     # 对每个主机的最大并发连接数（0表示无限制）
    max_workers: 5                  # 数据库连接池大小
    retry_limits:                   # 按错误类型配置重试次数
      api_error: 3                  # API错误最多重试3次
      content_error: 1              # 内容错误最多重试1次
      system_error: 2               # 系统错误最多重试2次
```

### 数据源特定配置

```yaml
# MySQL数据源配置
mysql:
  host: "localhost"
  port: 3306
  user: "root"
  password: "your_password"
  database: "ai_tasks"
  table_name: "tasks"
  pool_size: 10

# PostgreSQL数据源配置
postgresql:
  host: "localhost"
  port: 5432
  user: "postgres"
  password: "your_password"
  database: "ai_tasks"
  table_name: "tasks"
  schema_name: "public"
  pool_size: 10

# SQLite数据源配置
sqlite:
  db_path: "./data/tasks.db"
  table_name: "tasks"

# Excel数据源配置
excel:
  input_path: "./data/input.xlsx"
  output_path: "./data/output.xlsx"

# CSV数据源配置
csv:
  input_path: "./data/input.csv"
  output_path: "./data/output.csv"
```

### 字段配置

```yaml
# 从数据源提取的字段列表
columns_to_extract:
  - "question"
  - "context"

# 结果写回映射配置（别名 -> 实际字段名）
columns_to_write:
  answer: "ai_answer"
  category: "ai_category"
  confidence: "ai_confidence"
  sentiment: "ai_sentiment"
```

### AI模型和提示词配置

```yaml
# 提示词配置
prompt:
  required_fields:   # AI必须返回的字段列表
    - "answer"
    - "category"
    - "confidence"
    - "sentiment"
  use_json_schema: true  # 是否启用JSON Schema输出约束
  model: "auto"          # 使用的AI模型，auto表示自动选择
  temperature: 0.3       # 模型温度参数（0-1之间）
  temperature_override: true  # 是否覆盖模型默认温度
  system_prompt: |       # 系统提示词（可选）
    你是一个专业的数据分析师...
  template: |            # 提示词模板，{record_json}为数据占位符
    请分析以下数据并提供专业的回答:

    {record_json}

    系统要求:
    1. 请详细分析问题和上下文，提供准确、有深度的回答
    2. 回答应专业、清晰、有条理，避免冗余内容
    3. 必须返回JSON格式的结果，包含以下字段:
       - answer: 详细的回答内容
       - category: 问题类别，可选值为 "technical"、"business"、"general"
       - confidence: 置信度评分，范围0-100的整数
       - sentiment: 情感倾向，可选值为 "positive"、"neutral"、"negative"
```

### 验证配置

```yaml
# 字段值验证配置
validation:
  enabled: true       # 是否启用验证
  field_rules:        # 字段验证规则
    category:         # 字段名
      - "technical"   # 允许的值列表
      - "business"
      - "general"
    sentiment:
      - "positive"
      - "neutral"
      - "negative"
```

### 模型和通道配置

```yaml
# 模型配置
models:
  - id: 1
    name: "model-1"              # 模型显示名称
    model: "gpt-4-turbo"         # 实际模型名称
    channel_id: "1"              # 所属通道ID
    api_key: "your_api_key_1"    # API密钥
    timeout: 300                 # 超时时间（秒）
    weight: 10                   # 调度权重（使用加权随机算法）
    temperature: 0.3             # 模型默认温度（请求未提供时生效）
    safe_rps: 5                  # 每秒安全请求数（令牌桶限流）
    supports_json_schema: true   # 是否支持JSON Schema
    supports_advanced_params: false  # 是否支持高级参数（presence_penalty等）

# 通道配置
channels:
  "1":
    name: "openai-api"
    base_url: "https://api.openai.com"
    api_path: "/v1/chat/completions"
    timeout: 300
    proxy: ""  # 可选代理设置，例如 "http://127.0.0.1:7890"
    ssl_verify: true  # SSL证书验证开关，Mac上遇到证书问题可设为false
```

## 项目结构

```text
AI-DataFlux/
├── main.py              # 数据处理入口
├── gateway.py           # API 网关入口
├── cli.py               # 统一命令行入口
├── config.yaml          # 配置文件
├── config-example.yaml  # 配置示例
├── requirements.txt     # 依赖列表
├── src/
│   ├── __init__.py      # 版本信息
│   ├── config/          # 配置管理
│   │   └── settings.py  # 配置加载与日志初始化
│   ├── models/          # 数据模型
│   │   ├── errors.py    # 错误类型定义
│   │   └── task.py      # 任务元数据
│   ├── core/            # 核心处理逻辑 (组件化架构)
│   │   ├── clients/     # API 客户端
│   │   ├── content/     # 内容处理与解析
│   │   ├── retry/       # 重试决策策略
│   │   ├── state/       # 任务状态管理
│   │   ├── processor.py # 核心协调器 (UniversalAIProcessor)
│   │   ├── scheduler.py # 分片任务调度器
│   │   ├── validator.py # JSON 字段验证器
│   │   └── token_estimator.py # Token 估算器
│   ├── data/            # 数据源层
│   │   ├── base.py      # 任务池抽象基类
│   │   ├── excel.py     # Excel/CSV 任务池
│   │   ├── mysql.py     # MySQL 任务池
│   │   ├── postgresql.py # PostgreSQL 任务池
│   │   ├── sqlite.py    # SQLite 任务池
│   │   ├── factory.py   # 任务池工厂
│   │   └── engines/     # 可插拔数据引擎
│   │       ├── __init__.py     # 引擎工厂和库检测
│   │       ├── base.py         # 引擎抽象基类
│   │       ├── pandas_engine.py  # Pandas 实现
│   │       └── polars_engine.py  # Polars 高性能实现
│   ├── gateway/         # API 网关
│   │   ├── app.py       # FastAPI 应用
│   │   ├── service.py   # 核心服务逻辑
│   │   ├── dispatcher.py # 模型调度器
│   │   ├── limiter.py   # 限流组件
│   │   ├── session.py   # HTTP 连接池管理
│   │   ├── resolver.py  # 自定义 DNS 解析器
│   │   └── schemas.py   # Pydantic 模型
│   └── control/         # Web GUI 控制面板
│       ├── server.py    # FastAPI 控制服务器
│       ├── config_api.py # 配置文件 API
│       └── process_manager.py # 进程生命周期管理
├── web/                 # 前端代码 (React + TypeScript)
│   ├── src/             # 源代码
│   └── dist/            # 构建产物（需 npm run build）
├── docs/
│   ├── GUI.md           # Web GUI 使用指南
│   ├── DESIGN.md        # 详细设计文档
│   ├── LOGIC_FRAMEWORK.md # 逻辑架构图
│   └── architecture_diagram.md # 系统架构图
└── README.md            # 项目文档
```

## 系统架构

AI-DataFlux 采用双组件架构设计，由数据处理引擎和API网关两部分组成：

### 1. API 网关 (gateway.py)

`gateway.py` 是一个OpenAI兼容的API网关，充当AI模型的统一访问层：

- **多模型管理**：管理多个AI模型和厂商API
- **自动故障切换**：当某个模型暂时不可用或出错时自动切换到其他可用模型
- **智能负载均衡**：使用加权随机算法（`random.choices`）根据配置的权重分配请求
- **Session连接池**：按(ssl_verify, proxy)组合复用HTTP连接，避免重复创建ClientSession
- **令牌桶限流**：为每个模型单独实现基于令牌桶的限流策略
- **指数退避**：API错误时自动退避（2s→4s→6s→指数增长，最长60s）
- **流式响应支持**：完整支持流式和非流式响应模式
- **管理API**：提供模型状态和健康监控接口

启动方式：

```bash
python gateway.py --config config.yaml
```

默认监听 `http://0.0.0.0:8787`，提供以下API端点：

- `/` - 网关根路径与版本信息
- `/v1/chat/completions` - OpenAI兼容的聊天补全接口
- `/v1/models` - 可用模型列表
- `/admin/models` - 模型详细状态和指标
- `/admin/health` - 系统健康状态

### 2. 数据处理引擎 (main.py)

`main.py` 是主要的数据处理引擎，负责从数据源读取任务、调用API网关、处理结果：

#### 主要模块

1. **配置管理** (`src/config/`)
   - 配置加载与验证
   - 日志初始化

2. **数据模型** (`src/models/`)
   - `ErrorType`: 错误分类（API/内容/系统）
   - `TaskMetadata`: 任务元数据，分离业务数据与内部状态

3. **数据引擎** (`src/data/engines/`)
   - `BaseEngine`: 引擎抽象接口
   - `PandasEngine`: Pandas 实现，支持 calamine/xlsxwriter 高速读写
   - `PolarsEngine`: Polars 高性能实现，多线程并行处理
   - 引擎工厂: 自动检测可用库，支持 auto/pandas/polars 选择

4. **任务池** (`src/data/`)
   - `BaseTaskPool`: 任务池抽象基类
   - `ExcelTaskPool`: Excel/CSV数据源实现，支持向量化过滤
   - `MySQLTaskPool`: MySQL数据源实现，支持连接池
   - `PostgreSQLTaskPool`: PostgreSQL数据源实现，支持批量更新
   - `SQLiteTaskPool`: SQLite数据源实现，支持WAL模式

5. **核心处理** (`src/core/`)
   - `UniversalAIProcessor`: 主处理引擎（协调器）
   - `ContentProcessor`: 提示词生成与响应解析
   - `RetryStrategy`: 智能重试策略
   - `TaskStateManager`: 任务状态与并发控制
   - `ShardedTaskManager`: 分片任务调度器
   - `JsonValidator`: AI 输出字段验证器

## Excel 数据源格式

对于Excel数据源，您的输入文件应至少包含在`columns_to_extract`中指定的列。程序会自动创建配置的写回字段列。

## 数据库数据源格式

对于 MySQL/PostgreSQL/SQLite 数据源，您的表应包含：

- **id**: 主键列（必须，支持整数或字符串）
- **提取字段**: 在`columns_to_extract`中指定的列
- **写回字段**: 在`columns_to_write`中映射的目标列

## 高级用法

### 高性能引擎配置

AI-DataFlux 支持多种高性能库来加速数据处理：

```yaml
datasource:
  # 引擎选择 (推荐 auto)
  engine: auto        # auto | pandas | polars

  # 读取器选择 (推荐 auto)
  excel_reader: auto  # auto | openpyxl | calamine

  # 写入器选择 (推荐 auto)
  excel_writer: auto  # auto | openpyxl | xlsxwriter
```

**自动回退机制**：

| 配置值 | 优先选择 | 回退选择 | 触发条件 |
| --- | --- | --- | --- |
| `engine: auto` | polars | pandas | polars 未安装 |
| `excel_reader: auto` | calamine | openpyxl | fastexcel 未安装 |
| `excel_writer: auto` | xlsxwriter | openpyxl | xlsxwriter 未安装 |

**说明**：

- PandasEngine 会按 `excel_reader` / `excel_writer` 选择具体读写器。
- PolarsEngine 读取优先使用 fastexcel（可用时自动使用），写入使用 `polars.DataFrame.write_excel`。`excel_reader`/`excel_writer` 目前主要影响引擎初始化与日志提示，未做强制切换。

**安装高性能库**：

```bash
# 推荐：安装所有高性能库
pip install numpy polars python-calamine fastexcel xlsxwriter

# 或单独安装
pip install polars          # Polars DataFrame 引擎
pip install fastexcel       # Calamine Excel 读取器
pip install xlsxwriter      # 高性能 Excel 写入器
```

### 连续任务流处理

最新版本采用连续任务流处理模式，比传统批处理更高效：

- 动态填充任务池，保持最大并发度
- 实时处理完成的任务，无需等待整批完成
- 更灵活的错误处理和任务重试机制

### API错误自适应暂停

针对API限流等常见问题，系统采用全局暂停机制：

```yaml
datasource:
  concurrency:
    api_pause_duration: 2.0         # 暂停秒数
    api_error_trigger_window: 2.0   # 触发窗口
```

当遇到API错误时：

1. 系统暂停所有新请求（api_pause_duration秒）
2. 暂停后系统进入"错误触发窗口期"
3. 窗口期内的新错误不会触发额外暂停
4. 窗口期过后的错误将再次触发暂停

### JSON Schema约束输出

通过启用JSON Schema，可以更严格地约束AI输出格式：

```yaml
prompt:
  use_json_schema: true
  required_fields:
    - answer
    - category
```

系统会自动根据required_fields和validation规则构建Schema，提高输出一致性。

### 模型权重和安全RPS

通过调整模型配置中的`weight`和`safe_rps`可以精确控制模型负载：

```yaml
models:
  - id: model1
    weight: 10     # 相对调用频率权重（使用加权随机算法）
    safe_rps: 5    # 每秒安全请求数限制（令牌桶容量为safe_rps*2）
```

较大的weight表示该模型被选中的概率更高（权重10的模型被选中概率是权重5的2倍），而safe_rps限制了对单个模型的请求频率。

### 分类重试机制

系统按错误类型独立管理重试：

```yaml
datasource:
  concurrency:
    retry_limits:
      api_error: 3      # API错误（超时、HTTP错误）最多重试3次
      content_error: 1  # 内容错误（JSON解析失败）最多重试1次
      system_error: 2   # 系统错误（内部异常）最多重试2次
```

- **API错误**：触发全局暂停后重试，从数据源重新加载原始数据
- **内容错误**：直接重试，不暂停
- **系统错误**：直接重试，不暂停

### 内存管理

系统会自动监控内存使用情况：

- 在处理大量数据时自动监控内存使用
- 当内存使用率高于85%或进程内存超过40GB时触发垃圾回收
- 动态调整分片大小以适应可用内存
- 任务元数据与业务数据完全分离，重试时从数据源重新加载，避免内存泄漏
- 定期清理超过24小时的过期任务元数据

## 调试与故障排除

### 常见问题

1. **找不到依赖模块**
   - 安装所需依赖: `pip install pyyaml aiohttp pandas openpyxl psutil pydantic fastapi uvicorn`
   - 数据库支持: `pip install mysql-connector-python psycopg2-binary`

2. **无法连接到API网关**
   - 确认gateway.py正在运行: `ps aux | grep gateway.py`
   - 检查global.flux_api_url配置是否正确指向运行的服务
   - 尝试用浏览器访问 `http://127.0.0.1:8787/` 确认服务可用

3. **API错误频繁**
   - 增加`api_pause_duration`值，给API更多恢复时间
   - 减少`batch_size`降低并发请求数
   - 检查模型的`safe_rps`设置是否合理
   - 查看gateway.py的日志了解具体错误原因

4. **处理速度慢**
   - 增加`batch_size`提高并发度
   - 调整分片大小参数适应数据量
   - 使用更多权重分配给响应更快的模型
   - 增加网关服务的工作进程数: `--workers 4`（多核服务器）

5. **内存使用过高**
   - 减小`max_shard_size`和`batch_size`
   - 降低`max_connections`值减少连接池内存占用
   - 分离运行gateway.py和main.py至不同服务器

6. **Mac上SSL证书验证失败**
   - 错误信息: `[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed`
   - 解决方案: 在通道配置中设置 `ssl_verify: false`
   - 示例:

     ```yaml
     channels:
       "1":
         name: "api-provider"
         base_url: "https://api.example.com"
         ssl_verify: false  # 临时禁用SSL验证
     ```

   - ⚠️ 注意: 这会降低安全性，仅建议在测试环境或Mac证书问题时使用
   - 永久解决方案:

     ```bash
     # 更新Python证书
     pip install --upgrade certifi
     # 或运行macOS自带的证书安装脚本
     /Applications/Python\ 3.*/Install\ Certificates.command
     ```

### 开启调试日志

修改配置文件中的日志级别以获取更详细的信息：

```yaml
global:
  log:
    level: "debug"
```

### 性能监控

处理过程中会输出实时性能指标：

- 每批次处理速率（记录/秒）
- 当前内存使用情况（MB）
- 分片加载和处理状态
- API错误统计和暂停触发情况
- 按错误类型的重试统计
- 重试超限任务计数

---

## AI-DataFlux v2.3

高效、智能的批量AI处理引擎。
