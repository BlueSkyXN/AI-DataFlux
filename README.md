# AI-DataFlux

AI-DataFlux 是一个高性能、可扩展的通用AI处理引擎，专为批量AI任务处理设计。支持从多种数据源读取任务，使用多种AI模型进行并行处理，并将结果写回数据源。

## 核心特性

- **多数据源支持**：同时支持MySQL和Excel作为数据源和结果存储
- **智能模型调度**：基于加权负载均衡的多模型调度系统，支持自动故障切换
- **高并发处理**：优化的异步架构实现高吞吐量任务处理
- **连续任务流**：采用连续任务流模式，比传统批处理模式更高效
- **API错误自适应**：全局暂停机制替代传统重试，更优雅地处理API限流
- **灵活的配置系统**：通过YAML文件实现全方位配置
- **字段值验证**：支持对返回结果的字段值进行枚举验证
- **JSON Schema支持**：通过Schema约束AI输出格式，提高结果一致性
- **读写锁优化**：使用读写锁分离，提高多线程下的并发性能
- **内存使用监控**：自动监控内存使用，在高内存使用时触发垃圾回收
- **可视化进度**：实时显示处理进度和统计信息

## 快速开始

### 安装依赖

```bash
# 基础依赖
pip install pyyaml aiohttp pandas openpyxl psutil

# 如果使用MySQL数据源，需要安装
pip install mysql-connector-python
```

### 配置文件

根据提供的`config-example.yaml`创建您的配置文件：

```bash
cp config-example.yaml config.yaml
```

然后编辑`config.yaml`，设置您的API密钥、数据源信息等配置。

### 启动 Flux-Api 服务

首先启动API网关（确保始终在后台运行）：

```bash
# 启动API网关
python Flux-Api.py --config config.yaml --port 8787

# 或者使用nohup在后台运行
nohup python Flux-Api.py --config config.yaml --port 8787 > flux-api.log 2>&1 &
```

### 运行数据处理引擎

确保API网关已启动，然后运行数据处理引擎：

```bash
python AI-DataFlux.py --config config.yaml
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

### 数据源配置

```yaml
datasource:
  type: excel    # 数据源类型: mysql, excel
  concurrency:   # 并发配置
    max_workers: 5          # 最大工作线程数
    batch_size: 100         # 批处理大小（也用作最大并发任务数）
    save_interval: 300      # Excel保存间隔（秒）
    shard_size: 10000       # 默认分片大小
    min_shard_size: 1000    # 最小分片大小
    max_shard_size: 50000   # 最大分片大小
    api_pause_duration: 2.0         # API错误时全局暂停秒数
    api_error_trigger_window: 2.0   # 多少秒内的API错误才会触发暂停
    max_connections: 1000           # aiohttp的最大并发连接数
    max_connections_per_host: 0     # 对每个主机的最大并发连接数（0表示无限制）
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

# Excel数据源配置
excel:
  input_path: "./data/input.xlsx"
  output_path: "./data/output.xlsx"
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
    weight: 10                   # 调度权重
    temperature: 0.3             # 模型温度
    safe_rps: 5                  # 每秒安全请求数
    supports_json_schema: true   # 是否支持JSON Schema

# 通道配置
channels:
  "1":
    name: "openai-api"
    base_url: "https://api.openai.com"
    api_path: "/v1/chat/completions"
    timeout: 300
    proxy: ""  # 可选代理设置，例如 "http://127.0.0.1:7890"
```

## 系统架构

AI-DataFlux 采用双组件架构设计，由数据处理引擎和API网关两部分组成：

### 1. Flux-Api 组件

`Flux-Api.py` 是一个OpenAI兼容的API网关，充当AI模型的统一访问层：

- **多模型管理**：管理多个AI模型和厂商API
- **自动故障切换**：当某个模型暂时不可用或出错时自动切换到其他可用模型
- **智能负载均衡**：根据配置的权重分配请求至不同模型
- **连接池优化**：高效管理HTTP连接，提高并发性能
- **令牌桶限流**：为每个模型单独实现基于令牌桶的限流策略
- **流式响应支持**：完整支持流式和非流式响应模式
- **管理API**：提供模型状态和健康监控接口

启动方式：
```bash
python Flux-Api.py --config config.yaml
```

默认监听 `http://127.0.0.1:8787`，提供以下API端点：
- `/v1/chat/completions` - OpenAI兼容的聊天补全接口
- `/v1/models` - 可用模型列表
- `/admin/models` - 模型详细状态和指标
- `/admin/health` - 系统健康状态

### 2. AI-DataFlux 引擎

`AI-DataFlux.py` 是主要的数据处理引擎，负责从数据源读取任务、调用Flux-Api、处理结果：

#### 主要组件

1. **任务池管理**：`BaseTaskPool`抽象基类及其具体实现
   - `MySQLTaskPool`: MySQL数据源实现
   - `ExcelTaskPool`: Excel数据源实现

2. **分片任务管理**：`ShardedTaskManager`类
   - 动态分片大小计算
   - 分片加载和进度跟踪
   - 内存使用监控

3. **错误处理机制**：`ErrorType`类
   - 区分API错误、内容错误和系统错误
   - 全局暂停替代传统重试

4. **字段验证系统**：`JsonValidator`类
   - 灵活配置字段值验证规则
   - 支持JSON Schema格式约束

5. **主处理流程**：`UniversalAIProcessor`类
   - 连续任务流处理
   - 智能错误处理和重试
   - 实时进度报告

## Excel 数据源格式

对于Excel数据源，您的输入文件应至少包含在`columns_to_extract`中指定的列。程序会自动创建配置的写回字段列。

## MySQL 数据源格式

对于MySQL数据源，您的表应包含：

- **id**: 主键列（必须）
- **提取字段**: 在`columns_to_extract`中指定的列
- **写回字段**: 在`columns_to_write`中映射的目标列

## 高级用法

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
    weight: 10     # 相对调用频率权重
    safe_rps: 5    # 每秒安全请求数限制
```

较大的weight表示该模型被选中的概率更高，而safe_rps限制了对单个模型的请求频率。

### 内存管理

系统会自动监控内存使用情况：

- 在处理大量数据时自动监控内存使用
- 当内存使用率高于85%或进程内存超过1.5GB时触发垃圾回收
- 动态调整分片大小以适应可用内存

## 调试与故障排除

### 常见问题

1. **找不到依赖模块**
   - 安装所需依赖: `pip install pyyaml aiohttp pandas openpyxl psutil`
   - MySQL支持: `pip install mysql-connector-python`

2. **无法连接到Flux-Api**
   - 确认Flux-Api.py正在运行: `ps aux | grep Flux-Api.py`
   - 检查global.flux_api_url配置是否正确指向运行的服务
   - 尝试用浏览器访问 http://127.0.0.1:8787/ 确认服务可用

3. **API错误频繁**
   - 增加`api_pause_duration`值，给API更多恢复时间
   - 减少`batch_size`降低并发请求数
   - 检查模型的`safe_rps`设置是否合理
   - 查看Flux-Api.py的日志了解具体错误原因

4. **处理速度慢**
   - 增加`batch_size`提高并发度
   - 调整分片大小参数适应数据量
   - 使用更多权重分配给响应更快的模型
   - 增加Flux-Api服务的工作进程数: `--workers 4`（多核服务器）

5. **内存使用过高**
   - 减小`max_shard_size`和`batch_size`
   - 降低`max_connections`值减少连接池内存占用
   - 分离运行Flux-Api和AI-DataFlux至不同服务器

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

---

*AI-DataFlux - 高效、智能的批量AI处理引擎*