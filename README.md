# AI-DataFlux

AI-DataFlux 是一个高性能、可扩展的通用AI处理引擎，专为批量AI任务处理设计。支持从多种数据源读取任务，使用多种AI模型进行并行处理，并将结果写回数据源。

## 核心特性

- **多数据源支持**：同时支持MySQL和Excel作为数据源和结果存储
- **智能模型调度**：基于加权负载均衡的多模型调度系统
- **高并发处理**：多线程架构实现高吞吐量任务处理
- **强大的错误处理**：区分API错误和内容错误，针对性实现智能退避
- **灵活的配置系统**：通过YAML文件实现全方位配置
- **字段值验证**：支持对返回结果的字段值进行枚举验证
- **读写锁优化**：使用读写锁分离，提高多线程下的并发性能
- **可视化进度**：实时显示处理进度和统计信息

## 快速开始

### 安装依赖

```bash
# 基础依赖
pip install pyyaml aiohttp pandas openpyxl

# 如果使用MySQL数据源，需要安装
pip install mysql-connector-python
```

### 配置文件

根据提供的`config-example.yaml`创建您的配置文件：

```bash
cp config-example.yaml config.yaml
```

然后编辑`config.yaml`，设置您的API密钥、数据源信息等配置。

### 运行程序

```bash
python AI-DataFlux.py config.yaml
```

## 详细配置说明

配置文件分为以下几个主要部分：

### 全局配置

```yaml
global:
  log:
    level: info        # 日志级别: debug, info, warning, error
    output: console    # 输出目标: console, file
    format: text       # 日志格式: text, json
    file_path: ./logs/universal_ai.log  # 日志文件路径
```

### 数据源配置

```yaml
datasource:
  type: excel    # 数据源类型: mysql, excel
  concurrency:   # 并发配置
    max_workers: 5          # 最大工作线程数
    batch_size: 300         # 批处理大小
    save_interval: 300      # 保存间隔
    retry_times: 3          # 重试次数
    backoff_factor: 2       # 退避因子
```

### 数据源特定配置

```yaml
# MySQL数据源配置
mysql:
  host: localhost
  port: 3306
  user: root
  password: password
  database: ai_tasks
  table_name: tasks

# Excel数据源配置
excel:
  input_path: ./data/input.xlsx
  output_path: ./data/output.xlsx
```

### 字段配置

```yaml
# 从数据源提取的字段列表
columns_to_extract:
  - question
  - context
  - query_type

# 结果写回映射配置（别名 -> 实际字段名）
columns_to_write:
  answer: ai_answer
  category: ai_category
  confidence: ai_confidence
```

### AI模型配置

```yaml
# 模型配置
models:
  - id: gpt4                     # 模型唯一标识
    name: GPT-4                  # 模型显示名称
    model: gpt-4-0125-preview    # 实际模型名称
    channel_id: openai           # 所属通道ID
    api_key: sk-xxxxx            # API密钥
    weight: 3                    # 调度权重
    temperature: 0.7             # 模型温度

  - id: claude3
    name: Claude-3
    model: claude-3-sonnet-20240229
    channel_id: anthropic
    api_key: sk-ant-xxxxx
    weight: 2
    temperature: 0.7
```

### 通道配置

```yaml
# 通道配置
channels:
  openai:
    name: OpenAI官方
    base_url: https://api.openai.com
    api_path: /v1/chat/completions
    timeout: 300
  
  anthropic:
    name: Anthropic官方
    base_url: https://api.anthropic.com
    api_path: /v1/messages
    timeout: 300
    proxy: http://127.0.0.1:7890  # 可选代理
```

### 提示词和验证配置

```yaml
# 提示词配置
prompt:
  template: |
    请根据以下数据进行分析:
    {record_json}
    
    要求:
    1. 返回JSON格式的回答，包含以下字段: answer, category, confidence
    2. answer字段为详细的答案内容
    3. category字段为问题类别，可选值: technical, business, general
    4. confidence字段为置信度，取值0-100
  required_fields:     # AI必须返回的字段列表
    - answer
    - category
    - confidence

# 字段值验证配置
validation:
  enabled: true        # 是否启用验证
  field_rules:         # 字段验证规则
    category:          # 字段名
      - technical      # 允许的值列表
      - business
      - general
```

## 项目架构

AI-DataFlux 采用模块化设计，主要组件包括：

1. **任务池管理**：`BaseTaskPool`抽象基类及其具体实现
   - `MySQLTaskPool`: MySQL数据源实现
   - `ExcelTaskPool`: Excel数据源实现

2. **模型调度系统**：`ModelDispatcher`类
   - 智能退避策略
   - 加权负载均衡
   - 读写锁优化

3. **错误处理机制**：`ErrorType`类
   - 区分API错误和内容错误
   - 只对API错误执行退避

4. **字段验证系统**：`JsonValidator`类
   - 灵活配置字段值验证规则
   - 插件式设计

5. **主处理流程**：`UniversalAIProcessor`类
   - 多线程并发处理
   - 批量结果更新
   - 实时进度报告

## Excel 数据源格式

对于Excel数据源，您的输入文件应至少包含在`columns_to_extract`中指定的列。程序会自动添加：

- **processed**: 标记处理状态的列（"yes"表示已处理）
- **写回字段**: 根据`columns_to_write`配置自动创建的结果列

## MySQL 数据源格式

对于MySQL数据源，您的表应包含：

- **id**: 主键列
- **processed**: 标记处理状态的列
- **提取字段**: 在`columns_to_extract`中指定的列
- **写回字段**: 在`columns_to_write`中映射的目标列

## 高级用法

### 自定义错误退避

通过配置`backoff_factor`可以调整退避策略：

```yaml
datasource:
  concurrency:
    backoff_factor: 2  # 退避时间为 2^(失败次数-1) 秒
```

### 模型权重调整

通过调整模型配置中的`weight`值可以控制不同模型的使用频率：

```yaml
models:
  - id: model1
    weight: 3  # 被选中概率是 weight 为 1 的模型的 3 倍
```

### 代理设置

对于需要通过代理访问的API，可以在通道配置中设置代理：

```yaml
channels:
  anthropic:
    proxy: http://127.0.0.1:7890
```

## 调试与故障排除

### 常见问题

1. **找不到MySQL模块**
   - 安装MySQL连接器: `pip install mysql-connector-python`

2. **无法连接API**
   - 检查API密钥和URL配置
   - 检查网络连接和代理设置

3. **处理结果未写入**
   - 确保`columns_to_write`配置正确
   - 检查返回的JSON字段是否包含必需字段

### 开启调试日志

修改配置文件中的日志级别以获取更详细的信息：

```yaml
global:
  log:
    level: debug
```

---

*AI-DataFlux - 高效、智能的批量AI处理引擎*
