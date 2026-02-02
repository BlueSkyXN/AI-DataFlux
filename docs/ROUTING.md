# 规则路由配置指南

## 概述

规则路由（Rule Routing）是 AI-DataFlux 的高级配置功能，允许在处理单个数据文件时，根据记录中某个字段的值动态选择不同的 `prompt` 和 `validation` 配置。

### 核心特性

- **按记录路由**：每条记录独立匹配规则，使用对应的配置处理
- **差异配置**：子配置文件只需定义与主配置不同的字段
- **深度合并**：子配置与主配置深度合并，未定义字段保留主配置值
- **优雅降级**：路由字段不存在或无匹配规则时，自动使用默认配置
- **向后兼容**：不启用路由时，配置文件格式与旧版本完全兼容

### 适用场景

- 单个数据文件包含多种数据类型（如不同部门、产品线、客户类型）
- 不同类型需要不同的分类标签体系
- 希望一次处理完成，无需手动拆分文件

---

## 配置结构

### 目录布局

```
project/
├── config.yaml                    # 主配置文件
└── .config/
    └── rules/                     # 规则配置目录
        ├── type_a.yaml            # 类型 A 的差异配置
        ├── type_b.yaml            # 类型 B 的差异配置
        └── type_c.yaml            # 类型 C 的差异配置
```

### 主配置文件

在主配置文件中添加 `routing` 节点启用规则路由：

```yaml
# config.yaml

# ... 其他配置（global, datasource, models, channels 等）...

# 规则路由配置
routing:
  enabled: true                    # 启用路由
  field: "category"                # 用于路由的字段名（任意字段皆可）
  subtasks:                        # 路由规则列表
    - match: "type_a"              # 当 category="type_a" 时
      profile: ".config/rules/type_a.yaml"
    - match: "type_b"              # 当 category="type_b" 时
      profile: ".config/rules/type_b.yaml"
    - match: "type_c"
      profile: ".config/rules/type_c.yaml"

# 默认配置（路由未匹配时使用）
validation:
  enabled: true
  field_rules:
    result: ["0", "1", "2", "3"]

prompt:
  required_fields: ["result"]
  template: |
    默认的提示词模板...
    {record_json}
```

### 子配置文件（规则配置）

子配置文件只需包含与主配置不同的字段：

```yaml
# .config/rules/type_a.yaml
# 类型 A 的差异配置

# 仅允许的顶层键: prompt, validation

validation:
  enabled: true
  field_rules:
    result:
      - "0"
      - "1"
      - "2"
      # ... type_a 专属的标签列表

prompt:
  # 可以只覆盖部分字段
  template: |
    type_a 专用的提示词模板...
    
    # 分类标签列表
    | ID | 类名 |
    |----|------|
    | 1 | type_a 分类 1 |
    | 2 | type_a 分类 2 |
    ...
    
    # 数据内容
    {record_json}
```

---

## 配置详解

### routing 配置项

| 配置项 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `enabled` | bool | 是 | - | 是否启用路由功能 |
| `field` | string | 是 | - | 用于路由判断的字段名（可自定义，如 `category`、`type`、`department` 等）<br>**注意**：无需手动添加到 `columns_to_extract`，系统会自动追加 |
| `subtasks` | list | 是 | - | 路由规则列表 |

### subtask 配置项

| 配置项 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `match` | string | 是 | 匹配值（精确字符串匹配） |
| `profile` | string | 是 | 子配置文件路径（相对于主配置文件目录） |

### 子配置文件限制

子配置文件**只允许**包含以下顶层键：

- `prompt` - 提示词相关配置
- `validation` - 结果验证配置

其他配置项（如 `datasource`、`models`、`channels` 等）**不支持**通过路由覆盖。

### 路由字段处理

**显式 vs 隐式路由字段**：

系统根据用户配置自动判断路由字段的用途：

#### 1. **隐式路由字段**（未在 `columns_to_extract` 中声明）

**行为**：
- ✅ 自动追加到 `columns_to_extract`（数据源加载时包含）
- ✅ 自动排除出 Prompt JSON（不发送给 AI）
- ✅ 仅用于内部路由决策

**适用场景**：路由字段纯粹是元数据，不属于业务字段（如部门代码、数据类型标签）

**示例**：
```yaml
# 配置
columns_to_extract:
  - "title"
  - "content"
  # 没有声明 "BGBU"

routing:
  enabled: true
  field: "BGBU"  # 隐式路由字段
  subtasks:
    - match: "type_a"
      profile: ".config/rules/type_a.yaml"

# 实际行为
# 1. 数据源加载: ["title", "content", "BGBU"]  ← 自动追加
# 2. Prompt 生成: {"title": "...", "content": "..."}  ← BGBU 被排除
# 3. 路由决策: BGBU = "type_a" → 使用 type_a.yaml 配置
```

#### 2. **显式路由字段**（在 `columns_to_extract` 中显式声明）

**行为**：
- ✅ 作为正常业务字段处理
- ✅ 包含在 Prompt JSON 中（发送给 AI）
- ✅ 同时用于路由决策

**适用场景**：路由字段本身也是业务字段，需要参与 AI 分析（如产品类别、文章主题）

**示例**：
```yaml
# 配置
columns_to_extract:
  - "title"
  - "content"
  - "category"  # 显式声明为业务字段

routing:
  enabled: true
  field: "category"  # 显式路由字段
  subtasks:
    - match: "tech"
      profile: ".config/rules/tech.yaml"

# 实际行为
# 1. 数据源加载: ["title", "content", "category"]
# 2. Prompt 生成: {"title": "...", "content": "...", "category": "tech"}  ← category 包含
# 3. 路由决策: category = "tech" → 使用 tech.yaml 配置
```

#### 对比总结

| 特性 | 隐式路由字段 | 显式路由字段 |
|------|-------------|-------------|
| **配置** | 不在 `columns_to_extract` 中 | 在 `columns_to_extract` 中显式声明 |
| **数据源加载** | ✅ 自动追加 | ✅ 正常提取 |
| **Prompt JSON** | ❌ 排除（不发给 AI） | ✅ 包含（发给 AI） |
| **路由决策** | ✅ 用于路由 | ✅ 用于路由 |
| **适用场景** | 纯元数据字段 | 业务字段 + 路由 |
| **Token 使用** | 节省（不发送） | 正常（发送） |

---

## 处理流程

```
┌─────────────────────────────────────────────────────────────┐
│                     处理器初始化                            │
├─────────────────────────────────────────────────────────────┤
│  1. 加载主配置文件                                          │
│  2. 检查 routing.enabled                                   │
│  3. 如果启用：                                              │
│     - 加载所有子配置文件                                    │
│     - 验证子配置键的合法性                                  │
│     - 为每个规则创建独立的 ContentProcessor 和 Validator   │
│  4. 创建默认的 ContentProcessor 和 Validator               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     处理每条记录                            │
├─────────────────────────────────────────────────────────────┤
│  对于每条记录:                                              │
│  1. 获取路由字段值: record[routing.field]                  │
│  2. 匹配规则:                                               │
│     - 字段不存在 → 使用默认配置                            │
│     - 无匹配规则 → 使用默认配置                            │
│     - 匹配成功 → 使用对应的 ContentProcessor 和 Validator │
│  3. 生成 Prompt → 调用 API → 解析响应 → 验证结果           │
│  4. 写回结果                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 使用示例

### 示例 1：隐式路由字段（元数据路由）

**场景**：数据文件包含多种部门的数据，部门代码 `BGBU` 仅用于路由，不需要 AI 分析。

**目录结构**：
```
project/
├── config.yaml
└── .config/
    └── rules/
        ├── type_a.yaml
        ├── type_b.yaml
        └── type_c.yaml
```

**config.yaml**：
```yaml
# 数据源配置
excel:
  input_path: "data/input.xlsx"
  output_path: "data/output.xlsx"

# 提取字段（BGBU 未声明，系统自动追加但不发给 AI）
columns_to_extract:
  - "title"
  - "description"
  - "content"
  # 无需添加 "BGBU"

# 写回字段
columns_to_write:
  result: "result"

# 规则路由配置
routing:
  enabled: true
  field: "BGBU"  # 隐式路由字段（自动追加 + 排除出 Prompt）
  subtasks:
    - match: "type_a"
      profile: ".config/rules/type_a.yaml"
    - match: "type_b"
      profile: ".config/rules/type_b.yaml"
    - match: "type_c"
      profile: ".config/rules/type_c.yaml"

# 默认配置（未匹配时使用）
validation:
  enabled: true
  field_rules:
    result: ["0", "1", "2", "3", "4", "5"]

prompt:
  required_fields: ["result"]
  use_json_schema: true
  template: |
    # 任务
    根据内容，选择最匹配的分类标签ID。

    # 默认分类列表
    | ID | 类名 |
    |----|------|
    | 1 | 通用分类1 |
    | 2 | 通用分类2 |
    | 0 | 以上都不是 |

    # 数据内容（BGBU 不会出现在这里）
    {record_json}
```

**实际行为**：
```
数据源加载: ["title", "description", "content", "BGBU"]  ← 自动追加
Prompt 生成: {"title": "...", "description": "...", "content": "..."}  ← BGBU 排除
路由决策: BGBU = "type_a" → 使用 type_a.yaml 配置
```

---

### 示例 2：显式路由字段（业务字段路由）

**场景**：数据文件包含多个类别的文章，`category` 字段既用于路由，又需要 AI 参考。

**目录结构**：
```
project/
├── config.yaml
└── .config/
    └── rules/
        ├── tech.yaml
        ├── business.yaml
        └── lifestyle.yaml
```

**config.yaml**：
```yaml
# 数据源配置
excel:
  input_path: "data/input.xlsx"
  output_path: "data/output.xlsx"

# 提取字段（category 显式声明，作为业务字段）
columns_to_extract:
  - "title"
  - "content"
  - "category"  # 显式声明，既用于路由又发给 AI

# 写回字段
columns_to_write:
  result: "result"

# 规则路由配置
routing:
  enabled: true
  field: "category"  # 显式路由字段（包含在 Prompt 中）
  subtasks:
    - match: "tech"
      profile: ".config/rules/tech.yaml"
    - match: "business"
      profile: ".config/rules/business.yaml"
    - match: "lifestyle"
      profile: ".config/rules/lifestyle.yaml"

# 默认配置（未匹配时使用）
validation:
  enabled: true
  field_rules:
    result: ["0", "1", "2", "3"]

prompt:
  required_fields: ["result"]
  use_json_schema: true
  template: |
    # 任务
    根据文章类别和内容，选择最合适的分类标签ID。

    # 默认分类列表
    | ID | 类名 |
    |----|------|
    | 1 | 通用分类1 |
    | 2 | 通用分类2 |
    | 0 | 以上都不是 |

    # 数据内容（category 会包含在这里）
    {record_json}
```

**实际行为**：
```
数据源加载: ["title", "content", "category"]
Prompt 生成: {"title": "...", "content": "...", "category": "tech"}  ← category 包含
路由决策: category = "tech" → 使用 tech.yaml 配置
```

**执行命令**：
```bash
python cli.py process --config config.yaml
```

**验证配置**：
```bash
python cli.py process --config config.yaml --validate
```

输出示例：
```
[OK] Config valid: config.yaml
  - Datasource: excel
  - Engine: auto
  - Input columns: ['category', 'title', 'description', 'content']
  - Output columns: ['result']
  - Routing: enabled on 'category' (3 rules)
```

---

## 配置合并规则

子配置与主配置采用**深度合并**策略：

```yaml
# 主配置
prompt:
  required_fields: ["result"]
  use_json_schema: true
  temperature: 0.1
  template: "默认模板"

# 子配置（只覆盖 template）
prompt:
  template: "type_a 专用模板"

# 合并结果
prompt:
  required_fields: ["result"]      # 保留
  use_json_schema: true            # 保留
  temperature: 0.1                 # 保留
  template: "type_a 专用模板"      # 被覆盖
```

---

## 错误处理

### 路由字段不存在

当记录中不存在 `routing.field` 指定的字段时，系统会：
1. 记录 DEBUG 级别日志
2. 使用默认配置继续处理

### 无匹配规则

当路由字段值没有匹配任何规则时，系统会：
1. 使用默认配置处理该记录
2. 不产生错误或警告

### 子配置文件错误

- **文件不存在**：启动时抛出 ConfigError，终止处理
- **非法顶层键**：启动时抛出 ConfigError，提示只允许 `prompt` 和 `validation`

---

## 性能考虑

- **预加载**：所有子配置在处理器初始化时一次性加载
- **缓存**：每个规则的 ContentProcessor 和 Validator 仅创建一次
- **无额外 I/O**：处理记录时不产生配置文件读取
- **内存开销**：与规则数量成正比，通常可忽略

---

## 向后兼容

不使用规则路由时，只需：
1. 不添加 `routing` 配置节点，或
2. 设置 `routing.enabled: false`

配置文件格式与旧版本完全兼容。
