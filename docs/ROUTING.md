# Canonical v4 规则路由

规则路由按记录字段选择局部 `prompt` / `validation` profile。Profile 不是 `RootConfig`，不带 `schema_version`，也不能覆盖 datasource、columns、model selection、retry、writeback、Gateway 或 Control。

## 主配置

```yaml
schema_version: 4
runtime: {}
job:
  datasource:
    type: csv
    input_path: ./input.csv
    output_path: ./output.csv
  columns:
    extract: [content]
    write: {result: result}
  prompt:
    template: "默认处理：{record_json}"
  validation:
    enabled: true
    field_rules:
      result: [default]
  routing:
    enabled: true
    field: category
    subtasks:
      - match: type_a
        profile: .config/rules/type_a.yaml
      - match: type_b
        profile: .config/rules/type_b.yaml
```

若 `category` 未显式列入 `job.columns.extract`，processor 会把它加入 datasource 读取字段，但不会把这个隐式字段发送给模型。显式列入时，它作为普通业务输入参与 prompt。

## Profile

```yaml
# .config/rules/type_a.yaml
prompt:
  temperature: 0.1
  required_fields: [result]
  template: "type_a：{record_json}"
validation:
  enabled: true
  field_rules:
    result: [a1, a2]
```

Profile 只覆盖声明字段，其他字段继承主 JobConfig。允许的 prompt 字段为 `required_fields`、`use_json_schema`、`temperature`、`temperature_override`、`system_prompt`、`template`；validation 允许 `enabled` 和 `field_rules`。

## 校验与执行

- `routing.enabled=true` 时 `field` 和至少一条 `subtasks` 必填。
- 每条 rule 必须包含 `match` 和非空 `profile`。
- 相对 profile 路径按主配置目录解析。
- RootConfig 加载时解析全部 profile；文件缺失、YAML 非 mapping、空 profile、未知字段都会在启动任何组件前失败。
- Execution config hash 包含解析后的 profile 内容，因此 profile 语义变化会阻断旧 Job 自动恢复。
- 字段缺失或没有匹配 rule 时使用主 `prompt` / `validation`。

```bash
python3 cli.py process --config config.yaml --validate
```
