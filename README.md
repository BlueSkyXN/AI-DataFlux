# AI-DataFlux

AI-DataFlux 是面向批量 AI 数据处理的 Python 3.10+ 引擎，包含 OpenAI-compatible Gateway、七类 datasource adapter、durable Job/Worker 和 React/Vite 本地 Control Plane。

> 当前活动开发版本为 **4.0.0-dev**。4.0 配置是 clean break：只接受 `schema_version: 4` 的 canonical `RootConfig`，不读取、不迁移、不兼容 3.x YAML。

当前工作区的实现、验证和剩余交付见 [4.0 验证记录](./docs/V4_LOCAL_VALIDATION.md)。Repository state 已使用独立 schema 2；旧 state schema 1 不自动迁移，也不能只改版本号后继续使用。

## 能力

- Excel、CSV、SQLite、MySQL、PostgreSQL、Feishu Bitable、Feishu Sheet 数据源
- Chat Completions 与 Responses Gateway
- 按 capability 的 route 调度、限流和连接池
- 批处理、分片、分类重试、结果校验和 durable Job checkpoint
- Control API、配置编辑器、进程管理、Job 状态与日志
- Pandas/Polars、openpyxl/calamine/xlsxwriter 可选运行路径

## 快速开始

```bash
python3 -m pip install -r requirements.txt
cp config-example.yaml config.yaml

# 严格校验 canonical v4 Job 配置
python3 cli.py process --config config.yaml --validate

# 启动组件
python3 cli.py gateway --config config.yaml
python3 cli.py gui --config config.yaml --no-browser
python3 cli.py process --config config.yaml

# 估算 token
python3 cli.py token --config config.yaml
```

`DATAFLUX_TOKEN` 优先于 `runtime.auth.token`。默认 Gateway 监听 `127.0.0.1:8787`，Control 监听 `127.0.0.1:8790`。绑定非 loopback 地址时必须显式提供 token。

## Canonical v4 配置

根结构固定为：

```yaml
schema_version: 4

runtime:
  log: {}
  auth: {}
  workspace: {}
  scheduler: {}
  token_estimation: {}

job:
  gateway_url: http://127.0.0.1:8787
  datasource:
    type: csv
    input_path: ./data/input.csv
    output_path: ./data/output.csv
  columns:
    extract: [question, context]
    write:
      answer: ai_answer
  model_selection:
    mode: auto
  prompt:
    template: "分析并返回 JSON：{record_json}"
  validation: {}
  routing: {}
  concurrency: {}
  retry: {}
  writeback: {}

gateway:
  listen: {}
  connection_pool: {}
  channels: {}
  routes: []
  fallback_groups: {}
  retry: {}
  affinity: {}

control:
  listen: {}
```

`runtime` 必填；`job`、`gateway`、`control` 至少存在一个。支持 job-only、gateway-only、control-only 和组合配置。命令还会执行组件检查：

| 命令 | 必需 section |
|---|---|
| `process`、`token`、本地 `job submit` | `job` |
| `gateway` | `gateway` |
| `gui` | `control` |

完整字段、默认值和 datasource discriminator 见 [docs/CONFIG.md](./docs/CONFIG.md)；可执行示例见 [config-example.yaml](./config-example.yaml)。旧顶层键如 `global`、`datasource`、`columns_to_extract`、`models`、`channels`、`server` 会作为 unknown key 拒绝。

## 配置与持久化边界

- 执行配置 hash：对包含默认值和已解析 routing profile 的 canonical JSON 计算 SHA-256，用于 Job 恢复一致性。
- Config API ETag：对原始 YAML bytes 计算 SHA-256，用于配置文件乐观并发；它与执行配置 hash 不是同一个值。
- Routing profile：是独立局部 YAML，只允许覆盖 `prompt` 和 `validation`，不带 `schema_version`。
- 配置错误使用 Pydantic path/code 输出，错误信息不回显完整 token、password、API key 或 app secret。

## 开发验证

```bash
python3 -m pytest tests/ -v -m "not integration"
python3 -m pytest tests/ -v -m integration
ruff check src/ tests/ cli.py main.py gateway.py
black --check src/ tests/ cli.py main.py gateway.py
mypy src/ --ignore-missing-imports

cd web
npm run lint
npm test -- --run
npm run build
```

数据库和 Feishu 的真实验收需要独立测试环境；mock/unit test 通过不能替代真实 MySQL/PostgreSQL contract evidence 或 Feishu UAT。

## 文档

- [配置合同](./docs/CONFIG.md)
- [系统架构](./docs/ARCH.md)
- [数据源](./docs/DATA_SOURCE.md)
- [Gateway API](./docs/GATEWAY_API.md)
- [Control API](./docs/CONTROL_API.md)
- [GUI](./docs/GUI.md)
- [Job/Worker](./docs/JOBS.md)
- [Routing](./docs/ROUTING.md)
- [构建变体](./docs/BUILD_VARIANTS.md)
- [发布门禁](./docs/RELEASE_GATES.md)
- [4.0 执行合同](./docs/V4_EXECUTION_CONTRACT.md)
- [Bootstrap Acceptance Ledger](./docs/BOOTSTRAP_ACCEPTANCE.md)

## 版本与交付

## AI-DataFlux 4.0.0-dev

这是本地研发版本，不代表已创建 GitHub Release、已部署或已完成业务 UAT。版本事实源为 `src.__version__`，Web `package.json` / `package-lock.json` 与用户入口必须保持一致。

## License

[GNU Affero General Public License v3.0](./LICENSE)
