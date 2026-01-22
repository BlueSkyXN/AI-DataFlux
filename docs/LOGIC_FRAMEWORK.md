# 逻辑框架图

```mermaid
flowchart TB
  User[用户/客户端]

  subgraph Entry["入口层"]
    CLI[cli.py]
    Main[main.py]
    Gateway[gateway.py]
  end

  subgraph Config["配置层"]
    ConfigFile[config.yaml]
    Settings[config/settings.py]
  end

  subgraph Models["模型层"]
    Errors[models/errors.py]
    TaskMeta[models/task.py]
  end

  subgraph Core["核心处理层"]
    Processor[core/processor.py]
    Scheduler[core/scheduler.py]
    Validator[core/validator.py]
  end

  subgraph Data["数据源层"]
    Factory[data/factory.py]
    BasePool[data/base.py]
    Excel[data/excel.py]
    MySQL[data/mysql.py]
    Engines[data/engines/*]
  end

  subgraph GatewayLayer["API 网关层"]
    App[gateway/app.py]
    Service[gateway/service.py]
    Dispatcher[gateway/dispatcher.py]
    Limiter[gateway/limiter.py]
    Session[gateway/session.py]
    Resolver[gateway/resolver.py]
    Schemas[gateway/schemas.py]
  end

  subgraph External["外部 AI API"]
    OpenAI[OpenAI 兼容 API]
    Claude[Claude API]
    Other[其他兼容 API]
  end

  User --> CLI
  User --> Gateway
  CLI --> Main
  CLI --> Gateway

  ConfigFile --> Settings
  Settings --> Main
  Settings --> Gateway
  Settings --> Processor

  Main --> Processor
  Processor --> Scheduler
  Processor --> Validator
  Processor --> Factory

  Factory --> BasePool
  Factory --> Excel
  Factory --> MySQL
  Factory --> Engines

  Errors --> Processor
  TaskMeta --> Processor
  Errors --> Service
  Schemas --> Service

  Gateway --> App
  App --> Service
  Service --> Dispatcher
  Service --> Limiter
  Service --> Session
  Service --> Resolver

  Dispatcher --> OpenAI
  Dispatcher --> Claude
  Dispatcher --> Other
```

## 外部使用视角

```mermaid
flowchart TB
  Customer[客户/业务系统]
  Config[配置文件\nconfig.yaml]
  DataSources[数据源\nExcel / MySQL / 其他]
  ExternalAI[外部 AI 服务]
  Logs[日志 / 统计报告]
  OutputData[处理结果\n写回数据源 / 导出文件]
  APIResponse[API 响应\n文本 / 结构化 JSON]

  subgraph Batch["批处理模式 (离线)"]
    CLI[CLI\ncli.py process]
    BatchEngine[AI-DataFlux\n批处理能力]
  end

  subgraph Online["API 网关模式 (在线)"]
    Gateway[网关服务\ncli.py gateway / gateway.py]
    GatewayEngine[AI-DataFlux\nOpenAI 兼容网关]
  end

  Customer --> CLI
  CLI --> BatchEngine
  Config --> BatchEngine
  DataSources --> BatchEngine
  BatchEngine --> ExternalAI
  BatchEngine --> OutputData
  BatchEngine --> Logs

  Customer --> Gateway
  Gateway --> GatewayEngine
  Config --> GatewayEngine
  GatewayEngine --> ExternalAI
  GatewayEngine --> APIResponse
  GatewayEngine --> Logs
```
