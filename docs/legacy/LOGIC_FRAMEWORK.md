# 逻辑框架图

## 系统内部依赖关系

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

  subgraph Core["核心处理层 (组件化)"]
    Processor[core/processor.py<br/>(协调者)]
    Scheduler[core/scheduler.py]
    Validator[core/validator.py]
    TokenEst[core/token_estimator.py]

    subgraph Components["功能组件"]
        Clients[core/clients/*<br/>API通信]
        Content[core/content/*<br/>Prompt/解析]
        State[core/state/*<br/>状态管理]
        Retry[core/retry/*<br/>重试策略]
    end
  end

  subgraph Data["数据源层"]
    Factory[data/factory.py]
    BasePool[data/base.py]
    Excel[data/excel.py<br/>(含 CSV)]
    MySQL[data/mysql.py]
    PostgreSQL[data/postgresql.py]
    SQLite[data/sqlite.py]
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

  %% 入口依赖
  User --> CLI
  User --> Gateway
  CLI --> Main
  CLI --> Gateway

  %% 配置依赖
  ConfigFile --> Settings
  Settings --> Main
  Settings --> Gateway
  Settings --> Processor

  %% 核心流程
  Main --> Processor

  %% Processor 协调组件
  Processor --> Scheduler
  Processor --> Validator
  Processor --> Factory
  Processor --> TokenEst
  Processor --> Clients
  Processor --> Content
  Processor --> State
  Processor --> Retry

  %% 数据源工厂
  Factory --> BasePool
  Factory --> Excel
  Factory --> MySQL
  Factory --> PostgreSQL
  Factory --> SQLite
  Factory --> Engines

  %% 模型使用
  Errors --> Processor
  Errors --> Retry
  TaskMeta --> State
  TaskMeta --> Scheduler
  Errors --> Service
  Schemas --> Service

  %% 网关内部流程
  Gateway --> App
  App --> Service
  Service --> Dispatcher
  Service --> Limiter
  Service --> Session
  Service --> Resolver

  %% 外部调用
  Dispatcher --> External
  Clients --> External
```

## 外部使用视角

```mermaid
flowchart TB
  Customer[客户/业务系统]
  Config[配置文件\\nconfig.yaml]
  DataSources[数据源\\nExcel / CSV / MySQL / PostgreSQL / SQLite]
  ExternalAI[外部 AI 服务]
  Logs[日志 / 统计报告]
  OutputData[处理结果\\n写回数据源 / 导出文件]
  APIResponse[API 响应\\n文本 / 结构化 JSON]

  subgraph Batch["批处理模式 (离线)"]
    CLI[CLI\\ncli.py process]
    BatchEngine[AI-DataFlux\\n批处理引擎]
  end

  subgraph Online["API 网关模式 (在线)"]
    Gateway[网关服务\\ncli.py gateway / gateway.py]
    GatewayEngine[AI-DataFlux\\nOpenAI 兼容网关]
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
