"""
AI-DataFlux 测试套件

测试目录结构:
    tests/
    ├── __init__.py                    # 本文件（测试套件索引）
    ├── conftest.py                    # pytest 共享 fixtures
    ├── test_cli.py                    # CLI 入口测试（version/check/help/process/token/gui）
    ├── test_config.py                 # 配置加载与验证测试
    ├── test_control.py                # Web GUI 控制面板测试（配置 API / 鉴权 / 进程管理）
    ├── test_csv_pool.py               # CSV 数据源测试（自动检测 / 编码 / 大文件）
    ├── test_engines.py                # 数据引擎测试（Pandas / Polars / 工厂）
    ├── test_factory.py                # 数据源工厂测试（Excel / MySQL / 引擎选择）
    ├── test_feishu_client_async.py    # 飞书客户端异步测试（Token / 重试 / 分块）
    ├── test_feishu_pool.py            # 飞书数据源测试（Bitable / Sheet / 队列操作）
    ├── test_integration.py            # 集成测试（端到端流程 / 引擎兼容性）
    ├── test_models.py                 # 数据模型测试（TaskMetadata / ErrorRecord / ErrorType）
    ├── test_postgresql_pool.py        # PostgreSQL 数据源测试（连接池 / 条件构建）
    ├── test_scheduler.py              # 分片调度器测试（分片计算 / 加载 / 指标 / 内存）
    ├── test_sqlite_pool.py            # SQLite 数据源测试（连接管理 / 任务读写）
    ├── test_token_estimator.py        # Token 估算器测试（模式 / 计数 / tiktoken）
    ├── test_validator.py              # JSON 验证器测试（枚举 / 规则 / 边界）
    └── core/                          # 核心模块子目录测试
        ├── clients/
        │   └── test_flux_client.py    # Flux AI 客户端测试（API 调用 / 错误处理）
        ├── content/
        │   └── test_processor.py      # 内容处理器测试（Prompt / 解析 / Schema）
        ├── retry/
        │   └── test_strategy.py       # 重试策略测试（决策 / 熔断 / 上限）
        └── state/
            └── test_manager.py        # 状态管理器测试（任务状态 / 元数据 / 并发）

覆盖率: 以最新 pytest --cov 报告为准
"""
