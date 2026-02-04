"""
AI-DataFlux 测试套件

测试目录结构:
    tests/
    ├── README.md            # 测试文档
    ├── __init__.py          # 本文件
    ├── conftest.py          # pytest fixtures
    ├── test_cli.py          # CLI 入口测试
    ├── test_config.py       # 配置加载测试
    ├── test_engines.py      # 数据引擎测试
    ├── test_factory.py      # 数据源工厂测试
    ├── test_integration.py  # 集成测试
    ├── test_models.py       # 数据模型测试
    ├── test_processor.py    # 处理器核心测试
    ├── test_scheduler.py    # 分片调度器测试
    ├── test_token_estimator.py # Token 估算器测试
    └── test_validator.py    # JSON 验证器测试

测试统计:
    - 测试总数: 164（基于 def test_，不含参数化展开与跳过统计）
    - 测试文件: 10
    - 测试代码行数: 2636（tests 目录 .py 总行数，含 conftest 与 __init__）
    - 覆盖率: 以最新覆盖率报告为准

参考项目: SuperBatchVideoCompressor
更新时间: 2026-01-12
"""
