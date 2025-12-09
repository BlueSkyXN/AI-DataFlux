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
    └── test_validator.py    # JSON 验证器测试

测试统计:
    - 测试总数: 143 个 (1 个跳过)
    - 测试文件: 9 个
    - 测试代码行数: ~2400 行
    - 覆盖率: 45.36% (核心模块 60-96%)

参考项目: SuperBatchVideoCompressor
更新时间: 2025-12-09
"""
