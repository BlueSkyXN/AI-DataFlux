# 🧪 测试快速参考

## 常用命令

```bash
# 运行所有测试
pytest tests/

# 详细输出
pytest tests/ -v

# 带覆盖率
pytest tests/ --cov=src --cov-report=term-missing

# 生成 HTML 报告
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# 跳过集成测试
pytest tests/ -v -m "not integration"

# 运行特定文件
pytest tests/test_engines.py -v

# 运行特定测试
pytest tests/test_engines.py::TestPandasEngine::test_read_excel -v

# 调试模式
pytest tests/ -x --pdb

# 显示最慢的 10 个测试
pytest tests/ --durations=10

# 并行执行 (需要 pytest-xdist)
pytest tests/ -n auto
```

## 测试统计

- **测试总数**: 143 个 (1 个跳过)
- **测试文件**: 9 个
- **代码行数**: 2295 行
- **覆盖率**: 45.36%

## 核心模块覆盖率

| 模块 | 覆盖率 |
|------|--------|
| models/task.py | 96% ⭐ |
| models/errors.py | 92% ⭐ |
| data/factory.py | 85% ⭐ |
| core/validator.py | 85% ⭐ |
| core/scheduler.py | 83% ⭐ |

## CI/CD 矩阵

- **操作系统**: 9 种 (Ubuntu/Windows/macOS)
- **Python**: 3.10-3.14
- **架构**: x64, ARM64
- **并行 job**: 70+

## 快速链接

- 📖 [完整测试文档](../tests/README.md)
- 📊 [测试总结报告](../TESTING_SUMMARY.md)
- 🔧 [pytest 配置](../pytest.ini)
- 📈 [覆盖率配置](../.coveragerc)
