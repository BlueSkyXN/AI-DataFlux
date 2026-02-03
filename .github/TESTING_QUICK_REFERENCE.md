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

- **测试总数**: 164（基于 `def test_`，不含参数化展开与跳过统计）
- **测试文件**: 10
- **代码行数**: 2636（tests 目录 `.py` 总行数，含 conftest 与 __init__）
- **覆盖率**: 以最新覆盖率报告为准

## 核心模块覆盖率

| 模块 | 覆盖率 |
|------|--------|
| models/task.py | 96% ⭐ |
| models/errors.py | 92% ⭐ |
| data/factory.py | 85% ⭐ |
| core/validator.py | 85% ⭐ |
| core/scheduler.py | 83% ⭐ |

## CI/CD 矩阵

- **操作系统**: 8 种 (Ubuntu/Windows/macOS，不含 Windows ARM)
- **Python**: 3.10-3.14
- **架构**: x64；ARM64 仅 Linux/macOS
- **并行 job**: 38+

## 快速链接

- 📖 [完整测试文档](../tests/README.md)
- 📊 [测试总结报告](../TESTING_SUMMARY.md)
- 🔧 [pytest 配置](../pytest.ini)
- 📈 [覆盖率配置](../.coveragerc)
