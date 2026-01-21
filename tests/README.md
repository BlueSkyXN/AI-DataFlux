# AI-DataFlux 测试文档

本目录包含 AI-DataFlux 项目的完整测试套件，采用 pytest 框架，参考 SuperBatchVideoCompressor 项目的最佳实践。

## 📋 测试目录结构

```
tests/
├── README.md              # 本文档
├── __init__.py            # 测试套件说明
├── conftest.py            # pytest fixtures 配置
│
├── test_cli.py            # CLI 命令行测试
├── test_config.py         # 配置加载和验证测试
├── test_engines.py        # 数据引擎测试 (Pandas/Polars)
├── test_factory.py        # 数据源工厂模式测试
├── test_integration.py    # 集成测试
├── test_models.py         # 数据模型测试 (TaskMetadata/ErrorType)
├── test_processor.py      # 处理器核心逻辑测试
├── test_scheduler.py      # 分片调度器测试
├── test_token_estimator.py # Token 估算器测试
└── test_validator.py      # JSON 验证器测试
```

## 🚀 快速开始

### 安装测试依赖

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### 运行所有测试

```bash
# 基础测试
pytest tests/

# 详细输出
pytest tests/ -v

# 带覆盖率报告
pytest tests/ --cov=src --cov-report=term-missing

# 生成 HTML 覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

### 运行特定测试

```bash
# 运行单个测试文件
pytest tests/test_engines.py -v

# 运行单个测试类
pytest tests/test_engines.py::TestPandasEngine -v

# 运行单个测试函数
pytest tests/test_engines.py::TestPandasEngine::test_read_excel -v

# 运行匹配模式的测试
pytest tests/ -k "engine" -v
```

### 运行带标记的测试

```bash
# 跳过集成测试
pytest tests/ -v -m "not integration"

# 只运行集成测试
pytest tests/ -v -m "integration"

# 跳过慢速测试
pytest tests/ -v -m "not slow"
```

## 📊 测试覆盖率

### 当前覆盖率统计

覆盖率以 `pytest --cov=src` 结果为准，默认排除 `src/gateway/*` 与 `src/utils/console.py`（见 `.coveragerc`）。

- **总体覆盖率**: 以最新覆盖率报告为准（文档中的历史数值可能过期）
- **核心模块覆盖率**:
  - 以覆盖率报告为准

### 生成覆盖率报告

```bash
# 终端报告
pytest tests/ --cov=src --cov-report=term-missing

# HTML 报告 (推荐)
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# XML 报告 (用于 CI/CD)
pytest tests/ --cov=src --cov-report=xml

# JSON 报告
pytest tests/ --cov=src --cov-report=json
```

## 🧪 测试模块说明

### test_cli.py
- **目的**: 测试 CLI 命令行接口
- **覆盖**: version, check, process, gateway 命令
- **测试数量**: 9 个测试

### test_config.py
- **目的**: 测试配置文件加载和验证
- **覆盖**: YAML 解析、配置验证、错误处理
- **测试数量**: 9 个测试

### test_engines.py
- **目的**: 测试数据引擎抽象和实现
- **覆盖**: PandasEngine, PolarsEngine, 引擎工厂
- **测试数量**: 28 个测试
- **特性**:
  - 引擎自动选择
  - 读写器性能库检测 (Calamine/xlsxwriter)
  - 向量化操作测试

### test_factory.py
- **目的**: 测试数据源任务池工厂
- **覆盖**: Excel 池创建、MySQL 池创建、引擎选择
- **测试数量**: 17 个测试
- **特性**:
  - 多种引擎配置 (auto/pandas/polars)
  - 读写器配置
  - 并发参数验证

### test_integration.py
- **目的**: 集成测试多模块协同
- **覆盖**: Excel 任务池、引擎兼容性、配置到池的完整流程
- **测试数量**: 7 个测试
- **标记**: `@pytest.mark.integration`

### test_models.py
- **目的**: 测试数据模型和数据类
- **覆盖**: TaskMetadata, ErrorRecord, ErrorType
- **测试数量**: 21 个测试
- **特性**:
  - 重试计数管理
  - 错误历史记录
  - 边界情况处理

### test_processor.py
- **目的**: 测试 AI 处理器核心逻辑
- **覆盖**: 提示词生成、JSON 提取、Schema 构建、任务状态管理
- **测试数量**: 23 个测试
- **特性**:
  - Markdown 代码块提取
  - 字段验证
  - 错误重试逻辑

### test_scheduler.py
- **目的**: 测试分片任务调度器
- **覆盖**: 分片计算、加载、进度跟踪、内存监控
- **测试数量**: 26 个测试
- **特性**:
  - 动态分片大小计算
  - 空分片跳过
  - 处理指标统计

### test_token_estimator.py
- **目的**: 测试 Token 估算器
- **覆盖**: mode 规范化、输入/输出估算、采样逻辑
- **测试数量**: 14 个测试

### test_validator.py
- **目的**: 测试 JSON 字段验证器
- **覆盖**: 字段规则验证、大小写敏感性、数值类型
- **测试数量**: 10 个测试

## 🔧 Fixtures 说明

`conftest.py` 提供以下共享 fixtures:

### 配置类
- `sample_config`: 示例配置字典
- `sample_config_file`: 临时配置文件

### 数据类
- `sample_dataframe`: 示例 Pandas DataFrame
- `sample_excel_file`: 临时 Excel 文件

### 引擎类
- `pandas_engine`: PandasEngine 实例
- `polars_engine`: PolarsEngine 实例 (如果可用)

### 环境类
- `temp_dir`: 临时目录
- `clean_temp_dir`: 自动清理的临时目录

### Mock 类
- `mock_api_response`: 模拟 API 响应

## 📝 编写测试指南

### 测试命名规范

```python
# 测试类以 Test 开头
class TestFeatureName:
    """功能说明"""

    # 测试方法以 test_ 开头
    def test_specific_behavior(self):
        """测试具体行为的文档字符串"""
        # Arrange
        # Act
        # Assert
```

### 使用 Fixtures

```python
def test_with_fixture(sample_dataframe):
    """测试使用 fixture"""
    assert len(sample_dataframe) == 5
    assert "question" in sample_dataframe.columns
```

### 异常测试

```python
def test_error_handling(self):
    """测试异常处理"""
    with pytest.raises(ValueError, match="错误消息"):
        raise ValueError("错误消息")
```

### Mock 外部依赖

```python
from unittest.mock import MagicMock, patch

def test_with_mock(self):
    """测试使用 Mock"""
    mock_pool = MagicMock()
    mock_pool.get_total_task_count.return_value = 100
    # 使用 mock_pool
```

### 测试标记

```python
@pytest.mark.integration
def test_full_workflow(self):
    """集成测试标记"""
    pass

@pytest.mark.slow
def test_performance(self):
    """慢速测试标记"""
    pass
```

## 🎯 CI/CD 集成

### GitHub Actions 工作流

项目配置了全面的 CI/CD 流程 (`.github/workflows/test.yml`):

#### 1. 代码质量检查 (lint)
- Ruff 代码检查
- Black 格式验证
- MyPy 类型检查
- Python 语法检查

#### 2. 单元测试矩阵 (unit-tests)
- **操作系统**: Ubuntu 22.04/24.04/24.04-ARM, Windows 2022/2025/11-ARM, macOS 15/26
- **Python 版本**: 3.10, 3.11, 3.12, 3.13, 3.14
- **架构**: x64, ARM64
- **总组合**: 70+ 并行测试 job

#### 3. CLI 功能测试 (cli-test)
- 测试所有 CLI 命令
- 配置验证

#### 4. 高性能库测试 (perf-libs)
- Polars 可用性检测
- Calamine/xlsxwriter 检测
- 引擎自动选择验证

#### 5. 集成测试 (integration-test)
- 手动触发 (workflow_dispatch)
- 完整工作流验证

### 覆盖率上传

覆盖率自动上传到 Codecov:
```yaml
- name: Upload coverage
  uses: codecov/codecov-action@v4
  with:
    file: ./coverage.xml
    flags: unittests
```

## 🛠️ 高级用法

### 调试测试

```bash
# 在第一个失败处停止
pytest tests/ -x

# 显示局部变量
pytest tests/ -l

# 进入调试器
pytest tests/ --pdb

# 详细回溯
pytest tests/ --tb=long
```

### 性能分析

```bash
# 显示最慢的 10 个测试
pytest tests/ --durations=10

# 显示所有测试耗时
pytest tests/ --durations=0
```

### 并行执行

```bash
# 安装 pytest-xdist
pip install pytest-xdist

# 自动检测 CPU 核心数
pytest tests/ -n auto

# 指定进程数
pytest tests/ -n 4
```

## 📈 测试最佳实践

1. **每个测试只测一件事**: 保持测试简单、专注
2. **使用描述性名称**: 测试名应清楚说明测试内容
3. **AAA 模式**: Arrange (准备), Act (执行), Assert (断言)
4. **避免测试间依赖**: 每个测试应独立运行
5. **使用 fixtures**: 复用测试数据和环境设置
6. **测试边界情况**: 不只测试正常流程
7. **保持测试快速**: 快速反馈提高开发效率
8. **及时更新测试**: 代码改变时同步更新测试

## 🔍 故障排查

### 常见问题

**Q: 测试导入失败**
```bash
# 确保 PYTHONPATH 包含项目根目录
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

**Q: Polars 相关测试被跳过**
```bash
# Polars 是可选依赖
pip install polars

# 或忽略这些测试
pytest tests/ -v  # 自动跳过不可用的库
```

**Q: 覆盖率数据不准确**
```bash
# 清理旧的覆盖率数据
rm -rf .coverage htmlcov/
pytest tests/ --cov=src --cov-report=html
```

## 📚 参考资源

- [pytest 官方文档](https://docs.pytest.org/)
- [pytest-cov 文档](https://pytest-cov.readthedocs.io/)
- [pytest-asyncio 文档](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock 文档](https://docs.python.org/3/library/unittest.mock.html)

## 🤝 贡献测试

欢迎提交新的测试用例! 请遵循以下步骤:

1. 在对应的 test_*.py 文件中添加测试
2. 确保测试通过: `pytest tests/ -v`
3. 检查覆盖率: `pytest tests/ --cov=src`
4. 运行代码质量检查: `ruff check tests/` 和 `black --check tests/`
5. 提交 Pull Request

---

**测试总数**: 164（基于 `def test_`，不含参数化展开与跳过统计）

**覆盖模块**: 9+ 个核心模块

**测试代码行数**: 2636（tests 目录 `.py` 总行数，含 conftest 与 __init__）

**最后更新**: 2026-01-12
