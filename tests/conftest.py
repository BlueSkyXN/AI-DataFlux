"""
pytest fixtures - 测试共享资源

Fixtures 是 pytest 的核心概念，用于:
1. 提供测试数据
2. 设置/清理测试环境
3. 在多个测试间共享资源
"""

import sys
from pathlib import Path

import pytest
import pandas as pd

# 确保可以导入 src 模块
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== 配置 Fixtures ====================

@pytest.fixture
def sample_config() -> dict:
    """提供示例配置字典"""
    return {
        "global": {
            "log": {
                "level": "info",
                "format": "text",
                "output": "console",
            },
            "flux_api_url": "http://127.0.0.1:8787",
        },
        "datasource": {
            "type": "excel",
            "engine": "auto",
            "excel_reader": "auto",
            "excel_writer": "auto",
            "require_all_input_fields": True,
            "concurrency": {
                "batch_size": 10,
                "save_interval": 60,
            },
        },
        "excel": {
            "input_path": "./test_input.xlsx",
            "output_path": "./test_output.xlsx",
        },
        "columns_to_extract": ["question", "context"],
        "columns_to_write": {"answer": "ai_answer"},
        "prompt": {
            "required_fields": ["answer"],
            "use_json_schema": True,
            "template": "Test: {record_json}",
        },
        "validation": {
            "enabled": False,
            "field_rules": {},
        },
        "models": [],
        "channels": {},
    }


@pytest.fixture
def sample_config_file(sample_config, tmp_path) -> Path:
    """创建临时配置文件"""
    import yaml
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(sample_config, f, allow_unicode=True)
    return config_path


# ==================== 数据 Fixtures ====================

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """提供示例 DataFrame"""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "question": ["Q1", "Q2", "", None, "Q5"],
        "context": ["C1", "C2", "C3", "C4", ""],
        "answer": [None, "A2", None, None, None],
    })


@pytest.fixture
def sample_excel_file(sample_dataframe, tmp_path) -> Path:
    """创建临时 Excel 文件"""
    excel_path = tmp_path / "test_data.xlsx"
    sample_dataframe.to_excel(excel_path, index=False, engine="openpyxl")
    return excel_path


# ==================== 引擎 Fixtures ====================

@pytest.fixture
def pandas_engine():
    """提供 PandasEngine 实例"""
    from src.data.engines import PandasEngine
    return PandasEngine()


@pytest.fixture
def polars_engine():
    """提供 PolarsEngine 实例 (如果可用)"""
    from src.data.engines import POLARS_AVAILABLE
    if not POLARS_AVAILABLE:
        pytest.skip("Polars not available")
    
    from src.data.engines.polars_engine import PolarsEngine
    return PolarsEngine()


# ==================== 临时目录 Fixtures ====================

@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """提供临时目录"""
    return tmp_path


@pytest.fixture
def clean_temp_dir(tmp_path) -> Path:
    """提供干净的临时目录，测试后自动清理"""
    test_dir = tmp_path / "test_workspace"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # pytest 会自动清理 tmp_path


# ==================== Mock Fixtures ====================

@pytest.fixture
def mock_api_response():
    """模拟 API 响应"""
    return {
        "choices": [
            {
                "message": {
                    "content": '{"answer": "Test answer"}'
                }
            }
        ]
    }
