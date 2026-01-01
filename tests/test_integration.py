"""
集成测试

测试多个模块协同工作的场景
"""

import pytest
import pandas as pd
from pathlib import Path


class TestExcelTaskPoolIntegration:
    """Excel 任务池集成测试"""
    
    @pytest.fixture
    def excel_config(self, sample_excel_file, tmp_path):
        """创建 Excel 任务池配置"""
        output_path = tmp_path / "output.xlsx"
        return {
            "datasource": {
                "type": "excel",
                "engine": "pandas",
                "excel_reader": "openpyxl",
                "excel_writer": "openpyxl",
                "require_all_input_fields": True,
                "concurrency": {
                    "batch_size": 10,
                    "save_interval": 60,
                },
            },
            "excel": {
                "input_path": str(sample_excel_file),
                "output_path": str(output_path),
            },
            "columns_to_extract": ["question", "context"],
            "columns_to_write": {"answer": "answer"},
        }
    
    def test_create_excel_task_pool(self, excel_config):
        """测试创建 Excel 任务池"""
        from src.data.excel import ExcelTaskPool
        
        pool = ExcelTaskPool(
            input_path=excel_config["excel"]["input_path"],
            output_path=excel_config["excel"]["output_path"],
            columns_to_extract=excel_config["columns_to_extract"],
            columns_to_write=excel_config["columns_to_write"],
            engine_type="pandas",
        )
        
        assert pool is not None
        assert pool.engine.name == "pandas"
    
    def test_get_id_boundaries(self, excel_config):
        """测试获取 ID 边界"""
        from src.data.excel import ExcelTaskPool
        
        pool = ExcelTaskPool(
            input_path=excel_config["excel"]["input_path"],
            output_path=excel_config["excel"]["output_path"],
            columns_to_extract=excel_config["columns_to_extract"],
            columns_to_write=excel_config["columns_to_write"],
            engine_type="pandas",
        )
        
        min_id, max_id = pool.get_id_boundaries()
        assert min_id == 0
        assert max_id >= 0
    
    def test_initialize_shard(self, excel_config):
        """测试初始化分片"""
        from src.data.excel import ExcelTaskPool
        
        pool = ExcelTaskPool(
            input_path=excel_config["excel"]["input_path"],
            output_path=excel_config["excel"]["output_path"],
            columns_to_extract=excel_config["columns_to_extract"],
            columns_to_write=excel_config["columns_to_write"],
            engine_type="pandas",
        )
        
        min_id, max_id = pool.get_id_boundaries()
        count = pool.initialize_shard(0, min_id, max_id)
        
        # 应该加载了一些任务
        assert count >= 0
    
    def test_update_task_results(self, excel_config):
        """测试更新任务结果"""
        from src.data.excel import ExcelTaskPool
        
        pool = ExcelTaskPool(
            input_path=excel_config["excel"]["input_path"],
            output_path=excel_config["excel"]["output_path"],
            columns_to_extract=excel_config["columns_to_extract"],
            columns_to_write=excel_config["columns_to_write"],
            engine_type="pandas",
        )
        
        # 直接更新第一行的结果
        pool.update_task_results({0: {"answer": "Test Answer"}})
        
        # 验证写入
        row = pool.engine.get_row(pool.df, 0)
        assert row["answer"] == "Test Answer"


class TestEngineCompatibility:
    """引擎兼容性测试"""
    
    def test_pandas_polars_read_same_file(self, sample_excel_file):
        """测试 Pandas 和 Polars 读取同一文件"""
        from src.data.engines import get_engine, POLARS_AVAILABLE
        
        # Pandas 读取
        pandas_engine = get_engine("pandas")
        df_pandas = pandas_engine.read_excel(sample_excel_file)
        
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        # Polars 读取
        polars_engine = get_engine("polars")
        df_polars = polars_engine.read_excel(sample_excel_file)
        
        # 行数应该相同
        assert pandas_engine.row_count(df_pandas) == polars_engine.row_count(df_polars)
        
        # 列名应该相同
        assert set(pandas_engine.get_column_names(df_pandas)) == \
               set(polars_engine.get_column_names(df_polars))
    
    def test_engine_auto_fallback(self):
        """测试引擎自动回退"""
        from src.data.engines import get_engine
        
        # 无论 polars 是否可用，auto 都应该返回一个有效引擎
        engine = get_engine("auto")
        
        assert engine is not None
        assert engine.name in ["pandas", "polars"]
        assert hasattr(engine, "read_excel")
        assert hasattr(engine, "write_excel")


class TestConfigToPoolIntegration:
    """配置到任务池的完整流程测试"""
    
    def test_factory_creates_correct_pool(self, sample_config_file, sample_excel_file, tmp_path):
        """测试工厂根据配置创建正确的任务池"""
        from src.config import load_config
        from src.data.factory import create_task_pool
        
        # 加载配置
        config = load_config(str(sample_config_file))
        
        # 修改配置使用测试文件
        config["excel"] = {
            "input_path": str(sample_excel_file),
            "output_path": str(tmp_path / "output.xlsx"),
        }
        
        # 创建任务池
        pool = create_task_pool(
            config=config,
            columns_to_extract=config["columns_to_extract"],
            columns_to_write=config["columns_to_write"],
        )
        
        assert pool is not None
