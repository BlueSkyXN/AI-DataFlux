"""
数据源工厂测试

测试 src/data/factory.py 的任务池工厂功能
"""

import pytest


class TestFactoryBasics:
    """工厂基础功能测试"""

    def test_factory_imports(self):
        """测试工厂模块可导入"""
        from src.data.factory import (
            create_task_pool,
            MYSQL_AVAILABLE,
            EXCEL_ENABLED,
        )

        assert callable(create_task_pool)
        assert isinstance(MYSQL_AVAILABLE, bool)
        assert isinstance(EXCEL_ENABLED, bool)

    def test_excel_enabled(self):
        """测试 Excel 功能可用"""
        from src.data.factory import EXCEL_ENABLED

        # 在测试环境中 Excel 应该可用
        assert EXCEL_ENABLED is True


class TestCreateExcelPool:
    """Excel 任务池创建测试"""

    @pytest.fixture
    def excel_config(self, sample_excel_file, tmp_path):
        """创建 Excel 配置"""
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
                "output_path": str(tmp_path / "output.xlsx"),
            },
        }

    def test_create_excel_pool_success(self, excel_config):
        """测试成功创建 Excel 任务池"""
        from src.data.factory import create_task_pool
        from src.data.excel import ExcelTaskPool

        pool = create_task_pool(
            config=excel_config,
            columns_to_extract=["question", "context"],
            columns_to_write={"answer": "answer"},
        )

        assert isinstance(pool, ExcelTaskPool)

    def test_create_excel_pool_with_auto_engine(self, excel_config):
        """测试使用自动引擎创建"""
        from src.data.factory import create_task_pool

        excel_config["datasource"]["engine"] = "auto"

        pool = create_task_pool(
            config=excel_config,
            columns_to_extract=["question"],
            columns_to_write={"answer": "answer"},
        )

        assert pool is not None
        assert pool.engine.name in ["pandas", "polars"]

    def test_create_excel_pool_missing_input_path(self, excel_config):
        """测试缺少输入路径抛出异常"""
        from src.data.factory import create_task_pool

        del excel_config["excel"]["input_path"]

        with pytest.raises(ValueError, match="input_path"):
            create_task_pool(
                config=excel_config,
                columns_to_extract=["question"],
                columns_to_write={"answer": "answer"},
            )

    def test_create_excel_pool_default_output_path(self, excel_config):
        """测试默认输出路径使用输入路径"""
        from src.data.factory import create_task_pool

        del excel_config["excel"]["output_path"]

        pool = create_task_pool(
            config=excel_config,
            columns_to_extract=["question"],
            columns_to_write={"answer": "answer"},
        )

        # 输出路径应该等于输入路径
        assert pool.output_path == pool.input_path


class TestCreateMySQLPool:
    """MySQL 任务池创建测试"""

    @pytest.fixture
    def mysql_config(self):
        """创建 MySQL 配置"""
        return {
            "datasource": {
                "type": "mysql",
                "require_all_input_fields": True,
                "concurrency": {
                    "max_workers": 5,
                },
            },
            "mysql": {
                "host": "localhost",
                "port": 3306,
                "user": "test_user",
                "password": "test_pass",
                "database": "test_db",
                "table_name": "test_table",
            },
        }

    def test_create_mysql_pool_missing_config(self, mysql_config):
        """测试缺少 MySQL 配置字段"""
        from src.data.factory import create_task_pool, MYSQL_AVAILABLE

        if not MYSQL_AVAILABLE:
            pytest.skip("MySQL connector not available")

        del mysql_config["mysql"]["host"]

        with pytest.raises(ValueError, match="host"):
            create_task_pool(
                config=mysql_config,
                columns_to_extract=["question"],
                columns_to_write={"answer": "answer"},
            )

    def test_create_mysql_pool_connector_not_available(self, mysql_config):
        """测试 MySQL connector 不可用时抛出异常"""
        from src.data.factory import MYSQL_AVAILABLE

        if MYSQL_AVAILABLE:
            pytest.skip("MySQL connector is available")

        from src.data.factory import create_task_pool

        with pytest.raises(ImportError, match="MySQL Connector"):
            create_task_pool(
                config=mysql_config,
                columns_to_extract=["question"],
                columns_to_write={"answer": "answer"},
            )


class TestUnsupportedDataSource:
    """不支持的数据源测试"""

    def test_unsupported_datasource_type(self, tmp_path):
        """测试不支持的数据源类型"""
        from src.data.factory import create_task_pool

        config = {
            "datasource": {
                "type": "mongodb",  # 不支持
            },
        }

        with pytest.raises(ValueError, match="不支持的数据源类型"):
            create_task_pool(
                config=config,
                columns_to_extract=["question"],
                columns_to_write={"answer": "answer"},
            )

    def test_empty_datasource_type_defaults_to_excel(self, sample_excel_file, tmp_path):
        """测试空数据源类型默认为 Excel"""
        from src.data.factory import create_task_pool
        from src.data.excel import ExcelTaskPool

        config = {
            "datasource": {
                # 不指定 type，应默认为 excel
                "engine": "pandas",
            },
            "excel": {
                "input_path": str(sample_excel_file),
                "output_path": str(tmp_path / "output.xlsx"),
            },
        }

        pool = create_task_pool(
            config=config,
            columns_to_extract=["question"],
            columns_to_write={"answer": "answer"},
        )

        assert isinstance(pool, ExcelTaskPool)


class TestEngineSelection:
    """引擎选择测试"""

    @pytest.fixture
    def excel_config(self, sample_excel_file, tmp_path):
        """创建 Excel 配置"""
        return {
            "datasource": {
                "type": "excel",
                "engine": "pandas",
                "excel_reader": "openpyxl",
                "excel_writer": "openpyxl",
            },
            "excel": {
                "input_path": str(sample_excel_file),
                "output_path": str(tmp_path / "output.xlsx"),
            },
        }

    def test_pandas_engine_selection(self, excel_config):
        """测试选择 Pandas 引擎"""
        from src.data.factory import create_task_pool

        excel_config["datasource"]["engine"] = "pandas"

        pool = create_task_pool(
            config=excel_config,
            columns_to_extract=["question"],
            columns_to_write={"answer": "answer"},
        )

        assert pool.engine.name == "pandas"

    def test_polars_engine_selection(self, excel_config):
        """测试选择 Polars 引擎"""
        from src.data.factory import create_task_pool
        from src.data.engines import POLARS_AVAILABLE

        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")

        excel_config["datasource"]["engine"] = "polars"

        pool = create_task_pool(
            config=excel_config,
            columns_to_extract=["question"],
            columns_to_write={"answer": "answer"},
        )

        assert pool.engine.name == "polars"

    def test_reader_writer_selection(self, excel_config):
        """测试读写器选择"""
        from src.data.factory import create_task_pool

        excel_config["datasource"]["excel_reader"] = "openpyxl"
        excel_config["datasource"]["excel_writer"] = "openpyxl"

        pool = create_task_pool(
            config=excel_config,
            columns_to_extract=["question"],
            columns_to_write={"answer": "answer"},
        )

        assert pool.engine.excel_reader == "openpyxl"
        assert pool.engine.excel_writer == "openpyxl"


class TestConcurrencyConfig:
    """并发配置测试"""

    @pytest.fixture
    def excel_config(self, sample_excel_file, tmp_path):
        """创建 Excel 配置"""
        return {
            "datasource": {
                "type": "excel",
                "engine": "pandas",
                "concurrency": {
                    "save_interval": 120,
                },
            },
            "excel": {
                "input_path": str(sample_excel_file),
                "output_path": str(tmp_path / "output.xlsx"),
            },
        }

    def test_save_interval_config(self, excel_config):
        """测试保存间隔配置"""
        from src.data.factory import create_task_pool

        pool = create_task_pool(
            config=excel_config,
            columns_to_extract=["question"],
            columns_to_write={"answer": "answer"},
        )

        assert pool.save_interval == 120

    def test_default_save_interval(self, excel_config):
        """测试默认保存间隔"""
        from src.data.factory import create_task_pool

        del excel_config["datasource"]["concurrency"]["save_interval"]

        pool = create_task_pool(
            config=excel_config,
            columns_to_extract=["question"],
            columns_to_write={"answer": "answer"},
        )

        # 默认为 300 秒
        assert pool.save_interval == 300


class TestRequireAllInputFields:
    """输入字段要求测试"""

    @pytest.fixture
    def excel_config(self, sample_excel_file, tmp_path):
        """创建 Excel 配置"""
        return {
            "datasource": {
                "type": "excel",
                "engine": "pandas",
                "require_all_input_fields": True,
            },
            "excel": {
                "input_path": str(sample_excel_file),
                "output_path": str(tmp_path / "output.xlsx"),
            },
        }

    def test_require_all_input_fields_true(self, excel_config):
        """测试要求所有输入字段"""
        from src.data.factory import create_task_pool

        pool = create_task_pool(
            config=excel_config,
            columns_to_extract=["question", "context"],
            columns_to_write={"answer": "answer"},
        )

        assert pool.require_all_input_fields is True

    def test_require_all_input_fields_false(self, excel_config):
        """测试不要求所有输入字段"""
        from src.data.factory import create_task_pool

        excel_config["datasource"]["require_all_input_fields"] = False

        pool = create_task_pool(
            config=excel_config,
            columns_to_extract=["question"],
            columns_to_write={"answer": "answer"},
        )

        assert pool.require_all_input_fields is False
