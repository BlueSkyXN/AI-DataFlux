"""
CSV 数据源测试

测试 CSV 文件处理功能，包括：
- CSV 自动检测 (通过文件扩展名)
- 编码处理 (UTF-8/GBK)
- 任务读取与写入
- 工厂模式创建

CSV 数据源复用 ExcelTaskPool，通过文件扩展名自动检测。
本测试验证 CSV 读写功能的正确性。
"""

import pytest

from src.data.excel import ExcelTaskPool
from src.data.factory import create_task_pool


class TestCSVAutoDetection:
    """CSV 自动检测测试"""

    @pytest.fixture
    def temp_csv(self, tmp_path):
        """创建临时 CSV 文件"""
        csv_path = tmp_path / "test_data.csv"
        csv_content = """id,input_text,context,output_result,output_summary
1,文本一,上下文一,,
2,文本二,上下文二,,
3,文本三,上下文三,已处理,摘要三
4,,上下文四,,
"""
        csv_path.write_text(csv_content, encoding="utf-8")
        yield csv_path

    def test_csv_auto_detection(self, temp_csv):
        """测试 CSV 文件自动检测"""
        pool = ExcelTaskPool(
            input_path=temp_csv,
            output_path=temp_csv,
            columns_to_extract=["input_text", "context"],
            columns_to_write={"result": "output_result", "summary": "output_summary"},
            engine_type="pandas",
        )

        # 验证 CSV 被正确检测
        assert pool._is_csv is True
        assert pool.engine.row_count(pool.df) == 4

        pool.close()

    def test_excel_not_detected_as_csv(self, tmp_path):
        """测试 Excel 文件不被误检测为 CSV"""
        import pandas as pd

        excel_path = tmp_path / "test_data.xlsx"
        df = pd.DataFrame({"col1": [1, 2, 3]})
        df.to_excel(excel_path, index=False)

        pool = ExcelTaskPool(
            input_path=excel_path,
            output_path=excel_path,
            columns_to_extract=["col1"],
            columns_to_write={"out": "col1"},
            engine_type="pandas",
        )

        assert pool._is_csv is False

        pool.close()


class TestCSVTaskPool:
    """CSV 任务池功能测试"""

    @pytest.fixture
    def temp_csv(self, tmp_path):
        """创建临时 CSV 文件"""
        csv_path = tmp_path / "tasks.csv"
        # 注意：空字段需要明确表示，pandas 默认将空字符串视为空
        csv_content = """input_text,context,output_result
输入文本1,上下文1,
输入文本2,上下文2,
输入文本3,上下文3,已处理
"""
        csv_path.write_text(csv_content, encoding="utf-8")
        yield csv_path

    @pytest.fixture
    def csv_pool(self, temp_csv):
        """创建 CSV 任务池"""
        pool = ExcelTaskPool(
            input_path=temp_csv,
            output_path=temp_csv,
            columns_to_extract=["input_text", "context"],
            columns_to_write={"result": "output_result"},
            require_all_input_fields=True,
            engine_type="pandas",
        )
        yield pool
        pool.close()

    def test_get_total_task_count(self, csv_pool):
        """测试获取未处理任务数"""
        count = csv_pool.get_total_task_count()
        assert count == 2  # 索引 0, 1 未处理

    def test_get_processed_task_count(self, csv_pool):
        """测试获取已处理任务数"""
        count = csv_pool.get_processed_task_count()
        assert count == 1  # 索引 2 已处理

    def test_get_id_boundaries(self, csv_pool):
        """测试获取 ID 边界"""
        min_idx, max_idx = csv_pool.get_id_boundaries()
        # CSV 使用 DataFrame 索引，从 0 开始
        assert min_idx == 0
        assert max_idx == 2

    def test_initialize_shard(self, csv_pool):
        """测试初始化分片"""
        loaded = csv_pool.initialize_shard(0, 0, 2)
        assert loaded == 2  # 2 个未处理任务

    def test_get_task_batch(self, csv_pool):
        """测试获取任务批次"""
        csv_pool.initialize_shard(0, 0, 2)
        batch = csv_pool.get_task_batch(1)

        assert len(batch) == 1
        task_id, data = batch[0]
        assert "input_text" in data
        assert "context" in data

    def test_update_and_save(self, csv_pool, temp_csv):
        """测试更新并保存 CSV"""
        csv_pool.initialize_shard(0, 0, 2)
        batch = csv_pool.get_task_batch(1)
        task_id, _ = batch[0]

        # 更新结果
        csv_pool.update_task_results({task_id: {"result": "测试结果"}})

        # 强制保存
        csv_pool.close()

        # 读取并验证
        content = temp_csv.read_text(encoding="utf-8")
        assert "测试结果" in content


class TestCSVFactoryIntegration:
    """CSV 工厂方法集成测试"""

    @pytest.fixture
    def temp_csv(self, tmp_path):
        """创建临时 CSV 文件"""
        csv_path = tmp_path / "factory_test.csv"
        csv_content = """id,input_text,output_result
1,测试输入,
"""
        csv_path.write_text(csv_content, encoding="utf-8")
        yield csv_path

    def test_create_csv_pool_via_factory(self, temp_csv):
        """测试通过工厂方法创建 CSV 任务池"""
        config = {
            "datasource": {
                "type": "csv",
                "engine": "pandas",
                "require_all_input_fields": True,
                "concurrency": {"save_interval": 60},
            },
            "csv": {
                "input_path": str(temp_csv),
                "output_path": str(temp_csv),
            },
        }

        pool = create_task_pool(
            config=config,
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
        )

        assert pool is not None
        assert pool._is_csv is True

        pool.close()

    def test_create_csv_pool_missing_path(self):
        """测试缺少路径配置时抛出异常"""
        config = {
            "datasource": {"type": "csv"},
            "csv": {},  # 缺少 input_path
        }

        with pytest.raises(ValueError, match="input_path"):
            create_task_pool(
                config=config,
                columns_to_extract=["input_text"],
                columns_to_write={"result": "output"},
            )


class TestCSVEncodingHandling:
    """CSV 编码处理测试"""

    def test_utf8_encoding(self, tmp_path):
        """测试 UTF-8 编码"""
        csv_path = tmp_path / "utf8.csv"
        csv_content = """id,text
1,中文文本
2,日本語テキスト
3,한국어 텍스트
"""
        csv_path.write_text(csv_content, encoding="utf-8")

        pool = ExcelTaskPool(
            input_path=csv_path,
            output_path=csv_path,
            columns_to_extract=["text"],
            columns_to_write={"out": "text"},
            engine_type="pandas",
        )

        # 验证可以正确读取多语言文本
        count = pool.engine.row_count(pool.df)
        assert count == 3

        pool.close()

    def test_special_characters(self, tmp_path):
        """测试特殊字符"""
        csv_path = tmp_path / "special.csv"
        csv_content = """id,text
1,"包含,逗号"
2,"包含""引号"
3,"多行
文本"
"""
        csv_path.write_text(csv_content, encoding="utf-8")

        pool = ExcelTaskPool(
            input_path=csv_path,
            output_path=csv_path,
            columns_to_extract=["text"],
            columns_to_write={"out": "text"},
            engine_type="pandas",
        )

        # 验证特殊字符被正确处理
        count = pool.engine.row_count(pool.df)
        assert count == 3

        pool.close()


class TestCSVLargeFile:
    """CSV 大文件处理测试"""

    def test_large_csv_performance(self, tmp_path):
        """测试大 CSV 文件处理性能"""
        csv_path = tmp_path / "large.csv"

        # 创建 1000 行数据
        rows = ["id,input_text,output"]
        for i in range(1000):
            rows.append(f'{i},"输入文本{i}",')
        csv_path.write_text("\n".join(rows), encoding="utf-8")

        pool = ExcelTaskPool(
            input_path=csv_path,
            output_path=csv_path,
            columns_to_extract=["input_text"],
            columns_to_write={"out": "output"},
            engine_type="pandas",
        )

        count = pool.get_total_task_count()
        assert count == 1000

        pool.close()
