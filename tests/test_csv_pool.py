"""
CSV 数据源测试

被测模块: src/data/excel.py (ExcelTaskPool CSV 模式), src/data/factory.py

测试 CSV 文件处理功能，包括：
- CSV 自动检测 (通过文件扩展名)
- 编码处理 (UTF-8/GBK)
- 任务读取与写入
- 工厂模式创建

CSV 数据源复用 ExcelTaskPool，通过文件扩展名自动检测。
本测试验证 CSV 读写功能的正确性。

测试类/函数清单:
    TestCSVAutoDetection           CSV 自动检测测试
        test_csv_auto_detection    验证 .csv 文件被正确检测为 CSV 模式
        test_excel_not_detected_as_csv  验证 .xlsx 文件不被误检测为 CSV
    TestCSVTaskPool                CSV 任务池功能测试
        test_get_total_task_count  验证未处理任务计数
        test_get_processed_task_count  验证已处理任务计数
        test_get_id_boundaries     验证 ID 边界（从 0 开始的索引）
        test_initialize_shard      验证分片加载未处理任务
        test_get_task_batch        验证获取任务批次的数据格式
        test_update_and_save       验证更新结果并保存到文件
    TestCSVFactoryIntegration      CSV 工厂方法集成测试
        test_create_csv_pool_via_factory  验证通过工厂创建 CSV 任务池
        test_create_csv_pool_missing_path 验证缺少路径时抛出 ValueError
    TestCSVEncodingHandling        CSV 编码处理测试
        test_utf8_encoding         验证 UTF-8 多语言文本正确读取
        test_special_characters    验证逗号、引号、换行等特殊字符处理
    TestCSVLargeFile               CSV 大文件处理测试
        test_large_csv_performance 验证 1000 行 CSV 文件正常加载
"""

import pytest
from unittest import mock

pd = pytest.importorskip("pandas")


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
        from src.data.excel import ExcelTaskPool

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
        from src.data.excel import ExcelTaskPool

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
        from src.data.excel import ExcelTaskPool

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
        receipt = csv_pool.update_task_results(
            "csv-update",
            {task_id: {"result": "测试结果"}},
        )
        assert receipt.committed_ids == (task_id,)

        # 强制保存
        csv_pool.close()

        # 读取并验证
        content = temp_csv.read_text(encoding="utf-8")
        assert "测试结果" in content

    def test_persisted_receipt_requires_successful_atomic_replace(self, temp_csv):
        from src.data.excel import ExcelTaskPool

        pool = ExcelTaskPool(
            input_path=temp_csv,
            output_path=temp_csv,
            columns_to_extract=["input_text", "context"],
            columns_to_write={"result": "output_result"},
            engine_type="pandas",
        )
        original = temp_csv.read_bytes()
        try:
            with mock.patch(
                "src.data.excel.os.replace", side_effect=OSError("replace failed")
            ):
                receipt = pool.update_task_results(
                    "replace-failed",
                    {0: {"result": "not durable"}},
                )

            assert receipt.committed_ids == ()
            assert receipt.items[0].disposition.value == "rejected"
            assert receipt.items[0].retryable is True
            assert temp_csv.read_bytes() == original
            assert list(temp_csv.parent.glob(f".{temp_csv.stem}.*.tmp.csv")) == []
        finally:
            pool.close()

    def test_partial_result_preserves_existing_output_columns(self, tmp_path):
        from src.data.excel import ExcelTaskPool

        csv_path = tmp_path / "partial.csv"
        pd.DataFrame(
            [
                {
                    "input_text": "question",
                    "output_result": "keep-me",
                    "output_summary": None,
                }
            ]
        ).to_csv(csv_path, index=False)
        pool = ExcelTaskPool(
            input_path=csv_path,
            output_path=csv_path,
            columns_to_extract=["input_text"],
            columns_to_write={
                "result": "output_result",
                "summary": "output_summary",
            },
            engine_type="pandas",
        )
        try:
            receipt = pool.update_task_results(
                "partial-result",
                {0: {"summary": "new-summary"}},
            )
            row = pd.read_csv(csv_path).iloc[0]
            assert receipt.committed_ids == (0,)
            assert row["output_result"] == "keep-me"
            assert row["output_summary"] == "new-summary"
        finally:
            pool.close()

    def test_excel_unicode_failure_does_not_fallback_or_acknowledge(self, tmp_path):
        from src.data.excel import ExcelTaskPool

        excel_path = tmp_path / "target.xlsx"
        pd.DataFrame([{"input_text": "question", "output_result": None}]).to_excel(
            excel_path,
            index=False,
        )
        pool = ExcelTaskPool(
            input_path=excel_path,
            output_path=excel_path,
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            engine_type="pandas",
        )
        encoding_error = UnicodeEncodeError("utf-8", "x", 0, 1, "invalid")
        writes = []

        def fail_only_configured_excel(_df, destination, *, csv):
            writes.append((destination, csv))
            if not csv:
                raise encoding_error

        try:
            with mock.patch.object(
                pool,
                "_atomic_write",
                side_effect=fail_only_configured_excel,
            ):
                receipt = pool.update_task_results(
                    "unicode-failed",
                    {0: {"result": "not-durable"}},
                )
            assert receipt.items[0].disposition.value == "rejected"
            assert writes == [(excel_path, False)]
            assert not excel_path.with_suffix(".csv").exists()
            assert pool.engine.is_empty(
                pool.engine.get_row(pool.df, 0)["output_result"]
            )
        finally:
            pool.close()

    def test_directory_fsync_failure_after_replace_is_indeterminate(self, temp_csv):
        from src.data.contracts import CommitDisposition
        from src.data.excel import ExcelTaskPool

        pool = ExcelTaskPool(
            input_path=temp_csv,
            output_path=temp_csv,
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            engine_type="pandas",
        )
        try:
            with mock.patch.object(
                pool,
                "_fsync_parent_directory",
                side_effect=OSError("directory fsync failed"),
            ):
                receipt = pool.update_task_results(
                    "directory-fsync",
                    {0: {"result": "possibly durable"}},
                )

            assert receipt.items[0].disposition == CommitDisposition.INDETERMINATE
            assert pd.read_csv(temp_csv).iloc[0]["output_result"] == "possibly durable"
        finally:
            pool.close()

    def test_reconciliation_reads_target_file(self, temp_csv):
        from src.data.contracts import CommitDisposition
        from src.data.excel import ExcelTaskPool

        pool = ExcelTaskPool(
            input_path=temp_csv,
            output_path=temp_csv,
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            engine_type="pandas",
        )
        results = {0: {"result": "read-from-disk"}}
        try:
            with mock.patch.object(
                pool,
                "_read_output_file",
                side_effect=IOError("readback unavailable"),
            ):
                uncertain = pool.update_task_results("write-readback", results)
            assert uncertain.items[0].disposition == CommitDisposition.INDETERMINATE

            pool.df = pool.engine.set_value(
                pool.df,
                0,
                "output_result",
                "memory-only-wrong-value",
            )
            reconciled = pool.reconcile_task_results("reconcile-file", results)
            assert reconciled.committed_ids == (0,)

            rejected = pool.reconcile_task_results(
                "reconcile-mismatch",
                {0: {"result": "different-value"}},
            )
            assert rejected.items[0].disposition == CommitDisposition.REJECTED
        finally:
            pool.close()

    def test_receipt_covers_no_fields_and_missing_row(self, temp_csv):
        from src.data.contracts import CommitDisposition
        from src.data.excel import ExcelTaskPool

        pool = ExcelTaskPool(
            input_path=temp_csv,
            output_path=temp_csv,
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            engine_type="pandas",
        )
        try:
            receipt = pool.update_task_results(
                "preflight",
                {0: {"unmapped": "value"}, 999: {"result": "missing"}},
            )
            assert receipt.submitted_ids == (0, 999)
            assert [item.record_id for item in receipt.items] == [0, 999]
            assert all(
                item.disposition == CommitDisposition.REJECTED for item in receipt.items
            )
        finally:
            pool.close()

    def test_set_value_failure_does_not_block_other_records(self, temp_csv):
        from src.data.contracts import CommitDisposition
        from src.data.excel import ExcelTaskPool

        pool = ExcelTaskPool(
            input_path=temp_csv,
            output_path=temp_csv,
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            engine_type="pandas",
        )
        original_set_value = pool.engine.set_value

        def selective_failure(df, idx, column, value):
            if idx == 0:
                raise ValueError("cannot set row zero")
            return original_set_value(df, idx, column, value)

        try:
            with mock.patch.object(
                pool.engine,
                "set_value",
                side_effect=selective_failure,
            ):
                receipt = pool.update_task_results(
                    "set-value-failure",
                    {
                        0: {"result": "rejected"},
                        1: {"result": "committed"},
                    },
                )
            assert [item.disposition for item in receipt.items] == [
                CommitDisposition.REJECTED,
                CommitDisposition.COMMITTED,
            ]
            persisted = pd.read_csv(temp_csv)
            assert pd.isna(persisted.iloc[0]["output_result"])
            assert persisted.iloc[1]["output_result"] == "committed"
        finally:
            pool.close()


@pytest.mark.parametrize("engine_type", ["pandas", "polars"])
@pytest.mark.parametrize("suffix", ["csv", "xlsx"])
def test_file_writeback_is_equivalent_across_engines_and_formats(
    tmp_path,
    engine_type,
    suffix,
):
    from src.data.excel import ExcelTaskPool
    from src.data.engines import POLARS_AVAILABLE

    if engine_type == "polars" and not POLARS_AVAILABLE:
        pytest.skip("polars is unavailable")

    path = tmp_path / f"equivalent.{suffix}"
    source = pd.DataFrame(
        [
            {
                "input_text": "question",
                "output_result": None,
                "output_summary": "keep-me",
            }
        ]
    )
    if suffix == "csv":
        source.to_csv(path, index=False)
    else:
        source.to_excel(path, index=False)

    pool = ExcelTaskPool(
        input_path=path,
        output_path=path,
        columns_to_extract=["input_text"],
        columns_to_write={
            "result": "output_result",
            "summary": "output_summary",
        },
        engine_type=engine_type,
    )
    try:
        results = {0: {"result": "written"}}
        receipt = pool.update_task_results(f"{engine_type}-{suffix}", results)
        assert receipt.committed_ids == (0,)
        assert pool.reconcile_task_results("reconcile", results).committed_ids == (0,)

        if suffix == "csv":
            persisted = pd.read_csv(path)
        else:
            persisted = pd.read_excel(path)
        assert persisted.iloc[0]["output_result"] == "written"
        assert persisted.iloc[0]["output_summary"] == "keep-me"
    finally:
        pool.close()


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
        from src.data.factory import create_task_pool

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
        from src.data.factory import create_task_pool

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
        from src.data.excel import ExcelTaskPool

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
        from src.data.excel import ExcelTaskPool

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
        from src.data.excel import ExcelTaskPool

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
