"""
飞书数据源测试

测试飞书多维表格（Bitable）和电子表格（Sheet）TaskPool 的核心功能，
包括工厂函数注册、配置验证、快照读取、ID 映射、写回等。

由于飞书 API 需要真实的 app_id/app_secret，以下测试通过 Mock 隔离网络调用。
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ==================== 工厂集成测试 ====================


class TestFeishuFactoryIntegration:
    """飞书数据源工厂集成测试"""

    def test_feishu_available_flag(self):
        """测试飞书可用性标志"""
        from src.data.factory import FEISHU_AVAILABLE

        assert isinstance(FEISHU_AVAILABLE, bool)
        # aiohttp 已在 requirements.txt 中，应该可用
        assert FEISHU_AVAILABLE is True

    def test_create_feishu_bitable_missing_config(self):
        """测试缺少飞书多维表格配置字段"""
        from src.data.factory import create_task_pool

        config = {
            "datasource": {"type": "feishu_bitable"},
            "feishu": {
                "app_id": "cli_test",
                # 缺少 app_secret, app_token, table_id
            },
        }

        with pytest.raises(ValueError, match="app_secret"):
            create_task_pool(
                config=config,
                columns_to_extract=["question"],
                columns_to_write={"answer": "ai_answer"},
            )

    def test_create_feishu_sheet_missing_config(self):
        """测试缺少飞书电子表格配置字段"""
        from src.data.factory import create_task_pool

        config = {
            "datasource": {"type": "feishu_sheet"},
            "feishu": {
                "app_id": "cli_test",
                "app_secret": "secret",
                # 缺少 spreadsheet_token, sheet_id
            },
        }

        with pytest.raises(ValueError, match="spreadsheet_token"):
            create_task_pool(
                config=config,
                columns_to_extract=["question"],
                columns_to_write={"answer": "ai_answer"},
            )

    def test_unsupported_type_includes_feishu(self):
        """测试不支持类型的错误消息包含飞书选项"""
        from src.data.factory import create_task_pool

        config = {"datasource": {"type": "unknown_db"}}

        with pytest.raises(ValueError, match="feishu_bitable"):
            create_task_pool(
                config=config,
                columns_to_extract=["question"],
                columns_to_write={"answer": "ai_answer"},
            )


# ==================== 客户端测试 ====================


class TestFeishuClient:
    """飞书客户端基础测试"""

    def test_client_import(self):
        """测试客户端模块可导入"""
        from src.data.feishu.client import (
            FeishuClient,
            FeishuAPIError,
            FeishuRateLimitError,
            FeishuPermissionError,
        )

        assert callable(FeishuClient)
        assert issubclass(FeishuRateLimitError, FeishuAPIError)
        assert issubclass(FeishuPermissionError, FeishuAPIError)

    def test_client_init(self):
        """测试客户端初始化"""
        from src.data.feishu.client import FeishuClient

        client = FeishuClient(
            app_id="cli_test",
            app_secret="test_secret",
            max_retries=5,
            qps_limit=10,
        )

        assert client.app_id == "cli_test"
        assert client.app_secret == "test_secret"
        assert client.max_retries == 5
        assert client.qps_limit == 10
        assert client._token is None

    def test_client_error_hierarchy(self):
        """测试错误类层级"""
        from src.data.feishu.client import (
            FeishuAPIError,
            FeishuRateLimitError,
            FeishuPermissionError,
        )

        err = FeishuAPIError(code=100, msg="test error", url="https://test.com")
        assert err.code == 100
        assert "test error" in str(err)

        rate_err = FeishuRateLimitError(msg="rate limit", retry_after=2.0)
        assert rate_err.code == 429
        assert rate_err.retry_after == 2.0
        assert isinstance(rate_err, FeishuAPIError)

        perm_err = FeishuPermissionError(code=99991403, msg="no permission")
        assert isinstance(perm_err, FeishuAPIError)


# ==================== Bitable TaskPool 测试 ====================


class TestFeishuBitableTaskPool:
    """飞书多维表格任务池测试"""

    @pytest.fixture
    def mock_snapshot(self):
        """模拟飞书多维表格记录快照"""
        return [
            {
                "record_id": "recAAABBB001",
                "fields": {
                    "question": "什么是 AI？",
                    "context": "人工智能概述",
                    "ai_answer": "",
                    "ai_category": "",
                },
            },
            {
                "record_id": "recAAABBB002",
                "fields": {
                    "question": "什么是机器学习？",
                    "context": "ML 基础",
                    "ai_answer": "",
                    "ai_category": "",
                },
            },
            {
                "record_id": "recAAABBB003",
                "fields": {
                    "question": "什么是深度学习？",
                    "context": "DL 概述",
                    "ai_answer": "已有回答",
                    "ai_category": "technical",
                },
            },
        ]

    @pytest.fixture
    def bitable_pool(self, mock_snapshot):
        """创建带模拟快照的 Bitable 任务池"""
        from src.data.feishu.bitable import FeishuBitableTaskPool

        pool = FeishuBitableTaskPool(
            app_id="cli_test",
            app_secret="test_secret",
            app_token="basc_test",
            table_id="tbl_test",
            columns_to_extract=["question", "context"],
            columns_to_write={"answer": "ai_answer", "category": "ai_category"},
        )

        # 注入模拟快照，跳过实际 API 调用
        pool._snapshot = mock_snapshot
        pool._snapshot_loaded = True
        pool._id_map = {i: rec["record_id"] for i, rec in enumerate(mock_snapshot)}
        pool._reverse_map = {rec["record_id"]: i for i, rec in enumerate(mock_snapshot)}

        return pool

    def test_get_total_task_count(self, bitable_pool):
        """测试未处理任务计数"""
        # 3 条记录，其中 1 条已处理
        assert bitable_pool.get_total_task_count() == 2

    def test_get_processed_task_count(self, bitable_pool):
        """测试已处理任务计数"""
        assert bitable_pool.get_processed_task_count() == 1

    def test_get_id_boundaries(self, bitable_pool):
        """测试 ID 边界"""
        min_id, max_id = bitable_pool.get_id_boundaries()
        assert min_id == 0
        assert max_id == 2  # 3 条记录，索引 0-2

    def test_initialize_shard(self, bitable_pool):
        """测试分片初始化"""
        loaded = bitable_pool.initialize_shard(0, 0, 2)
        assert loaded == 2  # 只加载未处理的 2 条

    def test_get_task_batch(self, bitable_pool):
        """测试获取任务批次"""
        bitable_pool.initialize_shard(0, 0, 2)
        batch = bitable_pool.get_task_batch(10)
        assert len(batch) == 2

        # 验证任务数据
        task_id, data = batch[0]
        assert task_id == 0
        assert data["question"] == "什么是 AI？"
        assert data["context"] == "人工智能概述"

    def test_reload_task_data(self, bitable_pool):
        """测试重新加载任务数据"""
        data = bitable_pool.reload_task_data(0)
        assert data is not None
        assert data["question"] == "什么是 AI？"

        # 超出范围返回 None
        assert bitable_pool.reload_task_data(100) is None

    def test_id_mapping(self, bitable_pool):
        """测试 ID 映射表"""
        assert bitable_pool._id_map[0] == "recAAABBB001"
        assert bitable_pool._id_map[1] == "recAAABBB002"
        assert bitable_pool._reverse_map["recAAABBB001"] == 0

    def test_field_not_empty(self):
        """测试字段非空判断"""
        from src.data.feishu.bitable import FeishuBitableTaskPool

        assert FeishuBitableTaskPool._field_not_empty("hello") is True
        assert FeishuBitableTaskPool._field_not_empty("") is False
        assert FeishuBitableTaskPool._field_not_empty("  ") is False
        assert FeishuBitableTaskPool._field_not_empty(None) is False
        assert FeishuBitableTaskPool._field_not_empty([]) is False
        assert FeishuBitableTaskPool._field_not_empty(["a"]) is True
        assert FeishuBitableTaskPool._field_not_empty(42) is True

    def test_convert_field_value(self):
        """测试字段值转换"""
        from src.data.feishu.bitable import FeishuBitableTaskPool

        assert FeishuBitableTaskPool._convert_field_value(None) == ""
        assert FeishuBitableTaskPool._convert_field_value("hello") == "hello"
        assert FeishuBitableTaskPool._convert_field_value(42) == "42"
        assert FeishuBitableTaskPool._convert_field_value(3.14) == "3.14"
        assert FeishuBitableTaskPool._convert_field_value([{"text": "A"}, {"text": "B"}]) == "A, B"
        assert FeishuBitableTaskPool._convert_field_value({"text": "link"}) == "link"

    def test_sample_unprocessed_rows(self, bitable_pool):
        """测试采样未处理行"""
        samples = bitable_pool.sample_unprocessed_rows(10)
        assert len(samples) == 2
        assert "question" in samples[0]

    def test_sample_processed_rows(self, bitable_pool):
        """测试采样已处理行"""
        samples = bitable_pool.sample_processed_rows(10)
        assert len(samples) == 1


# ==================== Sheet TaskPool 测试 ====================


class TestFeishuSheetTaskPool:
    """飞书电子表格任务池测试"""

    @pytest.fixture
    def mock_sheet_data(self):
        """模拟电子表格数据"""
        return {
            "header": ["question", "context", "ai_answer", "ai_category"],
            "rows": [
                ["什么是 AI？", "人工智能概述", "", ""],
                ["什么是 ML？", "机器学习基础", "", ""],
                ["什么是 DL？", "深度学习概述", "已有回答", "technical"],
            ],
        }

    @pytest.fixture
    def sheet_pool(self, mock_sheet_data):
        """创建带模拟快照的 Sheet 任务池"""
        from src.data.feishu.sheet import FeishuSheetTaskPool

        pool = FeishuSheetTaskPool(
            app_id="cli_test",
            app_secret="test_secret",
            spreadsheet_token="shtcn_test",
            sheet_id="Sheet1",
            columns_to_extract=["question", "context"],
            columns_to_write={"answer": "ai_answer", "category": "ai_category"},
        )

        # 注入模拟快照
        pool._header_row = mock_sheet_data["header"]
        pool._data_rows = mock_sheet_data["rows"]
        pool._col_name_to_index = {
            name: idx for idx, name in enumerate(mock_sheet_data["header"])
        }
        pool._snapshot_loaded = True

        return pool

    def test_get_total_task_count(self, sheet_pool):
        """测试未处理任务计数"""
        assert sheet_pool.get_total_task_count() == 2

    def test_get_processed_task_count(self, sheet_pool):
        """测试已处理任务计数"""
        assert sheet_pool.get_processed_task_count() == 1

    def test_get_id_boundaries(self, sheet_pool):
        """测试 ID 边界"""
        min_id, max_id = sheet_pool.get_id_boundaries()
        assert min_id == 0
        assert max_id == 2

    def test_initialize_shard(self, sheet_pool):
        """测试分片初始化"""
        loaded = sheet_pool.initialize_shard(0, 0, 2)
        assert loaded == 2

    def test_get_task_batch(self, sheet_pool):
        """测试获取任务批次"""
        sheet_pool.initialize_shard(0, 0, 2)
        batch = sheet_pool.get_task_batch(10)
        assert len(batch) == 2

        task_id, data = batch[0]
        assert task_id == 0
        assert data["question"] == "什么是 AI？"

    def test_reload_task_data(self, sheet_pool):
        """测试重新加载任务数据"""
        data = sheet_pool.reload_task_data(0)
        assert data is not None
        assert data["question"] == "什么是 AI？"

        assert sheet_pool.reload_task_data(100) is None

    def test_sample_unprocessed_rows(self, sheet_pool):
        """测试采样未处理行"""
        samples = sheet_pool.sample_unprocessed_rows(10)
        assert len(samples) == 2

    def test_sample_processed_rows(self, sheet_pool):
        """测试采样已处理行"""
        samples = sheet_pool.sample_processed_rows(10)
        assert len(samples) == 1


class TestColIndexToLetter:
    """列号转换测试"""

    def test_single_letters(self):
        """测试单字母列号"""
        from src.data.feishu.sheet import _col_index_to_letter

        assert _col_index_to_letter(0) == "A"
        assert _col_index_to_letter(1) == "B"
        assert _col_index_to_letter(25) == "Z"

    def test_double_letters(self):
        """测试双字母列号"""
        from src.data.feishu.sheet import _col_index_to_letter

        assert _col_index_to_letter(26) == "AA"
        assert _col_index_to_letter(27) == "AB"
        assert _col_index_to_letter(51) == "AZ"
        assert _col_index_to_letter(52) == "BA"


# ==================== 队列操作测试 ====================


class TestFeishuQueueOperations:
    """飞书任务池队列操作测试（继承自 BaseTaskPool）"""

    @pytest.fixture
    def bitable_pool_with_tasks(self):
        """创建有任务的 Bitable 池"""
        from src.data.feishu.bitable import FeishuBitableTaskPool

        pool = FeishuBitableTaskPool(
            app_id="cli_test",
            app_secret="test_secret",
            app_token="basc_test",
            table_id="tbl_test",
            columns_to_extract=["q"],
            columns_to_write={"a": "answer"},
        )

        pool._snapshot_loaded = True
        pool._snapshot = []
        pool._id_map = {}
        pool._reverse_map = {}

        # 手动设置任务
        pool.tasks = [
            (0, {"q": "Q1"}),
            (1, {"q": "Q2"}),
        ]

        return pool

    def test_has_tasks(self, bitable_pool_with_tasks):
        """测试队列非空检查"""
        assert bitable_pool_with_tasks.has_tasks() is True

    def test_get_remaining_count(self, bitable_pool_with_tasks):
        """测试剩余任务数"""
        assert bitable_pool_with_tasks.get_remaining_count() == 2

    def test_add_task_to_front(self, bitable_pool_with_tasks):
        """测试任务放回队头"""
        bitable_pool_with_tasks.add_task_to_front(99, {"q": "urgent"})
        assert bitable_pool_with_tasks.tasks[0] == (99, {"q": "urgent"})
        assert bitable_pool_with_tasks.get_remaining_count() == 3

    def test_add_task_to_back(self, bitable_pool_with_tasks):
        """测试任务放回队尾"""
        bitable_pool_with_tasks.add_task_to_back(99, {"q": "delayed"})
        assert bitable_pool_with_tasks.tasks[-1] == (99, {"q": "delayed"})

    def test_clear_tasks(self, bitable_pool_with_tasks):
        """测试清空队列"""
        bitable_pool_with_tasks.clear_tasks()
        assert bitable_pool_with_tasks.has_tasks() is False
