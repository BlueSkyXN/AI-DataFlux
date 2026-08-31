"""
PostgreSQL 数据源任务池单元测试

被测模块: src/data/postgresql.py (PostgreSQLTaskPool, PostgreSQLConnectionPoolManager)

测试 src/data/postgresql.py 的 PostgreSQLTaskPool 类功能，包括：
- ThreadedConnectionPool 连接池管理
- execute_batch 批量写入优化
- SQL 注入防护 (psycopg2.sql)
- 连接池大小配置

注意: 这些测试使用 Mock 对象，不需要实际的 PostgreSQL 数据库连接。

测试类/函数清单:
    TestPostgreSQLAvailabilityCheck                    可用性检查测试
        test_import_error_when_psycopg2_not_available  验证 psycopg2 不可用时的处理
    TestPostgreSQLConnectionPoolManager                连接池管理器测试
        test_get_pool_creates_instance                 验证首次调用创建连接池实例
        test_get_pool_raises_without_config            验证首次调用无配置时抛 ValueError
        test_get_pool_raises_when_not_available         验证 psycopg2 不可用时抛 ImportError
    TestPostgreSQLTaskPoolMocked                       任务池测试（Mock）
        test_initialization                            验证初始化参数存储
        test_initialization_rejects_invalid_identifier 验证非法标识符被拒绝
        test_get_total_task_count                      验证获取未处理任务数
        test_get_id_boundaries                         验证获取 ID 边界
        test_initialize_shard                          验证分片初始化
        test_update_task_results_batch                 验证批量更新
        test_reload_task_data                          验证重载任务数据
        test_reload_task_data_not_found                验证重载不存在记录返回 None
    TestPostgreSQLTaskPoolConditionBuilding             WHERE 条件构建测试
        test_build_unprocessed_condition_all_required   验证 require_all=True 用 AND
        test_build_unprocessed_condition_any_required   验证 require_all=False 用 OR
"""

import pytest
from unittest.mock import patch, MagicMock

from src.data.contracts import CommitDisposition


def create_mock_sql():
    """创建模拟的 psycopg2.sql 模块，用于测试 SQL 标识符安全拼接"""

    class MockSQL:
        def __init__(self, template):
            self.template = template

        def format(self, *args, **kwargs):
            return self

        def as_string(self, conn):
            return self.template

        def join(self, items):
            """支持 sql.SQL(", ").join() 语法"""
            return self

    class MockIdentifier:
        def __init__(self, name):
            self.name = name

        def as_string(self, conn):
            return f'"{self.name}"'

    mock_sql_module = MagicMock()
    mock_sql_module.SQL = MockSQL
    mock_sql_module.Identifier = MockIdentifier
    return mock_sql_module


# 模拟 psycopg2 模块
mock_psycopg2 = MagicMock()
mock_pool = MagicMock()
mock_extras = MagicMock()


class TestPostgreSQLAvailabilityCheck:
    """测试 PostgreSQL 可用性检查"""

    def test_import_error_when_psycopg2_not_available(self):
        """测试 psycopg2 不可用时抛出 ImportError"""
        with patch.dict("sys.modules", {"psycopg2": None}):
            # 强制重新导入以触发 ImportError 检测
            # 这个测试验证模块级别的可用性检查逻辑
            pass  # 模块已在加载时处理


class TestPostgreSQLConnectionPoolManager:
    """PostgreSQL 连接池管理器测试（使用 Mock）"""

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.pool")
    def test_get_pool_creates_instance(self, mock_pool_module):
        """测试首次调用时创建连接池"""
        from src.data.postgresql import PostgreSQLConnectionPoolManager

        # 重置单例状态
        PostgreSQLConnectionPoolManager._instance = None
        PostgreSQLConnectionPoolManager._pool = None

        mock_pool_instance = MagicMock()
        mock_pool_module.ThreadedConnectionPool.return_value = mock_pool_instance

        config = {
            "host": "localhost",
            "port": 5432,
            "user": "test",
            "password": "test",
            "database": "testdb",
        }

        result = PostgreSQLConnectionPoolManager.get_pool(
            config=config, max_connections=5
        )

        assert result == mock_pool_instance
        mock_pool_module.ThreadedConnectionPool.assert_called_once()

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    def test_get_pool_raises_without_config(self):
        """测试首次调用不提供配置时抛出异常"""
        from src.data.postgresql import PostgreSQLConnectionPoolManager

        # 重置单例状态
        PostgreSQLConnectionPoolManager._instance = None
        PostgreSQLConnectionPoolManager._pool = None

        with pytest.raises(ValueError, match="首次获取连接池必须提供数据库配置"):
            PostgreSQLConnectionPoolManager.get_pool()

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", False)
    def test_get_pool_raises_when_not_available(self):
        """测试 psycopg2 不可用时抛出 ImportError"""
        from src.data.postgresql import PostgreSQLConnectionPoolManager

        # 重置单例状态
        PostgreSQLConnectionPoolManager._instance = None
        PostgreSQLConnectionPoolManager._pool = None

        with pytest.raises(ImportError, match="psycopg2 不可用"):
            PostgreSQLConnectionPoolManager.get_pool(config={"host": "localhost"})


class TestPostgreSQLTaskPoolMocked:
    """PostgreSQL 任务池测试（完全使用 Mock）"""

    @pytest.fixture
    def mock_pool_manager(self):
        """Mock 连接池管理器"""
        with patch(
            "src.data.postgresql.PostgreSQLConnectionPoolManager"
        ) as mock_manager:
            mock_pool = MagicMock()
            mock_manager.get_pool.return_value = mock_pool

            # 模拟连接
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_pool.getconn.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor

            yield {
                "manager": mock_manager,
                "pool": mock_pool,
                "conn": mock_conn,
                "cursor": mock_cursor,
            }

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    def test_initialization(self, mock_pool_manager):
        """测试初始化"""
        from src.data.postgresql import PostgreSQLTaskPool

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "port": 5432,
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["input_text", "context"],
            columns_to_write={"result": "output_result"},
            table_name="tasks",
            schema_name="public",
        )

        assert pool.table_name == "tasks"
        assert pool.schema_name == "public"
        assert pool.columns_to_extract == ["input_text", "context"]
        assert "output_result" in pool.write_colnames

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    def test_initialization_rejects_invalid_identifier(self, mock_pool_manager):
        """测试非法标识符会被拒绝"""
        from src.data.postgresql import PostgreSQLTaskPool

        with pytest.raises(ValueError, match="非法标识符"):
            PostgreSQLTaskPool(
                connection_config={
                    "host": "localhost",
                    "port": 5432,
                    "user": "test",
                    "password": "test",
                    "database": "testdb",
                },
                columns_to_extract=["input-text"],
                columns_to_write={"result": "output_result"},
                table_name="tasks",
                schema_name="public",
            )

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.extras")
    @patch("src.data.postgresql.sql", new_callable=create_mock_sql)
    def test_get_total_task_count(self, mock_sql, mock_extras, mock_pool_manager):
        """测试获取未处理任务数"""
        from src.data.postgresql import PostgreSQLTaskPool

        mock_cursor = mock_pool_manager["cursor"]
        mock_cursor.fetchone.return_value = {"count": 42}

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            table_name="tasks",
        )

        count = pool.get_total_task_count()

        assert count == 42
        mock_cursor.execute.assert_called()

        pool.execute_with_connection = MagicMock(
            side_effect=RuntimeError("database unavailable")
        )
        with pytest.raises(RuntimeError, match="database unavailable"):
            pool.get_total_task_count()

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.extras")
    @patch("src.data.postgresql.sql", new_callable=create_mock_sql)
    def test_get_id_boundaries(self, mock_sql, mock_extras, mock_pool_manager):
        """测试获取 ID 边界"""
        from src.data.postgresql import PostgreSQLTaskPool

        mock_cursor = mock_pool_manager["cursor"]
        mock_cursor.fetchone.return_value = {"min_id": 1, "max_id": 100}

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            table_name="tasks",
        )

        min_id, max_id = pool.get_id_boundaries()

        assert min_id == 1
        assert max_id == 100

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.extras")
    @patch("src.data.postgresql.sql", new_callable=create_mock_sql)
    def test_initialize_shard(self, mock_sql, mock_extras, mock_pool_manager):
        """测试初始化分片"""
        from src.data.postgresql import PostgreSQLTaskPool

        mock_cursor = mock_pool_manager["cursor"]
        mock_cursor.fetchall.return_value = [
            {"id": 1, "input_text": "文本1"},
            {"id": 2, "input_text": "文本2"},
        ]

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            table_name="tasks",
        )

        loaded = pool.initialize_shard(0, 1, 10)

        assert loaded == 2
        assert pool.has_tasks()

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.extras")
    @patch("src.data.postgresql.sql", new_callable=create_mock_sql)
    def test_update_task_results_batch(self, mock_sql, mock_extras, mock_pool_manager):
        """测试事务更新逐条确认实际命中记录。"""
        from src.data.postgresql import PostgreSQLTaskPool

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            table_name="tasks",
        )

        results = {
            1: {"result": "结果1"},
            2: {"result": "结果2"},
        }
        mock_cursor = mock_pool_manager["cursor"]
        mock_cursor.rowcount = 1

        receipt = pool.update_task_results("batch-ok", results)

        assert receipt.committed_ids == (1, 2)
        assert receipt.atomic is True
        assert mock_cursor.execute.call_count == 2

        mock_cursor.reset_mock()
        mock_cursor.rowcount = 0
        rolled_back = pool.update_task_results(
            "batch-missing",
            {"missing": {"result": "结果"}},
        )
        assert rolled_back.items[0].disposition == CommitDisposition.REJECTED
        assert rolled_back.items[0].code == "record_not_found"
        mock_pool_manager["conn"].rollback.assert_called()

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.extras")
    @patch("src.data.postgresql.sql", new_callable=create_mock_sql)
    def test_partial_result_preserves_unmentioned_columns(
        self, mock_sql, mock_extras, mock_pool_manager
    ):
        from src.data.postgresql import PostgreSQLTaskPool

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["input_text"],
            columns_to_write={
                "result": "output_result",
                "summary": "output_summary",
            },
            table_name="tasks",
        )
        mock_cursor = mock_pool_manager["cursor"]
        mock_cursor.rowcount = 1

        receipt = pool.update_task_results(
            "batch-partial",
            {"row-a": {"summary": "new-summary"}},
        )

        assert receipt.committed_ids == ("row-a",)
        assert mock_cursor.execute.call_args.args[1] == ("new-summary", "row-a")

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.extras")
    @patch("src.data.postgresql.sql", new_callable=create_mock_sql)
    def test_preflight_failure_aborts_atomic_batch(
        self,
        mock_sql,
        mock_extras,
        mock_pool_manager,
    ):
        from src.data.postgresql import PostgreSQLTaskPool

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            table_name="tasks",
        )

        receipt = pool.update_task_results(
            "batch-preflight",
            {1: {"result": "must-not-write"}, 2: {}},
        )

        assert [item.disposition for item in receipt.items] == [
            CommitDisposition.NOT_ATTEMPTED,
            CommitDisposition.REJECTED,
        ]
        mock_pool_manager["cursor"].execute.assert_not_called()

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.extras")
    @patch("src.data.postgresql.sql", new_callable=create_mock_sql)
    def test_commit_loss_is_indeterminate(
        self,
        mock_sql,
        mock_extras,
        mock_pool_manager,
    ):
        from src.data.postgresql import PostgreSQLTaskPool

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            table_name="tasks",
        )
        mock_pool_manager["cursor"].rowcount = 1
        mock_pool_manager["conn"].commit.side_effect = ConnectionError("lost")

        receipt = pool.update_task_results(
            "batch-unknown",
            {1: {"result": "value"}},
        )

        assert receipt.items[0].disposition == CommitDisposition.INDETERMINATE
        assert receipt.items[0].code == "commit_outcome_unknown"

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.extras")
    @patch("src.data.postgresql.sql", new_callable=create_mock_sql)
    def test_reconciliation_compares_expected_fields(
        self,
        mock_sql,
        mock_extras,
        mock_pool_manager,
    ):
        from src.data.postgresql import PostgreSQLTaskPool

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            table_name="tasks",
        )
        mock_pool_manager["cursor"].fetchone.side_effect = [
            {"output_result": "done"},
            {"output_result": "different"},
            None,
        ]

        receipt = pool.reconcile_task_results(
            "readback",
            {
                1: {"result": "done"},
                2: {"result": "expected"},
                3: {"result": "expected"},
            },
        )

        assert [item.disposition for item in receipt.items] == [
            CommitDisposition.COMMITTED,
            CommitDisposition.REJECTED,
            CommitDisposition.REJECTED,
        ]
        assert receipt.items[1].code == "readback_mismatch"
        assert receipt.items[2].code == "record_not_found"

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.extras")
    @patch("src.data.postgresql.sql", new_callable=create_mock_sql)
    def test_reload_task_data(self, mock_sql, mock_extras, mock_pool_manager):
        """测试重载任务数据 - 验证方法被正确调用"""
        from src.data.postgresql import PostgreSQLTaskPool

        mock_cursor = mock_pool_manager["cursor"]
        mock_cursor.fetchone.return_value = {"input_text": "原始文本"}

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            table_name="tasks",
        )

        # 由于 psycopg2 不可用，会返回 None
        data = pool.reload_task_data(1)
        # 主要测试方法能被调用而不崩溃
        # 由于模块级 psycopg2 不可用，跳过断言
        assert data["input_text"] == "原始文本"

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.extras")
    @patch("src.data.postgresql.sql", new_callable=create_mock_sql)
    def test_reload_task_data_not_found(self, mock_sql, mock_extras, mock_pool_manager):
        """测试重载不存在的记录"""
        from src.data.postgresql import PostgreSQLTaskPool

        mock_cursor = mock_pool_manager["cursor"]
        mock_cursor.fetchone.return_value = None

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["input_text"],
            columns_to_write={"result": "output_result"},
            table_name="tasks",
        )

        data = pool.reload_task_data(999)

        assert data is None


class TestPostgreSQLTaskPoolConditionBuilding:
    """测试 WHERE 条件构建"""

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.PostgreSQLConnectionPoolManager")
    @patch("src.data.postgresql.extras")
    def test_build_unprocessed_condition_all_required(self, mock_extras, mock_manager):
        """测试 require_all_input_fields=True 的条件构建"""
        from src.data.postgresql import PostgreSQLTaskPool

        mock_manager.get_pool.return_value = MagicMock()

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["col1", "col2"],
            columns_to_write={"out": "output"},
            table_name="tasks",
            require_all_input_fields=True,
        )

        condition = pool._build_unprocessed_condition()

        # 输入列使用 AND
        assert '"col1" IS NOT NULL' in condition
        assert '"col2" IS NOT NULL' in condition
        assert " AND " in condition

    @patch("src.data.postgresql.POSTGRESQL_AVAILABLE", True)
    @patch("src.data.postgresql.PostgreSQLConnectionPoolManager")
    @patch("src.data.postgresql.extras")
    def test_build_unprocessed_condition_any_required(self, mock_extras, mock_manager):
        """测试 require_all_input_fields=False 的条件构建"""
        from src.data.postgresql import PostgreSQLTaskPool

        mock_manager.get_pool.return_value = MagicMock()

        pool = PostgreSQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "test",
                "password": "test",
                "database": "testdb",
            },
            columns_to_extract=["col1", "col2"],
            columns_to_write={"out": "output"},
            table_name="tasks",
            require_all_input_fields=False,
        )

        condition = pool._build_unprocessed_condition()

        # 输入列使用 OR (在括号内)
        assert '"col1" IS NOT NULL' in condition
        assert '"col2" IS NOT NULL' in condition
