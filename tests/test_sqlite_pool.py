"""
SQLite 数据源任务池单元测试

测试 src/data/sqlite.py 的 SQLiteTaskPool 类功能，包括：
- SQLiteConnectionManager 连接管理
- 线程本地连接隔离
- WAL 模式配置
- 任务读取与批量写入
"""

import pytest
import sqlite3
import time
from unittest.mock import MagicMock
from src.data.sqlite import SQLiteTaskPool, SQLiteConnectionManager


class TestSQLiteConnectionManager:
    """SQLite 连接管理器测试"""

    def test_set_db_path(self):
        """测试设置数据库路径"""
        SQLiteConnectionManager.set_db_path("/test/path.db")
        assert SQLiteConnectionManager._db_path == "/test/path.db"

    def test_get_connection_without_path_raises(self):
        """测试未设置路径时获取连接抛出异常"""
        # 重置状态
        SQLiteConnectionManager._db_path = None
        SQLiteConnectionManager._thread_local = type("local", (), {"conn": None})()

        with pytest.raises(ValueError, match="数据库路径未设置"):
            SQLiteConnectionManager.get_connection()

    def test_close_connection(self):
        """测试关闭连接"""
        # 模拟已存在连接
        mock_conn = MagicMock()
        SQLiteConnectionManager._thread_local.conn = mock_conn

        SQLiteConnectionManager.close_connection()

        mock_conn.close.assert_called_once()
        assert SQLiteConnectionManager._thread_local.conn is None


class TestSQLiteTaskPool:
    """SQLite 任务池测试"""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """创建临时 SQLite 数据库"""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # 创建测试表
        cursor.execute(
            """
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY,
                input_text TEXT,
                context TEXT,
                output_result TEXT,
                output_summary TEXT
            )
        """
        )

        # 插入测试数据
        test_data = [
            (1, "输入文本1", "上下文1", None, None),
            (2, "输入文本2", "上下文2", None, None),
            (3, "输入文本3", "上下文3", "已处理", "摘要"),
            (4, "", "上下文4", None, None),  # 空输入
            (5, "输入文本5", "", None, None),  # 空上下文
        ]
        cursor.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?)", test_data)
        conn.commit()
        conn.close()

        yield db_path

        # 清理
        # 确保连接彻底关闭，避免 Windows 文件锁
        try:
            SQLiteConnectionManager.close_connection()
        finally:
            if db_path.exists():
                for _ in range(5):
                    try:
                        db_path.unlink()
                        break
                    except PermissionError:
                        SQLiteConnectionManager.close_connection()
                        time.sleep(0.1)

    @pytest.fixture
    def task_pool(self, temp_db):
        """创建任务池实例"""
        pool = SQLiteTaskPool(
            db_path=temp_db,
            table_name="tasks",
            columns_to_extract=["input_text", "context"],
            columns_to_write={"result": "output_result", "summary": "output_summary"},
            require_all_input_fields=True,
        )
        yield pool
        pool.close()

    def test_initialization(self, task_pool):
        """测试初始化"""
        assert task_pool.table_name == "tasks"
        assert task_pool.columns_to_extract == ["input_text", "context"]
        assert "output_result" in task_pool.write_colnames
        assert "output_summary" in task_pool.write_colnames

    def test_initialization_file_not_found(self, tmp_path):
        """测试数据库文件不存在时抛出异常"""
        with pytest.raises(FileNotFoundError, match="数据库文件不存在"):
            SQLiteTaskPool(
                db_path=tmp_path / "nonexistent.db",
                table_name="tasks",
                columns_to_extract=["input_text"],
                columns_to_write={"result": "output"},
            )

    def test_initialization_table_not_found(self, temp_db):
        """测试表不存在时抛出异常"""
        try:
            with pytest.raises(ValueError, match="不存在"):
                SQLiteTaskPool(
                    db_path=temp_db,
                    table_name="nonexistent_table",
                    columns_to_extract=["input_text"],
                    columns_to_write={"result": "output"},
                )
        finally:
            # 确保关闭连接，否则 Windows 下 fixture 清理文件会报 PermissionError
            SQLiteConnectionManager.close_connection()

    def test_initialization_rejects_invalid_identifier(self, temp_db):
        """测试非法标识符会被拒绝"""
        with pytest.raises(ValueError, match="非法标识符"):
            SQLiteTaskPool(
                db_path=temp_db,
                table_name="tasks;drop",
                columns_to_extract=["input_text"],
                columns_to_write={"result": "output_result"},
            )

    def test_get_total_task_count(self, task_pool):
        """测试获取未处理任务数"""
        # require_all_input_fields=True，只有 id=1,2 满足条件
        count = task_pool.get_total_task_count()
        assert count == 2  # id=1, id=2 (输入列都非空且输出列为空)

    def test_get_processed_task_count(self, task_pool):
        """测试获取已处理任务数"""
        count = task_pool.get_processed_task_count()
        assert count == 1  # id=3 已处理

    def test_get_id_boundaries(self, task_pool):
        """测试获取 ID 边界"""
        min_id, max_id = task_pool.get_id_boundaries()
        assert min_id == 1
        assert max_id == 5

    def test_initialize_shard(self, task_pool):
        """测试初始化分片"""
        loaded = task_pool.initialize_shard(0, 1, 3)
        assert loaded == 2  # id=1, id=2 (未处理且输入有效)

        # 验证任务队列
        assert task_pool.has_tasks()
        assert task_pool.get_remaining_count() == 2

    def test_get_task_batch(self, task_pool):
        """测试获取任务批次"""
        task_pool.initialize_shard(0, 1, 5)
        batch = task_pool.get_task_batch(1)

        assert len(batch) == 1
        task_id, data = batch[0]
        assert task_id in [1, 2]
        assert "input_text" in data
        assert "context" in data

    def test_update_task_results(self, task_pool, temp_db):
        """测试更新任务结果"""
        # 准备结果
        results = {
            1: {"result": "处理结果1", "summary": "摘要1"},
            2: {"result": "处理结果2", "summary": "摘要2"},
        }

        task_pool.update_task_results(results)

        # 验证数据库
        conn = sqlite3.connect(str(temp_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE id IN (1, 2)")
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            if row["id"] == 1:
                assert row["output_result"] == "处理结果1"
                assert row["output_summary"] == "摘要1"
            elif row["id"] == 2:
                assert row["output_result"] == "处理结果2"
                assert row["output_summary"] == "摘要2"

    def test_update_task_results_skip_error(self, task_pool):
        """测试跳过错误结果"""
        results = {
            1: {"result": "成功", "summary": "OK"},
            2: {"_error": "API Error", "result": "失败"},  # 有错误标记
        }

        task_pool.update_task_results(results)

        # 不应该抛出异常，错误记录被跳过

    def test_reload_task_data(self, task_pool):
        """测试重载任务数据"""
        data = task_pool.reload_task_data(1)

        assert data is not None
        assert data["input_text"] == "输入文本1"
        assert data["context"] == "上下文1"

    def test_reload_task_data_not_found(self, task_pool):
        """测试重载不存在的记录"""
        data = task_pool.reload_task_data(999)
        assert data is None

    def test_sample_unprocessed_rows(self, task_pool):
        """测试采样未处理行"""
        samples = task_pool.sample_unprocessed_rows(10)

        assert len(samples) == 2
        assert "input_text" in samples[0]

    def test_sample_processed_rows(self, task_pool):
        """测试采样已处理行"""
        samples = task_pool.sample_processed_rows(10)

        assert len(samples) == 1
        assert "output_result" in samples[0]

    def test_fetch_all_rows(self, task_pool):
        """测试获取所有行"""
        rows = task_pool.fetch_all_rows(["input_text"])

        assert len(rows) == 5  # 所有行

    def test_fetch_all_processed_rows(self, task_pool):
        """测试获取所有已处理行"""
        rows = task_pool.fetch_all_processed_rows(["output_result"])

        assert len(rows) == 1
        assert rows[0]["output_result"] == "已处理"


class TestSQLiteTaskPoolWithRequireAny:
    """测试 require_all_input_fields=False 的情况"""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """创建临时数据库"""
        db_path = tmp_path / "test_any.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY,
                input_text TEXT,
                context TEXT,
                output_result TEXT
            )
        """
        )

        test_data = [
            (1, "有文本", "", None),  # 只有 input_text
            (2, "", "有上下文", None),  # 只有 context
            (3, "", "", None),  # 都为空
        ]
        cursor.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?)", test_data)
        conn.commit()
        conn.close()

        yield db_path

        # 清理，避免 Windows 下文件锁导致 tmp_path 回收失败
        try:
            SQLiteConnectionManager.close_connection()
        finally:
            if db_path.exists():
                for _ in range(5):
                    try:
                        db_path.unlink()
                        break
                    except PermissionError:
                        SQLiteConnectionManager.close_connection()
                        time.sleep(0.1)

    def test_require_any_input_field(self, temp_db):
        """测试 require_all_input_fields=False"""
        pool = SQLiteTaskPool(
            db_path=temp_db,
            table_name="tasks",
            columns_to_extract=["input_text", "context"],
            columns_to_write={"result": "output_result"},
            require_all_input_fields=False,  # 任一非空即可
        )

        count = pool.get_total_task_count()
        assert count == 2  # id=1 和 id=2 (至少有一个输入列非空)

        pool.close()
