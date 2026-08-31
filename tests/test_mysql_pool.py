from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.data.base import BaseTaskPool
from src.data.mysql import MySQLConnectionPoolManager, MySQLTaskPool


class _Cursor:
    def __init__(self):
        self.fetchone_values = []
        self.fetchall_values = []
        self.executions = []
        self.rowcount = 1
        self.closed = False

    def execute(self, query, params=None):
        self.executions.append((str(query), params))

    def fetchone(self):
        return self.fetchone_values.pop(0) if self.fetchone_values else None

    def fetchall(self):
        return self.fetchall_values.pop(0) if self.fetchall_values else []

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, cursor):
        self.test_cursor = cursor
        self.commit = MagicMock()
        self.rollback = MagicMock()
        self.close = MagicMock()

    def cursor(self, **_kwargs):
        self.test_cursor.closed = False
        return self.test_cursor


@pytest.fixture
def mysql_pool():
    cursor = _Cursor()
    connection = _Connection(cursor)
    connection_pool = MagicMock()
    connection_pool.get_connection.return_value = connection
    with patch.object(
        MySQLConnectionPoolManager, "get_pool", return_value=connection_pool
    ):
        pool = MySQLTaskPool(
            connection_config={
                "host": "localhost",
                "user": "user",
                "password": "password",
                "database": "database",
            },
            columns_to_extract=["input_text"],
            columns_to_write={"answer": "output_text"},
            table_name="tasks",
        )
    pool._test_cursor = cursor
    pool._test_connection = connection
    yield pool


def test_mysql_connection_pool_manager_lifecycle(monkeypatch):
    from src.data import mysql as mysql_module

    fake_pool = MagicMock(pool_name="pool")
    constructor = MagicMock(return_value=fake_pool)
    monkeypatch.setattr(mysql_module, "MYSQL_AVAILABLE", True)
    monkeypatch.setattr(mysql_module.pooling, "MySQLConnectionPool", constructor)
    MySQLConnectionPoolManager._instance = None
    MySQLConnectionPoolManager._pool = None
    config = {
        "host": "localhost",
        "user": "user",
        "password": "password",
        "database": "database",
    }
    try:
        assert MySQLConnectionPoolManager.get_pool(config, pool_size=3) is fake_pool
        assert MySQLConnectionPoolManager.get_pool(config) is fake_pool
        assert constructor.call_args.kwargs["pool_size"] == 3
    finally:
        MySQLConnectionPoolManager.close_pool()
    assert MySQLConnectionPoolManager._pool is None


def test_mysql_connection_pool_manager_requires_config(monkeypatch):
    from src.data import mysql as mysql_module

    monkeypatch.setattr(mysql_module, "MYSQL_AVAILABLE", True)
    MySQLConnectionPoolManager._instance = None
    MySQLConnectionPoolManager._pool = None
    with pytest.raises(ValueError, match="必须提供"):
        MySQLConnectionPoolManager.get_pool()


def test_mysql_execute_with_connection_commit_rollback_and_cleanup(mysql_pool):
    cursor = mysql_pool._test_cursor
    connection = mysql_pool._test_connection

    assert (
        mysql_pool.execute_with_connection(lambda _conn, _cursor: "ok", is_write=True)
        == "ok"
    )
    connection.commit.assert_called_once()
    connection.close.assert_called()
    assert cursor.closed is True

    connection.rollback.reset_mock()
    with pytest.raises(ValueError, match="callback"):
        mysql_pool.execute_with_connection(
            lambda _conn, _cursor: (_ for _ in ()).throw(ValueError("callback")),
            is_write=True,
        )
    connection.rollback.assert_called_once()


def test_mysql_counts_boundaries_shards_reload_and_conditions(mysql_pool):
    cursor = mysql_pool._test_cursor
    cursor.fetchone_values = [
        {"count": 2},
        {"count": 3},
        {"min_id": "a", "max_id": "z"},
        {"input_text": "reloaded"},
        None,
    ]
    cursor.fetchall_values = [
        [
            {"id": "a", "input_text": "one"},
            {"id": None, "input_text": "skip"},
            {"id": "b", "input_text": "two"},
        ]
    ]

    assert mysql_pool.get_total_task_count() == 2
    assert mysql_pool.get_processed_task_count() == 3
    assert mysql_pool.get_id_boundaries() == ("a", "z")
    assert mysql_pool.initialize_shard(0, "a", "z") == 2
    assert mysql_pool.get_task_batch(1)[0][0] == "a"
    assert mysql_pool.reload_task_data("a") == {"input_text": "reloaded"}
    assert mysql_pool.reload_task_data("missing") is None
    assert "input_text" in mysql_pool._build_unprocessed_condition()
    assert "output_text" in mysql_pool._build_processed_condition()
    assert mysql_pool.capabilities.atomic_batch is True


@pytest.mark.asyncio
async def test_mysql_keyset_scan_and_write_receipts(mysql_pool):
    cursor = mysql_pool._test_cursor
    cursor.fetchall_values = [
        [
            {"id": "a", "input_text": "one"},
            {"id": "b", "input_text": "two"},
        ],
        [{"id": "c", "input_text": "three"}],
    ]

    first = await mysql_pool.scan(None, 2)
    second = await mysql_pool.scan(first.next_cursor, 2)
    assert [item.record_id for item in first.records] == ["a", "b"]
    assert first.next_cursor == {"last_id": "b"}
    assert [item.record_id for item in second.records] == ["c"]

    receipt = mysql_pool.update_task_results(
        {"a": {"answer": "done"}, "b": {"_error": "skip"}}
    )
    assert receipt.persisted_ids == ("a",)
    assert receipt.atomic is True
    assert mysql_pool.update_task_results({}).persisted_ids == ()
    assert mysql_pool.update_task_results({"b": {"_error": "skip"}}).persisted_ids == ()

    cursor.rowcount = 0
    with pytest.raises(RuntimeError, match="更新行数异常"):
        mysql_pool.update_task_results({"missing": {"answer": "x"}})


def test_mysql_sampling_and_full_scans(mysql_pool):
    cursor = mysql_pool._test_cursor
    cursor.fetchall_values = [
        [{"input_text": "one"}],
        [{"output_text": "done"}],
        [{"input_text": "one"}, {"input_text": "two"}],
        [{"output_text": "done"}],
    ]

    assert mysql_pool.sample_unprocessed_rows(1) == [{"input_text": "one"}]
    assert mysql_pool.sample_processed_rows(1) == [{"output_text": "done"}]
    assert mysql_pool.fetch_all_rows(["input_text"]) == [
        {"input_text": "one"},
        {"input_text": "two"},
    ]
    assert mysql_pool.fetch_all_processed_rows(["output_text"]) == [
        {"output_text": "done"}
    ]


@pytest.mark.parametrize(
    "method,args,expected",
    [
        ("get_id_boundaries", (), (0, 0)),
        ("initialize_shard", (0, 1, 2), 0),
        ("reload_task_data", (1,), None),
        ("sample_unprocessed_rows", (1,), []),
        ("sample_processed_rows", (1,), []),
        ("fetch_all_rows", (["input_text"],), []),
        ("fetch_all_processed_rows", (["output_text"],), []),
    ],
)
def test_mysql_read_helpers_degrade_on_query_error(mysql_pool, method, args, expected):
    mysql_pool.execute_with_connection = MagicMock(side_effect=RuntimeError("query"))
    assert getattr(mysql_pool, method)(*args) == expected


@pytest.mark.parametrize("method", ["get_total_task_count", "get_processed_task_count"])
def test_mysql_count_errors_are_not_reported_as_zero(mysql_pool, method):
    mysql_pool.execute_with_connection = MagicMock(side_effect=RuntimeError("query"))
    with pytest.raises(RuntimeError, match="query"):
        getattr(mysql_pool, method)()


def test_mysql_base_contract_helpers_without_columns():
    pool = object.__new__(MySQLTaskPool)
    BaseTaskPool.__init__(pool, [], {}, require_all_input_fields=False)
    pool.table_name = "tasks"
    pool.select_columns = ["id"]
    pool.write_colnames = []
    pool.write_aliases = []
    assert pool._build_unprocessed_condition() == "(1=1) AND (1=0)"
    assert pool._build_processed_condition() == "1=1"
