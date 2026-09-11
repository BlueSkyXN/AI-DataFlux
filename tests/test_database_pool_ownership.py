from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(params=["mysql", "postgresql"])
def database_adapter(request, monkeypatch):
    from src.data import mysql, postgresql

    if request.param == "mysql":
        module, available = mysql, mysql.MYSQL_AVAILABLE
        manager, adapter = mysql.MySQLConnectionPoolManager, mysql.MySQLTaskPool
        driver, factory = mysql.pooling, "MySQLConnectionPool"
    else:
        module, available = postgresql, postgresql.POSTGRESQL_AVAILABLE
        manager = postgresql.PostgreSQLConnectionPoolManager
        adapter = postgresql.PostgreSQLTaskPool
        driver, factory = postgresql.pool, "ThreadedConnectionPool"
    if not available:
        pytest.skip(f"{request.param} driver is unavailable")

    class DriverPool:
        def __init__(self, **kwargs):
            self.settings = kwargs
            self.pool_name = kwargs.get("pool_name", "test-pool")
            self.closed = False

        def getconn(self):
            if self.closed:
                raise RuntimeError("connection pool is closed")
            return SimpleNamespace(database=self.settings["database"])

        get_connection = getconn

        def closeall(self):
            self.closed = True

        _remove_connections = closeall

    manager.close_pool()
    constructor = MagicMock(side_effect=DriverPool)
    monkeypatch.setattr(driver, factory, constructor)
    created = []

    def create(**overrides):
        config = {
            "host": "example.invalid",
            "port": 3306 if module is mysql else 5432,
            "user": "test",
            "password": "test-only",
            "database": "db_a",
            **overrides,
        }
        task_pool = adapter(config, ["input"], {"answer": "result"}, "tasks")
        created.append(task_pool)
        return task_pool

    yield create, constructor
    for task_pool in created:
        task_pool.close()
    manager.close_pool()


@pytest.mark.parametrize(
    "override",
    [
        {"host": "other.invalid"},
        {"port": 12345},
        {"user": "other-user"},
        {"password": "other-test-only"},
        {"database": "db_b"},
    ],
)
def test_task_pools_isolate_connection_settings(database_adapter, override):
    create, constructor = database_adapter
    first = create()
    second = create(**override)

    assert first.pool is not second.pool
    assert constructor.call_count == 2
    for field, value in override.items():
        assert second.pool.settings[field] == value
    first.close()
    assert second._get_connection().database == override.get("database", "db_a")


def test_closing_one_shared_pool_owner_keeps_other_job_alive(database_adapter):
    create, constructor = database_adapter
    first, second = create(), create()
    shared = second.pool
    assert first.pool is shared
    assert constructor.call_count == 1

    first.close()
    first.close()
    assert second._get_connection().database == "db_a"
    assert not shared.closed

    second.close()
    assert shared.closed
    replacement = create()
    assert replacement.pool is not shared
    assert constructor.call_count == 2


def test_concurrent_pool_owners_release_only_after_last_close(database_adapter):
    create, constructor = database_adapter
    with ThreadPoolExecutor(max_workers=4) as executor:
        task_pools = list(executor.map(lambda _: create(), range(8)))
        shared = task_pools[0].pool
        assert all(task_pool.pool is shared for task_pool in task_pools)
        assert constructor.call_count == 1
        list(executor.map(lambda task_pool: task_pool.close(), task_pools[:-1]))
        assert not shared.closed
        assert task_pools[-1]._get_connection().database == "db_a"
        task_pools[-1].close()
        assert shared.closed


@pytest.mark.asyncio
async def test_database_reload_propagates_query_failure(database_adapter, monkeypatch):
    create, _ = database_adapter
    task_pool = create()
    monkeypatch.setattr(
        task_pool,
        "execute_with_connection",
        MagicMock(side_effect=ConnectionError("temporary query failure")),
    )
    with pytest.raises(ConnectionError, match="temporary query failure"):
        await task_pool.reload([1])


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("operation", ["scan", "reload", "write", "reconcile"])
async def test_cancelled_database_operation_drains_before_pool_release(
    database_adapter, monkeypatch, shared, operation
):
    from src.core.processor import UniversalAIProcessor

    create, _ = database_adapter
    task_pool = create()
    other = create() if shared else None
    driver = task_pool.pool
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    returned = threading.Event()
    connection = MagicMock()
    connection.cursor.return_value.rowcount = 1
    connection.cursor.return_value.fetchone.return_value = {
        "id": 1,
        "input": "original",
        "result": "done",
    }
    connection.cursor.return_value.fetchall.return_value = [
        {"id": 1, "input": "original"}
    ]

    def execute(*_args):
        entered.set()
        assert release.wait(5)
        assert not driver.closed, "in-flight connection must remain usable"

    connection.cursor.return_value.execute.side_effect = execute
    connection.close.side_effect = returned.set
    driver.putconn = MagicMock(side_effect=lambda _connection: returned.set())
    monkeypatch.setattr(driver, "getconn", lambda: connection)
    monkeypatch.setattr(driver, "get_connection", lambda: connection)
    method_name = {
        "scan": "execute_with_connection",
        "reload": "execute_with_connection",
        "write": "update_task_results",
        "reconcile": "reconcile_task_results",
    }[operation]
    method = getattr(task_pool, method_name)

    def observed_operation(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(task_pool, method_name, observed_operation)
    close_entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    close = task_pool.close

    def observed_close():
        loop.call_soon_threadsafe(close_entered.set)
        close()

    monkeypatch.setattr(task_pool, "close", observed_close)
    processor = object.__new__(UniversalAIProcessor)
    processor.task_pool = task_pool
    processor._close_task = None
    invoke = {
        "scan": lambda: task_pool.scan(None, 2),
        "reload": lambda: task_pool.reload([1]),
        "write": lambda: task_pool.write_results("audit", {1: {"answer": "done"}}),
        "reconcile": lambda: task_pool.reconcile_results(
            "audit", {1: {"answer": "done"}}
        ),
    }[operation]
    running = asyncio.create_task(invoke())
    cleanup = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        cleanup = asyncio.create_task(processor.aclose())
        await asyncio.wait_for(close_entered.wait(), 2)
        await asyncio.sleep(0.02)
        assert not cleanup.done(), "cleanup must wait for the database operation"
        assert task_pool.pool is driver
        assert not driver.closed
        release.set()
        await asyncio.wait_for(cleanup, 2)
        assert finished.is_set()
        assert returned.is_set(), "borrowed connection must be returned before close"
        assert task_pool.pool is None
        assert driver.closed is (not shared)
        if other is not None:
            assert other.pool is driver
            other.close()
            assert driver.closed
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 2)
        await asyncio.gather(
            running, *([cleanup] if cleanup is not None else []), return_exceptions=True
        )
        monkeypatch.setattr(task_pool, "close", close)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during_cleanup", [False, True])
async def test_processor_cleanup_does_not_block_on_other_pool_creation(
    database_adapter, monkeypatch, cancel_during_cleanup
):
    from src.core.processor import UniversalAIProcessor
    from src.core.scheduler import ShardedTaskManager

    create, constructor = database_adapter
    first = create()
    entered, release = threading.Event(), threading.Event()
    close = first.close
    factory = constructor.side_effect

    def slow_factory(**kwargs):
        if kwargs["database"] == "slow_db":
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test pool creation timed out")
        return factory(**kwargs)

    constructor.side_effect = slow_factory
    second = asyncio.create_task(asyncio.to_thread(create, database="slow_db"))
    processing = None
    watchdog = threading.Timer(2, release.set)
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        processor = object.__new__(UniversalAIProcessor)
        processor.task_pool = first
        processor.task_manager = ShardedTaskManager(first)
        processor.source_max_attempts = 1
        processor._close_task = None
        monkeypatch.setattr(first, "get_total_task_count", lambda: 0)
        close_started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def observed_close():
            loop.call_soon_threadsafe(close_started.set)
            close()

        monkeypatch.setattr(first, "close", observed_close)
        watchdog.start()
        processing = asyncio.create_task(processor.process_shard_async_continuous())
        await asyncio.wait_for(close_started.wait(), timeout=3)
        assert (
            not release.is_set()
        ), "cleanup blocked the event loop until watchdog release"
        assert not processing.done()
        if cancel_during_cleanup:
            for _ in range(2):
                processing.cancel()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not processing.done(), "cancel must drain the running cleanup"
        release.set()
        if cancel_during_cleanup:
            with pytest.raises(asyncio.CancelledError):
                await processing
        else:
            assert await processing is True
        second_pool = await second
        assert second_pool._get_connection().database == "slow_db"
    finally:
        release.set()
        watchdog.cancel()
        if watchdog.ident is not None:
            watchdog.join()
        await asyncio.gather(
            second,
            *([processing] if processing is not None else []),
            return_exceptions=True,
        )
        monkeypatch.setattr(first, "close", close)
