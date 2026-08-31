"""Real MySQL/PostgreSQL contract tests enabled only by the CI service job."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Iterator

import pytest

from src.data.contracts import CommitDisposition

pytestmark = [pytest.mark.integration]
requires_live_databases = pytest.mark.skipif(
    os.getenv("DATAFLUX_DB_INTEGRATION") != "1",
    reason="set DATAFLUX_DB_INTEGRATION=1 with disposable databases",
)

TABLE_NAME = "dataflux_adapter_contract"
MYSQL_APP_USER = "dataflux_contract"
MYSQL_APP_PASSWORD = "dataflux_contract"


def _assert_receipt(receipt: Any, batch_id: str) -> None:
    assert receipt.batch_id == batch_id
    assert receipt.committed_ids == ("alpha-001", "alpha-002")
    assert all(
        item.disposition == CommitDisposition.COMMITTED for item in receipt.items
    )
    assert receipt.atomic is True
    assert receipt.succeeded is True
    assert datetime.fromisoformat(receipt.completed_at).tzinfo is not None


async def _assert_keyset_writeback_and_rollback(
    adapter: Any,
    *,
    reset_outputs: Any,
    read_outputs: Any,
    read_summaries: Any,
) -> None:
    first_page = await adapter.scan(None, 2)
    assert [record.record_id for record in first_page.records] == [
        "alpha-001",
        "alpha-002",
    ]
    assert all(isinstance(record.record_id, str) for record in first_page.records)
    assert first_page.next_cursor == {"last_id": "alpha-002"}

    second_page = await adapter.scan(first_page.next_cursor, 2)
    assert [record.record_id for record in second_page.records] == ["beta-001"]
    assert second_page.next_cursor is None

    receipt = await adapter.write_results(
        "successful-batch",
        {
            "alpha-001": {"result": "persisted-one"},
            "alpha-002": {"result": "persisted-two"},
        },
    )
    _assert_receipt(receipt, "successful-batch")
    assert read_outputs() == {
        "alpha-001": "persisted-one",
        "alpha-002": "persisted-two",
    }
    assert read_summaries() == {
        "alpha-001": "keep-one",
        "alpha-002": "keep-two",
    }

    reconciled = await adapter.reconcile_results(
        "reconcile-success",
        {
            "alpha-001": {"result": "persisted-one"},
            "alpha-002": {"result": "persisted-two"},
        },
    )
    assert reconciled.committed_ids == ("alpha-001", "alpha-002")

    reset_outputs()
    rolled_back = await adapter.write_results(
        "rollback-batch",
        {
            "alpha-001": {"result": "must-be-rolled-back"},
            "missing-string-id": {"result": "must-not-be-acknowledged"},
        },
    )
    assert [item.disposition for item in rolled_back.items] == [
        CommitDisposition.NOT_ATTEMPTED,
        CommitDisposition.REJECTED,
    ]
    assert read_outputs() == {"alpha-001": None, "alpha-002": None}
    assert read_summaries() == {
        "alpha-001": "keep-one",
        "alpha-002": "keep-two",
    }

    mismatched = await adapter.reconcile_results(
        "reconcile-mismatch",
        {
            "alpha-001": {"result": "must-be-rolled-back"},
            "alpha-002": {"result": "persisted-two"},
        },
    )
    assert all(
        item.disposition == CommitDisposition.REJECTED for item in mismatched.items
    )


@pytest.mark.asyncio
async def test_sqlite_string_keyset_atomic_rollback_and_receipt(tmp_path) -> None:
    from src.data.sqlite import SQLiteTaskPool

    db_path = tmp_path / "adapter-contract.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        f'CREATE TABLE "{TABLE_NAME}" ('
        "id TEXT PRIMARY KEY, "
        "input_text TEXT NOT NULL, "
        "output_result TEXT NULL, "
        "output_summary TEXT NULL"
        ")"
    )
    connection.executemany(
        f'INSERT INTO "{TABLE_NAME}" (id, input_text, output_summary) '
        "VALUES (?, ?, ?)",
        [
            ("alpha-001", "first", "keep-one"),
            ("alpha-002", "second", "keep-two"),
            ("beta-001", "third", None),
        ],
    )
    connection.commit()
    connection.close()
    adapter = SQLiteTaskPool(
        db_path=db_path,
        columns_to_extract=["input_text"],
        columns_to_write={
            "result": "output_result",
            "summary": "output_summary",
        },
        table_name=TABLE_NAME,
    )

    def connect() -> sqlite3.Connection:
        return sqlite3.connect(db_path)

    def reset_outputs() -> None:
        connection = connect()
        connection.execute(f'UPDATE "{TABLE_NAME}" SET output_result = NULL')
        connection.commit()
        connection.close()

    def read_column(column: str) -> dict[str, str | None]:
        connection = connect()
        rows = connection.execute(
            f'SELECT id, "{column}" FROM "{TABLE_NAME}" '
            "WHERE id IN (?, ?) ORDER BY id",
            ("alpha-001", "alpha-002"),
        ).fetchall()
        connection.close()
        return dict(rows)

    try:
        await _assert_keyset_writeback_and_rollback(
            adapter,
            reset_outputs=reset_outputs,
            read_outputs=lambda: read_column("output_result"),
            read_summaries=lambda: read_column("output_summary"),
        )
    finally:
        adapter.close()


@pytest.fixture
def mysql_contract() -> Iterator[dict[str, Any]]:
    import mysql.connector

    from src.data.mysql import MySQLConnectionPoolManager, MySQLTaskPool

    host = os.environ["DATAFLUX_MYSQL_HOST"]
    port = int(os.getenv("DATAFLUX_MYSQL_PORT", "3306"))
    database = os.environ["DATAFLUX_MYSQL_DATABASE"]
    root_password = os.environ["DATAFLUX_MYSQL_ROOT_PASSWORD"]

    admin = mysql.connector.connect(
        host=host,
        port=port,
        user="root",
        password=root_password,
        database=database,
    )
    cursor = admin.cursor()
    cursor.execute(
        f"CREATE USER IF NOT EXISTS '{MYSQL_APP_USER}'@'%' "
        f"IDENTIFIED WITH mysql_native_password BY '{MYSQL_APP_PASSWORD}'"
    )
    cursor.execute(
        f"ALTER USER '{MYSQL_APP_USER}'@'%' "
        f"IDENTIFIED WITH mysql_native_password BY '{MYSQL_APP_PASSWORD}'"
    )
    cursor.execute(f"GRANT ALL PRIVILEGES ON `{database}`.* TO '{MYSQL_APP_USER}'@'%'")
    cursor.execute(f"DROP TABLE IF EXISTS `{TABLE_NAME}`")
    cursor.execute(
        f"CREATE TABLE `{TABLE_NAME}` ("
        "id VARCHAR(64) PRIMARY KEY, "
        "input_text VARCHAR(255) NOT NULL, "
        "output_result VARCHAR(255) NULL, "
        "output_summary VARCHAR(255) NULL"
        ")"
    )
    cursor.executemany(
        f"INSERT INTO `{TABLE_NAME}` (id, input_text, output_summary) "
        "VALUES (%s, %s, %s)",
        [
            ("alpha-001", "first", "keep-one"),
            ("alpha-002", "second", "keep-two"),
            ("beta-001", "third", None),
        ],
    )
    admin.commit()
    cursor.close()
    admin.close()

    MySQLConnectionPoolManager.close_pool()
    adapter = MySQLTaskPool(
        connection_config={
            "host": host,
            "port": port,
            "user": MYSQL_APP_USER,
            "password": MYSQL_APP_PASSWORD,
            "database": database,
        },
        columns_to_extract=["input_text"],
        columns_to_write={
            "result": "output_result",
            "summary": "output_summary",
        },
        table_name=TABLE_NAME,
        pool_size=2,
    )

    def connect() -> Any:
        return mysql.connector.connect(
            host=host,
            port=port,
            user="root",
            password=root_password,
            database=database,
        )

    try:
        yield {"adapter": adapter, "connect": connect}
    finally:
        adapter.close()
        connection = connect()
        cleanup_cursor = connection.cursor()
        cleanup_cursor.execute(f"DROP TABLE IF EXISTS `{TABLE_NAME}`")
        connection.commit()
        cleanup_cursor.close()
        connection.close()


@pytest.mark.asyncio
@requires_live_databases
async def test_mysql_string_keyset_atomic_rollback_and_receipt(mysql_contract) -> None:
    adapter = mysql_contract["adapter"]

    def reset_outputs() -> None:
        connection = mysql_contract["connect"]()
        cursor = connection.cursor()
        cursor.execute(f"UPDATE `{TABLE_NAME}` SET output_result = NULL")
        connection.commit()
        cursor.close()
        connection.close()

    def read_outputs() -> dict[str, str | None]:
        connection = mysql_contract["connect"]()
        cursor = connection.cursor()
        cursor.execute(
            f"SELECT id, output_result FROM `{TABLE_NAME}` "
            "WHERE id IN (%s, %s) ORDER BY id",
            ("alpha-001", "alpha-002"),
        )
        result = dict(cursor.fetchall())
        cursor.close()
        connection.close()
        return result

    def read_summaries() -> dict[str, str | None]:
        connection = mysql_contract["connect"]()
        cursor = connection.cursor()
        cursor.execute(
            f"SELECT id, output_summary FROM `{TABLE_NAME}` "
            "WHERE id IN (%s, %s) ORDER BY id",
            ("alpha-001", "alpha-002"),
        )
        result = dict(cursor.fetchall())
        cursor.close()
        connection.close()
        return result

    await _assert_keyset_writeback_and_rollback(
        adapter,
        reset_outputs=reset_outputs,
        read_outputs=read_outputs,
        read_summaries=read_summaries,
    )


@pytest.fixture
def postgresql_contract() -> Iterator[dict[str, Any]]:
    import psycopg2

    from src.data.postgresql import (
        PostgreSQLConnectionPoolManager,
        PostgreSQLTaskPool,
    )

    config = {
        "host": os.environ["DATAFLUX_POSTGRES_HOST"],
        "port": int(os.getenv("DATAFLUX_POSTGRES_PORT", "5432")),
        "user": os.environ["DATAFLUX_POSTGRES_USER"],
        "password": os.environ["DATAFLUX_POSTGRES_PASSWORD"],
        "database": os.environ["DATAFLUX_POSTGRES_DATABASE"],
    }

    def connect() -> Any:
        return psycopg2.connect(**config)

    connection = connect()
    connection.autocommit = True
    cursor = connection.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS "{TABLE_NAME}"')
    cursor.execute(
        f'CREATE TABLE "{TABLE_NAME}" ('
        "id TEXT PRIMARY KEY, "
        "input_text TEXT NOT NULL, "
        "output_result TEXT NULL, "
        "output_summary TEXT NULL"
        ")"
    )
    cursor.executemany(
        f'INSERT INTO "{TABLE_NAME}" (id, input_text, output_summary) '
        "VALUES (%s, %s, %s)",
        [
            ("alpha-001", "first", "keep-one"),
            ("alpha-002", "second", "keep-two"),
            ("beta-001", "third", None),
        ],
    )
    cursor.close()
    connection.close()

    PostgreSQLConnectionPoolManager.close_pool()
    adapter = PostgreSQLTaskPool(
        connection_config=config,
        columns_to_extract=["input_text"],
        columns_to_write={
            "result": "output_result",
            "summary": "output_summary",
        },
        table_name=TABLE_NAME,
        schema_name="public",
        pool_size=2,
    )

    try:
        yield {"adapter": adapter, "connect": connect}
    finally:
        adapter.close()
        connection = connect()
        connection.autocommit = True
        cleanup_cursor = connection.cursor()
        cleanup_cursor.execute(f'DROP TABLE IF EXISTS "{TABLE_NAME}"')
        cleanup_cursor.close()
        connection.close()


@pytest.mark.asyncio
@requires_live_databases
async def test_postgresql_string_keyset_atomic_rollback_and_receipt(
    postgresql_contract,
) -> None:
    adapter = postgresql_contract["adapter"]

    def reset_outputs() -> None:
        connection = postgresql_contract["connect"]()
        cursor = connection.cursor()
        cursor.execute(f'UPDATE "{TABLE_NAME}" SET output_result = NULL')
        connection.commit()
        cursor.close()
        connection.close()

    def read_outputs() -> dict[str, str | None]:
        connection = postgresql_contract["connect"]()
        cursor = connection.cursor()
        cursor.execute(
            f'SELECT id, output_result FROM "{TABLE_NAME}" '
            "WHERE id IN (%s, %s) ORDER BY id",
            ("alpha-001", "alpha-002"),
        )
        result = dict(cursor.fetchall())
        cursor.close()
        connection.close()
        return result

    def read_summaries() -> dict[str, str | None]:
        connection = postgresql_contract["connect"]()
        cursor = connection.cursor()
        cursor.execute(
            f'SELECT id, output_summary FROM "{TABLE_NAME}" '
            "WHERE id IN (%s, %s) ORDER BY id",
            ("alpha-001", "alpha-002"),
        )
        result = dict(cursor.fetchall())
        cursor.close()
        connection.close()
        return result

    await _assert_keyset_writeback_and_rollback(
        adapter,
        reset_outputs=reset_outputs,
        read_outputs=read_outputs,
        read_summaries=read_summaries,
    )
