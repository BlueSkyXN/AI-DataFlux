"""
数据源任务池工厂

根据配置创建对应的任务池实例。
支持的数据源类型:
- mysql: MySQL 数据库
- postgresql: PostgreSQL 数据库
- sqlite: SQLite 数据库
- excel: Excel 文件（含 CSV 支持）
- csv: CSV 文件（独立实现，支持高级特性）
"""

import logging
from typing import Any

from .base import BaseTaskPool


# 可用性标志
MYSQL_AVAILABLE = False
POSTGRESQL_AVAILABLE = False
SQLITE_AVAILABLE = True  # SQLite 是 Python 标准库，始终可用
EXCEL_ENABLED = False

try:
    import mysql.connector  # noqa: F401

    MYSQL_AVAILABLE = True
except ImportError:
    pass

try:
    import psycopg2  # noqa: F401

    POSTGRESQL_AVAILABLE = True
except ImportError:
    pass

try:
    import pandas  # noqa: F401
    import openpyxl  # noqa: F401

    EXCEL_ENABLED = True
except ImportError:
    pass


def create_task_pool(
    config: dict[str, Any],
    columns_to_extract: list[str],
    columns_to_write: dict[str, str],
) -> BaseTaskPool:
    """
    根据配置创建数据源任务池

    Args:
        config: 完整配置字典
        columns_to_extract: 需要提取的列名列表
        columns_to_write: 写回映射 {别名: 实际列名}

    Returns:
        任务池实例 (MySQLTaskPool, PostgreSQLTaskPool, SQLiteTaskPool, ExcelTaskPool 或 CSVTaskPool)

    Raises:
        ValueError: 配置无效或数据源类型不支持
        ImportError: 所需库不可用
    """
    datasource_config = config.get("datasource", {})
    datasource_type = datasource_config.get("type", "excel").lower()

    # 获取公共配置
    require_all_input_fields = datasource_config.get("require_all_input_fields", True)
    concurrency_config = datasource_config.get("concurrency", {})

    # 高性能引擎配置
    engine_type = datasource_config.get("engine", "auto")
    excel_reader = datasource_config.get("excel_reader", "auto")
    excel_writer = datasource_config.get("excel_writer", "auto")

    logging.info(f"正在创建数据源任务池，类型: {datasource_type}")

    if datasource_type == "mysql":
        return _create_mysql_pool(
            config,
            columns_to_extract,
            columns_to_write,
            require_all_input_fields,
            concurrency_config,
        )

    elif datasource_type == "postgresql":
        return _create_postgresql_pool(
            config,
            columns_to_extract,
            columns_to_write,
            require_all_input_fields,
            concurrency_config,
        )

    elif datasource_type == "sqlite":
        return _create_sqlite_pool(
            config,
            columns_to_extract,
            columns_to_write,
            require_all_input_fields,
        )

    elif datasource_type == "excel":
        return _create_excel_pool(
            config,
            columns_to_extract,
            columns_to_write,
            require_all_input_fields,
            concurrency_config,
            engine_type,
            excel_reader,
            excel_writer,
        )

    elif datasource_type == "csv":
        return _create_csv_pool(
            config,
            columns_to_extract,
            columns_to_write,
            require_all_input_fields,
            concurrency_config,
            engine_type,
        )

    else:
        raise ValueError(
            f"不支持的数据源类型: {datasource_type}，"
            f"可选: mysql, postgresql, sqlite, excel, csv"
        )


def _create_mysql_pool(
    config: dict[str, Any],
    columns_to_extract: list[str],
    columns_to_write: dict[str, str],
    require_all_input_fields: bool,
    concurrency_config: dict[str, Any],
) -> BaseTaskPool:
    """创建 MySQL 任务池"""
    if not MYSQL_AVAILABLE:
        raise ImportError(
            "MySQL Connector 不可用，请安装: pip install mysql-connector-python"
        )

    from .mysql import MySQLTaskPool

    mysql_config = config.get("mysql", {})

    # 验证必需配置
    required_keys = ["host", "user", "password", "database", "table_name"]
    missing_keys = [k for k in required_keys if not mysql_config.get(k)]
    if missing_keys:
        raise ValueError(f"MySQL 配置缺少必需字段: {missing_keys}")

    return MySQLTaskPool(
        connection_config={
            "host": mysql_config["host"],
            "port": mysql_config.get("port", 3306),
            "user": mysql_config["user"],
            "password": mysql_config["password"],
            "database": mysql_config["database"],
        },
        columns_to_extract=columns_to_extract,
        columns_to_write=columns_to_write,
        table_name=mysql_config["table_name"],
        # 连接池大小: 优先使用 mysql.pool_size，否则使用 batch_size 的 1/10，最小为 5
        pool_size=mysql_config.get(
            "pool_size", max(5, concurrency_config.get("batch_size", 100) // 10)
        ),
        require_all_input_fields=require_all_input_fields,
    )


def _create_excel_pool(
    config: dict[str, Any],
    columns_to_extract: list[str],
    columns_to_write: dict[str, str],
    require_all_input_fields: bool,
    concurrency_config: dict[str, Any],
    engine_type: str,
    excel_reader: str = "auto",
    excel_writer: str = "auto",
) -> BaseTaskPool:
    """创建 Excel 任务池"""
    if not EXCEL_ENABLED:
        raise ImportError(
            "pandas 或 openpyxl 不可用，请安装: pip install pandas openpyxl"
        )

    from .excel import ExcelTaskPool

    excel_config = config.get("excel", {})

    # 获取文件路径
    input_path = excel_config.get("input_path")
    output_path = excel_config.get("output_path")

    if not input_path:
        raise ValueError("Excel 配置缺少 'input_path'")

    # 如果未指定输出路径，使用输入路径
    if not output_path:
        output_path = input_path
        logging.warning(f"未指定 Excel 输出路径，将使用输入路径: {output_path}")

    return ExcelTaskPool(
        input_path=input_path,
        output_path=output_path,
        columns_to_extract=columns_to_extract,
        columns_to_write=columns_to_write,
        save_interval=concurrency_config.get("save_interval", 300),
        require_all_input_fields=require_all_input_fields,
        engine_type=engine_type,
        excel_reader=excel_reader,
        excel_writer=excel_writer,
    )


def _create_postgresql_pool(
    config: dict[str, Any],
    columns_to_extract: list[str],
    columns_to_write: dict[str, str],
    require_all_input_fields: bool,
    concurrency_config: dict[str, Any],
) -> BaseTaskPool:
    """创建 PostgreSQL 任务池"""
    if not POSTGRESQL_AVAILABLE:
        raise ImportError(
            "psycopg2 不可用，请安装: pip install psycopg2-binary"
        )

    from .postgresql import PostgreSQLTaskPool

    pg_config = config.get("postgresql", {})

    # 验证必需配置
    required_keys = ["host", "user", "password", "database", "table_name"]
    missing_keys = [k for k in required_keys if not pg_config.get(k)]
    if missing_keys:
        raise ValueError(f"PostgreSQL 配置缺少必需字段: {missing_keys}")

    return PostgreSQLTaskPool(
        connection_config={
            "host": pg_config["host"],
            "port": pg_config.get("port", 5432),
            "user": pg_config["user"],
            "password": pg_config["password"],
            "database": pg_config["database"],
        },
        columns_to_extract=columns_to_extract,
        columns_to_write=columns_to_write,
        table_name=pg_config["table_name"],
        schema_name=pg_config.get("schema_name", "public"),
        # 连接池大小: 优先使用 postgresql.pool_size，否则使用 batch_size 的 1/10，最小为 5
        pool_size=pg_config.get(
            "pool_size", max(5, concurrency_config.get("batch_size", 100) // 10)
        ),
        require_all_input_fields=require_all_input_fields,
    )


def _create_sqlite_pool(
    config: dict[str, Any],
    columns_to_extract: list[str],
    columns_to_write: dict[str, str],
    require_all_input_fields: bool,
) -> BaseTaskPool:
    """创建 SQLite 任务池"""
    from .sqlite import SQLiteTaskPool

    sqlite_config = config.get("sqlite", {})

    db_path = sqlite_config.get("db_path")
    table_name = sqlite_config.get("table_name")

    if not db_path:
        raise ValueError("SQLite 配置缺少 'db_path'")
    if not table_name:
        raise ValueError("SQLite 配置缺少 'table_name'")

    return SQLiteTaskPool(
        db_path=db_path,
        table_name=table_name,
        columns_to_extract=columns_to_extract,
        columns_to_write=columns_to_write,
        require_all_input_fields=require_all_input_fields,
    )


def _create_csv_pool(
    config: dict[str, Any],
    columns_to_extract: list[str],
    columns_to_write: dict[str, str],
    require_all_input_fields: bool,
    concurrency_config: dict[str, Any],
    engine_type: str,
) -> BaseTaskPool:
    """
    创建 CSV 任务池

    CSV 数据源复用 ExcelTaskPool 实现，因为 BaseEngine 已内置 CSV 读写支持。
    这种方式可以:
    - 复用向量化过滤、批量更新等优化
    - 自动支持 Pandas/Polars 引擎切换
    - 零额外代码实现

    如需 CSV 特有功能（自定义分隔符、特殊编码、压缩支持），
    可以创建独立的 CSVTaskPool 类。
    """
    if not EXCEL_ENABLED:
        raise ImportError(
            "pandas 或 openpyxl 不可用，请安装: pip install pandas openpyxl"
        )

    from .excel import ExcelTaskPool

    csv_config = config.get("csv", {})

    # 获取文件路径
    input_path = csv_config.get("input_path")
    output_path = csv_config.get("output_path")

    if not input_path:
        raise ValueError("CSV 配置缺少 'input_path'")

    # 如果未指定输出路径，使用输入路径
    if not output_path:
        output_path = input_path
        logging.warning(f"未指定 CSV 输出路径，将使用输入路径: {output_path}")

    # 使用 ExcelTaskPool 处理 CSV（BaseEngine 支持 CSV 读写）
    return ExcelTaskPool(
        input_path=input_path,
        output_path=output_path,
        columns_to_extract=columns_to_extract,
        columns_to_write=columns_to_write,
        save_interval=concurrency_config.get("save_interval", 300),
        require_all_input_fields=require_all_input_fields,
        engine_type=engine_type,
        excel_reader="auto",  # CSV 自动检测
        excel_writer="auto",
    )


__all__ = [
    "create_task_pool",
    "MYSQL_AVAILABLE",
    "POSTGRESQL_AVAILABLE",
    "SQLITE_AVAILABLE",
    "EXCEL_ENABLED",
]
