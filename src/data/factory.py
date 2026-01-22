"""
数据源任务池工厂

根据配置创建对应的任务池实例。
"""

import logging
from typing import Any

from .base import BaseTaskPool


# 可用性标志
MYSQL_AVAILABLE = False
EXCEL_ENABLED = False

try:
    import mysql.connector  # noqa: F401

    MYSQL_AVAILABLE = True
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
        任务池实例 (MySQLTaskPool 或 ExcelTaskPool)

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

    else:
        raise ValueError(f"不支持的数据源类型: {datasource_type}，可选: mysql, excel")


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
        pool_size=concurrency_config.get("max_workers", 5),
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


__all__ = [
    "create_task_pool",
    "MYSQL_AVAILABLE",
    "EXCEL_ENABLED",
]
