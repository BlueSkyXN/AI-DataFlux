"""
数据源任务池工厂模块

本模块提供任务池的工厂函数，根据配置动态创建对应类型的数据源实例。
采用工厂模式解耦任务池的创建和使用，便于扩展新的数据源类型。

支持的数据源类型:
    - mysql: MySQL 关系型数据库
        - 使用 mysql-connector-python 连接
        - 支持连接池复用
        - 适合大规模数据处理

    - postgresql: PostgreSQL 数据库
        - 使用 psycopg2 连接
        - 支持 ThreadedConnectionPool
        - 使用 execute_batch() 批量更新
        - 适合企业级高性能场景

    - sqlite: SQLite 数据库
        - Python 标准库，无需额外安装
        - 线程级连接管理
        - 支持 WAL 模式
        - 适合开发测试和中小规模数据

    - excel: Excel 电子表格文件
        - 支持 .xlsx 和 .xls 格式
        - 可选高性能读取器 (calamine)
        - 可选高性能写入器 (xlsxwriter)
        - 支持 Pandas/Polars 引擎

    - csv: CSV 逗号分隔值文件
        - 复用 ExcelTaskPool 实现
        - 自动检测文件类型

    - feishu_bitable: 飞书多维表格
        - 原生异步 HTTP 客户端（基于 aiohttp）
        - 快照读取，ID 映射，批量更新
        - Token 自动刷新，限流重试

    - feishu_sheet: 飞书电子表格
        - 原生异步 HTTP 客户端（基于 aiohttp）
        - 快照读取，行号映射，串行写入
        - Token 自动刷新，限流重试

函数/变量清单:
    公开函数:
        - create_task_pool(config, columns_to_extract, columns_to_write) -> BaseTaskPool
            工厂函数，根据 datasource.type 创建对应任务池实例
            输入: config (完整 YAML 配置字典), columns_to_extract, columns_to_write
            输出: BaseTaskPool 子类实例
            异常: ValueError (配置无效), ImportError (依赖缺失)

    内部函数:
        - _normalize_nonempty_str(value) -> str | None
            规范化配置值为非空字符串，None/bool/空白 -> None
        - _create_mysql_pool(...) -> MySQLTaskPool
            创建 MySQL 任务池，验证必需配置字段
        - _create_excel_pool(...) -> ExcelTaskPool
            创建 Excel 任务池，支持引擎和读写器配置
        - _create_postgresql_pool(...) -> PostgreSQLTaskPool
            创建 PostgreSQL 任务池，支持 Schema 配置
        - _create_sqlite_pool(...) -> SQLiteTaskPool
            创建 SQLite 任务池，验证数据库路径和表名
        - _create_csv_pool(...) -> ExcelTaskPool
            创建 CSV 任务池，复用 ExcelTaskPool 实现
        - _create_feishu_bitable_pool(...) -> FeishuBitableTaskPool
            创建飞书多维表格任务池
        - _create_feishu_sheet_pool(...) -> FeishuSheetTaskPool
            创建飞书电子表格任务池

    模块级变量（可用性标志）:
        - MYSQL_AVAILABLE (bool): mysql-connector-python 是否已安装
        - POSTGRESQL_AVAILABLE (bool): psycopg2 是否已安装
        - SQLITE_AVAILABLE (bool): 始终为 True（标准库）
        - EXCEL_ENABLED (bool): pandas + openpyxl 是否已安装
        - FEISHU_AVAILABLE (bool): aiohttp 是否已安装

配置示例:
    datasource:
      type: mysql  # 或 postgresql、sqlite、excel、csv、feishu_bitable、feishu_sheet
      engine: auto  # pandas、polars 或 auto
      excel_reader: auto  # openpyxl、calamine 或 auto
      excel_writer: auto  # openpyxl、xlsxwriter 或 auto
      require_all_input_fields: true
      concurrency:
        batch_size: 100

使用示例:
    from src.data.factory import create_task_pool

    pool = create_task_pool(
        config=config,
        columns_to_extract=["name", "content"],
        columns_to_write={"result": "ai_result"},
    )

扩展指南:
    添加新数据源：
    1. 在 src/data/ 下创建新的实现类（继承 BaseTaskPool）
    2. 在本模块添加可用性检测
    3. 添加 _create_xxx_pool() 辅助函数
    4. 在 create_task_pool() 中添加分支
"""

import logging
from typing import Any

from .base import BaseTaskPool


# ==================== 可用性检测标志 ====================
# 通过尝试导入来检测各依赖库是否可用

MYSQL_AVAILABLE = False
POSTGRESQL_AVAILABLE = False
SQLITE_AVAILABLE = True  # SQLite 是 Python 标准库，始终可用
EXCEL_ENABLED = False
FEISHU_AVAILABLE = False

# 检测 MySQL 连接器
try:
    import mysql.connector  # noqa: F401

    MYSQL_AVAILABLE = True
except ImportError:
    pass

# 检测 PostgreSQL 连接器
try:
    import psycopg2  # noqa: F401

    POSTGRESQL_AVAILABLE = True
except ImportError:
    pass

# 检测 Excel 依赖（pandas + openpyxl）
try:
    import pandas  # noqa: F401
    import openpyxl  # noqa: F401

    EXCEL_ENABLED = True
except ImportError:
    pass

# 检测飞书依赖（aiohttp）
try:
    import aiohttp  # noqa: F401

    FEISHU_AVAILABLE = True
except ImportError:
    pass


def _normalize_nonempty_str(value: Any) -> str | None:
    """
    规范化配置值为非空字符串。

    - None / bool / 空白字符串 -> None
    - 其他类型 -> str(value).strip()
    """
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text if text else None


def create_task_pool(
    config: dict[str, Any],
    columns_to_extract: list[str],
    columns_to_write: dict[str, str],
) -> BaseTaskPool:
    """
    根据配置创建数据源任务池

    工厂函数，读取配置中的 datasource.type 字段，
    创建对应类型的任务池实例。

    Args:
        config: 完整配置字典，需包含 datasource 配置节
        columns_to_extract: 需要从数据源提取的列名列表
        columns_to_write: 写回映射 {别名: 实际列名}

    Returns:
        BaseTaskPool: 具体的任务池实例
            - MySQLTaskPool
            - PostgreSQLTaskPool
            - SQLiteTaskPool
            - ExcelTaskPool（也用于 CSV）

    Raises:
        ValueError: 配置无效或数据源类型不支持
        ImportError: 所需依赖库不可用

    配置读取:
        datasource.type: 数据源类型（必需）
        datasource.engine: DataFrame 引擎（auto/pandas/polars）
        datasource.excel_reader: Excel 读取器（auto/openpyxl/calamine）
        datasource.excel_writer: Excel 写入器（auto/openpyxl/xlsxwriter）
        datasource.require_all_input_fields: 是否要求所有输入字段非空

    """
    # 提取数据源配置
    datasource_config = config.get("datasource", {})
    datasource_type = datasource_config.get("type", "excel").lower()

    # 获取公共配置
    require_all_input_fields = datasource_config.get("require_all_input_fields", True)
    concurrency_config = datasource_config.get("concurrency", {})

    # 高性能引擎配置（用于 Excel/CSV）
    engine_type = datasource_config.get("engine", "auto")
    excel_reader = datasource_config.get("excel_reader", "auto")
    excel_writer = datasource_config.get("excel_writer", "auto")

    logging.info(f"正在创建数据源任务池，类型: {datasource_type}")

    # 根据类型创建对应的任务池
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

    elif datasource_type == "feishu_bitable":
        return _create_feishu_bitable_pool(
            config,
            columns_to_extract,
            columns_to_write,
            require_all_input_fields,
        )

    elif datasource_type == "feishu_sheet":
        return _create_feishu_sheet_pool(
            config,
            columns_to_extract,
            columns_to_write,
            require_all_input_fields,
        )

    else:
        raise ValueError(
            f"不支持的数据源类型: {datasource_type}，"
            f"可选: mysql, postgresql, sqlite, excel, csv, feishu_bitable, feishu_sheet"
        )


def _create_mysql_pool(
    config: dict[str, Any],
    columns_to_extract: list[str],
    columns_to_write: dict[str, str],
    require_all_input_fields: bool,
    concurrency_config: dict[str, Any],
) -> BaseTaskPool:
    """
    创建 MySQL 任务池

    Args:
        config: 完整配置（需包含 mysql 配置节）
        columns_to_extract: 提取列
        columns_to_write: 写回映射
        require_all_input_fields: 是否要求所有输入字段非空
        concurrency_config: 并发配置

    Returns:
        MySQLTaskPool 实例

    Raises:
        ImportError: mysql-connector-python 未安装
        ValueError: 缺少必需配置字段
    """
    if not MYSQL_AVAILABLE:
        raise ImportError(
            "MySQL Connector 不可用，请安装: pip install mysql-connector-python"
        )

    from .mysql import MySQLTaskPool

    mysql_config = config.get("mysql", {})

    # 验证必需配置字段
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
    """
    创建 Excel 任务池

    Args:
        config: 完整配置（需包含 excel 配置节）
        columns_to_extract: 提取列
        columns_to_write: 写回映射
        require_all_input_fields: 是否要求所有输入字段非空
        concurrency_config: 并发配置
        engine_type: DataFrame 引擎类型
        excel_reader: Excel 读取器类型
        excel_writer: Excel 写入器类型

    Returns:
        ExcelTaskPool 实例

    Raises:
        ImportError: pandas 或 openpyxl 未安装
        ValueError: 缺少必需配置字段
    """
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

    # 如果未指定输出路径，使用输入路径（原地修改）
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
    """
    创建 PostgreSQL 任务池

    Args:
        config: 完整配置（需包含 postgresql 配置节）
        columns_to_extract: 提取列
        columns_to_write: 写回映射
        require_all_input_fields: 是否要求所有输入字段非空
        concurrency_config: 并发配置

    Returns:
        PostgreSQLTaskPool 实例

    Raises:
        ImportError: psycopg2 未安装
        ValueError: 缺少必需配置字段
    """
    if not POSTGRESQL_AVAILABLE:
        raise ImportError("psycopg2 不可用，请安装: pip install psycopg2-binary")

    from .postgresql import PostgreSQLTaskPool

    pg_config = config.get("postgresql", {})

    # 验证必需配置字段
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
        # 连接池大小计算逻辑同 MySQL
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
    """
    创建 SQLite 任务池

    Args:
        config: 完整配置（需包含 sqlite 配置节）
        columns_to_extract: 提取列
        columns_to_write: 写回映射
        require_all_input_fields: 是否要求所有输入字段非空

    Returns:
        SQLiteTaskPool 实例

    Raises:
        ValueError: 缺少必需配置字段
    """
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
    这种设计的优点：
    - 复用向量化过滤、批量更新等优化
    - 自动支持 Pandas/Polars 引擎切换
    - 零额外代码实现

    如需 CSV 特有功能（自定义分隔符、特殊编码、压缩支持），
    可以创建独立的 CSVTaskPool 类。

    Args:
        config: 完整配置（需包含 csv 配置节）
        columns_to_extract: 提取列
        columns_to_write: 写回映射
        require_all_input_fields: 是否要求所有输入字段非空
        concurrency_config: 并发配置
        engine_type: DataFrame 引擎类型

    Returns:
        ExcelTaskPool 实例（处理 CSV 文件）
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


def _create_feishu_bitable_pool(
    config: dict[str, Any],
    columns_to_extract: list[str],
    columns_to_write: dict[str, str],
    require_all_input_fields: bool,
) -> BaseTaskPool:
    """
    创建飞书多维表格任务池

    Args:
        config: 完整配置（需包含 feishu 和 datasource 配置节）
        columns_to_extract: 提取列
        columns_to_write: 写回映射
        require_all_input_fields: 是否要求所有输入字段非空

    Returns:
        FeishuBitableTaskPool 实例

    Raises:
        ImportError: aiohttp 未安装
        ValueError: 缺少必需配置字段
    """
    if not FEISHU_AVAILABLE:
        raise ImportError("aiohttp 不可用，请安装: pip install aiohttp")

    from .feishu.bitable import FeishuBitableTaskPool

    feishu_config = config.get("feishu", {})
    datasource_config = config.get("datasource", {})

    # 验证全局凭据
    if not feishu_config.get("app_id") or not feishu_config.get("app_secret"):
        raise ValueError("缺少飞书全局配置: feishu.app_id 和 feishu.app_secret")

    # 兼容两种配置路径:
    # 1) feishu.app_token/table_id（GUI 当前写入路径）
    # 2) datasource.app_token/table_id（旧路径）
    app_token = _normalize_nonempty_str(feishu_config.get("app_token"))
    if app_token is None:
        app_token = _normalize_nonempty_str(datasource_config.get("app_token"))
    table_id = _normalize_nonempty_str(feishu_config.get("table_id"))
    if table_id is None:
        table_id = _normalize_nonempty_str(datasource_config.get("table_id"))
    if app_token is None:
        raise ValueError(
            "缺少飞书多维表格配置: feishu.app_token 或 datasource.app_token"
        )
    if table_id is None:
        raise ValueError("缺少飞书多维表格配置: feishu.table_id 或 datasource.table_id")

    return FeishuBitableTaskPool(
        app_id=feishu_config["app_id"],
        app_secret=feishu_config["app_secret"],
        app_token=app_token,
        table_id=table_id,
        columns_to_extract=columns_to_extract,
        columns_to_write=columns_to_write,
        require_all_input_fields=require_all_input_fields,
        max_retries=feishu_config.get("max_retries", 3),
        qps_limit=feishu_config.get("qps_limit", 0),
    )


def _create_feishu_sheet_pool(
    config: dict[str, Any],
    columns_to_extract: list[str],
    columns_to_write: dict[str, str],
    require_all_input_fields: bool,
) -> BaseTaskPool:
    """
    创建飞书电子表格任务池

    Args:
        config: 完整配置（需包含 feishu 和 datasource 配置节）
        columns_to_extract: 提取列
        columns_to_write: 写回映射
        require_all_input_fields: 是否要求所有输入字段非空

    Returns:
        FeishuSheetTaskPool 实例

    Raises:
        ImportError: aiohttp 未安装
        ValueError: 缺少必需配置字段
    """
    if not FEISHU_AVAILABLE:
        raise ImportError("aiohttp 不可用，请安装: pip install aiohttp")

    from .feishu.sheet import FeishuSheetTaskPool

    feishu_config = config.get("feishu", {})
    datasource_config = config.get("datasource", {})

    # 验证全局凭据
    if not feishu_config.get("app_id") or not feishu_config.get("app_secret"):
        raise ValueError("缺少飞书全局配置: feishu.app_id 和 feishu.app_secret")

    # 兼容两种配置路径:
    # 1) feishu.spreadsheet_token/sheet_id（GUI 当前写入路径）
    # 2) datasource.spreadsheet_token/sheet_id（旧路径）
    spreadsheet_token = _normalize_nonempty_str(feishu_config.get("spreadsheet_token"))
    if spreadsheet_token is None:
        spreadsheet_token = _normalize_nonempty_str(
            datasource_config.get("spreadsheet_token")
        )
    sheet_id = _normalize_nonempty_str(feishu_config.get("sheet_id"))
    if sheet_id is None:
        sheet_id = _normalize_nonempty_str(datasource_config.get("sheet_id"))
    if spreadsheet_token is None:
        raise ValueError(
            "缺少飞书电子表格配置: feishu.spreadsheet_token 或 datasource.spreadsheet_token"
        )
    if sheet_id is None:
        raise ValueError("缺少飞书电子表格配置: feishu.sheet_id 或 datasource.sheet_id")

    return FeishuSheetTaskPool(
        app_id=feishu_config["app_id"],
        app_secret=feishu_config["app_secret"],
        spreadsheet_token=spreadsheet_token,
        sheet_id=sheet_id,
        columns_to_extract=columns_to_extract,
        columns_to_write=columns_to_write,
        require_all_input_fields=require_all_input_fields,
        max_retries=feishu_config.get("max_retries", 3),
        qps_limit=feishu_config.get("qps_limit", 0),
    )


__all__ = [
    "create_task_pool",
    "MYSQL_AVAILABLE",
    "POSTGRESQL_AVAILABLE",
    "SQLITE_AVAILABLE",
    "EXCEL_ENABLED",
    "FEISHU_AVAILABLE",
]
