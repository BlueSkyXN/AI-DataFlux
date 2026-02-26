"""
PostgreSQL 数据源任务池实现模块

本模块提供 PostgreSQL 数据库的任务池实现，专为企业级高性能场景设计。
支持大规模数据处理（10万+ 行），具备优秀的并发性能和数据一致性保证。

核心特性:
    - 高性能连接池: ThreadedConnectionPool 支持多线程并发
    - 批量更新优化: execute_batch() 比逐条更新快 5-10 倍
    - MVCC 并发: 读写互不阻塞，适合高并发场景
    - SQL 注入防护: 使用 psycopg2.sql 模块构建安全查询
    - Schema 支持: 支持指定数据库 Schema（默认 public）

架构设计:
    ┌─────────────────────────────────────────────────────────────┐
    │                  PostgreSQLTaskPool                          │
    │  ┌─────────────┐   ┌──────────────────────────────────────┐ │
    │  │ 任务队列    │   │ PostgreSQLConnectionPoolManager      │ │
    │  │ tasks[]     │   │ (ThreadedConnectionPool)             │ │
    │  └─────────────┘   └──────────────────────────────────────┘ │
    │         │                        │                          │
    │         ▼                        ▼                          │
    │  ┌─────────────┐   ┌──────────────────────────────────────┐ │
    │  │ 分片状态    │   │ PostgreSQL Database                  │ │
    │  │ shard_id    │   │ Schema: {schema_name}                │ │
    │  │ min_id      │   │ Table: {table_name}                  │ │
    │  │ max_id      │   │ 必需列: id (主键)                    │ │
    │  └─────────────┘   └──────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────┘

类清单:
    PostgreSQLConnectionPoolManager:
        PostgreSQL 连接池管理器（单例模式），管理全局唯一的 ThreadedConnectionPool。
        类方法:
            - get_pool(config, pool_name, min_connections, max_connections) -> ThreadedConnectionPool
                获取或创建连接池实例（双检锁线程安全）
            - close_pool() -> None
                关闭连接池并释放所有连接（closeall）

    PostgreSQLTaskPool(BaseTaskPool):
        PostgreSQL 数据源任务池，针对大规模数据和高并发场景优化。

        公开方法（BaseTaskPool 接口实现）:
            - __init__(connection_config, columns_to_extract, columns_to_write,
                       table_name, schema_name, pool_size, require_all_input_fields)
                初始化：检查依赖 → 校验标识符 → 获取连接池
            - get_total_task_count() -> int
                COUNT(*) 统计未处理任务数（psycopg2.sql 安全构建）
            - get_processed_task_count() -> int
                COUNT(*) 统计已处理任务数
            - get_id_boundaries() -> tuple[int, int]
                查询 MIN(id)/MAX(id) 获取 ID 边界
            - initialize_shard(shard_id, min_id, max_id) -> int
                SELECT 指定 ID 范围的未处理记录到内存队列
            - get_task_batch(batch_size) -> list[tuple]
                从内存队列弹出一批任务
            - update_task_results(results) -> None
                使用 execute_batch() 批量写回结果（性能 5-10x）
            - reload_task_data(record_id) -> dict | None
                SELECT 重新加载指定记录的输入数据
            - close() -> None
                关闭全局连接池

        Token 估算方法:
            - sample_unprocessed_rows(sample_size) -> list[dict]
            - sample_processed_rows(sample_size) -> list[dict]
            - fetch_all_rows(columns) -> list[dict]
            - fetch_all_processed_rows(columns) -> list[dict]

        内部方法:
            - _get_connection() -> Connection
                从连接池获取连接（getconn）
            - _put_connection(conn) -> None
                归还连接到连接池（putconn）
            - execute_with_connection(callback, is_write) -> Any
                封装连接获取/归还/事务管理（RealDictCursor）
            - _validate_identifier(identifier, field_name) -> str
                校验 SQL 标识符安全性
            - _validate_identifiers(identifiers, field_name) -> list[str]
                批量校验 SQL 标识符
            - _build_unprocessed_condition() -> str
                构建未处理任务的 WHERE 条件（双引号标识符）
            - _build_processed_condition() -> str
                构建已处理任务的 WHERE 条件

        关键属性:
            - table_name (str): 目标数据表名
            - schema_name (str): 数据库 Schema 名（默认 public）
            - pool: ThreadedConnectionPool 实例
            - select_columns (list[str]): 查询列（id + 输入列）
            - write_colnames (list[str]): 写入列名列表

    模块依赖:
        - base.BaseTaskPool: 抽象基类
        - psycopg2: PostgreSQL 连接器（可选依赖）
        - psycopg2.pool: 线程安全连接池
        - psycopg2.extras: RealDictCursor, execute_batch
        - psycopg2.sql: 安全 SQL 构建器

与 MySQL 版本的差异:
    1. 标识符引用: PostgreSQL 用双引号，MySQL 用反引号
    2. 连接池: ThreadedConnectionPool vs MySQLConnectionPool
    3. 批量更新: execute_batch() vs 逐条 execute()
    4. Schema: 支持多 Schema，MySQL 只有 database
    5. 字符串比较: 大小写敏感（MySQL 默认不敏感）

性能优化:
    - execute_batch(page_size=100): 批量发送，减少网络往返
    - statement_timeout=30000: 防止慢查询阻塞
    - 连接池大小: 建议 batch_size/10，最小 5

使用示例:
    from src.data.postgresql import PostgreSQLTaskPool

    pool = PostgreSQLTaskPool(
        connection_config={
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "password",
            "database": "my_db",
        },
        columns_to_extract=["title", "content"],
        columns_to_write={"result": "ai_result"},
        table_name="tasks",
        schema_name="public",  # 可选，默认 public
        pool_size=10,
    )

表结构要求:
    - 必须有 id 列作为主键（整数类型，SERIAL 或 BIGSERIAL）
    - 输入列和输出列类型应为 VARCHAR 或 TEXT
    - 建议在 id 列上建立索引

依赖:
    - psycopg2: pip install psycopg2-binary

注意事项:
    1. 连接池是全局单例，多次创建会共享同一池
    2. 首次创建时的配置有效，后续配置会被忽略
    3. 使用 RealDictCursor 返回字典格式结果
    4. 查询超时默认 30 秒
"""

import logging
import re
import threading
from typing import Any

from .base import BaseTaskPool

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# ==================== 条件导入 psycopg2 ====================
# psycopg2 是可选依赖，不可用时提供占位符类

try:
    import psycopg2
    from psycopg2 import pool, extras, sql

    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

    # 定义占位符类，避免类型检查和代码引用报错
    class pool:  # type: ignore
        """连接池占位符"""

        pass

    class extras:  # type: ignore
        """扩展功能占位符"""

        pass

    class sql:  # type: ignore
        """SQL 构建器占位符"""

        @staticmethod
        def SQL(*args: Any, **kwargs: Any) -> Any:
            pass

        @staticmethod
        def Identifier(*args: Any, **kwargs: Any) -> Any:
            pass


class PostgreSQLConnectionPoolManager:
    """
    PostgreSQL 连接池管理器（单例模式）

    管理全局唯一的 PostgreSQL 连接池实例。
    使用 psycopg2 的 ThreadedConnectionPool 支持多线程环境。

    ThreadedConnectionPool 特点:
        - 线程安全的连接获取和归还
        - 支持设置最小和最大连接数
        - 连接失效时自动重建

    Attributes:
        _instance: 单例实例引用
        _lock: 创建锁，保证线程安全
        _pool: ThreadedConnectionPool 实例
    """

    _instance: "PostgreSQLConnectionPoolManager | None" = None
    _lock = threading.Lock()
    _pool: Any = None

    @classmethod
    def get_pool(
        cls,
        config: dict[str, Any] | None = None,
        pool_name: str = "ai_dataflux_pg_pool",
        min_connections: int = 1,
        max_connections: int = 10,
    ) -> Any:
        """
        获取连接池实例（单例）

        首次调用必须提供配置，后续调用可省略。
        使用双检锁保证线程安全的单例创建。

        Args:
            config: 数据库连接配置字典
                - host: 数据库主机地址
                - port: 端口号（默认 5432）
                - user: 用户名
                - password: 密码
                - database: 数据库名
            pool_name: 连接池名称（仅用于日志识别）
            min_connections: 最小连接数（空闲时保持的连接数）
            max_connections: 最大连接数（并发连接数上限）

        Returns:
            ThreadedConnectionPool: 连接池实例

        Raises:
            ImportError: psycopg2 未安装
            ValueError: 首次调用未提供配置
            RuntimeError: 连接池创建失败

        连接选项:
            - connect_timeout=10: 连接超时 10 秒
            - statement_timeout=30000: 查询超时 30 秒（防止慢查询）
        """
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("psycopg2 不可用，请安装: pip install psycopg2-binary")

        with cls._lock:
            if cls._instance is None:
                if config is None:
                    raise ValueError("首次获取连接池必须提供数据库配置")

                cls._instance = cls()

                try:
                    logging.info(
                        f"正在创建 PostgreSQL 连接池 '{pool_name}' "
                        f"(大小: {min_connections}-{max_connections})..."
                    )

                    cls._pool = pool.ThreadedConnectionPool(
                        minconn=min_connections,
                        maxconn=max_connections,
                        host=config["host"],
                        port=config.get("port", 5432),
                        user=config["user"],
                        password=config["password"],
                        database=config["database"],
                        # 性能优化选项
                        connect_timeout=10,
                        options="-c statement_timeout=30000",  # 30s 查询超时
                    )

                    logging.info(f"PostgreSQL 连接池 '{pool_name}' 创建成功")

                except psycopg2.Error as err:
                    logging.error(f"创建 PostgreSQL 连接池失败: {err}")
                    cls._instance = None
                    raise RuntimeError(f"PostgreSQL 连接池创建失败: {err}") from err

            elif config is not None:
                logging.warning("连接池已存在，将忽略新的配置")

            if cls._pool is None:
                raise RuntimeError("连接池实例已创建但内部池对象为 None")

            return cls._pool

    @classmethod
    def close_pool(cls) -> None:
        """
        关闭连接池并释放所有资源

        使用 closeall() 关闭池中所有连接。
        关闭后连接池实例会被清空，下次 get_pool() 会创建新实例。
        """
        with cls._lock:
            if cls._instance is not None and cls._pool:
                logging.info("正在关闭 PostgreSQL 连接池...")
                try:
                    cls._pool.closeall()
                except Exception as e:
                    logging.warning(f"关闭 PostgreSQL 连接池时出错: {e}")
                cls._pool = None
                cls._instance = None
                logging.info("PostgreSQL 连接池已关闭")


class PostgreSQLTaskPool(BaseTaskPool):
    """
    PostgreSQL 数据源任务池

    从 PostgreSQL 数据库表读取未处理的任务数据，AI 处理后将结果写回。
    针对大规模数据和高并发场景优化。

    与 MySQL 版本的主要差异:
        1. SQL 构建: 使用 psycopg2.sql 模块（更安全）
        2. 批量更新: 使用 execute_batch()（更高效）
        3. 游标: 使用 RealDictCursor（更易用）
        4. Schema: 支持多 Schema（更灵活）

    Attributes:
        table_name (str): 目标数据表名
        schema_name (str): 数据库 Schema 名（默认 public）
        pool: ThreadedConnectionPool 实例
        select_columns (list[str]): 查询列（id + 输入列）
        write_colnames (list[str]): 写入列名列表

    SQL 安全:
        - 使用 sql.Identifier() 转义表名和列名
        - 使用 %s 参数化查询防止注入
    """

    def __init__(
        self,
        connection_config: dict[str, Any],
        columns_to_extract: list[str],
        columns_to_write: dict[str, str],
        table_name: str,
        schema_name: str = "public",
        pool_size: int = 10,
        require_all_input_fields: bool = True,
    ):
        """
        初始化 PostgreSQL 任务池

        Args:
            connection_config: 数据库连接配置
                - host: 主机地址
                - port: 端口号（默认 5432）
                - user: 用户名
                - password: 密码
                - database: 数据库名
            columns_to_extract: 需要提取的输入列名列表
            columns_to_write: AI 输出字段映射 {别名: 实际列名}
            table_name: 目标数据表名
            schema_name: 数据库 Schema 名（默认 "public"）
                PostgreSQL 使用 Schema 组织表，常见值:
                - public: 默认 Schema
                - 自定义 Schema 名
            pool_size: 连接池最大连接数
            require_all_input_fields: 是否要求所有输入字段都非空

        Raises:
            ImportError: psycopg2 未安装
            RuntimeError: 连接池创建失败
        """
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("psycopg2 不可用，请安装: pip install psycopg2-binary")

        super().__init__(columns_to_extract, columns_to_write, require_all_input_fields)

        self.table_name = self._validate_identifier(table_name, "table_name")
        self.schema_name = self._validate_identifier(schema_name, "schema_name")
        self.columns_to_extract = self._validate_identifiers(
            self.columns_to_extract,
            "columns_to_extract",
        )
        self.write_aliases = list(self.columns_to_write.keys())
        self.write_colnames = self._validate_identifiers(
            list(self.columns_to_write.values()),
            "columns_to_write",
        )
        self.select_columns = list(set(["id"] + self.columns_to_extract))

        # 获取连接池
        try:
            self.pool = PostgreSQLConnectionPoolManager.get_pool(
                config=connection_config,
                max_connections=pool_size,
            )
        except Exception as e:
            logging.error(f"初始化 PostgreSQLTaskPool 时无法获取连接池: {e}")
            raise RuntimeError(f"无法连接到数据库: {e}") from e

        # 分片状态
        self.current_shard_id = -1
        self.current_min_id = 0
        self.current_max_id = 0

        logging.info(
            f"PostgreSQLTaskPool 初始化完成，目标表: {self.schema_name}.{self.table_name}"
        )

    # ==================== 连接管理 ====================

    def _get_connection(self) -> Any:
        """
        从连接池获取数据库连接

        使用 getconn() 从池中获取可用连接。
        如果池已满，会等待其他线程归还连接。

        Returns:
            psycopg2 连接对象

        Raises:
            RuntimeError: 连接池未初始化
            ConnectionError: 无法获取连接
        """
        if not self.pool:
            raise RuntimeError("数据库连接池未初始化")

        try:
            return self.pool.getconn()
        except psycopg2.Error as err:
            logging.error(f"从连接池获取连接失败: {err}")
            raise ConnectionError(f"无法获取数据库连接: {err}") from err

    def _put_connection(self, conn: Any) -> None:
        """
        归还连接到连接池

        使用 putconn() 将连接归还到池中供其他线程使用。

        Args:
            conn: 要归还的连接对象
        """
        if self.pool and conn:
            try:
                self.pool.putconn(conn)
            except Exception as e:
                logging.warning(f"归还连接时出错: {e}")

    def execute_with_connection(self, callback: Any, is_write: bool = False) -> Any:
        """
        使用连接池连接执行数据库操作

        封装连接的获取、使用、归还以及事务管理。
        使用 RealDictCursor 使结果可以通过列名访问。

        Args:
            callback: 回调函数，签名为 (conn, cursor) -> result
            is_write: 是否为写操作
                - True: 成功时自动 commit，失败时自动 rollback
                - False: 只读操作，无事务管理

        Returns:
            回调函数的返回值

        Raises:
            RuntimeError: 数据库操作失败

        游标特点:
            使用 extras.RealDictCursor，查询结果为字典格式:
            {"column_name": value, ...}
        """
        conn = None
        cursor = None

        try:
            conn = self._get_connection()

            # 使用 RealDictCursor 以字典形式返回结果
            cursor = conn.cursor(cursor_factory=extras.RealDictCursor)

            result = callback(conn, cursor)

            if is_write:
                conn.commit()

            return result

        except psycopg2.Error as err:
            logging.error(f"数据库操作失败: {err}")

            if conn and is_write:
                try:
                    conn.rollback()
                    logging.warning("数据库事务已回滚")
                except Exception as rb_err:
                    logging.error(f"数据库回滚失败: {rb_err}")

            raise RuntimeError(f"PostgreSQL 错误: {err}") from err

        except Exception as e:
            logging.error(f"执行数据库回调时发生错误: {e}", exc_info=True)

            if conn and is_write:
                try:
                    conn.rollback()
                    logging.warning("数据库事务已回滚")
                except Exception as rb_err:
                    logging.error(f"数据库回滚失败: {rb_err}")

            raise

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception as e:
                    logging.warning(f"关闭游标时出错: {e}")

            if conn:
                self._put_connection(conn)

    # ==================== 核心接口实现 ====================

    def get_total_task_count(self) -> int:
        """
        获取未处理任务总数

        使用 psycopg2.sql 模块安全构建 COUNT(*) 查询。

        Returns:
            int: 未处理任务数量，查询失败返回 0
        """

        def _get_count(conn: Any, cursor: Any) -> int:
            where_clause = self._build_unprocessed_condition()
            query = sql.SQL("SELECT COUNT(*) as count FROM {}.{} WHERE {}").format(
                sql.Identifier(self.schema_name),
                sql.Identifier(self.table_name),
                sql.SQL(where_clause),
            )

            logging.debug("执行计数查询")

            cursor.execute(query)
            result = cursor.fetchone()
            count = result["count"] if result else 0

            logging.info(f"PostgreSQL 中未处理的任务总数: {count}")
            return count

        try:
            return self.execute_with_connection(_get_count)
        except Exception as e:
            logging.error(f"获取总任务数时出错: {e}")
            return 0

    def get_processed_task_count(self) -> int:
        """
        获取已处理任务总数

        统计所有输出列都非空的记录数。

        Returns:
            int: 已处理任务数量，查询失败返回 0
        """

        def _get_count(conn: Any, cursor: Any) -> int:
            where_clause = self._build_processed_condition()
            query = sql.SQL("SELECT COUNT(*) as count FROM {}.{} WHERE {}").format(
                sql.Identifier(self.schema_name),
                sql.Identifier(self.table_name),
                sql.SQL(where_clause),
            )

            cursor.execute(query)
            result = cursor.fetchone()
            count = result["count"] if result else 0

            logging.info(f"PostgreSQL 中已处理的任务总数: {count}")
            return count

        try:
            return self.execute_with_connection(_get_count)
        except Exception as e:
            logging.error(f"获取已处理任务数时出错: {e}")
            return 0

    def get_id_boundaries(self) -> tuple[int, int]:
        """
        获取表中 ID 的边界值

        Returns:
            tuple[int, int]: (最小ID, 最大ID)，表为空返回 (0, 0)
        """

        def _get_boundaries(conn: Any, cursor: Any) -> tuple[int, int]:
            query = sql.SQL(
                "SELECT MIN(id) as min_id, MAX(id) as max_id FROM {}.{}"
            ).format(
                sql.Identifier(self.schema_name),
                sql.Identifier(self.table_name),
            )

            cursor.execute(query)
            result = cursor.fetchone()

            if result and result["min_id"] is not None and result["max_id"] is not None:
                min_id = int(result["min_id"])
                max_id = int(result["max_id"])
                logging.info(f"PostgreSQL ID 范围: {min_id} - {max_id}")
                return (min_id, max_id)
            else:
                logging.warning("无法获取 PostgreSQL 表的 ID 范围，返回 (0, 0)")
                return (0, 0)

        try:
            return self.execute_with_connection(_get_boundaries)
        except Exception as e:
            logging.error(f"获取 ID 边界时出错: {e}")
            return (0, 0)

    def initialize_shard(self, shard_id: int, min_id: int, max_id: int) -> int:
        """
        初始化分片，从数据库加载指定 ID 范围的未处理任务

        使用 psycopg2.sql 模块安全构建 SELECT 查询，
        将结果加载到内存任务队列中。

        Args:
            shard_id: 分片标识符（用于日志）
            min_id: ID 范围起始（包含）
            max_id: ID 范围结束（包含）

        Returns:
            int: 实际加载的任务数量
        """

        def _load_shard(conn: Any, cursor: Any) -> int:
            shard_tasks: list[tuple[Any, dict[str, Any]]] = []

            try:
                columns_identifiers = [
                    sql.Identifier(col) for col in self.select_columns
                ]
                where_clause = self._build_unprocessed_condition()

                query = sql.SQL(
                    """
                    SELECT {}
                    FROM {}.{}
                    WHERE id BETWEEN %s AND %s AND {}
                    ORDER BY id ASC
                """
                ).format(
                    sql.SQL(", ").join(columns_identifiers),
                    sql.Identifier(self.schema_name),
                    sql.Identifier(self.table_name),
                    sql.SQL(where_clause),
                )

                logging.debug(
                    f"执行分片加载查询 (分片 {shard_id}, ID: {min_id}-{max_id})"
                )
                cursor.execute(query, (min_id, max_id))
                rows = cursor.fetchall()

                logging.debug(f"查询到 {len(rows)} 条原始记录")

                for row in rows:
                    record_id = row.get("id")
                    if record_id is None:
                        logging.warning(f"分片 {shard_id}: 查询结果缺少 'id'，跳过此行")
                        continue

                    record_dict = {col: row.get(col) for col in self.columns_to_extract}
                    shard_tasks.append((record_id, record_dict))

            except Exception as e:
                logging.error(
                    f"加载分片 {shard_id} (ID: {min_id}-{max_id}) 失败: {e}",
                    exc_info=True,
                )
                shard_tasks = []

            # 更新任务队列
            with self.lock:
                self.tasks = shard_tasks

            # 更新分片状态
            self.current_shard_id = shard_id
            self.current_min_id = min_id
            self.current_max_id = max_id

            loaded_count = len(shard_tasks)
            logging.info(
                f"分片 {shard_id} (ID: {min_id}-{max_id}) 加载完成，任务数: {loaded_count}"
            )

            return loaded_count

        try:
            return self.execute_with_connection(_load_shard, is_write=False)
        except Exception as e:
            logging.error(f"执行加载分片 {shard_id} 操作时出错: {e}")
            return 0

    def get_task_batch(self, batch_size: int) -> list[tuple[Any, dict[str, Any]]]:
        """
        从内存队列获取一批任务

        Args:
            batch_size: 请求的任务数量

        Returns:
            list[tuple[Any, dict]]: 任务列表 [(record_id, data_dict), ...]
        """
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch

    def update_task_results(self, results: dict[int, dict[str, Any]]) -> None:
        """
        批量写回任务结果（使用 execute_batch 优化性能）

        使用 psycopg2.extras.execute_batch() 批量执行 UPDATE 语句，
        比逐条执行快 5-10 倍（减少网络往返）。

        Args:
            results: 结果字典 {记录ID: {别名: 值, ...}}

        优化参数:
            - page_size=100: 每批发送 100 条语句

        SQL 示例:
            UPDATE schema.table SET "out1" = %s, "out2" = %s WHERE id = %s
        """
        if not results:
            return

        # 准备更新数据
        updates_data: list[tuple[int, dict[str, Any]]] = []
        for record_id, row_result in results.items():
            if "_error" in row_result:
                continue

            update_values = {
                col_name: row_result.get(alias)
                for alias, col_name in self.columns_to_write.items()
                if alias in row_result
            }

            if update_values:
                updates_data.append((record_id, update_values))

        if not updates_data:
            logging.info("没有成功的记录需要更新到数据库")
            return

        logging.info(f"准备将 {len(updates_data)} 条记录的结果更新回 PostgreSQL...")

        def _perform_batch_update(conn: Any, cursor: Any) -> None:
            """使用 psycopg2.extras.execute_batch 批量更新"""
            try:
                # 构建批量更新语句
                set_parts = []
                for col_name in self.write_colnames:
                    set_parts.append(
                        sql.SQL("{} = %s").format(sql.Identifier(col_name))
                    )

                update_query = sql.SQL("UPDATE {}.{} SET {} WHERE id = %s").format(
                    sql.Identifier(self.schema_name),
                    sql.Identifier(self.table_name),
                    sql.SQL(", ").join(set_parts),
                )

                # 准备批量参数
                batch_params = []
                for record_id, values_dict in updates_data:
                    params = [values_dict.get(col) for col in self.write_colnames]
                    params.append(record_id)
                    batch_params.append(tuple(params))

                # 执行批量更新（page_size=100 性能最佳）
                extras.execute_batch(cursor, update_query, batch_params, page_size=100)
                success_count = len(batch_params)

                logging.info(
                    f"PostgreSQL 批量更新完成，成功更新 {success_count} 条记录"
                )

            except psycopg2.Error as err:
                logging.error(f"批量更新失败: {err}")
                raise

        try:
            self.execute_with_connection(_perform_batch_update, is_write=True)
        except Exception as e:
            logging.error(f"更新 PostgreSQL 记录失败: {e}")

    def reload_task_data(self, record_id: int) -> dict[str, Any] | None:
        """
        重新加载任务的原始输入数据

        Args:
            record_id: 记录的主键 ID

        Returns:
            dict[str, Any] | None: 输入数据字典，记录不存在返回 None
        """

        def _reload(conn: Any, cursor: Any) -> dict[str, Any] | None:
            if not self.columns_to_extract:
                logging.warning(f"无法重载记录 {record_id}，未指定输入列")
                return None

            cols_identifiers = [sql.Identifier(c) for c in self.columns_to_extract]
            query = sql.SQL("SELECT {} FROM {}.{} WHERE id = %s").format(
                sql.SQL(", ").join(cols_identifiers),
                sql.Identifier(self.schema_name),
                sql.Identifier(self.table_name),
            )

            cursor.execute(query, (record_id,))
            row = cursor.fetchone()

            if row:
                return {c: row.get(c) for c in self.columns_to_extract}
            else:
                logging.warning(f"记录 {record_id} 在 PostgreSQL 中未找到")
                return None

        try:
            return self.execute_with_connection(_reload, is_write=False)
        except Exception as e:
            logging.error(f"执行重载记录 {record_id} 数据操作失败: {e}")
            return None

    def close(self) -> None:
        """
        关闭连接池

        释放所有 PostgreSQL 数据库连接。
        连接池是全局单例，关闭后其他实例也会受影响。
        """
        logging.info("请求关闭 PostgreSQL 连接池...")
        PostgreSQLConnectionPoolManager.close_pool()

    # ==================== 内部方法 ====================

    @staticmethod
    def _validate_identifier(identifier: str, field_name: str) -> str:
        """校验 SQL 标识符，仅允许字母/数字/下划线"""
        if not IDENTIFIER_PATTERN.fullmatch(identifier):
            raise ValueError(
                f"{field_name} 包含非法标识符: {identifier!r}，仅允许字母/数字/下划线，且不能以数字开头"
            )
        return identifier

    def _validate_identifiers(
        self, identifiers: list[str], field_name: str
    ) -> list[str]:
        """批量校验 SQL 标识符"""
        return [self._validate_identifier(col, field_name) for col in identifiers]

    def _build_unprocessed_condition(self) -> str:
        """
        构建未处理任务的 WHERE 条件

        注意: PostgreSQL 使用双引号引用标识符，与 MySQL 的反引号不同。

        Returns:
            str: WHERE 条件字符串

        示例输出:
            '(("col1" IS NOT NULL AND "col1" <> '')) AND (("out1" IS NULL OR "out1" = ''))'
        """
        # 输入条件
        input_conditions = []
        for col in self.columns_to_extract:
            input_conditions.append(f'("{col}" IS NOT NULL AND "{col}" <> \'\')')

        if self.require_all_input_fields:
            input_clause = " AND ".join(input_conditions) if input_conditions else "1=1"
        else:
            input_clause = " OR ".join(input_conditions) if input_conditions else "1=1"

        # 输出条件（任一为空）
        output_conditions = []
        for col in self.write_colnames:
            output_conditions.append(f'("{col}" IS NULL OR "{col}" = \'\')')

        output_clause = " OR ".join(output_conditions) if output_conditions else "1=0"

        return f"({input_clause}) AND ({output_clause})"

    def _build_processed_condition(self) -> str:
        """
        构建已处理任务的 WHERE 条件

        已处理定义: 所有输出列都非空。

        Returns:
            str: WHERE 条件字符串
        """
        output_conditions = []
        for col in self.write_colnames:
            output_conditions.append(f'("{col}" IS NOT NULL AND "{col}" <> \'\')')

        output_clause = " AND ".join(output_conditions) if output_conditions else "1=1"
        return output_clause

    # ==================== Token 估算采样 ====================

    def sample_unprocessed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """
        采样未处理的行 (用于输入 token 估算)

        Args:
            sample_size: 采样数量

        Returns:
            采样数据列表 [{column: value, ...}, ...]
        """

        def _sample(conn: Any, cursor: Any) -> list[dict[str, Any]]:
            if not self.columns_to_extract:
                return []

            cols_identifiers = [sql.Identifier(c) for c in self.columns_to_extract]
            where_clause = self._build_unprocessed_condition()

            query = sql.SQL(
                """
                SELECT {}
                FROM {}.{}
                WHERE {}
                LIMIT %s
            """
            ).format(
                sql.SQL(", ").join(cols_identifiers),
                sql.Identifier(self.schema_name),
                sql.Identifier(self.table_name),
                sql.SQL(where_clause),
            )

            cursor.execute(query, (sample_size,))
            rows = cursor.fetchall()

            samples = []
            for row in rows:
                record_dict = {col: row.get(col) for col in self.columns_to_extract}
                samples.append(record_dict)

            logging.info(f"采样 {len(samples)} 条未处理记录用于输入 token 估算")
            return samples

        try:
            return self.execute_with_connection(_sample, is_write=False)
        except Exception as e:
            logging.error(f"采样未处理行失败: {e}")
            return []

    def sample_processed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """
        采样已处理的行 (用于输出 token 估算)

        Args:
            sample_size: 采样数量

        Returns:
            采样数据列表 [{column: value, ...}, ...]，包含输出列
        """

        def _sample(conn: Any, cursor: Any) -> list[dict[str, Any]]:
            if not self.write_colnames:
                return []

            cols_identifiers = [sql.Identifier(c) for c in self.write_colnames]
            where_clause = self._build_processed_condition()

            query = sql.SQL(
                """
                SELECT {}
                FROM {}.{}
                WHERE {}
                LIMIT %s
            """
            ).format(
                sql.SQL(", ").join(cols_identifiers),
                sql.Identifier(self.schema_name),
                sql.Identifier(self.table_name),
                sql.SQL(where_clause),
            )

            cursor.execute(query, (sample_size,))
            rows = cursor.fetchall()

            samples = []
            for row in rows:
                record_dict = {col: row.get(col) for col in self.write_colnames}
                samples.append(record_dict)

            logging.info(f"采样 {len(samples)} 条已处理记录用于输出 token 估算")
            return samples

        try:
            return self.execute_with_connection(_sample, is_write=False)
        except Exception as e:
            logging.error(f"采样已处理行失败: {e}")
            return []

    def fetch_all_rows(self, columns: list[str]) -> list[dict[str, Any]]:
        """
        获取所有行 (忽略处理状态)

        Args:
            columns: 需要提取的列名列表

        Returns:
            所有行的数据列表 [{column: value, ...}, ...]
        """

        def _fetch_all(conn: Any, cursor: Any) -> list[dict[str, Any]]:
            if not columns:
                return []

            safe_columns = self._validate_identifiers(columns, "columns")
            cols_identifiers = [sql.Identifier(c) for c in safe_columns]

            query = sql.SQL("SELECT {} FROM {}.{}").format(
                sql.SQL(", ").join(cols_identifiers),
                sql.Identifier(self.schema_name),
                sql.Identifier(self.table_name),
            )

            logging.info("正在查询所有记录")
            cursor.execute(query)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                record_dict = {col: row.get(col) for col in safe_columns}
                results.append(record_dict)

            logging.info(f"已获取 {len(results)} 条记录 (忽略处理状态)")
            return results

        try:
            return self.execute_with_connection(_fetch_all, is_write=False)
        except Exception as e:
            logging.error(f"获取所有行失败: {e}")
            return []

    def fetch_all_processed_rows(self, columns: list[str]) -> list[dict[str, Any]]:
        """
        获取所有已处理行 (仅输出已完成的记录)

        Args:
            columns: 需要提取的列名列表

        Returns:
            已处理行的数据列表 [{column: value, ...}, ...]
        """

        def _fetch_all(conn: Any, cursor: Any) -> list[dict[str, Any]]:
            if not columns:
                return []

            safe_columns = self._validate_identifiers(columns, "columns")
            cols_identifiers = [sql.Identifier(c) for c in safe_columns]
            where_clause = self._build_processed_condition()

            query = sql.SQL("SELECT {} FROM {}.{} WHERE {}").format(
                sql.SQL(", ").join(cols_identifiers),
                sql.Identifier(self.schema_name),
                sql.Identifier(self.table_name),
                sql.SQL(where_clause),
            )

            logging.info("正在查询已处理记录")
            cursor.execute(query)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                record_dict = {col: row.get(col) for col in safe_columns}
                results.append(record_dict)

            logging.info(f"已获取 {len(results)} 条已处理记录")
            return results

        try:
            return self.execute_with_connection(_fetch_all, is_write=False)
        except Exception as e:
            logging.error(f"获取已处理行失败: {e}")
            return []
