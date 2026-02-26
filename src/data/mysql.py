"""
MySQL 数据源任务池实现模块

本模块提供 MySQL 数据库的任务池实现，支持大规模数据的分片加载和结果写回。
采用连接池复用机制，适用于高并发场景下的数据库访问。

核心特性:
    - 连接池管理: 单例模式的 MySQLConnectionPool，全局复用
    - 自动重连: 连接失效时自动从池中获取新连接
    - 事务支持: 写操作自动管理事务（commit/rollback）
    - SQL 注入防护: 参数化查询，防止 SQL 注入攻击
    - 分片加载: 按 ID 范围分片，适合大表处理

架构设计:
    ┌─────────────────────────────────────────────────────────┐
    │                    MySQLTaskPool                         │
    │  ┌─────────────┐   ┌──────────────────────────────────┐ │
    │  │ 任务队列    │   │ MySQLConnectionPoolManager       │ │
    │  │ tasks[]     │   │ (单例连接池)                     │ │
    │  └─────────────┘   └──────────────────────────────────┘ │
    │         │                        │                       │
    │         ▼                        ▼                       │
    │  ┌─────────────┐   ┌──────────────────────────────────┐ │
    │  │ 分片状态    │   │ MySQL Database                   │ │
    │  │ shard_id    │   │ Table: {table_name}              │ │
    │  │ min_id      │   │ 必需列: id (主键)                │ │
    │  │ max_id      │   └──────────────────────────────────┘ │
    │  └─────────────┘                                        │
    └─────────────────────────────────────────────────────────┘

类清单:
    MySQLConnectionPoolManager:
        MySQL 连接池管理器（单例模式），管理全局唯一的连接池实例。
        类方法:
            - get_pool(config, pool_name, pool_size) -> MySQLConnectionPool
                获取或创建连接池实例（双检锁线程安全）
            - close_pool() -> None
                关闭连接池并释放所有资源（幂等操作）

    MySQLTaskPool(BaseTaskPool):
        MySQL 数据源任务池，从数据库表读取/写入任务数据。

        公开方法（BaseTaskPool 接口实现）:
            - __init__(connection_config, columns_to_extract, columns_to_write,
                       table_name, pool_size, require_all_input_fields)
                初始化：检查依赖 → 配置列 → 获取连接池
            - get_total_task_count() -> int
                COUNT(*) 统计未处理任务数
            - get_processed_task_count() -> int
                COUNT(*) 统计已处理任务数
            - get_id_boundaries() -> tuple[int, int]
                查询 MIN(id)/MAX(id) 获取 ID 边界
            - initialize_shard(shard_id, min_id, max_id) -> int
                SELECT 指定 ID 范围的未处理记录到内存队列
            - get_task_batch(batch_size) -> list[tuple]
                从内存队列弹出一批任务
            - update_task_results(results) -> None
                逐条 UPDATE 写回结果（同一事务）
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
                从连接池获取数据库连接
            - execute_with_connection(callback, is_write) -> Any
                封装连接获取/归还/事务管理的通用执行器
            - _build_unprocessed_condition() -> str
                构建未处理任务的 WHERE 条件
            - _build_processed_condition() -> str
                构建已处理任务的 WHERE 条件

        关键属性:
            - table_name (str): 目标数据表名
            - pool: MySQLConnectionPool 实例
            - select_columns (list[str]): 查询列（id + 输入列）
            - write_colnames (list[str]): 写入列名列表
            - write_aliases (list[str]): 写入别名列表

    模块依赖:
        - base.BaseTaskPool: 抽象基类
        - mysql.connector: MySQL 连接器（可选依赖）
        - mysql.connector.pooling: 连接池管理

连接池配置:
    - pool_size: 连接池大小，默认为 batch_size/10（最小 5）
    - pool_reset_session: 归还时重置会话状态
    - auth_plugin: 使用 mysql_native_password 认证

SQL 查询示例:
    # 未处理任务查询
    SELECT id, col1, col2 FROM table
    WHERE (col1 IS NOT NULL AND col1 <> '')  -- 输入条件
      AND (out1 IS NULL OR out1 = '')         -- 输出条件
      AND id BETWEEN ? AND ?                   -- 分片范围
    ORDER BY id ASC

    # 结果写回
    UPDATE table SET out1 = ?, out2 = ? WHERE id = ?

使用示例:
    from src.data.mysql import MySQLTaskPool

    pool = MySQLTaskPool(
        connection_config={
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "password",
            "database": "my_db",
        },
        columns_to_extract=["title", "content"],
        columns_to_write={"result": "ai_result"},
        table_name="tasks",
        pool_size=10,
    )

    # 初始化分片
    count = pool.initialize_shard(0, 1, 1000)

    # 获取任务批次
    batch = pool.get_task_batch(100)

    # 处理后更新结果
    results = {1: {"result": "分析结果"}, 2: {"result": "另一个结果"}}
    pool.update_task_results(results)

    # 关闭连接池
    pool.close()

表结构要求:
    - 必须有 id 列作为主键（整数类型）
    - 输入列和输出列类型应为 VARCHAR 或 TEXT
    - 建议在 id 列上建立索引（通常主键自带）

依赖:
    - mysql-connector-python: pip install mysql-connector-python

注意事项:
    1. 连接池是全局单例，多次创建 MySQLTaskPool 会共享同一池
    2. 首次创建时的配置有效，后续配置会被忽略
    3. 表名和列名使用反引号转义，支持特殊字符
    4. 大批量更新建议分批进行，避免长事务
"""

import logging
import threading
from typing import Any

from .base import BaseTaskPool

# ==================== 条件导入 MySQL 连接器 ====================
# mysql-connector-python 是可选依赖，不可用时提供占位符类

try:
    import mysql.connector
    from mysql.connector import pooling, errorcode

    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

    # 定义占位符类，避免类型检查和代码引用报错
    class errorcode:  # type: ignore
        """MySQL 错误码占位符"""

        ER_ACCESS_DENIED_ERROR = 1045  # 访问被拒绝
        ER_BAD_DB_ERROR = 1049  # 数据库不存在

    class pooling:  # type: ignore
        """连接池占位符"""

        pass


class MySQLConnectionPoolManager:
    """
    MySQL 连接池管理器（单例模式）

    管理全局唯一的 MySQL 连接池实例，确保整个应用共享同一组数据库连接。
    使用双检锁（Double-Checked Locking）保证线程安全的单例创建。

    设计原因:
        - 数据库连接创建开销大，需要复用
        - 多个 MySQLTaskPool 实例应共享同一连接池
        - 应用结束时需要统一释放所有连接

    Attributes:
        _instance: 单例实例引用
        _lock: 创建锁，保证线程安全
        _pool: MySQLConnectionPool 实例

    使用模式:
        # 首次获取（需要配置）
        pool = MySQLConnectionPoolManager.get_pool(config={...}, pool_size=10)

        # 后续获取（无需配置）
        pool = MySQLConnectionPoolManager.get_pool()

        # 关闭连接池
        MySQLConnectionPoolManager.close_pool()
    """

    _instance: "MySQLConnectionPoolManager | None" = None
    _lock = threading.Lock()
    _pool: Any = None

    @classmethod
    def get_pool(
        cls,
        config: dict[str, Any] | None = None,
        pool_name: str = "ai_dataflux_pool",
        pool_size: int = 5,
    ) -> Any:
        """
        获取连接池实例（单例）

        首次调用必须提供配置，后续调用可省略。
        使用双检锁保证线程安全的单例创建。

        Args:
            config: 数据库连接配置字典
                - host: 数据库主机地址
                - port: 端口号（默认 3306）
                - user: 用户名
                - password: 密码
                - database: 数据库名
            pool_name: 连接池名称（用于日志识别）
            pool_size: 连接池大小（并发连接数上限）

        Returns:
            MySQLConnectionPool: 连接池实例

        Raises:
            ImportError: mysql-connector-python 未安装
            ValueError: 首次调用未提供配置
            RuntimeError: 连接池创建失败（连接被拒、数据库不存在等）

        示例:
            # 首次创建
            pool = MySQLConnectionPoolManager.get_pool(
                config={"host": "localhost", "user": "root", ...},
                pool_size=10
            )

            # 后续获取
            pool = MySQLConnectionPoolManager.get_pool()
        """
        if not MYSQL_AVAILABLE:
            raise ImportError(
                "MySQL Connector 不可用，请安装: pip install mysql-connector-python"
            )

        with cls._lock:
            if cls._instance is None:
                if config is None:
                    raise ValueError("首次获取连接池必须提供数据库配置")

                cls._instance = cls()

                try:
                    logging.info(
                        f"正在创建 MySQL 连接池 '{pool_name}' (大小: {pool_size})..."
                    )
                    cls._pool = pooling.MySQLConnectionPool(
                        pool_name=pool_name,
                        pool_size=pool_size,
                        pool_reset_session=True,
                        host=config["host"],
                        port=config.get("port", 3306),
                        user=config["user"],
                        password=config["password"],
                        database=config["database"],
                        auth_plugin="mysql_native_password",
                    )
                    logging.info(f"MySQL 连接池 '{pool_name}' 创建成功")

                except mysql.connector.Error as err:
                    logging.error(f"创建 MySQL 连接池失败: {err}")
                    cls._instance = None
                    raise RuntimeError(f"MySQL 连接池创建失败: {err}") from err

            elif config is not None:
                logging.warning("连接池已存在，将忽略新的配置")

            if cls._pool is None:
                raise RuntimeError("连接池实例已创建但内部池对象为 None")

            return cls._pool

    @classmethod
    def close_pool(cls) -> None:
        """
        关闭连接池并释放所有资源

        关闭后连接池实例会被清空，下次 get_pool() 会创建新实例。
        应用退出前应调用此方法确保连接正确释放。

        注意:
            - 关闭后所有持有连接的操作都会失败
            - 多次调用是安全的（幂等操作）
        """
        with cls._lock:
            if cls._instance is not None:
                pool_name = cls._pool.pool_name if cls._pool else "unknown"
                logging.info(f"正在关闭 MySQL 连接池 '{pool_name}'...")
                cls._pool = None
                cls._instance = None
                logging.info("MySQL 连接池已关闭")


class MySQLTaskPool(BaseTaskPool):
    """
    MySQL 数据源任务池

    从 MySQL 数据库表读取未处理的任务数据，AI 处理后将结果写回。
    使用连接池管理数据库连接，支持高并发访问。

    工作流程:
        1. 初始化: 获取连接池 → 配置查询列和写入列
        2. 分片加载: 查询指定 ID 范围的未处理记录 → 填充任务队列
        3. 任务获取: 从队列弹出批次
        4. 结果更新: 执行 UPDATE 语句写回结果
        5. 关闭: 释放连接池

    Attributes:
        table_name (str): 目标数据表名
        pool: MySQLConnectionPool 实例
        select_columns (list[str]): 查询列（id + 输入列）
        write_colnames (list[str]): 写入列名列表
        write_aliases (list[str]): 写入别名列表
        current_shard_id (int): 当前分片 ID
        current_min_id (int): 当前分片最小 ID
        current_max_id (int): 当前分片最大 ID

    SQL 安全:
        - 表名和列名使用反引号转义
        - 值使用参数化查询（%s 占位符）
        - 防止 SQL 注入攻击

    事务管理:
        - 读操作无事务
        - 写操作自动提交/回滚
    """

    def __init__(
        self,
        connection_config: dict[str, Any],
        columns_to_extract: list[str],
        columns_to_write: dict[str, str],
        table_name: str,
        pool_size: int = 5,
        require_all_input_fields: bool = True,
    ):
        """
        初始化 MySQL 任务池

        创建流程:
            1. 检查 MySQL 连接器可用性
            2. 初始化基类
            3. 配置查询和写入列
            4. 获取或创建连接池
            5. 初始化分片状态

        Args:
            connection_config: 数据库连接配置
                - host: 主机地址
                - port: 端口号（默认 3306）
                - user: 用户名
                - password: 密码
                - database: 数据库名
            columns_to_extract: 需要提取的输入列名列表
                注: 会自动添加 id 列用于结果写回定位
            columns_to_write: AI 输出字段映射 {别名: 实际列名}
            table_name: 目标数据表名
            pool_size: 连接池大小
                建议: batch_size / 10，最小 5
            require_all_input_fields: 是否要求所有输入字段都非空

        Raises:
            ImportError: mysql-connector-python 未安装
            RuntimeError: 连接池创建失败或无法连接数据库
        """
        if not MYSQL_AVAILABLE:
            raise ImportError(
                "MySQL Connector 不可用，请安装: pip install mysql-connector-python"
            )

        super().__init__(columns_to_extract, columns_to_write, require_all_input_fields)

        self.table_name = table_name
        self.select_columns = list(set(["id"] + self.columns_to_extract))
        self.write_aliases = list(self.columns_to_write.keys())
        self.write_colnames = list(self.columns_to_write.values())

        # 获取连接池
        try:
            self.pool = MySQLConnectionPoolManager.get_pool(
                config=connection_config, pool_size=pool_size
            )
        except Exception as e:
            logging.error(f"初始化 MySQLTaskPool 时无法获取数据库连接池: {e}")
            raise RuntimeError(f"无法连接到数据库: {e}") from e

        # 分片状态
        self.current_shard_id = -1
        self.current_min_id = 0
        self.current_max_id = 0

        logging.info(f"MySQLTaskPool 初始化完成，目标表: {self.table_name}")

    # ==================== 连接管理 ====================

    def _get_connection(self) -> Any:
        """
        从连接池获取数据库连接

        Returns:
            MySQL 连接对象

        Raises:
            RuntimeError: 连接池未初始化
            ConnectionError: 无法获取连接（池耗尽或连接失效）
        """
        if not self.pool:
            raise RuntimeError("数据库连接池未初始化")

        try:
            return self.pool.get_connection()
        except mysql.connector.Error as err:
            logging.error(f"从连接池获取连接失败: {err}")
            raise ConnectionError(f"无法获取数据库连接: {err}") from err

    def execute_with_connection(self, callback: Any, is_write: bool = False) -> Any:
        """
        使用连接池连接执行数据库操作

        封装连接的获取、使用、归还以及事务管理。
        确保连接在任何情况下都会被正确归还到池中。

        Args:
            callback: 回调函数，签名为 (conn, cursor) -> result
                - conn: 数据库连接对象
                - cursor: 游标对象（dictionary=True，返回字典）
            is_write: 是否为写操作
                - True: 成功时自动 commit，失败时自动 rollback
                - False: 只读操作，无事务管理

        Returns:
            回调函数的返回值

        Raises:
            RuntimeError: 数据库操作失败
                包含原始错误信息，如访问被拒绝、数据库不存在等

        使用示例:
            def _query(conn, cursor):
                cursor.execute("SELECT * FROM table WHERE id = %s", (1,))
                return cursor.fetchone()

            result = self.execute_with_connection(_query, is_write=False)
        """
        conn = None
        cursor = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            result = callback(conn, cursor)

            if is_write:
                conn.commit()

            return result

        except mysql.connector.Error as err:
            logging.error(f"数据库操作失败: {err}")

            if conn and is_write:
                try:
                    conn.rollback()
                    logging.warning("数据库事务已回滚")
                except Exception as rb_err:
                    logging.error(f"数据库回滚失败: {rb_err}")

            # 特定错误处理
            if hasattr(err, "errno"):
                if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                    logging.critical("数据库访问被拒绝，请检查凭据")
                elif err.errno == errorcode.ER_BAD_DB_ERROR:
                    logging.critical("数据库不存在或无法访问")

            raise RuntimeError(f"数据库错误: {err}") from err

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
                try:
                    conn.close()  # 归还连接到池
                except Exception as e:
                    logging.warning(f"归还连接时出错: {e}")

    # ==================== 核心接口实现 ====================

    def get_total_task_count(self) -> int:
        """
        获取未处理任务总数

        执行 COUNT(*) 查询统计满足未处理条件的记录数。
        未处理条件: 输入列有值 AND 任一输出列为空。

        Returns:
            int: 未处理任务数量，查询失败返回 0
        """

        def _get_count(conn: Any, cursor: Any) -> int:
            where_clause = self._build_unprocessed_condition()
            sql = f"SELECT COUNT(*) as count FROM `{self.table_name}` WHERE {where_clause}"
            logging.debug(f"执行计数查询: {sql}")

            cursor.execute(sql)
            result = cursor.fetchone()
            count = result["count"] if result else 0

            logging.info(f"数据库中未处理的任务总数: {count}")
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
        用于进度统计和 Token 估算采样。

        Returns:
            int: 已处理任务数量，查询失败返回 0
        """

        def _get_count(conn: Any, cursor: Any) -> int:
            where_clause = self._build_processed_condition()
            sql = f"SELECT COUNT(*) as count FROM `{self.table_name}` WHERE {where_clause}"
            logging.debug(f"执行已处理计数查询: {sql}")

            cursor.execute(sql)
            result = cursor.fetchone()
            count = result["count"] if result else 0

            logging.info(f"数据库中已处理的任务总数: {count}")
            return count

        try:
            return self.execute_with_connection(_get_count)
        except Exception as e:
            logging.error(f"获取已处理任务数时出错: {e}")
            return 0

    def get_id_boundaries(self) -> tuple[int, int]:
        """
        获取表中 ID 的边界值

        查询 MIN(id) 和 MAX(id)，用于分片调度器划分工作区间。

        Returns:
            tuple[int, int]: (最小ID, 最大ID)
            表为空或查询失败返回 (0, 0)
        """

        def _get_boundaries(conn: Any, cursor: Any) -> tuple[int, int]:
            sql = (
                f"SELECT MIN(id) as min_id, MAX(id) as max_id FROM `{self.table_name}`"
            )
            logging.debug(f"执行边界查询: {sql}")

            cursor.execute(sql)
            result = cursor.fetchone()

            if result and result["min_id"] is not None and result["max_id"] is not None:
                min_id = int(result["min_id"])
                max_id = int(result["max_id"])
                logging.info(f"数据库 ID 范围: {min_id} - {max_id}")
                return (min_id, max_id)
            else:
                logging.warning("无法获取数据库表的 ID 范围，返回 (0, 0)")
                return (0, 0)

        try:
            return self.execute_with_connection(_get_boundaries)
        except Exception as e:
            logging.error(f"获取 ID 边界时出错: {e}")
            return (0, 0)

    def initialize_shard(self, shard_id: int, min_id: int, max_id: int) -> int:
        """
        初始化分片，从数据库加载指定 ID 范围的未处理任务

        执行 SELECT 查询获取指定范围内的未处理记录，
        并将结果加载到内存任务队列中。

        Args:
            shard_id: 分片标识符（用于日志）
            min_id: ID 范围起始（包含）
            max_id: ID 范围结束（包含）

        Returns:
            int: 实际加载的任务数量

        SQL 示例:
            SELECT id, col1, col2 FROM table
            WHERE id BETWEEN 1 AND 1000
              AND (col1 IS NOT NULL AND col1 <> '')
              AND (out1 IS NULL OR out1 = '')
            ORDER BY id ASC
        """

        def _load_shard(conn: Any, cursor: Any) -> int:
            shard_tasks: list[tuple[Any, dict[str, Any]]] = []

            try:
                if not self.select_columns:
                    logging.warning(f"分片 {shard_id}: 未指定查询列，无法加载数据")
                    return 0

                # 构建查询
                columns_str = ", ".join(
                    f"`{col.replace('`', '``')}`" for col in self.select_columns
                )
                where_clause = self._build_unprocessed_condition()

                sql = f"""
                    SELECT {columns_str}
                    FROM `{self.table_name}`
                    WHERE id BETWEEN %s AND %s AND {where_clause}
                    ORDER BY id ASC
                """

                logging.debug(
                    f"执行分片加载查询 (分片 {shard_id}, ID: {min_id}-{max_id})"
                )
                cursor.execute(sql, (min_id, max_id))
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
        从内存任务队列获取一批任务

        Args:
            batch_size: 请求的任务数量

        Returns:
            list[tuple[Any, dict[str, Any]]]: 任务列表
                - 元组第一个元素是记录 ID
                - 元组第二个元素是输入数据字典
        """
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch

    def update_task_results(self, results: dict[int, dict[str, Any]]) -> None:
        """
        批量写回任务结果到数据库

        为每条结果执行 UPDATE 语句。
        所有更新在同一事务中执行，全部成功才提交。

        Args:
            results: 结果字典 {记录ID: {别名: 值, ...}}

        处理逻辑:
            1. 跳过包含 "_error" 键的失败结果
            2. 根据 columns_to_write 映射构建 SET 子句
            3. 执行参数化 UPDATE 语句
            4. 统一提交或回滚

        SQL 示例:
            UPDATE table SET out1 = %s, out2 = %s WHERE id = %s
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

        logging.info(f"准备将 {len(updates_data)} 条记录的结果更新回数据库...")

        def _perform_updates(conn: Any, cursor: Any) -> None:
            success_count = 0
            error_count = 0

            for record_id, values_dict in updates_data:
                set_parts = []
                params = []

                for col_name, value in values_dict.items():
                    set_parts.append(f"`{col_name.replace('`', '``')}` = %s")
                    params.append(value)

                if not set_parts:
                    continue

                sql = f"UPDATE `{self.table_name}` SET {', '.join(set_parts)} WHERE id = %s"
                params.append(record_id)

                try:
                    cursor.execute(sql, tuple(params))
                    if cursor.rowcount >= 0:
                        success_count += 1
                    else:
                        error_count += 1
                except mysql.connector.Error as err:
                    logging.error(f"更新记录 {record_id} 失败: {err}")
                    error_count += 1

            if error_count > 0:
                logging.error(f"数据库更新完成，但有 {error_count} 次更新失败")

            logging.info(f"数据库更新完成，成功更新 {success_count} 条记录")

        try:
            self.execute_with_connection(_perform_updates, is_write=True)
        except Exception as e:
            logging.error(f"更新数据库记录失败: {e}")

    def reload_task_data(self, record_id: int) -> dict[str, Any] | None:
        """
        重新加载任务的原始输入数据

        从数据库重新查询指定记录的输入列数据。
        用于 API 错误重试时获取干净的原始数据。

        Args:
            record_id: 记录的主键 ID

        Returns:
            dict[str, Any] | None: 输入数据字典，记录不存在返回 None
        """

        def _reload(conn: Any, cursor: Any) -> dict[str, Any] | None:
            if not self.columns_to_extract:
                logging.warning(f"无法重载记录 {record_id}，未指定输入列")
                return None

            cols_str = ", ".join(
                f"`{c.replace('`', '``')}`" for c in self.columns_to_extract
            )
            sql = f"SELECT {cols_str} FROM `{self.table_name}` WHERE id = %s"

            cursor.execute(sql, (record_id,))
            row = cursor.fetchone()

            if row:
                return {c: row.get(c) for c in self.columns_to_extract}
            else:
                logging.warning(f"尝试重载数据失败: 记录 {record_id} 在数据库中未找到")
                return None

        try:
            return self.execute_with_connection(_reload, is_write=False)
        except Exception as e:
            logging.error(f"执行重载记录 {record_id} 数据操作失败: {e}")
            return None

    def close(self) -> None:
        """
        关闭连接池

        释放所有数据库连接。调用后不应再使用此任务池实例。

        注意:
            连接池是全局单例，关闭后其他 MySQLTaskPool 实例也会受影响。
        """
        logging.info("请求关闭 MySQL 连接池...")
        MySQLConnectionPoolManager.close_pool()

    # ==================== 内部方法 ====================

    def _build_unprocessed_condition(self) -> str:
        """
        构建未处理任务的 WHERE 条件

        未处理的定义:
            - 输入列: 根据 require_all_input_fields 配置
                - True: 所有输入列都非空（AND 连接）
                - False: 任一输入列非空（OR 连接）
            - 输出列: 任一输出列为空（OR 连接）

        Returns:
            str: WHERE 条件字符串（不含 WHERE 关键字）

        示例输出:
            "((col1 IS NOT NULL AND col1 <> '') AND (col2 IS NOT NULL AND col2 <> ''))
             AND ((out1 IS NULL OR out1 = '') OR (out2 IS NULL OR out2 = ''))"
        """
        # 输入条件
        input_conditions = []
        for col in self.columns_to_extract:
            safe_col = f"`{col.replace('`', '``')}`"
            input_conditions.append(f"({safe_col} IS NOT NULL AND {safe_col} <> '')")

        if self.require_all_input_fields:
            input_clause = " AND ".join(input_conditions) if input_conditions else "1=1"
        else:
            input_clause = " OR ".join(input_conditions) if input_conditions else "1=1"

        # 输出条件 (任一为空)
        output_conditions = []
        for col in self.write_colnames:
            safe_col = f"`{col.replace('`', '``')}`"
            output_conditions.append(f"({safe_col} IS NULL OR {safe_col} = '')")

        output_clause = " OR ".join(output_conditions) if output_conditions else "1=0"

        return f"({input_clause}) AND ({output_clause})"

    def _build_processed_condition(self) -> str:
        """
        构建已处理任务的 WHERE 条件

        已处理的定义: 所有输出列都非空。
        用于统计已完成数量和输出 Token 采样。

        Returns:
            str: WHERE 条件字符串

        示例输出:
            "((out1 IS NOT NULL AND out1 <> '') AND (out2 IS NOT NULL AND out2 <> ''))"
        """
        # 所有输出列都非空
        output_conditions = []
        for col in self.write_colnames:
            safe_col = f"`{col.replace('`', '``')}`"
            output_conditions.append(f"({safe_col} IS NOT NULL AND {safe_col} <> '')")

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

            cols_str = ", ".join(
                f"`{c.replace('`', '``')}`" for c in self.columns_to_extract
            )
            where_clause = self._build_unprocessed_condition()

            sql = f"""
                SELECT {cols_str}
                FROM `{self.table_name}`
                WHERE {where_clause}
                LIMIT %s
            """

            cursor.execute(sql, (sample_size,))
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

            cols_str = ", ".join(
                f"`{c.replace('`', '``')}`" for c in self.write_colnames
            )
            where_clause = self._build_processed_condition()

            sql = f"""
                SELECT {cols_str}
                FROM `{self.table_name}`
                WHERE {where_clause}
                LIMIT %s
            """

            cursor.execute(sql, (sample_size,))
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

            cols_str = ", ".join(f"`{c.replace('`', '``')}`" for c in columns)

            # 不带 WHERE 条件，直接查询所有行
            sql = f"SELECT {cols_str} FROM `{self.table_name}`"

            logging.info(f"正在查询所有记录: {sql}")
            cursor.execute(sql)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                record_dict = {col: row.get(col) for col in columns}
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

            cols_str = ", ".join(f"`{c.replace('`', '``')}`" for c in columns)
            where_clause = self._build_processed_condition()
            sql = f"SELECT {cols_str} FROM `{self.table_name}` WHERE {where_clause}"

            logging.info(f"正在查询已处理记录: {sql}")
            cursor.execute(sql)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                record_dict = {col: row.get(col) for col in columns}
                results.append(record_dict)

            logging.info(f"已获取 {len(results)} 条已处理记录")
            return results

        try:
            return self.execute_with_connection(_fetch_all, is_write=False)
        except Exception as e:
            logging.error(f"获取已处理行失败: {e}")
            return []
