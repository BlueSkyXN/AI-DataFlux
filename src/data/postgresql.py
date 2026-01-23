"""
PostgreSQL 数据源任务池实现

提供 PostgreSQL 数据库的任务池实现，包括连接池管理、分片加载、批量更新等功能。
适用于企业级高性能场景和大规模数据处理（10万+ 行）。

特点:
- 使用 ThreadedConnectionPool 支持多线程并发
- 使用 execute_batch() 批量更新（性能提升 5-10 倍）
- MVCC 并发控制，读写无阻塞
- 支持 JSON 类型、分区表等高级特性
"""

import logging
import threading
from typing import Any

from .base import BaseTaskPool

# 条件导入 psycopg2
try:
    import psycopg2
    from psycopg2 import pool, extras, sql

    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

    # 定义占位符类，避免代码报错
    class pool:  # type: ignore
        pass

    class extras:  # type: ignore
        pass

    class sql:  # type: ignore
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
    使用 ThreadedConnectionPool 支持多线程环境。

    Attributes:
        _instance: 单例实例
        _lock: 线程锁
        _pool: 连接池实例
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
        获取连接池实例

        Args:
            config: 数据库配置（首次调用时必需）
            pool_name: 连接池名称（仅用于日志）
            min_connections: 最小连接数
            max_connections: 最大连接数

        Returns:
            ThreadedConnectionPool 实例

        Raises:
            ImportError: psycopg2 不可用
            ValueError: 首次调用未提供配置
            RuntimeError: 连接池创建失败
        """
        if not POSTGRESQL_AVAILABLE:
            raise ImportError(
                "psycopg2 不可用，请安装: pip install psycopg2-binary"
            )

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
        """关闭连接池"""
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

    从 PostgreSQL 数据库读取任务数据，处理后写回结果。
    支持连接池、分片加载、事务处理、批量更新等功能。

    Attributes:
        table_name: 目标表名
        schema_name: 模式名（默认 public）
        pool: 数据库连接池
        select_columns: 查询列列表
        write_colnames: 写入列名列表
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
            columns_to_extract: 需要提取的列名列表
            columns_to_write: 写回映射 {别名: 实际列名}
            table_name: 目标表名
            schema_name: 模式名（默认 public）
            pool_size: 连接池大小
            require_all_input_fields: 是否要求所有输入字段都非空

        Raises:
            ImportError: psycopg2 不可用
            RuntimeError: 连接池创建失败
        """
        if not POSTGRESQL_AVAILABLE:
            raise ImportError(
                "psycopg2 不可用，请安装: pip install psycopg2-binary"
            )

        super().__init__(columns_to_extract, columns_to_write, require_all_input_fields)

        self.table_name = table_name
        self.schema_name = schema_name
        self.select_columns = list(set(["id"] + self.columns_to_extract))
        self.write_aliases = list(self.columns_to_write.keys())
        self.write_colnames = list(self.columns_to_write.values())

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
        """从连接池获取连接"""
        if not self.pool:
            raise RuntimeError("数据库连接池未初始化")

        try:
            return self.pool.getconn()
        except psycopg2.Error as err:
            logging.error(f"从连接池获取连接失败: {err}")
            raise ConnectionError(f"无法获取数据库连接: {err}") from err

    def _put_connection(self, conn: Any) -> None:
        """归还连接到连接池"""
        if self.pool and conn:
            try:
                self.pool.putconn(conn)
            except Exception as e:
                logging.warning(f"归还连接时出错: {e}")

    def execute_with_connection(self, callback: Any, is_write: bool = False) -> Any:
        """
        使用连接执行回调

        自动管理连接的获取、使用和归还，以及事务的提交和回滚。

        Args:
            callback: 回调函数 (conn, cursor) -> result
            is_write: 是否为写操作（需要提交事务）

        Returns:
            回调函数的返回值
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
        """获取未处理任务总数"""

        def _get_count(conn: Any, cursor: Any) -> int:
            where_clause = self._build_unprocessed_condition()
            query = sql.SQL("SELECT COUNT(*) as count FROM {}.{} WHERE {}").format(
                sql.Identifier(self.schema_name),
                sql.Identifier(self.table_name),
                sql.SQL(where_clause),
            )

            logging.debug(f"执行计数查询")

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
        """获取已处理任务总数"""

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
        """获取 ID 边界"""

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
        """初始化分片"""

        def _load_shard(conn: Any, cursor: Any) -> int:
            shard_tasks: list[tuple[Any, dict[str, Any]]] = []

            try:
                columns_identifiers = [sql.Identifier(col) for col in self.select_columns]
                where_clause = self._build_unprocessed_condition()

                query = sql.SQL("""
                    SELECT {}
                    FROM {}.{}
                    WHERE id BETWEEN %s AND %s AND {}
                    ORDER BY id ASC
                """).format(
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
        """从内存队列获取一批任务"""
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch

    def update_task_results(self, results: dict[int, dict[str, Any]]) -> None:
        """批量写回任务结果（使用 execute_batch 优化性能）"""
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

                logging.info(f"PostgreSQL 批量更新完成，成功更新 {success_count} 条记录")

            except psycopg2.Error as err:
                logging.error(f"批量更新失败: {err}")
                raise

        try:
            self.execute_with_connection(_perform_batch_update, is_write=True)
        except Exception as e:
            logging.error(f"更新 PostgreSQL 记录失败: {e}")

    def reload_task_data(self, record_id: int) -> dict[str, Any] | None:
        """重新加载任务的原始输入数据"""

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
        """关闭连接池"""
        logging.info("请求关闭 PostgreSQL 连接池...")
        PostgreSQLConnectionPoolManager.close_pool()

    # ==================== 内部方法 ====================

    def _build_unprocessed_condition(self) -> str:
        """
        构建未处理任务的 WHERE 条件

        未处理定义:
        - 输入列: 根据 require_all_input_fields 决定 AND 或 OR
        - 输出列: 任一为空
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
        """构建已处理任务的 WHERE 条件（所有输出列都非空）"""
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

            query = sql.SQL("""
                SELECT {}
                FROM {}.{}
                WHERE {}
                LIMIT %s
            """).format(
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

            query = sql.SQL("""
                SELECT {}
                FROM {}.{}
                WHERE {}
                LIMIT %s
            """).format(
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

            cols_identifiers = [sql.Identifier(c) for c in columns]

            query = sql.SQL("SELECT {} FROM {}.{}").format(
                sql.SQL(", ").join(cols_identifiers),
                sql.Identifier(self.schema_name),
                sql.Identifier(self.table_name),
            )

            logging.info(f"正在查询所有记录")
            cursor.execute(query)
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

            cols_identifiers = [sql.Identifier(c) for c in columns]
            where_clause = self._build_processed_condition()

            query = sql.SQL("SELECT {} FROM {}.{} WHERE {}").format(
                sql.SQL(", ").join(cols_identifiers),
                sql.Identifier(self.schema_name),
                sql.Identifier(self.table_name),
                sql.SQL(where_clause),
            )

            logging.info(f"正在查询已处理记录")
            cursor.execute(query)
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
