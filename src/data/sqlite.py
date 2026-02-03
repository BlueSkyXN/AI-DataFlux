"""
SQLite 数据源任务池实现模块

本模块提供 SQLite 数据库的任务池实现，适用于本地开发测试和中小规模数据处理。
SQLite 是 Python 标准库自带的嵌入式数据库，无需额外安装依赖。

核心特性:
    - 零配置: Python 标准库自带，即装即用
    - 线程级连接: 每个线程独立连接，避免线程安全问题
    - WAL 模式: 提升并发读取性能
    - 轻量级: 单文件数据库，便于部署和备份
    - 事务支持: 手动事务管理，批量更新原子性

架构设计:
    ┌─────────────────────────────────────────────────────────┐
    │                    SQLiteTaskPool                        │
    │  ┌─────────────┐   ┌──────────────────────────────────┐ │
    │  │ 任务队列    │   │ SQLiteConnectionManager          │ │
    │  │ tasks[]     │   │ (线程级连接管理)                 │ │
    │  └─────────────┘   └──────────────────────────────────┘ │
    │         │                        │                       │
    │         ▼                        ▼                       │
    │  ┌─────────────┐   ┌──────────────────────────────────┐ │
    │  │ 分片状态    │   │ SQLite Database File             │ │
    │  │ shard_id    │   │ {db_path}                        │ │
    │  │ min_id      │   │ Table: {table_name}              │ │
    │  │ max_id      │   │ 必需列: id (主键)                │ │
    │  └─────────────┘   └──────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────┘

SQLite 限制:
    - 单写多读: 同一时刻只能有一个写操作
    - 并发限制: 高并发写入性能较差
    - 文件锁定: 数据库文件会被锁定
    - 数据量限制: 建议 < 10万行（大数据用 PostgreSQL）

WAL 模式优化:
    - journal_mode=WAL: 写前日志，提升并发读
    - synchronous=NORMAL: 平衡性能与安全
    - cache_size=-64000: 64MB 缓存

使用示例:
    from src.data.sqlite import SQLiteTaskPool

    pool = SQLiteTaskPool(
        db_path="data/tasks.db",
        table_name="tasks",
        columns_to_extract=["title", "content"],
        columns_to_write={"result": "ai_result"},
    )

    # 初始化分片
    count = pool.initialize_shard(0, 1, 1000)

    # 获取任务批次
    batch = pool.get_task_batch(100)

    # 处理后更新结果
    results = {1: {"result": "分析结果"}}
    pool.update_task_results(results)

    # 关闭连接
    pool.close()

表结构要求:
    - 必须有 id 列作为主键（INTEGER PRIMARY KEY）
    - 输入列和输出列类型通常为 TEXT

适用场景:
    ✓ 本地开发和测试
    ✓ 小规模数据处理（< 10万行）
    ✓ 单用户应用
    ✓ 嵌入式场景
    ✗ 高并发写入
    ✗ 大规模数据
    ✗ 多用户并发

注意事项:
    1. SQLite 连接不能跨线程共享
    2. 使用 threading.local() 管理线程级连接
    3. 事务需要手动 BEGIN/COMMIT/ROLLBACK
    4. 数据库文件需要写权限
"""

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .base import BaseTaskPool


class SQLiteConnectionManager:
    """
    SQLite 连接管理器（线程级单例）

    SQLite 连接不能在线程间共享，因此每个线程需要独立的连接。
    使用 threading.local() 存储线程级连接，确保线程安全。

    WAL 模式:
        Write-Ahead Logging 模式允许并发读取，提升性能:
        - 读操作不阻塞写操作
        - 多个读操作可以并发执行
        - 写操作仍然是串行的

    Attributes:
        _thread_local: 线程本地存储，保存连接实例
        _db_path: 数据库文件路径（类级别，所有实例共享）
        _lock: 设置路径时的线程锁

    使用模式:
        # 设置全局数据库路径（可选）
        SQLiteConnectionManager.set_db_path("data/tasks.db")

        # 获取当前线程的连接
        conn = SQLiteConnectionManager.get_connection()

        # 关闭当前线程的连接
        SQLiteConnectionManager.close_connection()
    """

    _thread_local = threading.local()
    _db_path: str | None = None
    _lock = threading.Lock()

    @classmethod
    def set_db_path(cls, db_path: str) -> None:
        """
        设置数据库路径（全局配置）

        在首次使用前设置，后续所有线程共享此路径。

        Args:
            db_path: 数据库文件的路径
        """
        with cls._lock:
            cls._db_path = db_path

    @classmethod
    def get_connection(cls, db_path: str | None = None) -> sqlite3.Connection:
        """
        获取当前线程的数据库连接

        如果当前线程没有连接，会创建新连接并配置 WAL 模式。
        连接存储在 threading.local() 中，每个线程独立。

        Args:
            db_path: 数据库文件路径
                - 如果提供，使用此路径
                - 如果未提供，使用 set_db_path() 设置的路径

        Returns:
            sqlite3.Connection: 当前线程的数据库连接

        Raises:
            ValueError: 数据库路径未设置

        连接配置:
            - check_same_thread=False: 允许跨线程（需谨慎使用）
            - timeout=30.0: 锁等待超时 30 秒
            - isolation_level=None: 自动提交模式
            - row_factory=sqlite3.Row: 支持通过列名访问
        """
        # 确定使用的路径
        path_to_use = db_path or cls._db_path
        if not path_to_use:
            raise ValueError(
                "数据库路径未设置，请先调用 set_db_path() 或传入 db_path 参数"
            )

        # 检查是否需要创建新连接
        if not hasattr(cls._thread_local, "conn") or cls._thread_local.conn is None:
            logging.debug(f"为线程 {threading.current_thread().name} 创建 SQLite 连接")
            conn = sqlite3.connect(
                path_to_use,
                check_same_thread=False,  # 允许跨线程（需谨慎）
                timeout=30.0,  # 锁等待超时
                isolation_level=None,  # 自动提交模式，事务需手动管理
            )
            conn.row_factory = sqlite3.Row  # 字典式访问

            # 启用 WAL 模式提升并发性能
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB 缓存

            cls._thread_local.conn = conn
            cls._thread_local.db_path = path_to_use

        return cls._thread_local.conn

    @classmethod
    def close_connection(cls) -> None:
        """
        关闭当前线程的连接

        在线程结束前调用，释放数据库资源。
        关闭后可以重新获取连接（会创建新连接）。
        """
        if hasattr(cls._thread_local, "conn") and cls._thread_local.conn:
            try:
                cls._thread_local.conn.close()
                logging.debug(
                    f"线程 {threading.current_thread().name} 的 SQLite 连接已关闭"
                )
            except Exception as e:
                logging.warning(f"关闭 SQLite 连接时出错: {e}")
            finally:
                cls._thread_local.conn = None


class SQLiteTaskPool(BaseTaskPool):
    """
    SQLite 数据源任务池

    从 SQLite 数据库表读取未处理的任务数据，AI 处理后将结果写回。
    轻量级实现，适合本地开发和中小规模数据处理。

    与 MySQL/PostgreSQL 版本的差异:
        1. 无连接池: 使用线程级单例连接
        2. 手动事务: 需要显式 BEGIN/COMMIT
        3. 标识符引用: 使用方括号 [column]
        4. 类型系统: 动态类型，无需严格匹配

    Attributes:
        db_path (Path): SQLite 数据库文件路径
        table_name (str): 目标数据表名
        select_columns (list[str]): 查询列（id + 输入列）
        write_colnames (list[str]): 写入列名列表

    SQL 安全:
        - 表名和列名使用方括号转义
        - 值使用参数化查询（? 占位符）
    """

    def __init__(
        self,
        db_path: str | Path,
        table_name: str,
        columns_to_extract: list[str],
        columns_to_write: dict[str, str],
        require_all_input_fields: bool = True,
    ):
        """
        初始化 SQLite 任务池

        创建流程:
            1. 验证数据库文件存在
            2. 初始化基类
            3. 配置查询和写入列
            4. 设置全局数据库路径
            5. 验证目标表存在
            6. 初始化分片状态

        Args:
            db_path: SQLite 数据库文件路径
                - 必须是已存在的文件
                - 需要读写权限
            table_name: 目标数据表名
            columns_to_extract: 需要提取的输入列名列表
            columns_to_write: AI 输出字段映射 {别名: 实际列名}
            require_all_input_fields: 是否要求所有输入字段都非空

        Raises:
            FileNotFoundError: 数据库文件不存在
            ValueError: 目标表不存在
            sqlite3.Error: 数据库连接失败
        """
        super().__init__(columns_to_extract, columns_to_write, require_all_input_fields)

        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite 数据库文件不存在: {self.db_path}")

        self.table_name = table_name
        self.select_columns = list(set(["id"] + self.columns_to_extract))
        self.write_aliases = list(self.columns_to_write.keys())
        self.write_colnames = list(self.columns_to_write.values())

        # 设置全局数据库路径
        SQLiteConnectionManager.set_db_path(str(self.db_path))

        # 验证数据库和表
        self._validate_table()

        # 分片状态
        self.current_shard_id = -1
        self.current_min_id = 0
        self.current_max_id = 0

        logging.info(
            f"SQLiteTaskPool 初始化完成，数据库: {self.db_path}, 表: {self.table_name}"
        )

    def _validate_table(self) -> None:
        """
        验证目标表是否存在

        查询 sqlite_master 系统表检查表是否存在。

        Raises:
            ValueError: 表不存在
        """
        conn = SQLiteConnectionManager.get_connection(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (self.table_name,),
        )

        if cursor.fetchone() is None:
            raise ValueError(f"表 '{self.table_name}' 在数据库中不存在")

        cursor.close()

    # ==================== 核心接口实现 ====================

    def get_total_task_count(self) -> int:
        """
        获取未处理任务总数

        Returns:
            int: 未处理任务数量，查询失败返回 0
        """
        try:
            conn = SQLiteConnectionManager.get_connection()
            cursor = conn.cursor()

            where_clause = self._build_unprocessed_condition()
            sql = f"SELECT COUNT(*) as count FROM [{self.table_name}] WHERE {where_clause}"

            cursor.execute(sql)
            result = cursor.fetchone()
            count = result["count"] if result else 0
            cursor.close()

            logging.info(f"SQLite 中未处理的任务总数: {count}")
            return count

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
        try:
            conn = SQLiteConnectionManager.get_connection()
            cursor = conn.cursor()

            where_clause = self._build_processed_condition()
            sql = f"SELECT COUNT(*) as count FROM [{self.table_name}] WHERE {where_clause}"

            cursor.execute(sql)
            result = cursor.fetchone()
            count = result["count"] if result else 0
            cursor.close()

            logging.info(f"SQLite 中已处理的任务总数: {count}")
            return count

        except Exception as e:
            logging.error(f"获取已处理任务数时出错: {e}")
            return 0

    def get_id_boundaries(self) -> tuple[int, int]:
        """
        获取表中 ID 的边界值

        Returns:
            tuple[int, int]: (最小ID, 最大ID)
            表为空或查询失败返回 (0, 0)
        """
        try:
            conn = SQLiteConnectionManager.get_connection()
            cursor = conn.cursor()

            sql = (
                f"SELECT MIN(id) as min_id, MAX(id) as max_id FROM [{self.table_name}]"
            )
            cursor.execute(sql)
            result = cursor.fetchone()
            cursor.close()

            if result and result["min_id"] is not None and result["max_id"] is not None:
                min_id = int(result["min_id"])
                max_id = int(result["max_id"])
                logging.info(f"SQLite ID 范围: {min_id} - {max_id}")
                return (min_id, max_id)
            else:
                logging.warning("无法获取 SQLite 表的 ID 范围，返回 (0, 0)")
                return (0, 0)

        except Exception as e:
            logging.error(f"获取 ID 边界时出错: {e}")
            return (0, 0)

    def initialize_shard(self, shard_id: int, min_id: int, max_id: int) -> int:
        """
        初始化分片，从数据库加载指定 ID 范围的未处理任务

        Args:
            shard_id: 分片标识符（用于日志）
            min_id: ID 范围起始（包含）
            max_id: ID 范围结束（包含）

        Returns:
            int: 实际加载的任务数量
        """
        shard_tasks: list[tuple[Any, dict[str, Any]]] = []

        try:
            conn = SQLiteConnectionManager.get_connection()
            cursor = conn.cursor()

            # 构建查询
            columns_str = ", ".join(f"[{col}]" for col in self.select_columns)
            where_clause = self._build_unprocessed_condition()

            sql = f"""
                SELECT {columns_str}
                FROM [{self.table_name}]
                WHERE id BETWEEN ? AND ? AND {where_clause}
                ORDER BY id ASC
            """

            cursor.execute(sql, (min_id, max_id))
            rows = cursor.fetchall()
            cursor.close()

            for row in rows:
                record_id = row["id"]
                record_dict = {col: row[col] for col in self.columns_to_extract}
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

    def get_task_batch(self, batch_size: int) -> list[tuple[Any, dict[str, Any]]]:
        """
        从内存任务队列获取一批任务

        Args:
            batch_size: 请求的任务数量

        Returns:
            list[tuple[Any, dict[str, Any]]]: 任务列表
        """
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch

    def update_task_results(self, results: dict[int, dict[str, Any]]) -> None:
        """
        批量写回任务结果到数据库

        使用显式事务（BEGIN TRANSACTION / COMMIT）确保原子性。
        所有更新要么全部成功，要么全部回滚。

        Args:
            results: 结果字典 {记录ID: {别名: 值, ...}}

        事务管理:
            - BEGIN TRANSACTION: 开始事务
            - COMMIT: 全部成功时提交
            - ROLLBACK: 发生错误时回滚
        """
        if not results:
            return

        try:
            conn = SQLiteConnectionManager.get_connection()
            cursor = conn.cursor()

            success_count = 0
            error_count = 0

            # 开始事务
            cursor.execute("BEGIN TRANSACTION")

            try:
                for record_id, row_result in results.items():
                    if "_error" in row_result:
                        continue

                    # 构建 UPDATE 语句
                    set_parts = []
                    params = []

                    for alias, col_name in self.columns_to_write.items():
                        if alias in row_result:
                            set_parts.append(f"[{col_name}] = ?")
                            params.append(row_result[alias])

                    if not set_parts:
                        continue

                    sql = f"UPDATE [{self.table_name}] SET {', '.join(set_parts)} WHERE id = ?"
                    params.append(record_id)

                    try:
                        cursor.execute(sql, tuple(params))
                        success_count += 1
                    except sqlite3.Error as e:
                        logging.error(f"更新记录 {record_id} 失败: {e}")
                        error_count += 1

                # 提交事务
                cursor.execute("COMMIT")
                logging.info(
                    f"SQLite 更新完成，成功: {success_count}, 失败: {error_count}"
                )

            except Exception as e:
                cursor.execute("ROLLBACK")
                logging.error(f"SQLite 批量更新失败，已回滚: {e}")
                raise

            finally:
                cursor.close()

        except Exception as e:
            logging.error(f"更新 SQLite 记录失败: {e}")

    def reload_task_data(self, record_id: int) -> dict[str, Any] | None:
        """
        重新加载任务的原始输入数据

        Args:
            record_id: 记录的主键 ID

        Returns:
            dict[str, Any] | None: 输入数据字典，记录不存在返回 None
        """
        try:
            conn = SQLiteConnectionManager.get_connection()
            cursor = conn.cursor()

            cols_str = ", ".join(f"[{c}]" for c in self.columns_to_extract)
            sql = f"SELECT {cols_str} FROM [{self.table_name}] WHERE id = ?"

            cursor.execute(sql, (record_id,))
            row = cursor.fetchone()
            cursor.close()

            if row:
                return {col: row[col] for col in self.columns_to_extract}
            else:
                logging.warning(f"记录 {record_id} 在 SQLite 中未找到")
                return None

        except Exception as e:
            logging.error(f"重载记录 {record_id} 数据失败: {e}")
            return None

    def close(self) -> None:
        """
        关闭当前线程的数据库连接

        释放数据库资源。其他线程的连接不受影响。
        """
        logging.info("关闭 SQLite 连接...")
        SQLiteConnectionManager.close_connection()

    # ==================== 内部方法 ====================

    def _build_unprocessed_condition(self) -> str:
        """
        构建未处理任务的 WHERE 条件

        SQLite 使用方括号 [] 引用标识符（也支持双引号）。

        Returns:
            str: WHERE 条件字符串

        示例输出:
            "(([col1] IS NOT NULL AND [col1] <> '')) AND (([out1] IS NULL OR [out1] = ''))"
        """
        # 输入条件
        input_conditions = []
        for col in self.columns_to_extract:
            input_conditions.append(f"([{col}] IS NOT NULL AND [{col}] <> '')")

        if self.require_all_input_fields:
            input_clause = " AND ".join(input_conditions) if input_conditions else "1=1"
        else:
            input_clause = " OR ".join(input_conditions) if input_conditions else "1=1"

        # 输出条件（任一为空）
        output_conditions = []
        for col in self.write_colnames:
            output_conditions.append(f"([{col}] IS NULL OR [{col}] = '')")

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
            output_conditions.append(f"([{col}] IS NOT NULL AND [{col}] <> '')")

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
        try:
            conn = SQLiteConnectionManager.get_connection()
            cursor = conn.cursor()

            if not self.columns_to_extract:
                return []

            cols_str = ", ".join(f"[{c}]" for c in self.columns_to_extract)
            where_clause = self._build_unprocessed_condition()

            sql = f"""
                SELECT {cols_str}
                FROM [{self.table_name}]
                WHERE {where_clause}
                LIMIT ?
            """

            cursor.execute(sql, (sample_size,))
            rows = cursor.fetchall()
            cursor.close()

            samples = []
            for row in rows:
                record_dict = {col: row[col] for col in self.columns_to_extract}
                samples.append(record_dict)

            logging.info(f"采样 {len(samples)} 条未处理记录用于输入 token 估算")
            return samples

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
        try:
            conn = SQLiteConnectionManager.get_connection()
            cursor = conn.cursor()

            if not self.write_colnames:
                return []

            cols_str = ", ".join(f"[{c}]" for c in self.write_colnames)
            where_clause = self._build_processed_condition()

            sql = f"""
                SELECT {cols_str}
                FROM [{self.table_name}]
                WHERE {where_clause}
                LIMIT ?
            """

            cursor.execute(sql, (sample_size,))
            rows = cursor.fetchall()
            cursor.close()

            samples = []
            for row in rows:
                record_dict = {col: row[col] for col in self.write_colnames}
                samples.append(record_dict)

            logging.info(f"采样 {len(samples)} 条已处理记录用于输出 token 估算")
            return samples

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
        try:
            conn = SQLiteConnectionManager.get_connection()
            cursor = conn.cursor()

            if not columns:
                return []

            cols_str = ", ".join(f"[{c}]" for c in columns)
            sql = f"SELECT {cols_str} FROM [{self.table_name}]"

            logging.info("正在查询所有记录")
            cursor.execute(sql)
            rows = cursor.fetchall()
            cursor.close()

            results = []
            for row in rows:
                record_dict = {col: row[col] for col in columns}
                results.append(record_dict)

            logging.info(f"已获取 {len(results)} 条记录 (忽略处理状态)")
            return results

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
        try:
            conn = SQLiteConnectionManager.get_connection()
            cursor = conn.cursor()

            if not columns:
                return []

            cols_str = ", ".join(f"[{c}]" for c in columns)
            where_clause = self._build_processed_condition()
            sql = f"SELECT {cols_str} FROM [{self.table_name}] WHERE {where_clause}"

            logging.info("正在查询已处理记录")
            cursor.execute(sql)
            rows = cursor.fetchall()
            cursor.close()

            results = []
            for row in rows:
                record_dict = {col: row[col] for col in columns}
                results.append(record_dict)

            logging.info(f"已获取 {len(results)} 条已处理记录")
            return results

        except Exception as e:
            logging.error(f"获取已处理行失败: {e}")
            return []
