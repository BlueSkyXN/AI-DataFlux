"""
SQLite 数据源任务池实现

提供 SQLite 数据库的任务池实现，包括线程级连接管理、分片加载、结果写回等功能。
适用于本地开发测试和中小规模数据处理（< 10万行）。

特点:
- Python 标准库自带，无需额外安装
- 线程级连接管理（SQLite 不支持多线程共享连接）
- 支持 WAL 模式提升并发读性能
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

    SQLite 使用线程级连接，每个线程维护独立连接。
    支持 WAL 模式以提升并发读取性能。

    Attributes:
        _thread_local: 线程本地存储
        _db_path: 数据库文件路径（类级别共享）
    """

    _thread_local = threading.local()
    _db_path: str | None = None
    _lock = threading.Lock()

    @classmethod
    def set_db_path(cls, db_path: str) -> None:
        """设置数据库路径（首次使用前调用）"""
        with cls._lock:
            cls._db_path = db_path

    @classmethod
    def get_connection(cls, db_path: str | None = None) -> sqlite3.Connection:
        """
        获取当前线程的数据库连接

        Args:
            db_path: 数据库文件路径（可选，首次调用时设置）

        Returns:
            sqlite3.Connection 实例

        Raises:
            ValueError: 数据库路径未设置
        """
        # 确定使用的路径
        path_to_use = db_path or cls._db_path
        if not path_to_use:
            raise ValueError("数据库路径未设置，请先调用 set_db_path() 或传入 db_path 参数")

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
        """关闭当前线程的连接"""
        if hasattr(cls._thread_local, "conn") and cls._thread_local.conn:
            try:
                cls._thread_local.conn.close()
                logging.debug(f"线程 {threading.current_thread().name} 的 SQLite 连接已关闭")
            except Exception as e:
                logging.warning(f"关闭 SQLite 连接时出错: {e}")
            finally:
                cls._thread_local.conn = None


class SQLiteTaskPool(BaseTaskPool):
    """
    SQLite 数据源任务池

    从 SQLite 数据库读取任务数据，处理后写回结果。
    适用于本地开发测试和中小规模数据处理。

    Attributes:
        db_path: SQLite 数据库文件路径
        table_name: 目标表名
        select_columns: 查询列列表（包含 id + columns_to_extract）
        write_colnames: 写入列名列表
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

        Args:
            db_path: SQLite 数据库文件路径
            table_name: 目标表名
            columns_to_extract: 需要提取的列名列表
            columns_to_write: 写回映射 {别名: 实际列名}
            require_all_input_fields: 是否要求所有输入字段都非空

        Raises:
            FileNotFoundError: 数据库文件不存在
            ValueError: 表不存在
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
        """验证表是否存在"""
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
        """获取未处理任务总数"""
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
        """获取已处理任务总数"""
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
        """获取 ID 边界"""
        try:
            conn = SQLiteConnectionManager.get_connection()
            cursor = conn.cursor()

            sql = f"SELECT MIN(id) as min_id, MAX(id) as max_id FROM [{self.table_name}]"
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
        """初始化分片"""
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
        """从内存队列获取一批任务"""
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch

    def update_task_results(self, results: dict[int, dict[str, Any]]) -> None:
        """批量写回任务结果"""
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
                logging.info(f"SQLite 更新完成，成功: {success_count}, 失败: {error_count}")

            except Exception as e:
                cursor.execute("ROLLBACK")
                logging.error(f"SQLite 批量更新失败，已回滚: {e}")
                raise

            finally:
                cursor.close()

        except Exception as e:
            logging.error(f"更新 SQLite 记录失败: {e}")

    def reload_task_data(self, record_id: int) -> dict[str, Any] | None:
        """重新加载任务的原始输入数据"""
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
        """关闭连接"""
        logging.info("关闭 SQLite 连接...")
        SQLiteConnectionManager.close_connection()

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
        """构建已处理任务的 WHERE 条件（所有输出列都非空）"""
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
