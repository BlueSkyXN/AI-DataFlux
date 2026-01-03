"""
MySQL 数据源任务池实现

提供 MySQL 数据库的任务池实现，包括连接池管理、分片加载、结果写回等功能。
"""

import logging
import threading
from typing import Any

from .base import BaseTaskPool

# 条件导入 MySQL 连接器
try:
    import mysql.connector
    from mysql.connector import pooling, errorcode
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    
    # 定义占位符类，避免代码报错
    class errorcode:  # type: ignore
        ER_ACCESS_DENIED_ERROR = 1045
        ER_BAD_DB_ERROR = 1049
    
    class pooling:  # type: ignore
        pass


class MySQLConnectionPoolManager:
    """
    MySQL 连接池管理器 (单例模式)
    
    管理全局唯一的 MySQL 连接池实例，确保连接复用。
    """
    
    _instance: "MySQLConnectionPoolManager | None" = None
    _lock = threading.Lock()
    _pool: Any = None
    
    @classmethod
    def get_pool(
        cls, 
        config: dict[str, Any] | None = None,
        pool_name: str = "ai_dataflux_pool",
        pool_size: int = 5
    ) -> Any:
        """
        获取连接池实例
        
        Args:
            config: 数据库配置 (首次调用时必需)
            pool_name: 连接池名称
            pool_size: 连接池大小
            
        Returns:
            MySQLConnectionPool 实例
            
        Raises:
            ImportError: MySQL 连接器不可用
            ValueError: 首次调用未提供配置
            RuntimeError: 连接池创建失败
        """
        if not MYSQL_AVAILABLE:
            raise ImportError("MySQL Connector 不可用，请安装: pip install mysql-connector-python")
        
        with cls._lock:
            if cls._instance is None:
                if config is None:
                    raise ValueError("首次获取连接池必须提供数据库配置")
                
                cls._instance = cls()
                
                try:
                    logging.info(f"正在创建 MySQL 连接池 '{pool_name}' (大小: {pool_size})...")
                    cls._pool = pooling.MySQLConnectionPool(
                        pool_name=pool_name,
                        pool_size=pool_size,
                        pool_reset_session=True,
                        host=config["host"],
                        port=config.get("port", 3306),
                        user=config["user"],
                        password=config["password"],
                        database=config["database"],
                        auth_plugin="mysql_native_password"
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
        """关闭连接池"""
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
    
    从 MySQL 数据库读取任务数据，处理后写回结果。
    支持连接池、分片加载、事务处理等功能。
    
    Attributes:
        table_name: 目标表名
        pool: 数据库连接池
        select_columns: 查询列列表
    """
    
    def __init__(
        self,
        connection_config: dict[str, Any],
        columns_to_extract: list[str],
        columns_to_write: dict[str, str],
        table_name: str,
        pool_size: int = 5,
        require_all_input_fields: bool = True
    ):
        """
        初始化 MySQL 任务池
        
        Args:
            connection_config: 数据库连接配置
            columns_to_extract: 需要提取的列名列表
            columns_to_write: 写回映射 {别名: 实际列名}
            table_name: 目标表名
            pool_size: 连接池大小
            require_all_input_fields: 是否要求所有输入字段都非空
            
        Raises:
            ImportError: MySQL 连接器不可用
            RuntimeError: 连接池创建失败
        """
        if not MYSQL_AVAILABLE:
            raise ImportError("MySQL Connector 不可用，请安装: pip install mysql-connector-python")
        
        super().__init__(columns_to_extract, columns_to_write, require_all_input_fields)
        
        self.table_name = table_name
        self.select_columns = list(set(["id"] + self.columns_to_extract))
        self.write_aliases = list(self.columns_to_write.keys())
        self.write_colnames = list(self.columns_to_write.values())
        
        # 获取连接池
        try:
            self.pool = MySQLConnectionPoolManager.get_pool(
                config=connection_config,
                pool_size=pool_size
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
        """从连接池获取连接"""
        if not self.pool:
            raise RuntimeError("数据库连接池未初始化")
        
        try:
            return self.pool.get_connection()
        except mysql.connector.Error as err:
            logging.error(f"从连接池获取连接失败: {err}")
            raise ConnectionError(f"无法获取数据库连接: {err}") from err
    
    def execute_with_connection(
        self, 
        callback: Any, 
        is_write: bool = False
    ) -> Any:
        """
        使用连接执行回调
        
        自动管理连接的获取、使用和归还，以及事务的提交和回滚。
        
        Args:
            callback: 回调函数 (conn, cursor) -> result
            is_write: 是否为写操作 (需要提交事务)
            
        Returns:
            回调函数的返回值
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
        """获取未处理任务总数"""
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
    
    def get_id_boundaries(self) -> tuple[int, int]:
        """获取 ID 边界"""
        def _get_boundaries(conn: Any, cursor: Any) -> tuple[int, int]:
            sql = f"SELECT MIN(id) as min_id, MAX(id) as max_id FROM `{self.table_name}`"
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
        """初始化分片"""
        def _load_shard(conn: Any, cursor: Any) -> int:
            shard_tasks: list[tuple[Any, dict[str, Any]]] = []
            
            try:
                if not self.select_columns:
                    logging.warning(f"分片 {shard_id}: 未指定查询列，无法加载数据")
                    return 0
                
                # 构建查询
                columns_str = ", ".join(f"`{col.replace('`', '``')}`" for col in self.select_columns)
                where_clause = self._build_unprocessed_condition()
                
                sql = f"""
                    SELECT {columns_str}
                    FROM `{self.table_name}`
                    WHERE id BETWEEN %s AND %s AND {where_clause}
                    ORDER BY id ASC
                """
                
                logging.debug(f"执行分片加载查询 (分片 {shard_id}, ID: {min_id}-{max_id})")
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
                logging.error(f"加载分片 {shard_id} (ID: {min_id}-{max_id}) 失败: {e}", exc_info=True)
                shard_tasks = []
            
            # 更新任务队列
            with self.lock:
                self.tasks = shard_tasks
            
            # 更新分片状态
            self.current_shard_id = shard_id
            self.current_min_id = min_id
            self.current_max_id = max_id
            
            loaded_count = len(shard_tasks)
            logging.info(f"分片 {shard_id} (ID: {min_id}-{max_id}) 加载完成，任务数: {loaded_count}")
            
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
        """批量写回任务结果"""
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
        """重新加载任务的原始输入数据"""
        def _reload(conn: Any, cursor: Any) -> dict[str, Any] | None:
            if not self.columns_to_extract:
                logging.warning(f"无法重载记录 {record_id}，未指定输入列")
                return None
            
            cols_str = ", ".join(f"`{c.replace('`', '``')}`" for c in self.columns_to_extract)
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
        """关闭连接池"""
        logging.info("请求关闭 MySQL 连接池...")
        MySQLConnectionPoolManager.close_pool()
    
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
        构建已处理任务的 WHERE 条件 (用于输出 token 采样)
        
        已处理定义: 所有输出列都非空
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
