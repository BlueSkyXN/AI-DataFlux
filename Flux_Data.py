# Flux_Data.py
import logging
import threading
import time
import os
import pandas as pd
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import functools

# --- Conditional Import for MySQL ---
try:
    import mysql.connector
    from mysql.connector import pooling, errorcode
    MYSQL_AVAILABLE = True
    logging.debug("MySQL Connector库已加载。")
except ImportError:
    MYSQL_AVAILABLE = False
    logging.debug("MySQL Connector库未加载或不可用。")
    # Define dummy error codes if needed elsewhere, though unlikely now
    class errorcode:
        ER_ACCESS_DENIED_ERROR = 1045
        ER_BAD_DB_ERROR = 1049
    class pooling: pass # Dummy class

# --- Conditional Import/Check for Pandas/Excel ---
try:
    # Check if pandas is installed
    pd.__version__
    # Check if openpyxl is installed (needed by pandas for xlsx)
    import openpyxl
    EXCEL_ENABLED = True
    logging.debug("Pandas 和 openpyxl 库已加载。")
except (ImportError, AttributeError):
    EXCEL_ENABLED = False
    logging.debug("Pandas 或 openpyxl 库未加载或不可用。")


# --- Base Task Pool Abstract Base Class (Defined internally) ---
class BaseTaskPool(ABC):
    """Abstract base class defining the interface for data source task pools."""
    def __init__(self, columns_to_extract: List[str], columns_to_write: Dict[str, str]):
        self.columns_to_extract = columns_to_extract
        self.columns_to_write = columns_to_write
        self.tasks: List[Tuple[Any, Dict[str, Any]]] = [] # In-memory list of tasks for the current shard: (task_id, data_dict)
        self.lock = threading.Lock() # Protects access to self.tasks list and potentially df in Excel pool

    @abstractmethod
    def get_total_task_count(self) -> int:
        """Return the total number of unprocessed tasks in the data source."""
        pass

    @abstractmethod
    def get_id_boundaries(self) -> Tuple[Any, Any]:
        """Return the minimum and maximum ID or index for sharding."""
        pass

    @abstractmethod
    def initialize_shard(self, shard_id: int, min_id: Any, max_id: Any) -> int:
        """Load unprocessed tasks for the given shard range into memory (self.tasks). Return number loaded."""
        pass

    @abstractmethod
    def get_task_batch(self, batch_size: int) -> List[Tuple[Any, Dict[str, Any]]]:
        """Get a batch of tasks from the in-memory list (self.tasks) and remove them."""
        pass

    @abstractmethod
    def update_task_results(self, results: Dict[Any, Dict[str, Any]]):
        """Write back the processing results for multiple tasks to the data source."""
        pass

    @abstractmethod
    def reload_task_data(self, task_id: Any) -> Optional[Dict[str, Any]]:
        """Reload the original input data for a specific task ID, used for retries. Return None if reload fails."""
        pass

    @abstractmethod
    def close(self):
        """Clean up resources (e.g., close connections, save files)."""
        pass

    # --- Concrete methods using the lock for self.tasks manipulation ---
    def add_task_to_front(self, task_id: Any, record_dict: Optional[Dict[str, Any]]):
        """Add a task back to the front of the in-memory queue (for retries)."""
        if record_dict is None:
             logging.warning(f"尝试将任务 {task_id} 放回队列失败，因为重载的数据为 None。")
             return
        with self.lock:
            self.tasks.insert(0, (task_id, record_dict))
            logging.debug(f"任务 {task_id} 已放回队列头部。")

    def has_tasks(self) -> bool:
        """Check if there are any tasks remaining in the current in-memory shard."""
        with self.lock:
            return len(self.tasks) > 0

    def get_remaining_count(self) -> int:
        """Get the number of tasks remaining in the current in-memory shard."""
        with self.lock:
            return len(self.tasks)


# --- MySQL Specific Components ---
if MYSQL_AVAILABLE:
    class MySQLConnectionPoolManager:
        """Manages a single MySQL connection pool instance."""
        _instance = None
        _lock = threading.Lock()
        _pool = None

        @classmethod
        def get_pool(cls, config: Optional[Dict[str, Any]] = None, pool_name="mypool", pool_size=5):
            if not MYSQL_AVAILABLE: # Redundant check but safe
                raise ImportError("MySQL Connector不可用，无法创建连接池。")

            with cls._lock:
                if cls._instance is None:
                    if config is None:
                        raise ValueError("首次获取连接池必须提供数据库配置。")
                    cls._instance = cls()
                    try:
                        logging.info(f"正在创建MySQL连接池 '{pool_name}' (大小: {pool_size})...")
                        cls._pool = pooling.MySQLConnectionPool(
                            pool_name=pool_name,
                            pool_size=pool_size,
                            pool_reset_session=True,
                            host=config["host"],
                            port=config.get("port", 3306),
                            user=config["user"],
                            password=config["password"],
                            database=config["database"],
                            auth_plugin='mysql_native_password' # Or adjust as needed
                        )
                        logging.info(f"MySQL连接池 '{pool_name}' 创建成功。")
                    except mysql.connector.Error as err:
                        logging.error(f"创建MySQL连接池失败: {err}")
                        cls._instance = None # Reset instance if pool creation failed
                        raise # Re-raise the exception
                elif config is not None:
                    logging.warning("连接池已存在，将忽略新的配置并返回现有连接池。")

                if cls._pool is None:
                    raise RuntimeError("连接池实例已创建但内部池对象为None，发生内部错误。")

                return cls._pool

        @classmethod
        def close_pool(cls):
            with cls._lock:
                if cls._instance is not None and cls._pool is not None:
                    logging.info(f"正在关闭MySQL连接池 '{cls._pool.pool_name}'...")
                    cls._pool = None
                    cls._instance = None
                    logging.info("MySQL连接池实例已清理。")
                elif cls._instance is not None:
                    logging.warning("尝试关闭连接池，但内部池对象已为None。")
                    cls._instance = None

    class MySQLTaskPool(BaseTaskPool):
        """MySQL data source task pool implementation."""
        def __init__(
            self,
            connection_config: Dict[str, Any],
            columns_to_extract: List[str],
            columns_to_write: Dict[str, str],
            table_name: str,
            pool_size: int = 5
        ):
            if not MYSQL_AVAILABLE:
                raise ImportError("MySQL Connector库未安装或不可用，无法实例化 MySQLTaskPool。")

            super().__init__(columns_to_extract, columns_to_write) # Call super first

            self.table_name = table_name
            self.select_columns = list(set(['id'] + self.columns_to_extract))
            self.write_aliases = list(self.columns_to_write.keys())
            self.write_colnames = list(self.columns_to_write.values())

            try:
                self.pool = MySQLConnectionPoolManager.get_pool(
                    config=connection_config,
                    pool_size=pool_size
                )
            except Exception as e:
                logging.error(f"初始化 MySQLTaskPool 时无法获取数据库连接池: {e}")
                raise RuntimeError(f"无法连接到数据库或创建连接池: {e}") from e

            self.current_shard_id = -1
            self.current_min_id = 0
            self.current_max_id = 0
            logging.info(f"MySQLTaskPool 初始化完成，目标表: {self.table_name}")

        def _get_connection(self):
            """Gets a connection from the pool."""
            if not self.pool:
                raise RuntimeError("数据库连接池未初始化。")
            try:
                conn = self.pool.get_connection()
                return conn
            except mysql.connector.Error as err:
                logging.error(f"从连接池 '{self.pool.pool_name}' 获取连接失败: {err}")
                raise ConnectionError(f"无法从数据库连接池获取连接: {err}") from err

        def execute_with_connection(self, callback, is_write=False):
            """Executes a callback with a connection from the pool."""
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                # Use dictionary=True for easy access by column name
                cursor = conn.cursor(dictionary=True)
                result = callback(conn, cursor)
                if is_write:
                    conn.commit()
                return result
            except mysql.connector.Error as err:
                logging.error(f"数据库操作失败: {err}")
                if conn and is_write:
                    try: conn.rollback(); logging.warning("数据库事务已回滚。")
                    except Exception as rb_err: logging.error(f"数据库回滚失败: {rb_err}") # Catch generic exception on rollback
                # Specific error handling
                if hasattr(err, 'errno'): # Check if errno attribute exists
                    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                        logging.critical("数据库访问被拒绝，请检查凭据。")
                    elif err.errno == errorcode.ER_BAD_DB_ERROR:
                        db_name = 'N/A'
                        try:
                             if conn: db_name = conn.database
                             # Cannot get config easily here, rely on conn if available
                        except: pass
                        logging.critical(f"数据库 '{db_name}' 不存在或无法访问。")
                raise RuntimeError(f"数据库错误: {err}") from err
            except Exception as e:
                 logging.error(f"执行数据库回调时发生意外错误: {e}", exc_info=True)
                 if conn and is_write:
                     try: conn.rollback(); logging.warning("数据库事务已回滚。")
                     except Exception as rb_err: logging.error(f"数据库回滚时出错: {rb_err}")
                 raise
            finally:
                if cursor:
                    try: cursor.close()
                    except Exception as cur_err: logging.warning(f"关闭数据库游标时出错: {cur_err}")
                if conn:
                    try: conn.close() # Return connection to pool
                    except Exception as conn_err: logging.warning(f"关闭数据库连接时出错: {conn_err}")

        def _build_unprocessed_condition(self) -> str:
            """Constructs the WHERE clause for finding unprocessed rows."""
            input_conditions = []
            for col in self.columns_to_extract:
                safe_col = f"`{col.replace('`', '``')}`"
                input_conditions.append(f"({safe_col} IS NOT NULL AND {safe_col} <> '')")
            input_clause = " AND ".join(input_conditions) if input_conditions else "1=1" # True if no input cols

            output_conditions = []
            for col in self.write_colnames:
                safe_col = f"`{col.replace('`', '``')}`"
                output_conditions.append(f"({safe_col} IS NULL OR {safe_col} = '')")
            # True if AT LEAST ONE output is empty
            output_clause = " OR ".join(output_conditions) if output_conditions else "1=0" # False if no output cols

            # Final condition: all inputs valid AND at least one output empty
            return f"({input_clause}) AND ({output_clause})"


        def get_total_task_count(self) -> int:
            """Gets the total count of unprocessed tasks in the table."""
            def _get_count(conn, cursor):
                unprocessed_where = self._build_unprocessed_condition()
                sql = f"SELECT COUNT(*) as count FROM `{self.table_name}` WHERE {unprocessed_where}"
                logging.debug(f"Executing count query: {sql}")
                cursor.execute(sql)
                result = cursor.fetchone()
                count = result['count'] if result else 0
                logging.info(f"数据库中未处理的任务总数: {count}")
                return count
            try:
                 return self.execute_with_connection(_get_count)
            except Exception as e:
                 logging.error(f"获取总任务数时出错: {e}")
                 return 0 # Return 0 on error

        def get_id_boundaries(self) -> Tuple[int, int]:
            """Gets the minimum and maximum ID from the table."""
            def _get_boundaries(conn, cursor):
                sql = f"SELECT MIN(id) as min_id, MAX(id) as max_id FROM `{self.table_name}`"
                logging.debug(f"Executing boundary query: {sql}")
                cursor.execute(sql)
                result = cursor.fetchone()
                if result and result['min_id'] is not None and result['max_id'] is not None:
                    min_id = int(result['min_id'])
                    max_id = int(result['max_id'])
                    logging.info(f"数据库 ID 范围: {min_id} - {max_id}")
                    return (min_id, max_id)
                else:
                     logging.warning("无法获取数据库表的 ID 范围 (表可能为空或无'id'列?)。返回 (0, 0)。")
                     return (0, 0)
            try:
                return self.execute_with_connection(_get_boundaries)
            except Exception as e:
                 logging.error(f"获取ID边界时出错: {e}")
                 return (0, 0) # Return default on error

        def initialize_shard(self, shard_id: int, min_id: int, max_id: int) -> int:
            """Loads tasks for a specific shard based on ID range."""
            def _load_shard(conn, cursor):
                loaded_count = 0
                shard_tasks = []
                try:
                    # Ensure columns_to_extract is non-empty before creating cols string
                    if not self.select_columns:
                         logging.warning(f"分片 {shard_id}: columns_to_extract 为空，无法加载数据。")
                         return 0

                    columns_str = ", ".join(f"`{col.replace('`', '``')}`" for col in self.select_columns)
                    unprocessed_where = self._build_unprocessed_condition()
                    sql = f"""
                        SELECT {columns_str}
                        FROM `{self.table_name}`
                        WHERE id BETWEEN %s AND %s AND {unprocessed_where}
                        ORDER BY id ASC
                    """
                    logging.debug(f"Executing shard load query for shard {shard_id} (IDs {min_id}-{max_id})")
                    cursor.execute(sql, (min_id, max_id))
                    rows = cursor.fetchall()
                    logging.debug(f"查询到 {len(rows)} 条原始记录。")

                    for row in rows:
                        record_id = row.get("id") # Safely get id
                        if record_id is None:
                             logging.warning(f"分片 {shard_id}: 查询结果中缺少 'id'，跳过此行: {row}")
                             continue
                        record_dict = {col: row.get(col) for col in self.columns_to_extract}
                        shard_tasks.append((record_id, record_dict))
                    loaded_count = len(shard_tasks)

                except Exception as e:
                    logging.error(f"加载分片 {shard_id} (IDs {min_id}-{max_id}) 失败: {e}", exc_info=True)
                    loaded_count = 0
                    shard_tasks = []

                with self.lock:
                    self.tasks = shard_tasks
                self.current_shard_id = shard_id
                self.current_min_id = min_id
                self.current_max_id = max_id

                logging.info(f"分片 {shard_id} (ID范围: {min_id}-{max_id}) 加载完成，未处理任务数={loaded_count}")
                return loaded_count

            try:
                 return self.execute_with_connection(_load_shard, is_write=False)
            except Exception as e:
                 logging.error(f"执行加载分片 {shard_id} 操作时出错: {e}")
                 return 0 # Return 0 on error

        # --- ADDED get_task_batch for MySQLTaskPool ---
        def get_task_batch(self, batch_size: int) -> List[Tuple[Any, Dict[str, Any]]]:
            """Gets a batch of tasks from the in-memory list."""
            with self.lock: # Use the lock inherited from BaseTaskPool
                batch = self.tasks[:batch_size]
                self.tasks = self.tasks[batch_size:]
                # logging.debug(f"取出 {len(batch)} 个MySQL任务，剩余 {len(self.tasks)}")
                return batch
        # --- END ADDED METHOD ---

        def update_task_results(self, results: Dict[int, Dict[str, Any]]):
            """Updates task results back to the database."""
            if not results: return

            updates_data = []
            for record_id, row_result in results.items():
                if "_error" not in row_result:
                    update_values = {col_name: row_result.get(alias)
                                     for alias, col_name in self.columns_to_write.items()
                                     if alias in row_result} # Only include keys present in result
                    if update_values:
                         updates_data.append((record_id, update_values))

            if not updates_data:
                logging.info("没有成功的记录需要更新到数据库。")
                return

            update_count = len(updates_data)
            logging.info(f"准备将 {update_count} 条成功记录的结果更新回数据库...")

            def _perform_updates(conn, cursor):
                nonlocal update_count
                success_count = 0
                errors_count = 0
                # Use prepared statements for single updates (generally safer)
                sql_template = "UPDATE `{table}` SET {set_clause} WHERE id = %s"
                for record_id, values_dict in updates_data:
                    set_parts = []
                    params = []
                    for col_name, value in values_dict.items():
                        set_parts.append(f"`{col_name.replace('`', '``')}` = %s")
                        params.append(value)
                    if not set_parts: continue

                    sql = sql_template.format(
                        table=self.table_name,
                        set_clause=", ".join(set_parts)
                    )
                    params.append(record_id)
                    try:
                        # logging.debug(f"Executing update for ID {record_id}")
                        cursor.execute(sql, tuple(params))
                        if cursor.rowcount == 1:
                            success_count += 1
                        else:
                            logging.warning(f"更新记录 {record_id} 时影响行数为 {cursor.rowcount} (预期 1)。")
                            if cursor.rowcount == 0: success_count += 1 # Assume already updated
                            else: errors_count += 1 # Log unexpected multi-row update as error
                    except mysql.connector.Error as single_err:
                        logging.error(f"更新记录 {record_id} 失败: {single_err}")
                        errors_count += 1

                if errors_count > 0:
                     logging.error(f"数据库更新完成，但有 {errors_count} 次更新失败。")
                logging.info(f"数据库更新完成，成功更新 {success_count} 条记录。")

            try:
                 self.execute_with_connection(_perform_updates, is_write=True)
            except Exception as e:
                 logging.error(f"更新数据库记录的整体操作失败: {e}")


        def reload_task_data(self, record_id: int) -> Optional[Dict[str, Any]]:
            """Reloads the original input data for a specific task ID."""
            def _reload(conn, cursor):
                if not self.columns_to_extract:
                     logging.warning(f"无法重载记录 {record_id}，未指定输入列。")
                     return None
                cols_str = ", ".join(f"`{c.replace('`', '``')}`" for c in self.columns_to_extract)
                sql = f"SELECT {cols_str} FROM `{self.table_name}` WHERE id=%s"
                cursor.execute(sql, (record_id,))
                row = cursor.fetchone()
                if row:
                    return {c: row.get(c) for c in self.columns_to_extract}
                else:
                    logging.warning(f"尝试重载数据失败：记录 {record_id} 在数据库中未找到。")
                    return None
            try:
                 return self.execute_with_connection(_reload, is_write=False)
            except Exception as e:
                 logging.error(f"执行重载记录 {record_id} 数据操作失败: {e}")
                 return None # Return None on failure

        def close(self):
            """Closes the connection pool."""
            logging.info("请求关闭 MySQL 连接池...")
            MySQLConnectionPoolManager.close_pool()

# --- Excel Specific Components ---
if EXCEL_ENABLED:
    class ExcelTaskPool(BaseTaskPool):
        """Excel data source task pool implementation."""
        def __init__(
            self,
            input_excel: str,
            output_excel: str,
            columns_to_extract: List[str],
            columns_to_write: Dict[str, str],
            save_interval: int = 300
        ):
            if not os.path.exists(input_excel):
                raise FileNotFoundError(f"Excel输入文件不存在: {input_excel}")

            super().__init__(columns_to_extract, columns_to_write) # Call super first

            logging.info(f"正在读取Excel文件: {input_excel}")
            try:
                self.df = pd.read_excel(input_excel, engine='openpyxl')
                logging.info(f"Excel文件读取成功，共 {len(self.df)} 行。")
            except Exception as e:
                raise IOError(f"无法读取Excel文件 {input_excel}: {e}") from e

            self.output_excel = output_excel
            self.save_interval = save_interval
            self.last_save_time = time.time()
            # Using inherited lock from BaseTaskPool for simplicity
            # self.df_lock = threading.Lock()

            # --- Column Validation and Preparation ---
            missing_extract_cols = [c for c in self.columns_to_extract if c not in self.df.columns]
            if missing_extract_cols:
                logging.warning(f"输入列 {missing_extract_cols} 在Excel中不存在。")

            for alias, out_col in self.columns_to_write.items():
                if out_col not in self.df.columns:
                    logging.warning(f"输出列 '{out_col}' 不存在，将创建新列。")
                    self.df[out_col] = pd.NA if hasattr(pd, 'NA') else None

            self.current_shard_id = -1
            self.current_min_idx = 0
            self.current_max_idx = 0
            logging.info(f"ExcelTaskPool 初始化完成。输入: {input_excel}, 输出: {output_excel}")

        def _is_value_empty(self, value) -> bool:
            """Checks if a value is considered empty (NaN, None, or empty/whitespace string)."""
            if pd.isna(value): return True
            if isinstance(value, str) and not value.strip(): return True
            # Consider 0 or False as non-empty unless specifically required
            return False

        def _filter_unprocessed_indices(self, min_idx: int, max_idx: int) -> List[int]:
            """Filters DataFrame indices for unprocessed rows within the range."""
            unprocessed_indices = []
            start_idx = max(0, min_idx)
            end_idx = min(len(self.df), max_idx + 1)

            if start_idx >= end_idx: return []

            logging.debug(f"开始过滤索引范围 {start_idx} 到 {end_idx - 1}...")
            try:
                sub_df = self.df.iloc[start_idx:end_idx]

                # Input condition: All extract columns must be non-empty
                if self.columns_to_extract:
                    input_valid_mask = pd.Series(True, index=sub_df.index)
                    for col in self.columns_to_extract:
                        if col in sub_df.columns:
                             # Using internal _is_value_empty for consistency
                             input_valid_mask &= ~sub_df[col].apply(self._is_value_empty) # Invert empty check
                        else:
                            input_valid_mask &= False # Column missing = invalid input
                else:
                    input_valid_mask = pd.Series(True, index=sub_df.index) # No input cols = always valid

                # Output condition: At least one write column must be empty
                if self.columns_to_write:
                    output_empty_mask = pd.Series(False, index=sub_df.index)
                    for alias, out_col in self.columns_to_write.items():
                        if out_col in sub_df.columns:
                             output_empty_mask |= sub_df[out_col].apply(self._is_value_empty)
                        else:
                             output_empty_mask |= True # Column missing = considered empty
                else:
                    output_empty_mask = pd.Series(False, index=sub_df.index) # No output cols = never empty

                # Combine conditions
                final_mask = input_valid_mask & output_empty_mask
                unprocessed_indices = sub_df.index[final_mask].tolist()

            except Exception as e:
                 logging.error(f"过滤未处理索引时出错: {e}", exc_info=True)
                 return []

            logging.debug(f"过滤索引范围 {start_idx}-{end_idx-1} 完成，找到 {len(unprocessed_indices)} 个未处理索引。")
            return unprocessed_indices

        def get_total_task_count(self) -> int:
            """Gets the total count of unprocessed tasks in the DataFrame."""
            logging.info("正在计算 Excel 中未处理的任务总数...")
            unprocessed = self._filter_unprocessed_indices(0, len(self.df) - 1)
            count = len(unprocessed)
            logging.info(f"Excel 中未处理的任务总数: {count}")
            return count

        def get_id_boundaries(self) -> Tuple[int, int]:
            """Gets the index boundaries of the DataFrame."""
            if self.df.empty: return (0, -1)
            max_index = len(self.df) - 1
            logging.info(f"Excel DataFrame 索引范围: 0 - {max_index}")
            return (0, max_index)

        def initialize_shard(self, shard_id: int, min_idx: int, max_idx: int) -> int:
            """Loads tasks for a specific shard based on index range."""
            logging.info(f"开始初始化分片 {shard_id} (索引范围: {min_idx}-{max_idx})...")
            loaded_count = 0
            shard_tasks = []
            try:
                unprocessed_idx = self._filter_unprocessed_indices(min_idx, max_idx)

                if unprocessed_idx:
                    logging.debug(f"分片 {shard_id}: 找到 {len(unprocessed_idx)} 个未处理索引，正在提取数据...")
                    for idx in unprocessed_idx:
                        try:
                            row_data = self.df.loc[idx]
                            record_dict = {}
                            for col in self.columns_to_extract:
                                value = row_data.get(col)
                                record_dict[col] = str(value) if pd.notna(value) else "" # Ensure string for prompt
                            shard_tasks.append((idx, record_dict))
                        except Exception as row_err:
                            logging.error(f"分片 {shard_id}: 提取索引 {idx} 的数据时出错: {row_err}", exc_info=True)
                    loaded_count = len(shard_tasks)
                else:
                    logging.info(f"分片 {shard_id}: 在指定索引范围内未找到未处理的任务。")

            except Exception as e:
                logging.error(f"初始化分片 {shard_id} (索引 {min_idx}-{max_idx}) 失败: {e}", exc_info=True)
                loaded_count = 0
                shard_tasks = []

            with self.lock: # Lock for accessing self.tasks
                self.tasks = shard_tasks
            self.current_shard_id = shard_id
            self.current_min_idx = min_idx
            self.current_max_idx = max_idx

            logging.info(f"分片 {shard_id} (索引范围: {min_idx}-{max_idx}) 初始化完成，加载未处理任务数={loaded_count}")
            return loaded_count

        # --- ADDED get_task_batch implementation for ExcelTaskPool ---
        def get_task_batch(self, batch_size: int) -> List[Tuple[Any, Dict[str, Any]]]:
            """Gets a batch of tasks from the in-memory list."""
            with self.lock: # Use the lock inherited from BaseTaskPool
                batch = self.tasks[:batch_size]
                self.tasks = self.tasks[batch_size:]
                # logging.debug(f"取出 {len(batch)} 个Excel任务，剩余 {len(self.tasks)}")
                return batch
        # --- END ADDED METHOD ---

        def update_task_results(self, results: Dict[int, Dict[str, Any]]):
            """Updates task results back to the DataFrame and handles periodic saving."""
            if not results: return

            updated_indices = []
            needs_save = False # Flag to check if save is needed after updates

            try:
                # Use the single lock for DataFrame modifications
                with self.lock:
                    for idx, row_result in results.items():
                        if "_error" not in row_result:
                            if idx in self.df.index:
                                for alias, col_name in self.columns_to_write.items():
                                    if col_name in self.df.columns:
                                        value_to_write = row_result.get(alias, "") # Default to empty string
                                        try:
                                            self.df.loc[idx, col_name] = value_to_write
                                        except Exception as e_set:
                                             logging.warning(f"设置索引 {idx} 列 '{col_name}' 值 '{value_to_write}' 失败: {e_set}")
                                updated_indices.append(idx)
                            else:
                                logging.warning(f"尝试更新 Excel 中不存在的索引 {idx}，跳过。")

                    if updated_indices:
                        update_count = len(updated_indices)
                        logging.info(f"已在内存中更新 {update_count} 条 Excel 记录。")
                        current_time = time.time()
                        if current_time - self.last_save_time >= self.save_interval:
                             needs_save = True
                             self.last_save_time = current_time # Update time *before* save attempt

            except Exception as e:
                logging.error(f"更新 Excel DataFrame 时发生意外错误: {e}", exc_info=True)
                needs_save = False # Do not save if update itself failed

            # Perform save outside the main lock if needed
            if needs_save:
                logging.info(f"达到保存间隔 ({self.save_interval}s)，准备保存 Excel 文件...")
                try:
                    self._save_excel() # This method handles its own errors & locking for save
                except Exception as save_err:
                     # Log error but allow processing to continue, next interval will retry save
                     logging.error(f"自动保存Excel文件失败: {save_err}")
                     # Reset last_save_time to try saving again sooner? Or keep it updated? Keep updated for now.

        def _save_excel(self):
            """
            Saves the DataFrame to the output Excel file with backup and temp file strategy.
            Handles potential errors during saving and attempts recovery/cleanup.
            """
            backup_path = self.output_excel + ".bak"
            temp_path = self.output_excel + ".tmp" # Use a temporary file for the initial save

            logging.info(f"正在尝试将 DataFrame 保存到: {self.output_excel} (通过临时文件: {temp_path})")

            # --- Step 1: Save DataFrame to a temporary file ---
            try:
                # Acquire lock only during the actual save operation
                with self.lock:
                    # Ensure the output directory exists
                    output_dir = os.path.dirname(self.output_excel)
                    if output_dir and not os.path.exists(output_dir):
                        try:
                            os.makedirs(output_dir, exist_ok=True)
                            logging.info(f"已创建输出目录: {output_dir}")
                        except OSError as e:
                            logging.error(f"创建输出目录 {output_dir} 失败: {e}", exc_info=True)
                            # Decide if this is fatal or if saving to current dir is acceptable
                            # For now, raise the error as saving might fail anyway.
                            raise IOError(f"无法创建输出目录 {output_dir}") from e

                    # --- MODIFICATION: Remove explicit engine='openpyxl' ---
                    # Let pandas infer the engine based on the FINAL output path extension,
                    # or use the default engine for the temp file.
                    # self.df.to_excel(temp_path, index=False, engine='openpyxl')
                    self.df.to_excel(temp_path, index=False) # Use default engine or auto-detection
                    # --- END MODIFICATION ---
                    logging.debug(f"DataFrame 已成功写入临时文件: {temp_path}")

            except Exception as e:
                logging.error(f"写入临时 Excel 文件 {temp_path} 失败: {e}", exc_info=True)
                # Clean up the potentially corrupted temp file if it exists
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        logging.debug(f"已清理失败的临时文件: {temp_path}")
                    except Exception as rm_err:
                        logging.warning(f"清理失败的临时文件 {temp_path} 时出错: {rm_err}")
                # Do not proceed with rename/backup, re-raise or return to indicate failure
                # Re-raising makes the failure more explicit to the caller (update_task_results)
                raise IOError(f"无法将数据写入临时文件 {temp_path}") from e

            # --- Step 2: If temp save successful, perform atomic rename ---
            try:
                # Use lock again for the critical rename operations for consistency
                with self.lock:
                    # Rename existing output file to backup file (if it exists)
                    if os.path.exists(self.output_excel):
                        try:
                            os.replace(self.output_excel, backup_path) # Atomic replace if possible
                            logging.debug(f"已将现有文件重命名为备份: {backup_path}")
                        except OSError as e:
                            # Handle potential issues with renaming (e.g., permissions, file in use)
                            logging.error(f"将现有文件 {self.output_excel} 重命名为备份 {backup_path} 失败: {e}", exc_info=True)
                            # If backup fails, should we stop? Yes, it's safer not to overwrite without backup.
                            raise IOError(f"无法创建备份文件 {backup_path}") from e

                    # Rename temporary file to the final output file
                    try:
                        os.replace(temp_path, self.output_excel) # Atomic replace if possible
                        logging.info(f"DataFrame 已成功保存到: {self.output_excel}")
                    except OSError as e:
                         # Handle potential issues with the final rename
                         logging.error(f"将临时文件 {temp_path} 重命名为 {self.output_excel} 失败: {e}", exc_info=True)
                         # Attempt to restore the backup if the final rename failed
                         if os.path.exists(backup_path):
                             logging.warning(f"尝试从备份 {backup_path} 恢复原始文件...")
                             try:
                                 os.replace(backup_path, self.output_excel)
                                 logging.info(f"已成功从备份恢复文件到 {self.output_excel}")
                             except OSError as restore_err:
                                 logging.error(f"从备份 {backup_path} 恢复文件失败: {restore_err}. 文件可能已损坏或丢失！")
                         # Raise error after attempting recovery
                         raise IOError(f"无法将临时文件重命名为最终输出文件 {self.output_excel}") from e

                    # --- Step 3: Clean up backup file (only if Step 2 renaming was fully successful) ---
                    if os.path.exists(backup_path):
                        try:
                            os.remove(backup_path)
                            logging.debug(f"已成功删除备份文件: {backup_path}")
                        except Exception as rm_backup_err:
                            # Failure to remove backup is usually non-critical
                            logging.warning(f"删除备份文件 {backup_path} 失败: {rm_backup_err}")

            except Exception as e:
                 # Catch any unexpected errors during the locked rename/backup section
                 logging.error(f"保存 Excel 文件期间发生意外错误 (Rename/Backup): {e}", exc_info=True)
                 # Ensure temp file is cleaned up if it still exists
                 if os.path.exists(temp_path):
                     try:
                         os.remove(temp_path)
                         logging.debug(f"已清理未重命名的临时文件: {temp_path}")
                     except Exception as rm_final_tmp_err:
                         logging.warning(f"清理未重命名的临时文件 {temp_path} 失败: {rm_final_tmp_err}")
                 # Re-raise the error
                 raise
            # --- End of _save_excel ---


        def reload_task_data(self, idx: int) -> Optional[Dict[str, Any]]:
            """Reloads the original input data for a specific task index."""
            try:
                # Read access might be okay without lock if updates are careful
                if idx in self.df.index:
                    row_data = self.df.loc[idx]
                    record_dict = {
                        c: str(row_data.get(c)) if pd.notna(row_data.get(c)) else ""
                        for c in self.columns_to_extract
                    }
                    return record_dict
                else:
                    logging.warning(f"尝试重载数据失败：索引 {idx} 在 DataFrame 中不存在。")
                    return None
            except Exception as e:
                logging.error(f"重载索引 {idx} 数据时发生错误: {e}", exc_info=True)
                return None

        def close(self):
            """Forces a final save of the Excel file."""
            logging.info("正在执行 Excel 文件的最终保存操作...")
            try:
                self._save_excel()
            except Exception as e:
                 logging.error(f"最终保存 Excel 文件失败: {e}")


# --- Top-Level Factory Function ---
def create_task_pool(config: Dict[str, Any], columns_to_extract: List[str], columns_to_write: Dict[str, str]) -> BaseTaskPool:
    """
    Factory function to create the appropriate task pool (MySQL or Excel)
    based on the configuration.
    """
    datasource_config = config.get("datasource", {})
    ds_type = datasource_config.get("type", "excel").lower()
    concurrency_config = datasource_config.get("concurrency", {})

    logging.info(f"根据配置类型 '{ds_type}' 创建任务池...")

    if ds_type == "mysql":
        if not MYSQL_AVAILABLE:
            raise ImportError("MySQL 数据源已配置，但 MySQL Connector 库不可用。")
        mysql_config = config.get("mysql")
        if not mysql_config: raise ValueError("配置文件中缺少 [mysql] 配置段。")
        required_keys = ["host", "user", "password", "database", "table_name"]
        missing_keys = [k for k in required_keys if k not in mysql_config or not mysql_config[k]]
        if missing_keys: raise ValueError(f"MySQL配置不完整: {', '.join(missing_keys)}")

        connection_config = {k: mysql_config[k] for k in ["host", "user", "password", "database"]}
        connection_config["port"] = mysql_config.get("port", 3306)
        table_name = mysql_config["table_name"]
        pool_size = concurrency_config.get("db_pool_size", 5)

        logging.info(f"准备创建 MySQLTaskPool: 表={table_name}, 池大小={pool_size}")
        return MySQLTaskPool(connection_config, columns_to_extract, columns_to_write, table_name, pool_size)

    elif ds_type == "excel":
        if not EXCEL_ENABLED:
            raise ImportError("Excel 数据源已配置，但 Pandas 或 openpyxl 库不可用。")
        excel_config = config.get("excel")
        if not excel_config: raise ValueError("配置文件中缺少 [excel] 配置段。")
        input_excel = excel_config.get("input_path")
        if not input_excel: raise ValueError("Excel 配置缺少 'input_path'。")

        output_excel = excel_config.get("output_path")
        if not output_excel:
            base, ext = os.path.splitext(input_excel)
            output_excel = f"{base}_output{ext}"
            logging.info(f"未指定 Excel 输出路径，使用默认值: {output_excel}")

        save_interval = concurrency_config.get("save_interval", 300)
        logging.info(f"准备创建 ExcelTaskPool: 输入={input_excel}, 输出={output_excel}, 保存间隔={save_interval}s")
        return ExcelTaskPool(input_excel, output_excel, columns_to_extract, columns_to_write, save_interval)

    else:
        raise ValueError(f"不支持的数据源类型配置: '{ds_type}'")