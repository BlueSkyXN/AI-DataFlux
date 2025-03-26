#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import random
import aiohttp
import asyncio
import logging
import json
import re
import time
import os
import threading
import sys
import pandas as pd
import gc
import psutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor # Keep ThreadPoolExecutor for sync I/O
from abc import ABC, abstractmethod
from collections import defaultdict
from asyncio import Queue # Use asyncio Queue

# 尝试导入MySQL相关库，如果不可用则标记为None
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logging.warning("MySQL Connector库未安装，MySQL数据源将不可用")

###############################################################################
# 错误类型枚举 - 区分需要退避的错误和内容处理错误
###############################################################################
class ErrorType:
    # API调用错误 - 需要模型退避
    API_ERROR = "api_error"  # 网络、超时、HTTP 4xx/5xx 等服务端错误

    # 内容处理错误 - 不需要模型退避
    CONTENT_ERROR = "content_error"  # JSON解析失败、字段验证失败等

    # 系统错误 - 需要重新入队处理
    SYSTEM_ERROR = "system_error"  # 意外代码错误，需要重试

###############################################################################
# RWLock: 读写锁实现 - 用于模型调度器的锁竞争优化 (保持不变)
###############################################################################
class RWLock:
    """读写锁：允许多个读操作并发，但写操作需要独占"""

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers = 0
        self._write_ready = threading.Condition(threading.Lock())
        self._pending_writers = 0

    def read_acquire(self):
        """获取读锁"""
        with self._read_ready:
            while self._writers > 0 or self._pending_writers > 0:
                self._read_ready.wait()
            self._readers += 1

    def read_release(self):
        """释放读锁"""
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def write_acquire(self):
        """获取写锁"""
        with self._write_ready:
            self._pending_writers += 1

        with self._read_ready:
            while self._readers > 0 or self._writers > 0:
                self._read_ready.wait()
            self._writers += 1

        with self._write_ready:
            self._pending_writers -= 1

    def write_release(self):
        """释放写锁"""
        with self._read_ready:
            self._writers -= 1
            self._read_ready.notify_all()

    class ReadLock:
        """读锁上下文管理器"""
        def __init__(self, rwlock):
            self.rwlock = rwlock

        def __enter__(self):
            self.rwlock.read_acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.rwlock.read_release()

    class WriteLock:
        """写锁上下文管理器"""
        def __init__(self, rwlock):
            self.rwlock = rwlock

        def __enter__(self):
            self.rwlock.write_acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.rwlock.write_release()

    def read_lock(self):
        """返回读锁上下文管理器"""
        return self.ReadLock(self)

    def write_lock(self):
        """返回写锁上下文管理器"""
        return self.WriteLock(self)

###############################################################################
# JsonValidator: 插件式JSON字段值验证器 (保持不变)
###############################################################################
class JsonValidator:
    """用于验证JSON字段值是否在预定义的枚举范围内"""

    def __init__(self):
        """初始化验证器"""
        self.enabled = False
        self.field_rules = {}  # 字段名 -> 允许的值列表

    def configure(self, validation_config: Dict[str, Any]):
        """从配置中加载验证规则"""
        if not validation_config:
            self.enabled = False
            return

        self.enabled = validation_config.get("enabled", False)
        if not self.enabled:
            logging.info("字段值验证功能已禁用")
            return

        # 加载字段验证规则
        rules = validation_config.get("field_rules", {})
        for field, values in rules.items():
            if isinstance(values, list):
                self.field_rules[field] = values
                logging.info(f"加载字段验证规则: {field} -> {len(values)}个可选值")
            else:
                logging.warning(f"字段 {field} 的验证规则格式错误，应为列表")

        logging.info(f"字段值验证功能已启用，共加载 {len(self.field_rules)} 个字段规则")

    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证JSON数据字段值是否在允许范围内"""
        if not self.enabled or not self.field_rules:
            return True, []

        errors = []

        # 检查每个配置了规则的字段
        for field, allowed_values in self.field_rules.items():
            if field in data and allowed_values:
                value = data[field]
                if value not in allowed_values:
                    errors.append(f"字段 '{field}' 的值 '{value}' 不在允许的范围内")

        return len(errors) == 0, errors

###############################################################################
# 令牌桶限流器：用于模型级请求限流 (保持不变)
###############################################################################
class TokenBucket:
    """令牌桶限流器实现"""

    def __init__(self, capacity: float, refill_rate: float):
        """
        初始化令牌桶

        :param capacity: 桶容量（最大令牌数）
        :param refill_rate: 每秒补充的令牌数
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock() # Keep threading.Lock as it's simple state

    def refill(self):
        """根据经过的时间补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        # 根据经过时间计算新增令牌
        new_tokens = elapsed * self.refill_rate
        # 更新令牌数，不超过容量
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        """
        尝试消耗指定数量的令牌

        :param tokens: 要消耗的令牌数
        :return: 如果有足够令牌返回True，否则返回False
        """
        with self.lock:
            # 先补充令牌
            self.refill()
            # 检查是否有足够的令牌
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

###############################################################################
# 模型限流管理器：为每个模型维护独立的令牌桶 (保持不变)
###############################################################################
class ModelRateLimiter:
    """模型限流管理器"""

    def __init__(self):
        """初始化模型限流管理器"""
        self.limiters = {}  # 模型ID -> 令牌桶
        self.lock = threading.Lock()

    def configure(self, models_config: List[Dict[str, Any]]):
        """从模型配置中配置限流器"""
        with self.lock:
            for model in models_config:
                model_id = str(model.get("id"))
                # 从配置或估算中获取安全RPS
                weight = model.get("weight", 1)
                # 根据权重估算安全RPS，避免过于激进
                estimated_rps = max(0.5, min(weight / 10, 10))

                # 配置中显式定义的RPS优先
                safe_rps = model.get("safe_rps", estimated_rps)

                # 创建令牌桶，容量为安全RPS的2倍，允许短时突发
                self.limiters[model_id] = TokenBucket(
                    capacity=safe_rps * 2,
                    refill_rate=safe_rps
                )
                logging.info(f"为模型[{model_id}]配置限流: {safe_rps} RPS")

    def can_process(self, model_id: str) -> bool:
        """
        检查指定模型是否可以处理新请求 (blocking, but fast)

        :param model_id: 模型ID
        :return: 如果可以处理返回True，否则返回False
        """
        model_id = str(model_id)
        # No need for lock here, TokenBucket.consume is thread-safe
        if model_id not in self.limiters:
             return True # No limiter configured, allow processing
        limiter = self.limiters.get(model_id)
        if limiter:
             return limiter.consume(1.0)
        return True # Should not happen if configured correctly

###############################################################################
# ModelDispatcher: 仅对API错误进行退避处理 (保持不变)
###############################################################################
class ModelDispatcher:
    def __init__(self, models: List[Any], backoff_factor: int = 2):
        """
        使用改进的模型调度器:
        1. 读写锁分离 - 使用RWLock允许多个读操作并发
        2. 状态缓存 - 维护模型可用性缓存避免频繁锁操作

        仅对API_ERROR执行退避。

        :param models: 已解析的 ModelConfig 列表
        :param backoff_factor: 指数退避的基数
        """
        self.backoff_factor = backoff_factor

        # 模型状态记录表: model_id -> { fail_count, next_available_ts, ... }
        self._model_state = {}
        for m in models:
            self._model_state[m.id] = {
                "fail_count": 0,
                "next_available_ts": 0,  # 0 表示随时可用
                "success_count": 0,
                "error_count": 0,
                "avg_response_time": 0.0 # Initialize as float
            }

        self._rwlock = RWLock()

        # 可用性缓存和缓存时间戳
        self._availability_cache = {}  # model_id -> is_available
        self._cache_last_update = 0
        self._cache_ttl = 0.5  # 缓存有效期(秒)

        # 初始化缓存
        self._update_availability_cache()

    def _update_availability_cache(self):
        """更新可用性缓存 - 使用写锁保护缓存更新操作"""
        with self._rwlock.write_lock():
            current_time = time.time()
            new_cache = {}
            for model_id, state in self._model_state.items():
                new_cache[model_id] = (current_time >= state["next_available_ts"])
            self._availability_cache = new_cache
            self._cache_last_update = current_time

    def update_model_metrics(self, model_id: str, response_time: float, success: bool):
        """更新模型性能指标"""
        model_id = str(model_id) # Ensure key is string
        with self._rwlock.write_lock():
            if model_id not in self._model_state:
                return
            state = self._model_state[model_id]

            # 更新成功/失败计数
            if success:
                state["success_count"] += 1
            else:
                state["error_count"] += 1

            # 计算平均响应时间（指数移动平均）
            total_calls = state["success_count"] + state["error_count"]
            alpha = 0.1 # Smoothing factor for EMA
            if state["avg_response_time"] == 0:
                 state["avg_response_time"] = response_time
            else:
                 state["avg_response_time"] = alpha * response_time + (1 - alpha) * state["avg_response_time"]

            # Ensure cache is updated if availability changed due to success/failure reset
            current_time = time.time()
            if state["next_available_ts"] != 0 and success: # If it was unavailable and now succeeded
                 state["next_available_ts"] = 0
                 self._availability_cache[model_id] = True
            elif current_time < state["next_available_ts"]: # Still unavailable
                 self._availability_cache[model_id] = False
            else: # Now available
                 self._availability_cache[model_id] = True


    def get_model_success_rate(self, model_id: str) -> float:
        """获取模型的成功率"""
        model_id = str(model_id)
        with self._rwlock.read_lock():
            if model_id not in self._model_state:
                return 0.0
            state = self._model_state[model_id]
            total = state["success_count"] + state["error_count"]
            if total == 0:
                # Avoid division by zero, assume 100% if no calls yet
                return 1.0
            return state["success_count"] / total

    def get_model_avg_response_time(self, model_id: str) -> float:
        """获取模型的平均响应时间"""
        model_id = str(model_id)
        with self._rwlock.read_lock():
            if model_id not in self._model_state:
                # Return a high value if unknown, to deprioritize
                return 10.0
            # Return a default high value if no calls yet
            return self._model_state[model_id]["avg_response_time"] or 10.0

    def is_model_available(self, model_id: str) -> bool:
        """判断某个模型当前是否可用 - 优先使用缓存，减少锁操作"""
        model_id = str(model_id)
        current_time = time.time()
        cache_valid = False
        with self._rwlock.read_lock():
             cache_valid = (current_time - self._cache_last_update < self._cache_ttl)
             if cache_valid and model_id in self._availability_cache:
                  return self._availability_cache[model_id]

        # Cache invalid or model not in cache, need update or direct check
        if not cache_valid:
            self._update_availability_cache() # This acquires write lock internally

        # Re-check cache after update or check state directly
        with self._rwlock.read_lock():
            if model_id in self._availability_cache:
                return self._availability_cache[model_id]
            # Fallback if somehow still not in cache (shouldn't happen)
            if model_id in self._model_state:
                st = self._model_state[model_id]
                is_avail = (current_time >= st["next_available_ts"])
                # Optionally update cache here if needed, but _update_availability_cache should handle it
                return is_avail
            return False # Model ID unknown

    def mark_model_success(self, model_id: str):
        """模型调用成功时，重置其失败计数"""
        model_id = str(model_id)
        with self._rwlock.write_lock():
            if model_id in self._model_state:
                state = self._model_state[model_id]
                if state["fail_count"] > 0 or state["next_available_ts"] > 0:
                     state["fail_count"] = 0
                     state["next_available_ts"] = 0
                     self._availability_cache[model_id] = True
                     logging.debug(f"Model [{model_id}] marked as successful and available.")


    def mark_model_failed(self, model_id: str, error_type: str = ErrorType.API_ERROR):
        """模型调用失败时的处理，只有 API_ERROR 才会导致退避"""
        model_id = str(model_id)
        if error_type != ErrorType.API_ERROR:
            return

        with self._rwlock.write_lock():
            if model_id not in self._model_state:
                return
            st = self._model_state[model_id]
            st["fail_count"] += 1
            fail_count = st["fail_count"]

            # 更温和的退避算法（线性 + 指数的混合）
            if fail_count <= 3:
                backoff_seconds = fail_count * 2
            else:
                backoff_seconds = min(
                    6 + (self.backoff_factor ** (fail_count - 3)),
                    60  # Max backoff 60 seconds
                )
            st["next_available_ts"] = time.time() + backoff_seconds
            self._availability_cache[model_id] = False
            logging.warning(
                f"模型[{model_id}] API调用失败，第{fail_count}次，进入退避 {backoff_seconds:.1f} 秒"
            )

    def get_available_models(self, exclude_model_ids: Set[str] = None) -> List[str]:
        """获取所有当前可用的模型ID"""
        if exclude_model_ids is None:
            exclude_model_ids = set()
        else:
            # Ensure provided IDs are strings
            exclude_model_ids = set(map(str, exclude_model_ids))


        current_time = time.time()
        cache_valid = False
        with self._rwlock.read_lock():
             cache_valid = (current_time - self._cache_last_update < self._cache_ttl)

        if not cache_valid:
            self._update_availability_cache()

        available_models = []
        with self._rwlock.read_lock():
            for model_id, is_available in self._availability_cache.items():
                if is_available and model_id not in exclude_model_ids:
                    available_models.append(model_id)
        return available_models


###############################################################################
# 抽象任务池基类 (Modified: Added get_single_task)
###############################################################################
class BaseTaskPool(ABC):
    """任务池抽象基类：定义数据源的通用接口"""

    def __init__(self, columns_to_extract: List[str], columns_to_write: Dict[str, str]):
        self.columns_to_extract = columns_to_extract
        self.columns_to_write = columns_to_write
        # Use deque for efficient pop(0) if performance becomes an issue with large shards
        self.tasks: List[Tuple[Any, Dict[str, Any]]] = []
        self.lock = threading.Lock() # Lock for accessing self.tasks list

    @abstractmethod
    def get_total_task_count(self) -> int:
        """获取尚未处理的任务总数 (Estimate for progress)"""
        pass

    @abstractmethod
    def get_id_boundaries(self) -> Tuple[Optional[Any], Optional[Any]]:
        """获取可用于分片的记录ID或索引范围，返回(最小ID, 最大ID)"""
        pass

    @abstractmethod
    def initialize_shard(self, shard_id: int, min_id: Any, max_id: Any) -> int:
        """
        初始化指定分片，将任务加载到内部 'self.tasks' 列表.
        Returns the number of tasks loaded into the internal list.
        """
        pass

    def get_single_task(self) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """获取一个任务，如果没有可用任务返回None"""
        with self.lock:
            if not self.tasks:
                return None
            return self.tasks.pop(0) # Get from the front (FIFO)

    @abstractmethod
    def update_task_results(self, results: Dict[Any, Dict[str, Any]]):
        """批量更新任务处理结果 (Potentially blocking, run in executor)"""
        pass

    @abstractmethod
    def reload_task_data(self, task_id: Any) -> Optional[Dict[str, Any]]:
        """重新加载任务数据，用于系统错误重试 (Potentially blocking, run in executor)"""
        pass

    # Keep add_task_to_front for system error requeueing
    def add_task_to_front(self, task_id: Any, record_dict: Optional[Dict[str, Any]]):
        """将任务重新插回队列头部，用于系统错误重试"""
        if record_dict is None:
             logging.error(f"无法重新加载任务 {task_id} 的数据，任务将丢失。")
             return
        with self.lock:
            self.tasks.insert(0, (task_id, record_dict))

    def has_tasks(self) -> bool:
        """检查内部列表是否还有待处理任务"""
        with self.lock:
            return len(self.tasks) > 0

    def get_remaining_count_in_shard(self) -> int:
        """获取当前分片中剩余任务数量"""
        with self.lock:
            return len(self.tasks)

    @abstractmethod
    def close(self):
         """Close any resources like database connections or file handles."""
         pass


###############################################################################
# MySQL任务池实现 (Modified: Adapts to new BaseTaskPool structure)
###############################################################################
class MySQLTaskPool(BaseTaskPool):
    """
    MySQL数据源任务池，支持分片加载.
    """

    def __init__(
        self,
        connection_config: Dict[str, Any],
        columns_to_extract: List[str],
        columns_to_write: Dict[str, str],
        table_name: str,
        id_column: str = 'id' # Allow specifying ID column
    ):
        super().__init__(columns_to_extract, columns_to_write)

        if not MYSQL_AVAILABLE:
            raise ImportError("MySQL Connector库未安装，无法使用MySQL数据源")

        self.connection_config = connection_config
        self.table_name = table_name
        self.id_column = id_column # Store the ID column name

        # 创建连接池
        self.pool = MySQLConnectionPool(
            connection_config=connection_config,
            min_size=5,
            max_size=20
        )

        # 分片状态 (No longer needed here, managed by ShardedTaskManager)
        # self.current_shard_id = -1
        # self.current_min_id = 0
        # self.current_max_id = 0

    def execute_with_connection(self, callback):
        """Helper to execute DB operations with connection management."""
        conn = None
        try:
            conn = self.pool.get_connection()
            if conn is None:
                # Log error and maybe raise a specific exception?
                logging.error("无法从池中获取数据库连接")
                raise ConnectionError("无法获取数据库连接")
            return callback(conn)
        except mysql.connector.Error as err:
            logging.error(f"MySQL 操作失败: {err}")
            # Decide if the error is retryable or fatal
            # For now, re-raise to indicate failure
            raise
        finally:
            if conn:
                self.pool.release_connection(conn)

    def _build_unprocessed_condition(self) -> str:
        """
        构造"未处理"条件：
         - 输入列均非空
         - 输出列至少有一个为空
        """
        # Input columns not null or empty
        input_conditions = []
        for col in self.columns_to_extract:
            # Handle potential backticks in column names if needed, but usually not required
            c = f"`{col}` IS NOT NULL AND `{col}` <> ''"
            input_conditions.append(c)
        input_condition_str = "(" + " AND ".join(input_conditions) + ")" if input_conditions else "1"

        # At least one output column is null or empty
        output_conditions = []
        if not self.columns_to_write: # If no output columns, process everything matching input criteria
             output_condition_str = "1"
        else:
             for _, out_col in self.columns_to_write.items():
                 cond = f"(`{out_col}` IS NULL OR `{out_col}` = '')"
                 output_conditions.append(cond)
             output_condition_str = "(" + " OR ".join(output_conditions) + ")" if output_conditions else "0" # Should not be 0 if columns_to_write exists

        # Combine conditions
        unprocessed_where = f"{input_condition_str} AND {output_condition_str}"
        return unprocessed_where

    def get_total_task_count(self) -> int:
        """获取尚未处理的任务数 (Estimate)"""
        def _get_count(conn):
            try:
                cursor = conn.cursor()
                unprocessed_where = self._build_unprocessed_condition()
                # Use COUNT_BIG for potentially very large tables
                sql = f"SELECT COUNT(*) FROM `{self.table_name}` WHERE {unprocessed_where}"
                logging.debug(f"Executing count query: {sql}")
                cursor.execute(sql)
                result = cursor.fetchone()
                cursor.close()
                return int(result[0]) if result else 0
            except Exception as e:
                logging.error(f"统计未处理任务总数失败: {e}")
                return 0 # Return 0 on error to avoid blocking initialization
        try:
            return self.execute_with_connection(_get_count)
        except ConnectionError:
             logging.error("获取总任务数时连接数据库失败。")
             return 0


    def get_id_boundaries(self) -> Tuple[Optional[Any], Optional[Any]]:
        """获取全表ID范围 (Uses configured ID column)"""
        def _get_boundaries(conn):
            try:
                cursor = conn.cursor()
                # Use the configured ID column
                sql = f"SELECT MIN(`{self.id_column}`), MAX(`{self.id_column}`) FROM `{self.table_name}`"
                logging.debug(f"Executing ID boundary query: {sql}")
                cursor.execute(sql)
                result = cursor.fetchone()
                cursor.close()
                if result and result[0] is not None and result[1] is not None:
                    # Return raw values, let ShardedTaskManager handle type if needed
                    return (result[0], result[1])
                return (None, None) # Indicate empty or single row table
            except Exception as e:
                logging.error(f"获取ID范围失败: {e}")
                return (None, None)
        try:
            return self.execute_with_connection(_get_boundaries)
        except ConnectionError:
             logging.error("获取ID范围时连接数据库失败。")
             return (None, None)

    def initialize_shard(self, shard_id: int, min_id: Any, max_id: Any) -> int:
        """
        载入一个分片范围内的所有"未处理"记录到 self.tasks
        """
        def _load_shard(conn):
            loaded_tasks = []
            try:
                cursor = conn.cursor(dictionary=True) # Use dictionary cursor
                # Ensure all required columns exist
                all_cols = [self.id_column] + self.columns_to_extract
                columns_str = ", ".join(f"`{col}`" for col in all_cols)
                unprocessed_where = self._build_unprocessed_condition()

                # Use the configured ID column for filtering
                # Use placeholders for shard boundaries
                sql = f"""
                    SELECT {columns_str}
                    FROM `{self.table_name}`
                    WHERE `{self.id_column}` BETWEEN %s AND %s
                      AND {unprocessed_where}
                    ORDER BY `{self.id_column}`
                """
                logging.debug(f"Loading shard {shard_id} with query: {sql % (min_id, max_id)}") # Log query with params
                cursor.execute(sql, (min_id, max_id))

                rows = cursor.fetchall()
                cursor.close()

                for row in rows:
                    record_id = row[self.id_column]
                    record_dict = {c: (row.get(c, "") or "") for c in self.columns_to_extract}
                    loaded_tasks.append((record_id, record_dict))

                # --- Critical Section for self.tasks ---
                with self.lock:
                     # Prepend loaded tasks to allow requeued tasks to be processed first
                     # Or append if strict shard order is desired: self.tasks.extend(loaded_tasks)
                     self.tasks = loaded_tasks + self.tasks
                # --- End Critical Section ---

                loaded_count = len(loaded_tasks)
                logging.info(f"分片 {shard_id} ({self.id_column}范围: {min_id}-{max_id}), 加载了 {loaded_count} 个未处理任务")
                return loaded_count

            except Exception as e:
                logging.error(f"加载分片 {shard_id} 失败: {e}")
                # Consider implications: should it retry? Skip shard?
                return 0 # Return 0 to indicate failure for this shard

        try:
            return self.execute_with_connection(_load_shard)
        except ConnectionError:
             logging.error(f"加载分片 {shard_id} 时连接数据库失败。")
             return 0


    # get_task_batch is removed, use get_single_task instead

    def update_task_results(self, results: Dict[Any, Dict[str, Any]]):
        """
        将结果写回输出列 (Blocking - run in executor).
        Only updates results without "_error".
        Uses CASE WHEN for bulk updates, falls back to single updates.
        """
        if not results or not self.columns_to_write:
            return

        success_updates = {}
        for record_id, row_result in results.items():
            if "_error" not in row_result:
                update_data = {}
                for alias, out_col in self.columns_to_write.items():
                    update_data[out_col] = row_result.get(alias, None) # Use None for potential NULLs
                if update_data: # Only include if there are columns to write
                    success_updates[record_id] = update_data

        if not success_updates:
             logging.debug("No successful results to update in this batch.")
             return

        def _update(conn):
            updated_count = 0
            try:
                cursor = conn.cursor()
                set_clauses = []
                all_ids = list(success_updates.keys())
                ids_str = ', '.join(map(str, all_ids)) # Prepare for IN clause

                # Group values by output column for CASE WHEN
                values_by_col = defaultdict(dict)
                for rid, data in success_updates.items():
                    for col, val in data.items():
                        values_by_col[col][rid] = val

                for out_col, val_dict in values_by_col.items():
                    clause = f"`{out_col}` = CASE `{self.id_column}` "
                    params = []
                    for rid, val in val_dict.items():
                        clause += f"WHEN %s THEN %s "
                        params.extend([rid, val])
                    clause += f"ELSE `{out_col}` END"
                    set_clauses.append((clause, params)) # Store clause and its params

                if not set_clauses:
                     conn.rollback() # Should not happen if success_updates is not empty
                     logging.warning("No SET clauses generated for update.")
                     return 0

                # Combine SET clauses and parameters
                set_sql_parts = [c[0] for c in set_clauses]
                all_params = []
                for c in set_clauses:
                    all_params.extend(c[1])

                sql = f"""
                    UPDATE `{self.table_name}`
                    SET {', '.join(set_sql_parts)}
                    WHERE `{self.id_column}` IN ({','.join(['%s'] * len(all_ids))})
                """
                all_params.extend(all_ids) # Add IDs for the WHERE IN clause

                logging.debug(f"Executing bulk update SQL (params omitted for brevity): {sql}")
                # logging.debug(f"Bulk update params: {all_params}")
                cursor.execute(sql, all_params)
                updated_count = cursor.rowcount # Get affected rows
                conn.commit()
                logging.info(f"MySQL: 批量更新完成 (CASE WHEN), 更新数量={updated_count}")
                cursor.close()
                return updated_count

            except mysql.connector.Error as e:
                conn.rollback()
                logging.error(f"MySQL: 批量更新失败 (CASE WHEN), 将尝试单条更新: {e}")
                # Fallback to single updates
                single_update_count = 0
                try:
                    cursor = conn.cursor() # New cursor for fallback
                    for record_id, data in success_updates.items():
                        set_parts = []
                        params = []
                        for col, val in data.items():
                            set_parts.append(f"`{col}` = %s")
                            params.append(val)

                        if not set_parts: continue # Skip if no data for this ID

                        sql_single = f"""
                            UPDATE `{self.table_name}`
                            SET {', '.join(set_parts)}
                            WHERE `{self.id_column}` = %s
                        """
                        params.append(record_id)
                        try:
                             cursor.execute(sql_single, params)
                             single_update_count += cursor.rowcount
                        except mysql.connector.Error as e_single:
                             logging.error(f"MySQL: 单条更新记录 {record_id} 失败: {e_single}")
                             # Decide whether to rollback all single updates or just log
                             conn.rollback() # Rollback the single failed one, maybe continue?
                             # For simplicity, rollback and log for now
                             break # Stop fallback on first error

                    if single_update_count > 0:
                         conn.commit() # Commit successful single updates
                    logging.info(f"MySQL: 单条更新模式完成, 更新数量={single_update_count}")
                    cursor.close()
                    return single_update_count
                except Exception as e2:
                    conn.rollback()
                    logging.error(f"MySQL: 单条更新模式也失败: {e2}")
                    cursor.close()
                    return 0 # Indicate failure
            finally:
                 # Ensure cursor is closed if open
                 if 'cursor' in locals() and cursor and not cursor.is_closed():
                      cursor.close()

        try:
            self.execute_with_connection(_update)
        except (ConnectionError, mysql.connector.Error) as db_err:
             logging.error(f"更新结果时数据库连接或操作失败: {db_err}")
             # Here, the results are lost for this batch update.
             # Consider adding them back to a retry queue or logging failed IDs.


    def reload_task_data(self, record_id: Any) -> Optional[Dict[str, Any]]:
        """重新加载原始输入列 (Blocking - run in executor)"""
        def _reload(conn):
            record_dict = {}
            try:
                cursor = conn.cursor(dictionary=True)
                cols_str = ", ".join(f"`{c}`" for c in self.columns_to_extract)
                # Use configured ID column and placeholder
                sql = f"SELECT {cols_str} FROM `{self.table_name}` WHERE `{self.id_column}`=%s"
                cursor.execute(sql, (record_id,))
                row = cursor.fetchone()
                cursor.close()
                if row:
                    for c in self.columns_to_extract:
                        record_dict[c] = row.get(c, "") or ""
                    return record_dict
                else:
                     logging.warning(f"reload_task_data: 无法找到记录 {self.id_column}={record_id}")
                     return None
            except Exception as e:
                logging.error(f"reload_task_data for ID {record_id} 失败: {e}")
                # Ensure cursor is closed on error
                if 'cursor' in locals() and cursor and not cursor.is_closed():
                    cursor.close()
                return None # Indicate failure
        try:
            return self.execute_with_connection(_reload)
        except (ConnectionError, mysql.connector.Error) as db_err:
             logging.error(f"重新加载任务 {record_id} 时数据库连接或操作失败: {db_err}")
             return None


    def close(self):
        """关闭数据库连接池"""
        if hasattr(self, 'pool') and self.pool:
            logging.info("Closing MySQL connection pool...")
            self.pool.close_all()
            logging.info("MySQL connection pool closed.")


###############################################################################
# MySQL连接池管理 (保持不变)
###############################################################################
class MySQLConnectionPool:
    """线程安全的MySQL连接池管理器"""

    def __init__(self, connection_config: Dict[str, Any], min_size=5, max_size=20):
        self.config = connection_config
        self.min_size = min_size
        self.max_size = max_size
        self._pool: List[mysql.connector.MySQLConnection] = []
        self._in_use: Set[mysql.connector.MySQLConnection] = set()
        self._lock = threading.Lock()
        self._creating = 0 # Track connections being created

        # Initialize minimum connections
        for _ in range(min_size):
            self._create_and_add_connection()


    def _create_connection(self) -> Optional[mysql.connector.MySQLConnection]:
        """Creates a single MySQL connection."""
        try:
            logging.debug("Creating new MySQL connection...")
            conn = mysql.connector.connect(
                host=self.config["host"],
                port=self.config.get("port", 3306),
                user=self.config["user"],
                password=self.config["password"],
                database=self.config["database"],
                use_pure=True, # Recommended for cross-platform compatibility
                autocommit=False, # Important for transactions
                pool_reset_session=True, # Reset session state on release
                connection_timeout=10, # Connection attempt timeout
                # get_warnings=True, # Can be noisy
                # raise_on_warnings=False
            )
            logging.debug(f"MySQL connection created successfully (ID: {conn.connection_id}).")
            return conn
        except mysql.connector.Error as e:
            logging.error(f"创建数据库连接失败: {e}")
            return None

    def _create_and_add_connection(self):
        """Creates a connection and adds it to the pool under lock."""
        with self._lock:
             self._creating += 1
        conn = self._create_connection()
        with self._lock:
            self._creating -= 1
            if conn:
                self._pool.append(conn)
                # Notify waiting threads if any (though get_connection uses polling)
                # self._lock.notify_all() # If using Condition variable
                return True
            return False

    def get_connection(self, timeout=5) -> Optional[mysql.connector.MySQLConnection]:
        """Gets a connection from the pool, creating one if necessary and allowed."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                # 1. Try to find an available connection in the pool
                for conn in self._pool:
                    if conn not in self._in_use:
                        try:
                            # Ping the connection to check if it's alive
                            if conn.is_connected():
                                 conn.ping(reconnect=True, attempts=1, delay=0) # Try to reconnect once if needed
                                 if conn.is_connected():
                                      self._in_use.add(conn)
                                      logging.debug(f"Reusing connection {conn.connection_id} from pool.")
                                      return conn
                                 else: # Ping failed even after reconnect attempt
                                      logging.warning(f"Connection {conn.connection_id} is dead after ping/reconnect. Removing.")
                                      self._pool.remove(conn)
                                      # Try to replace it immediately if below min_size
                                      if len(self._pool) + len(self._in_use) + self._creating < self.min_size:
                                           threading.Thread(target=self._create_and_add_connection).start()
                                      continue # Try next connection in the pool
                            else: # Not connected initially
                                 logging.warning(f"Connection {conn.connection_id} not connected. Removing.")
                                 self._pool.remove(conn)
                                 if len(self._pool) + len(self._in_use) + self._creating < self.min_size:
                                     threading.Thread(target=self._create_and_add_connection).start()
                                 continue # Try next connection in the pool
                        except mysql.connector.Error as ping_err:
                            logging.warning(f"Ping failed for connection {conn.connection_id}. Removing. Error: {ping_err}")
                            try:
                                conn.close() # Attempt to close gracefully
                            except Exception: pass # Ignore errors during close
                            if conn in self._pool: self._pool.remove(conn) # Ensure removal
                            if conn in self._in_use: self._in_use.remove(conn) # Remove if somehow marked in use
                            if len(self._pool) + len(self._in_use) + self._creating < self.min_size:
                                threading.Thread(target=self._create_and_add_connection).start()
                            continue # Try next connection

                # 2. If no available connection, try to create a new one if below max_size
                current_total = len(self._pool) + len(self._in_use) + self._creating
                if current_total < self.max_size:
                    logging.debug(f"No available connections. Creating new one (Total: {current_total}/{self.max_size}).")
                    # Increment creating count under lock
                    self._creating += 1

            # Create connection outside the main lock to avoid blocking others
            new_conn = self._create_connection()

            with self._lock:
                 # Decrement creating count
                 self._creating -= 1
                 if new_conn:
                     # Add to pool and mark as in use
                     self._pool.append(new_conn)
                     self._in_use.add(new_conn)
                     logging.debug(f"Returning newly created connection {new_conn.connection_id}.")
                     return new_conn
                 else:
                      # Creation failed, loop will continue if timeout not exceeded
                      logging.warning("Failed to create new connection.")


            # Short sleep before retrying
            time.sleep(0.1)

        logging.error(f"获取数据库连接超时 ({timeout}秒)")
        return None


    def release_connection(self, conn: Optional[mysql.connector.MySQLConnection]):
        """Releases a connection back to the pool."""
        if conn is None:
            return
        with self._lock:
            if conn in self._in_use:
                try:
                    # Rollback any pending transaction before releasing
                    if conn.in_transaction:
                        conn.rollback()
                        logging.debug(f"Rolled back transaction for connection {conn.connection_id} on release.")
                    self._in_use.remove(conn)
                    logging.debug(f"Released connection {conn.connection_id} back to pool.")

                    # Optional: Prune excess idle connections if pool size > min_size
                    # (Could be done here or in a separate maintenance thread)

                except mysql.connector.Error as e:
                    logging.warning(f"Error releasing/resetting connection {conn.connection_id}: {e}. Closing and removing.")
                    self._in_use.remove(conn) # Ensure removed from in_use
                    if conn in self._pool: self._pool.remove(conn) # Remove from pool
                    try:
                        conn.close()
                    except Exception: pass # Ignore close errors
                    # Optionally try to replenish if below min_size
                    if len(self._pool) + len(self._in_use) + self._creating < self.min_size:
                         threading.Thread(target=self._create_and_add_connection).start()
                except Exception as e: # Catch unexpected errors during release
                     logging.error(f"Unexpected error releasing connection {conn.connection_id}: {e}. Attempting removal.")
                     if conn in self._in_use: self._in_use.remove(conn)
                     if conn in self._pool: self._pool.remove(conn)
                     try: conn.close()
                     except: pass
                     if len(self._pool) + len(self._in_use) + self._creating < self.min_size:
                         threading.Thread(target=self._create_and_add_connection).start()

            else:
                # Connection not recognized or already released
                logging.warning(f"Attempted to release connection {conn.connection_id} not marked as in use.")


    def close_all(self):
        """Closes all connections in the pool."""
        with self._lock:
            logging.info(f"Closing all {len(self._pool)} connections in the pool...")
            closed_count = 0
            # Close connections currently in the pool
            for conn in list(self._pool): # Iterate over a copy
                try:
                    conn.close()
                    closed_count += 1
                except Exception as e:
                    logging.warning(f"Error closing connection {conn.connection_id}: {e}")
            self._pool.clear()

            # Also attempt to close connections marked as "in_use" (they shouldn't be, ideally)
            if self._in_use:
                 logging.warning(f"Closing {len(self._in_use)} connections still marked as 'in_use'.")
                 for conn in list(self._in_use):
                      try:
                           conn.close()
                           closed_count += 1
                      except Exception as e:
                           logging.warning(f"Error closing 'in_use' connection {conn.connection_id}: {e}")
                 self._in_use.clear()
            logging.info(f"Closed {closed_count} total connections.")

###############################################################################
# Excel任务池实现 (Modified: Adapts to new BaseTaskPool structure)
###############################################################################
class ExcelTaskPool(BaseTaskPool):
    """
    Excel数据源任务池，支持分片加载. Uses DataFrame index as task ID.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        columns_to_extract: List[str],
        columns_to_write: Dict[str, str],
        output_excel: str,
        save_interval: int = 300 # Pass save interval from config
    ):
        super().__init__(columns_to_extract, columns_to_write)
        self.df = df
        self.output_excel = output_excel
        self.df_lock = threading.Lock() # Lock for DataFrame modifications and saving

        # Make sure output columns exist, convert to object type for flexibility
        for _, out_col in self.columns_to_write.items():
            if out_col not in self.df.columns:
                self.df[out_col] = pd.Series(dtype='object') # Create as object type
            elif self.df[out_col].dtype != 'object':
                 # Convert existing columns if they are not object type to avoid type issues
                 try:
                      self.df[out_col] = self.df[out_col].astype('object')
                 except Exception as e:
                      logging.warning(f"Could not convert column '{out_col}' to object type: {e}")


        # Periodic saving
        self.last_save_time = time.time()
        self.save_interval = save_interval

        self.total_rows = len(df)


    def _filter_unprocessed_indices(self, min_idx: int, max_idx: int) -> List[int]:
        """
        Returns [min_idx, max_idx] range indices of "unprocessed" rows.
        This is potentially slow for large slices, consider optimization if needed.
        """
        # Ensure slice boundaries are valid
        min_idx = max(0, min_idx)
        max_idx = min(len(self.df) - 1, max_idx)
        if min_idx > max_idx:
             return []

        try:
             # Operate on a slice, careful about SettingWithCopyWarning if modifying slice
             sub_df = self.df.iloc[min_idx : max_idx + 1]

             # Input columns not null or empty string
             cond_input = pd.Series(True, index=sub_df.index)
             for col in self.columns_to_extract:
                 if col not in sub_df.columns:
                     cond_input &= False # Treat missing columns as invalid input
                     break # No need to check other input cols if one is missing
                 else:
                      # Check for non-null AND non-empty string representation
                      # Convert to string first to handle various types correctly
                      is_valid = sub_df[col].notna() & (sub_df[col].astype(str).str.strip() != '')
                      cond_input &= is_valid

             if not cond_input.any(): # Optimization: if no rows satisfy input criteria
                 return []

             # At least one output column is null or empty string
             cond_output = pd.Series(False, index=sub_df.index)
             if not self.columns_to_write: # If no output cols, process all with valid input
                 cond_output = pd.Series(True, index=sub_df.index)
             else:
                 for _, out_col in self.columns_to_write.items():
                     if out_col not in sub_df.columns:
                         # If output column doesn't exist, it's considered empty
                         cond_output |= True
                     else:
                         # Check for null OR empty string representation
                         is_empty = sub_df[out_col].isna() | (sub_df[out_col].astype(str).str.strip() == '')
                         cond_output |= is_empty


             # Combine conditions: valid input AND empty output required
             final_cond = cond_input & cond_output
             return sub_df.index[final_cond].tolist()

        except Exception as e:
            logging.error(f"Error filtering unprocessed indices [{min_idx}-{max_idx}]: {e}")
            return []


    def get_total_task_count(self) -> int:
        """Estimates total task count by checking the whole DataFrame."""
        logging.info("Estimating total tasks for Excel... (This might take a while)")
        # This can be slow for very large excels. Consider sampling or alternative later.
        try:
             unprocessed_indices = self._filter_unprocessed_indices(0, len(self.df) - 1)
             count = len(unprocessed_indices)
             logging.info(f"Estimated total tasks: {count}")
             return count
        except Exception as e:
             logging.error(f"Failed to estimate total tasks: {e}")
             return 0 # Avoid blocking startup


    def get_id_boundaries(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Returns the index range (0 to len-1)"""
        if len(self.df) == 0:
             return (None, None)
        return (0, len(self.df) - 1)

    def initialize_shard(self, shard_id: int, min_idx: int, max_idx: int) -> int:
        """Load tasks for the given index range into self.tasks."""
        loaded_tasks = []
        try:
            unprocessed_idx = self._filter_unprocessed_indices(min_idx, max_idx)
            if not unprocessed_idx:
                logging.info(f"分片 {shard_id} (索引范围: {min_idx}-{max_idx}), 没有未处理任务")
                return 0

            for idx in unprocessed_idx:
                record_dict = {}
                valid_input = True
                for c in self.columns_to_extract:
                    if c not in self.df.columns:
                        logging.warning(f"分片 {shard_id}, 记录 {idx}: 缺少输入列 '{c}', 跳过")
                        valid_input = False
                        break
                    cell_val = self.df.at[idx, c]
                    # Ensure value is string, handle NaN/None
                    val = str(cell_val) if pd.notna(cell_val) else ""
                    record_dict[c] = val
                if valid_input:
                     loaded_tasks.append((idx, record_dict)) # Use index as ID

            # --- Critical Section for self.tasks ---
            with self.lock:
                 self.tasks = loaded_tasks + self.tasks # Prepend
            # --- End Critical Section ---

            loaded_count = len(loaded_tasks)
            logging.info(f"分片 {shard_id} (索引范围: {min_idx}-{max_idx}), 加载了 {loaded_count} 个未处理任务")
            return loaded_count
        except Exception as e:
            logging.error(f"初始化Excel分片 {shard_id} 失败: {e}")
            return 0

    def update_task_results(self, results: Dict[int, Dict[str, Any]]):
        """
        Update DataFrame with results (Blocking - run in executor).
        Handles periodic saving.
        """
        if not results or not self.columns_to_write:
            return

        updated_count = 0
        try:
            # --- Critical Section for DataFrame ---
            with self.df_lock:
                for idx, row_result in results.items():
                    if "_error" not in row_result:
                        if idx in self.df.index: # Check if index still exists
                            for alias, col_name in self.columns_to_write.items():
                                if col_name in self.df.columns:
                                     # Use .at for fast label-based scalar access
                                     # Convert result to string or handle types appropriately
                                     value_to_write = row_result.get(alias, "")
                                     # Ensure compatibility with object dtype if needed
                                     self.df.at[idx, col_name] = str(value_to_write) if pd.notna(value_to_write) else ""
                            updated_count += 1
                        else:
                             logging.warning(f"Index {idx} not found in DataFrame during update, skipping.")

                current_time = time.time()
                if current_time - self.last_save_time > self.save_interval:
                    self._save_excel_internal() # Save within the lock
                    self.last_save_time = current_time # Update time only on successful save
            # --- End Critical Section ---
            if updated_count > 0:
                 logging.debug(f"Updated {updated_count} rows in DataFrame.")

        except Exception as e:
            logging.error(f"更新Excel记录失败: {e}")
            # Results for this batch might be lost if save fails later

    def _save_excel_internal(self):
        """Internal save method, must be called under df_lock."""
        try:
            # Create backup before overwriting
            backup_path = self.output_excel + ".bak"
            if os.path.exists(self.output_excel):
                 # Simple backup, could be more robust (e.g., timestamped)
                 os.replace(self.output_excel, backup_path)
                 logging.info(f"Created backup: {backup_path}")

            logging.info(f"Saving data to Excel: {self.output_excel}...")
            # Consider using a more performant engine like 'openpyxl' if needed
            self.df.to_excel(self.output_excel, index=False, engine='openpyxl')
            logging.info(f"Successfully saved data to {self.output_excel}")
            # Optionally remove backup on successful save
            # if os.path.exists(backup_path): os.remove(backup_path)
        except Exception as e:
            logging.error(f"保存Excel文件失败: {e}")
            # Attempt to restore backup
            if os.path.exists(backup_path):
                 try:
                      os.replace(backup_path, self.output_excel)
                      logging.warning("Restored Excel from backup due to save failure.")
                 except Exception as restore_e:
                      logging.error(f"Failed to restore Excel from backup: {restore_e}")
            # Re-raise or handle the save error appropriately
            raise # Propagate the error


    def reload_task_data(self, idx: int) -> Optional[Dict[str, Any]]:
        """Reload input data for a given index (Blocking - run in executor)."""
        record_dict = {}
        try:
            # No need for lock for read-only .at access if df structure doesn't change
            if idx in self.df.index:
                valid_input = True
                for c in self.columns_to_extract:
                     if c not in self.df.columns:
                          logging.error(f"reload_task_data: 缺少列 '{c}' 无法重新加载任务 {idx}")
                          return None # Cannot reload if input column missing
                     cell_val = self.df.at[idx, c]
                     record_dict[c] = str(cell_val) if pd.notna(cell_val) else ""
                return record_dict
            else:
                logging.warning(f"reload_task_data: 行索引={idx} 不存在")
                return None
        except Exception as e:
            logging.error(f"reload_task_data for index {idx} 失败: {e}")
            return None

    def close(self):
        """Save the final Excel state."""
        logging.info("Performing final save for Excel data...")
        try:
             with self.df_lock: # Ensure no updates happen during final save
                  self._save_excel_internal()
             logging.info("Final Excel save complete.")
        except Exception as e:
             logging.error(f"Final Excel save failed: {e}")


###############################################################################
# 分片任务管理器 (Modified: Simplified, tracks progress)
###############################################################################
class ShardedTaskManager:
    """分片任务管理器，负责分片逻辑和进度跟踪"""

    def __init__(
        self,
        task_pool: BaseTaskPool,
        optimal_shard_size=10000,
        min_shard_size=1000,
        max_shard_size=50000
    ):
        self.task_pool = task_pool
        self.optimal_shard_size = optimal_shard_size
        self.min_shard_size = min_shard_size
        self.max_shard_size = max_shard_size

        self.current_shard_index = 0
        self.total_shards = 0
        self.shard_boundaries: List[Tuple[Any, Any]] = []

        self.total_estimated_tasks = 0 # Initial estimate
        self.total_processed_count = 0 # Updated by Processor
        self.start_time = time.time()

        # Memory monitoring (can be kept)
        self.memory_tracker = {
            'last_check_time': time.time(),
            'check_interval': 60, # Check every 60 seconds
            'peak_memory_mb': 0,
            'current_memory_mb': 0
        }

        self.all_shards_defined = False


    def calculate_optimal_shard_size(self, total_range: Optional[int]) -> int:
         """Dynamically calculate shard size based on memory and range."""
         calculated_size = self.optimal_shard_size # Start with default
         try:
              mem = psutil.virtual_memory()
              available_mb = mem.available / (1024 * 1024)
              # Estimate memory per record (highly dependent on data)
              # Let's assume roughly 1MB per 100 records as a safe starting point
              memory_based_size = int(available_mb * 0.1 / (1/100)) # Use 10% of available RAM
              memory_based_size = max(self.min_shard_size, memory_based_size) # Apply min limit

              calculated_size = min(self.max_shard_size, memory_based_size)

              # Adjust based on total range if available
              if total_range and total_range > 0:
                    # Avoid excessively small shards if range is large
                    num_shards_ideal = max(1, total_range / calculated_size)
                    if num_shards_ideal > 200: # Limit max number of shards
                         calculated_size = max(calculated_size, int(total_range / 200))
                         calculated_size = max(self.min_shard_size, calculated_size) # Ensure min size
                         calculated_size = min(self.max_shard_size, calculated_size) # Ensure max size


              logging.info(f"动态计算的分片大小: {calculated_size} (内存可用: {available_mb:.1f}MB, 范围: {total_range})")

         except Exception as e:
              logging.warning(f"计算分片大小出错，使用默认值 {self.optimal_shard_size}: {e}")
              calculated_size = self.optimal_shard_size

         # Final clamp
         shard_size = max(self.min_shard_size, min(calculated_size, self.max_shard_size))
         return shard_size


    def initialize(self) -> bool:
        """Defines shard boundaries based on ID range and calculated size."""
        try:
            self.total_estimated_tasks = self.task_pool.get_total_task_count()
            if self.total_estimated_tasks == 0:
                logging.info("预估未处理任务数为 0，可能已全部完成或数据源为空。")
                # Allow proceeding, load_next_shard will handle empty shards
                # return False

            min_id, max_id = self.task_pool.get_id_boundaries()

            if min_id is None or max_id is None:
                 logging.warning("无法获取有效的ID范围，将尝试加载整个数据源作为一个分片。")
                 self.total_shards = 1
                 self.shard_boundaries = [(None, None)] # Special case for single shard load
                 self.all_shards_defined = True
                 return True

            logging.info(f"数据ID范围: {min_id} - {max_id}, 预估未处理任务数: {self.total_estimated_tasks}")

            # Determine if IDs are numeric for range calculation
            is_numeric = isinstance(min_id, (int, float)) and isinstance(max_id, (int, float))
            total_range = None
            if is_numeric:
                 total_range = int(max_id) - int(min_id) + 1
                 if total_range <= 0:
                     logging.warning(f"ID范围无效或只有单条记录 ({total_range})。将加载为一个分片。")
                     self.total_shards = 1
                     self.shard_boundaries = [(min_id, max_id)]
                     self.all_shards_defined = True
                     return True
            else:
                 logging.info("ID范围非数字，将基于预估任务数和默认分片大小进行分片。")
                 # Cannot calculate range, rely on optimal_shard_size based on memory/defaults
                 total_range = None # Indicate non-numeric range


            shard_size = self.calculate_optimal_shard_size(total_range)

            if not is_numeric:
                 # Estimate shards based on total tasks if ID is not numeric
                 self.total_shards = max(1, (self.total_estimated_tasks + shard_size - 1) // shard_size)
                 logging.warning(f"ID非数字，无法精确分片。将尝试加载整个数据源，分 {self.total_shards} 次处理（估算）。")
                 # For non-numeric, we might have to load all at once or implement source-specific iteration
                 # Simplest is one big shard for now
                 self.total_shards = 1
                 self.shard_boundaries = [(min_id, max_id)]

            else: # Numeric IDs
                 self.total_shards = max(1, (total_range + shard_size - 1) // shard_size)
                 boundaries = []
                 current_id = int(min_id)
                 for i in range(self.total_shards):
                     end_id = min(current_id + shard_size - 1, int(max_id))
                     boundaries.append((current_id, end_id))
                     current_id = end_id + 1
                     if current_id > int(max_id):
                          break # Should not happen with correct calculation, but safe check
                 self.shard_boundaries = boundaries

            logging.info(f"数据将分成 {self.total_shards} 个分片处理。")
            self.all_shards_defined = True
            self.current_shard_index = 0 # Reset index
            return True

        except Exception as e:
            logging.error(f"初始化分片任务管理器失败: {e}", exc_info=True)
            return False

    def get_next_shard_params(self) -> Optional[Tuple[int, Any, Any]]:
         """Gets parameters for the next shard to load. Returns None if done."""
         if not self.all_shards_defined:
              logging.error("Shard boundaries not defined. Call initialize() first.")
              return None
         if self.current_shard_index >= self.total_shards:
             return None # All shards processed

         shard_idx = self.current_shard_index
         min_id, max_id = self.shard_boundaries[shard_idx]
         self.current_shard_index += 1
         return (shard_idx, min_id, max_id)

    def are_all_shards_loaded(self) -> bool:
         """Checks if the manager has iterated through all defined shards."""
         return self.all_shards_defined and (self.current_shard_index >= self.total_shards)


    def monitor_memory_usage(self):
        """Monitors memory and triggers GC if needed."""
        current_time = time.time()
        if current_time - self.memory_tracker['last_check_time'] < self.memory_tracker['check_interval']:
            return
        try:
            process = psutil.Process(os.getpid())
            current_mem_mb = process.memory_info().rss / (1024 * 1024)
            self.memory_tracker['current_memory_mb'] = current_mem_mb
            self.memory_tracker['peak_memory_mb'] = max(self.memory_tracker['peak_memory_mb'], current_mem_mb)
            self.memory_tracker['last_check_time'] = current_time

            mem = psutil.virtual_memory()
            # Trigger GC more aggressively if memory is high
            if mem.percent > 80 or current_mem_mb > 2048: # System > 80% or process > 2GB
                logging.info(
                    f"内存使用较高，触发GC: 当前={current_mem_mb:.1f}MB, "
                    f"峰值={self.memory_tracker['peak_memory_mb']:.1f}MB, 系统={mem.percent}%"
                )
                gc.collect()
            elif current_time - self.memory_tracker['last_check_time'] > 300: # Log memory periodically even if not high
                 logging.debug(f"内存使用: 当前={current_mem_mb:.1f}MB, 峰值={self.memory_tracker['peak_memory_mb']:.1f}MB")

        except Exception as e:
            logging.warning(f"内存监控失败: {e}")

    def log_progress(self, processed_count: int, current_queue_size: int, results_queue_size: int):
         """Logs processing progress."""
         elapsed = time.time() - self.start_time
         if elapsed == 0: elapsed = 1 # Avoid division by zero
         speed = processed_count / elapsed

         remaining_estimated = max(0, self.total_estimated_tasks - processed_count)
         eta_str = "未知"
         if speed > 0 and remaining_estimated > 0:
              eta_seconds = remaining_estimated / speed
              if eta_seconds < 60:
                   eta_str = f"{eta_seconds:.1f}秒"
              elif eta_seconds < 3600:
                   eta_str = f"{eta_seconds/60:.1f}分钟"
              else:
                   eta_str = f"{eta_seconds/3600:.2f}小时"

         logging.info(
             f"进度: {processed_count}/{self.total_estimated_tasks} ({processed_count/self.total_estimated_tasks*100:.1f}%) | "
             f"速度: {speed:.2f}条/秒 | 队列: {current_queue_size} | 结果队列: {results_queue_size} | "
             f"ETA: {eta_str} | "
             f"内存峰值: {self.memory_tracker['peak_memory_mb']:.1f}MB"
         )


    def finalize(self, total_processed: int):
        """Logs final summary and closes task pool."""
        elapsed = time.time() - self.start_time
        avg_speed = total_processed / elapsed if elapsed > 0 else 0
        logging.info(f"--- 任务处理完成 ---")
        logging.info(f"总处理记录: {total_processed}")
        logging.info(f"总耗时: {elapsed:.2f} 秒")
        logging.info(f"平均速度: {avg_speed:.2f} 条/秒")
        logging.info(f"峰值内存: {self.memory_tracker['peak_memory_mb']:.1f} MB")
        logging.info("关闭数据源...")
        try:
             self.task_pool.close()
        except Exception as e:
             logging.error(f"关闭数据源时出错: {e}")
        logging.info("数据源已关闭。")


###############################################################################
# 模型配置类 (保持不变)
###############################################################################
class ModelConfig:
    def __init__(self, model_dict: Dict[str, Any], channels: Dict[str, Any]):
        self.id = str(model_dict.get("id")) # Ensure ID is string
        self.name = model_dict.get("name", self.id) # Default name to ID
        self.model = model_dict.get("model")
        self.channel_id = str(model_dict.get("channel_id"))
        self.api_key = model_dict.get("api_key")
        self.timeout = model_dict.get("timeout", 60) # Default timeout 60s
        self.weight = int(model_dict.get("weight", 1)) # Ensure weight is int
        self.base_weight = self.weight
        self.max_weight = self.weight * 3 # Allow more dynamic range
        self.temperature = model_dict.get("temperature", 0.7)

        # Support for JSON Schema / Structured Output
        self.supports_json_schema = model_dict.get("supports_json_schema", False)

        # RPS limit for the model
        self.safe_rps = model_dict.get("safe_rps", max(0.5, min(self.weight / 5, 20))) # Adjusted estimation

        if not self.id or not self.model or not self.channel_id:
            raise ValueError(f"模型配置缺少必填字段: id={self.id}, model={self.model}, channel_id={self.channel_id}")

        if self.channel_id not in channels:
            raise ValueError(f"channel_id={self.channel_id} 在channels配置中不存在")

        channel_cfg = channels[self.channel_id]
        self.channel_name = channel_cfg.get("name", self.channel_id)
        self.base_url = channel_cfg.get("base_url")
        if not self.base_url:
             raise ValueError(f"Channel '{self.channel_id}' 缺少 base_url 配置")
        self.api_path = channel_cfg.get("api_path", "/v1/chat/completions")
        self.channel_timeout = channel_cfg.get("timeout", 120) # Channel level timeout
        self.channel_proxy = channel_cfg.get("proxy", None) # Use None if empty

        # Effective timeout is the minimum of model and channel timeout
        self.final_timeout = min(self.timeout, self.channel_timeout)
        self.connect_timeout = 10 # Standard connect timeout
        # Read timeout should be based on final_timeout minus connect allowance
        self.read_timeout = max(5, self.final_timeout - self.connect_timeout)

###############################################################################
# 加载配置文件 (保持不变)
###############################################################################
def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("配置文件格式错误，必须是字典类型")
            # Basic validation can be added here (e.g., check for 'models', 'channels')
            if 'models' not in config or 'channels' not in config:
                 logging.warning("配置文件可能缺少 'models' 或 'channels' 部分。")
            return config
    except FileNotFoundError:
        print(f"错误：配置文件 {config_path} 不存在！")
        sys.exit(1)
    except yaml.YAMLError as e:
         print(f"错误：解析配置文件 {config_path} 失败: {e}")
         sys.exit(1)
    except Exception as e:
        print(f"错误：加载配置文件失败: {e}")
        sys.exit(1)

###############################################################################
# 初始化日志 (保持不变)
###############################################################################
def init_logging(log_config: Optional[Dict[str, Any]]):
    if log_config is None: log_config = {}

    level_str = log_config.get("level", "info").upper()
    level = getattr(logging, level_str, logging.INFO)

    log_format = "%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S" # Add date format

    # JSON format option (can be useful for structured logging)
    if log_config.get("format") == "json":
        # Basic JSON format, more sophisticated formatters exist (e.g., python-json-logger)
        log_format = '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'


    output_type = log_config.get("output", "console")
    handlers = []

    if output_type == "console" or output_type == "both":
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(console_handler)

    if output_type == "file" or output_type == "both":
        file_path = log_config.get("file_path", "./universal_ai_processor.log") # Default log file name
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            # Use RotatingFileHandler for better log management
            from logging.handlers import RotatingFileHandler
            # Rotate logs at 10MB, keep 5 backup logs
            file_handler = RotatingFileHandler(file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
            handlers.append(file_handler)
        except Exception as e:
            print(f"错误：创建日志文件或目录失败: {e}")
            # Fallback to console logging if file setup fails
            if not handlers:
                 console_handler = logging.StreamHandler(sys.stdout)
                 console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
                 handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(level=level, handlers=handlers)

    # Set log levels for noisy libraries if needed
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info(f"日志级别设置为: {level_str}, 输出到: {output_type}")


###############################################################################
# 主处理类 - 采用新的资源池-任务池分离架构 (Refactored Core Logic)
###############################################################################
class UniversalAIProcessor:
    def __init__(self, config_path: str):
        try:
             self.config = load_config(config_path)
             global_cfg = self.config.get("global", {})
             init_logging(global_cfg.get("log", {})) # Initialize logging first

             models_cfg = self.config.get("models", [])
             channels_cfg = self.config.get("channels", {})
             if not models_cfg:
                 logging.error("配置文件中未找到 'models' 配置！")
                 raise ValueError("缺少 models 配置")
             if not channels_cfg:
                  logging.error("配置文件中未找到 'channels' 配置！")
                  raise ValueError("缺少 channels 配置")

             # Load model configurations
             self.models: List[ModelConfig] = []
             model_ids = set()
             for m_dict in models_cfg:
                 try:
                      model_conf = ModelConfig(m_dict, channels_cfg)
                      if model_conf.id in model_ids:
                           logging.warning(f"发现重复的模型ID: {model_conf.id}, 后者将覆盖前者。")
                      model_ids.add(model_conf.id)
                      self.models.append(model_conf)
                 except ValueError as e:
                      logging.error(f"加载模型配置失败: {e}. 跳过此模型。")

             if not self.models:
                  logging.error("没有成功加载任何模型配置。请检查配置文件。")
                  raise ValueError("无有效模型配置")

             # Datasource configuration
             datasource_cfg = self.config.get("datasource", {})
             self.datasource_type = datasource_cfg.get("type", "excel").lower()
             logging.info(f"使用数据源类型: {self.datasource_type}")

             # Concurrency settings (Look in datasource first, then specific type, then global)
             concurrency_cfg = datasource_cfg.get("concurrency",
                                                  self.config.get(self.datasource_type, {}).get("concurrency",
                                                                                               global_cfg.get("concurrency", {})))

             # Determine max_workers: Default to CPU count * 5, min 5, max 100
             default_workers = min(max(5, (os.cpu_count() or 1) * 5), 100)
             self.max_workers = concurrency_cfg.get("max_workers", default_workers)
             # Batch size for SAVING results, not processing
             self.result_save_batch_size = concurrency_cfg.get("result_save_batch_size", 100)
             self.save_interval = concurrency_cfg.get("save_interval", 60) # Default save interval 60s
             self.global_retry_times = concurrency_cfg.get("retry_times", 3)
             self.backoff_factor = concurrency_cfg.get("backoff_factor", 2)
             # Shard size settings
             self.shard_size = concurrency_cfg.get("shard_size", 10000)
             self.min_shard_size = concurrency_cfg.get("min_shard_size", 1000)
             self.max_shard_size = concurrency_cfg.get("max_shard_size", 50000)
             # Task queue size limit
             self.task_queue_max_size = concurrency_cfg.get("task_queue_max_size", self.max_workers * 5) # Buffer size


             # Prompt configuration
             prompt_cfg = self.config.get("prompt", {})
             self.prompt_template = prompt_cfg.get("template", "{record_json}") # Default if missing
             if not self.prompt_template:
                  logging.warning("提示词模板为空！")
             self.required_fields = prompt_cfg.get("required_fields", [])
             self.use_json_schema = prompt_cfg.get("use_json_schema", False)
             logging.info(f"JSON Schema/Structured Output 功能: {'启用' if self.use_json_schema else '禁用'}")
             if self.use_json_schema and not self.required_fields:
                  logging.warning("启用JSON Schema但未配置 'required_fields'，Schema可能无效。")


             # JSON Field Validation
             self.validator = JsonValidator()
             validation_cfg = self.config.get("validation", {})
             self.validator.configure(validation_cfg)

             # Column configuration
             self.columns_to_extract = self.config.get("columns_to_extract", [])
             self.columns_to_write = self.config.get("columns_to_write", {})
             if not self.columns_to_extract:
                  logging.warning("未配置 'columns_to_extract'，将无法从数据源提取数据生成提示词。")
             if not self.columns_to_write:
                  logging.warning("未配置 'columns_to_write'，处理结果将不会写回数据源。")


             # --- Initialize Core Components ---
             # Model Dispatcher (Resource Pool State)
             self.dispatcher = ModelDispatcher(self.models, backoff_factor=self.backoff_factor)
             # Model Rate Limiter (Resource Pool Control)
             self.rate_limiter = ModelRateLimiter()
             self.rate_limiter.configure([m.__dict__ for m in self.models]) # Pass model dicts

             # Model Map for quick lookup
             self.model_map = {m.id: m for m in self.models}

             # Dynamic Weight Adjustment Pool (Initialized based on base weights)
             self.models_pool: List[ModelConfig] = []
             self._rebuild_model_pool() # Initial build

             # Task Pool (Data Source Abstraction)
             self.task_pool = self._create_task_pool()

             # Sharded Task Manager (Shard Logic/Progress)
             self.task_manager = ShardedTaskManager(
                 task_pool=self.task_pool,
                 optimal_shard_size=self.shard_size,
                 min_shard_size=self.min_shard_size,
                 max_shard_size=self.max_shard_size
             )

             # --- Asynchronous Processing State ---
             self.task_queue: Queue[Optional[Tuple[Any, Dict[str, Any]]]] = Queue(maxsize=self.task_queue_max_size)
             self.results_queue: Queue[Optional[Tuple[Any, Dict[str, Any]]]] = Queue() # Unbounded results queue
             self.results_lock = asyncio.Lock() # Lock for managing pending_results batching
             self.pending_results: Dict[Any, Dict[str, Any]] = {} # Results waiting to be saved
             self.processed_count = 0 # Total tasks successfully processed and queued for saving
             self.system_error_count = 0 # Count tasks requeued due to system errors
             self.content_error_count = 0 # Count tasks failed due to content errors
             self.last_save_trigger_time = time.time() # Time when last save batch was initiated
             self.last_progress_log_time = time.time()
             self.progress_log_interval = 10 # Log progress every 10 seconds

             # Executor for synchronous I/O (DB writes, file saves)
             # Adjust max_workers if I/O becomes a bottleneck
             self.io_executor = ThreadPoolExecutor(max_workers=max(2, self.max_workers // 5), thread_name_prefix='IOExecutor')


             logging.info(f"初始化完成。模型数: {len(self.models)} | Worker数: {self.max_workers} | 任务队列大小: {self.task_queue_max_size}")
             logging.info(f"结果保存批次大小: {self.result_save_batch_size} | 保存间隔: {self.save_interval}秒")

        except Exception as e:
             logging.error(f"初始化 UniversalAIProcessor 失败: {e}", exc_info=True)
             # Clean up any partially initialized resources if necessary
             # e.g., close DB pool if created
             if hasattr(self, 'task_pool') and self.task_pool:
                 try: self.task_pool.close()
                 except: pass
             if hasattr(self, 'io_executor') and self.io_executor:
                  self.io_executor.shutdown(wait=False)
             raise # Re-raise the exception to stop execution


    def _rebuild_model_pool(self):
        """Rebuilds the weighted model pool based on current model weights."""
        new_pool = []
        total_weight = 0
        for model in self.models:
            # Only include models with weight > 0 and that are generally available
            # (We still check availability/rate limit per request)
            if model.weight > 0:
                new_pool.extend([model] * model.weight)
                total_weight += model.weight
        self.models_pool = new_pool
        logging.debug(f"重建模型池，大小: {len(self.models_pool)}, 总权重: {total_weight}")
        if not self.models_pool:
             logging.warning("模型池为空！所有模型权重可能为0或无有效模型。")


    def _create_task_pool(self) -> BaseTaskPool:
        """Factory method to create the appropriate task pool."""
        if self.datasource_type == "mysql":
            if not MYSQL_AVAILABLE:
                logging.error("MySQL Connector库未安装，无法使用MySQL数据源。")
                raise ImportError("MySQL Connector库未安装")

            mysql_config = self.config.get("mysql", {})
            required_mysql = ["host", "user", "password", "database", "table_name"]
            if any(k not in mysql_config for k in required_mysql):
                 raise ValueError(f"MySQL配置缺少必需项: {required_mysql}")

            connection_config = {
                "host": mysql_config["host"],
                "port": mysql_config.get("port", 3306),
                "user": mysql_config["user"],
                "password": mysql_config["password"],
                "database": mysql_config["database"]
            }
            id_column = mysql_config.get("id_column", "id") # Allow configuring ID column

            return MySQLTaskPool(
                connection_config=connection_config,
                columns_to_extract=self.columns_to_extract,
                columns_to_write=self.columns_to_write,
                table_name=mysql_config["table_name"],
                id_column=id_column
            )
        elif self.datasource_type == "excel":
            excel_config = self.config.get("excel", {})
            input_excel = excel_config.get("input_path")
            if not input_excel:
                 raise ValueError("Excel数据源缺少 'input_path' 配置。")

            # Default output path logic
            output_excel = excel_config.get("output_path")
            if not output_excel:
                 base, ext = os.path.splitext(input_excel)
                 output_excel = f"{base}_output{ext}"
                 logging.info(f"未指定Excel输出路径，将使用默认: {output_excel}")


            if not os.path.exists(input_excel):
                logging.error(f"Excel输入文件不存在: {input_excel}")
                raise FileNotFoundError(f"Excel文件不存在: {input_excel}")

            logging.info(f"开始读取Excel: {input_excel} ...")
            try:
                 # Specify engine and consider dtype='object' for robustness
                 df = pd.read_excel(input_excel, engine='openpyxl', dtype=str).fillna('')
                 logging.info(f"Excel读取完成，共 {len(df)} 行。")
            except Exception as e:
                 logging.error(f"读取Excel文件失败: {input_excel} - {e}")
                 raise

            return ExcelTaskPool(
                df=df,
                columns_to_extract=self.columns_to_extract,
                columns_to_write=self.columns_to_write,
                output_excel=output_excel,
                save_interval=self.save_interval
            )
        else:
            raise ValueError(f"不支持的数据源类型: {self.datasource_type}")

    def adjust_model_weights(self):
        """Dynamically adjusts model weights based on performance."""
        weights_changed = False
        if not self.model_map: return # No models to adjust

        # Calculate relative performance scores
        scores = {}
        max_score = 0
        min_avg_rt = float('inf')
        total_calls_overall = 0

        for model_id, model in self.model_map.items():
            if model.base_weight <= 0: continue # Skip fixed zero-weight models

            success_rate = self.dispatcher.get_model_success_rate(model_id)
            avg_response_time = self.dispatcher.get_model_avg_response_time(model_id)
            total_calls = self.dispatcher._model_state[model_id]["success_count"] + \
                          self.dispatcher._model_state[model_id]["error_count"]
            total_calls_overall += total_calls

            if total_calls < 5: # Don't adjust weights too early
                 scores[model_id] = model.weight # Keep current weight
                 continue

            # Score favors high success rate and low response time
            # Penalize high response time more heavily
            # Use base_weight as a reference point
            rt_factor = 1.0 / max(0.1, avg_response_time)**0.5 # Less sensitive to RT changes
            # Success rate is critical
            sr_factor = success_rate ** 2

            # Combine factors relative to base weight
            # score = model.base_weight * sr_factor * rt_factor
            # Alternative: Adjust current weight based on relative performance
            # Let's try a simpler approach: adjust based on SR and relative speed
            score = success_rate / max(0.1, avg_response_time)
            scores[model_id] = score
            if score > max_score: max_score = score
            if avg_response_time < min_avg_rt: min_avg_rt = avg_response_time


        if total_calls_overall < len(self.models) * 10: # Wait for more data overall
             return

        if max_score == 0: # Avoid division by zero if all scores are 0
             return

        # Normalize scores and calculate new weights
        for model_id, model in self.model_map.items():
             if model.base_weight <= 0 or model_id not in scores: continue

             normalized_score = scores[model_id] / max_score if max_score > 0 else 0

             # Calculate new weight based on normalized score, bounded by base/max
             # Target weight proportional to score, scaled by avg base weight?
             # Let's try scaling based on base_weight and score
             new_weight = int(model.base_weight * (0.5 + normalized_score * 1.5)) # Scale score effect
             # new_weight = int(model.base_weight * normalized_score * 2) # Simpler scaling


             # Clamp weight: ensure it's at least 1 (if base > 0) and not more than max_weight
             new_weight = max(1, min(new_weight, model.max_weight))

             if new_weight != model.weight:
                 logging.info(
                     f"调整模型[{model.name}]权重: {model.weight} -> {new_weight} "
                     f"(SR={self.dispatcher.get_model_success_rate(model_id):.2f}, "
                     f"RT={self.dispatcher.get_model_avg_response_time(model_id):.2f}s, "
                     f"Score={scores[model_id]:.3f}, NormScore={normalized_score:.3f})"
                 )
                 model.weight = new_weight
                 weights_changed = True

        if weights_changed:
            self._rebuild_model_pool()


    def get_available_model_randomly(self, exclude_model_ids: Optional[Set[str]] = None) -> Optional[ModelConfig]:
        """
        Selects an available model based on current weights and availability,
        respecting rate limits. Returns None if no suitable model found.
        """
        if exclude_model_ids is None:
            exclude_model_ids = set()
        else:
            exclude_model_ids = set(map(str, exclude_model_ids)) # Ensure strings

        # Get currently available models from dispatcher (checks backoff)
        available_now_ids = self.dispatcher.get_available_models(exclude_model_ids)

        if not available_now_ids:
            # logging.debug("No models available according to dispatcher (backoff).")
            return None

        # Filter further based on rate limits and positive weight
        eligible_models = []
        eligible_weights = []
        for model_id in available_now_ids:
            if model_id in self.model_map:
                model = self.model_map[model_id]
                # Check weight and rate limit
                if model.weight > 0 and self.rate_limiter.can_process(model_id):
                    eligible_models.append(model)
                    eligible_weights.append(model.weight)

        if not eligible_models:
             # logging.debug("No models passed weight and rate limit check.")
             return None

        # Weighted random choice
        try:
            selected_model = random.choices(eligible_models, weights=eligible_weights, k=1)[0]
            # logging.debug(f"Selected model {selected_model.id} based on weight and availability.")
            return selected_model
        except IndexError:
             # Should not happen if eligible_models is not empty
             logging.error("Error during weighted random choice for model selection.")
             return None
        except Exception as e:
             logging.error(f"Unexpected error selecting model: {e}")
             return None


    def create_prompt(self, record_data: Dict[str, Any]) -> str:
        """Creates the prompt string from the template and record data."""
        try:
            # Only include columns specified in columns_to_extract
            filtered_data = {k: v for k, v in record_data.items() if k in self.columns_to_extract}
            record_json_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError) as e:
            logging.error(f"行数据无法序列化为JSON: {e} - Data: {record_data}")
            # Return template without data, or maybe an error indicator?
            # Depending on template structure, this might be okay or cause issues.
            return self.prompt_template.replace("{record_json}", "{}") # Replace with empty JSON

        if "{record_json}" not in self.prompt_template:
            logging.warning("提示词模板中不包含 {record_json} 占位符。将直接使用模板。")
            return self.prompt_template # Return template as is

        try:
             return self.prompt_template.replace("{record_json}", record_json_str)
        except Exception as e:
             logging.error(f"替换提示词模板时出错: {e}")
             return self.prompt_template # Fallback to original template

    def extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """
        Attempts to extract a valid JSON object from the AI response.
        Prioritizes full parse, then regex for JSON within text.
        Validates required fields and optional enum values.
        """
        if not content:
             return {"_error": "empty_response", "_error_type": ErrorType.CONTENT_ERROR}

        # Ensure required_fields are strings if provided
        required = [str(f) for f in self.required_fields] if self.required_fields else []

        def check_and_validate(data_dict: Dict[str, Any]) -> Tuple[bool, Optional[List[str]]]:
            """Checks required fields and runs JSON validator."""
            if not isinstance(data_dict, dict):
                 return False, ["不是有效的JSON对象"]
            # Check required fields
            missing_fields = [f for f in required if f not in data_dict]
            if missing_fields:
                return False, [f"缺少必需字段: {', '.join(missing_fields)}"]

            # Perform enum validation if enabled
            if self.validator.enabled:
                is_valid, errors = self.validator.validate(data_dict)
                if not is_valid:
                    return False, errors
            return True, None

        # --- Extraction Logic ---
        parsed_data = None
        validation_errors = None

        # 1. Try direct JSON parsing (most common case)
        try:
            data = json.loads(content)
            is_valid, errors = check_and_validate(data)
            if is_valid:
                parsed_data = data
            else:
                 validation_errors = errors # Store errors for logging if regex fails too
                 logging.debug(f"直接JSON解析验证失败: {errors}. Raw: {content[:200]}")

        except json.JSONDecodeError:
            logging.debug(f"直接JSON解析失败. Raw: {content[:200]}")
            # Fall through to regex extraction

        # 2. If direct parse fails or validation fails, try regex for ```json ... ``` or {...}
        if parsed_data is None:
            # Regex for ```json ... ``` block
            code_block_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL | re.IGNORECASE)
            # Regex for bare { ... } (more greedy, try last)
            bare_json_match = re.search(r'(\{.*?\})', content, re.DOTALL)
            match_str = None

            if code_block_match:
                match_str = code_block_match.group(1)
                logging.debug("Found JSON in ```json code block.")
            elif bare_json_match:
                match_str = bare_json_match.group(1)
                logging.debug("Found potential JSON object using bare regex.")


            if match_str:
                try:
                    # Try parsing the extracted string
                    candidate = json.loads(match_str)
                    is_valid, errors = check_and_validate(candidate)
                    if is_valid:
                        parsed_data = candidate
                        logging.debug("Regex提取的JSON验证通过。")
                    else:
                         # Use original validation errors if regex validation also fails
                         validation_errors = validation_errors or errors
                         logging.warning(f"Regex提取的JSON验证失败: {errors}. Raw: {content[:200]}")

                except json.JSONDecodeError:
                    logging.warning(f"Regex提取的字符串无法解析为JSON. Extracted: {match_str[:200]}")
                except Exception as e:
                     logging.warning(f"Regex提取JSON时发生意外错误: {e}")


        # --- Return Result ---
        if parsed_data is not None:
            return parsed_data # Successfully parsed and validated
        else:
            # Return specific error based on what happened
            error_details = validation_errors if validation_errors else ["无法解析或验证JSON"]
            error_msg = f"invalid_json: {'; '.join(error_details)}"
            logging.warning(f"{error_msg}. Raw Content: {content[:500]}") # Log more context on failure
            return {"_error": error_msg, "_error_type": ErrorType.CONTENT_ERROR}


    def build_json_schema(self) -> Dict[str, Any]:
        """Builds the JSON Schema for structured output requests."""
        if not self.use_json_schema or not self.required_fields:
             # Return a minimal schema if not properly configured
             return {"type": "object"}

        schema = {
            "type": "object",
            "properties": {},
            "required": [str(f) for f in self.required_fields], # Ensure strings
             # "additionalProperties": False # Be lenient by default? Or make configurable?
             "additionalProperties": True # Allow extra fields unless specified otherwise
        }

        # Add properties based on required fields and validation rules
        validation_rules = self.validator.field_rules if self.validator.enabled else {}

        for field in schema["required"]:
            prop_schema = {}
            # Determine type based on validation rules if available
            if field in validation_rules:
                 allowed_values = validation_rules[field]
                 if allowed_values:
                      # Infer type from first value (basic heuristic)
                      first_val = allowed_values[0]
                      if isinstance(first_val, int): prop_schema["type"] = "integer"
                      elif isinstance(first_val, float): prop_schema["type"] = "number"
                      elif isinstance(first_val, bool): prop_schema["type"] = "boolean"
                      else: prop_schema["type"] = "string" # Default to string

                      # Add enum constraint
                      prop_schema["enum"] = allowed_values
                 else: # Rule exists but list is empty? Default to string.
                      prop_schema["type"] = "string"
            else:
                 # Default type if no validation rule exists
                 prop_schema["type"] = "string" # Default everything to string

            # Add description or other constraints if needed
            # prop_schema["description"] = f"Field {field}"

            schema["properties"][field] = prop_schema

        logging.debug(f"Generated JSON Schema: {json.dumps(schema)}")
        return schema


    async def call_ai_api_async(self, session: aiohttp.ClientSession, model_cfg: ModelConfig, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Calls the AI API asynchronously using the provided session.
        Returns (content, error_type) tuple. error_type is None on success.
        Handles API errors and timeouts specifically.
        """
        url = model_cfg.base_url.rstrip("/") + model_cfg.api_path
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model_cfg.api_key}"
        }

        # Build payload
        payload = {
            "model": model_cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": model_cfg.temperature
            # Add other parameters like max_tokens if needed
            # "max_tokens": 1024
        }

        # Add structured output format if enabled and supported
        if self.use_json_schema and model_cfg.supports_json_schema:
            payload["response_format"] = { "type": "json_object" }
            # Some models might use 'json_schema' with a schema, others 'json_object'
            # Check model documentation. Assuming 'json_object' for broader compatibility.
            # If schema is needed:
            # payload["response_format"] = {
            #     "type": "json_schema",
            #     "json_schema": self.build_json_schema()
            # }
            logging.debug(f"模型[{model_cfg.id}] 启用 JSON Object 输出模式")


        # Use timeout configured for the model/channel
        # Separate connect and total timeouts
        try:
             timeout = aiohttp.ClientTimeout(
                  connect=model_cfg.connect_timeout, # Timeout for establishing connection
                  total=model_cfg.final_timeout # Total time for entire operation (incl. connect, read)
             )
             proxy = model_cfg.channel_proxy or os.environ.get('HTTPS_PROXY') # Use env var as fallback

             logging.debug(f"Calling model {model_cfg.id} at {url} with timeout {model_cfg.final_timeout}s")
             async with session.post(url, headers=headers, json=payload, proxy=proxy, timeout=timeout) as resp:
                 # Check for HTTP errors (4xx, 5xx)
                 if resp.status >= 400:
                     error_text = await resp.text()
                     log_msg = f"模型[{model_cfg.name}] API错误: HTTP {resp.status} - {error_text[:500]}" # Limit error length
                     logging.warning(log_msg)
                     # Distinguish client (4xx) vs server (5xx) errors for backoff?
                     # For now, treat all >= 400 as API_ERROR requiring backoff.
                     return None, ErrorType.API_ERROR

                 # Process successful response (2xx)
                 response_text = await resp.text()
                 content_type = resp.headers.get('Content-Type', '')

                 # Attempt to parse JSON response (standard OpenAI format)
                 try:
                      data = json.loads(response_text)
                      if not isinstance(data, dict):
                            raise ValueError("响应不是JSON对象")

                      # Handle potential variations in response structure
                      if "choices" in data and data["choices"]:
                           message = data["choices"][0].get("message", {})
                           content = message.get("content")
                           if content is not None:
                                return str(content), None # SUCCESS
                           # Handle function/tool calls if needed later
                      elif "error" in data: # Handle explicit error messages in JSON
                           error_msg = data["error"].get("message", "未知API错误")
                           logging.warning(f"模型[{model_cfg.name}] API返回错误JSON: {error_msg}")
                           return None, ErrorType.API_ERROR # Treat as API error
                      # Add handling for other response formats if necessary (e.g., Anthropic)

                      # If content not found in expected structure
                      logging.warning(f"模型[{model_cfg.name}] 响应JSON结构未知或缺少内容: {response_text[:500]}")
                      return None, ErrorType.CONTENT_ERROR # Treat as content error if structure is wrong

                 except json.JSONDecodeError:
                      # Handle non-JSON responses (e.g., plain text, event stream if streaming was enabled)
                      logging.warning(f"模型[{model_cfg.name}] 响应不是有效JSON ({content_type}): {response_text[:500]}")
                      # If plain text might be valid, return it? Depends on expectation.
                      # Assuming JSON is expected, treat as content error.
                      return None, ErrorType.CONTENT_ERROR

        except asyncio.TimeoutError:
            logging.warning(f"模型[{model_cfg.name}] 请求超时 (>{model_cfg.final_timeout}s)")
            return None, ErrorType.API_ERROR # Timeout is an API/Network level error
        except aiohttp.ClientError as e:
            # Includes connection errors, proxy errors, etc.
            logging.warning(f"模型[{model_cfg.name}] 网络/客户端错误: {e}")
            return None, ErrorType.API_ERROR
        except Exception as e:
            # Catch any other unexpected errors during the API call
            logging.error(f"模型[{model_cfg.name}] 调用时发生意外错误: {e}", exc_info=True)
            return None, ErrorType.API_ERROR # Treat unexpected errors as API errors for backoff


    async def _process_record_with_retries(self, record_id: Any, row_data: Dict[str, Any], session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Processes a single record using available models with retries."""
        log_label = f"记录[{record_id}]"
        prompt = self.create_prompt(row_data)
        if not prompt: # Handle case where prompt creation failed
             return {"_error": "prompt_creation_failed", "_error_type": ErrorType.CONTENT_ERROR}

        used_model_ids_this_task = set()
        last_error = None
        last_error_type = None

        for attempt in range(self.global_retry_times):
            model_cfg = self.get_available_model_randomly(used_model_ids_this_task)

            if not model_cfg:
                # No models available (all in backoff or rate limited), wait and retry attempt
                wait_time = min(self.backoff_factor ** attempt, 30) # Exponential backoff for task retry
                logging.warning(f"{log_label}: 第{attempt+1}次尝试无可用模型，等待 {wait_time:.1f} 秒...")
                await asyncio.sleep(wait_time)
                continue # Go to next attempt

            logging.debug(f"{log_label}: 第{attempt+1}次尝试，使用模型 [{model_cfg.name}]")
            used_model_ids_this_task.add(model_cfg.id)
            start_time = time.time()

            content, error_type = await self.call_ai_api_async(session, model_cfg, prompt)
            response_time = time.time() - start_time

            if error_type is None and content is not None:
                # --- Success ---
                self.dispatcher.update_model_metrics(model_cfg.id, response_time, True)
                self.dispatcher.mark_model_success(model_cfg.id) # Reset failure count on success

                parsed_result = self.extract_json_from_response(content)

                # Check if JSON parsing/validation itself failed
                if "_error" in parsed_result and parsed_result.get("_error_type") == ErrorType.CONTENT_ERROR:
                    logging.warning(f"{log_label}: 模型[{model_cfg.name}] 调用成功但内容解析/验证失败: {parsed_result['_error']}")
                    last_error = parsed_result['_error']
                    last_error_type = ErrorType.CONTENT_ERROR
                    # Do NOT mark model as failed here, API call was okay. Continue to next attempt/model.
                    continue # Try next model or next attempt
                else:
                    # Actual Success
                    logging.info(f"{log_label}: 模型[{model_cfg.name}] 处理成功 ({response_time:.2f}s)")
                    parsed_result["_used_model_id"] = model_cfg.id
                    parsed_result["_used_model_name"] = model_cfg.name
                    parsed_result["_response_time_ms"] = int(response_time * 1000)
                    # Add excerpt for logging? Be careful with sensitive data.
                    # parsed_result["_response_excerpt"] = (content[:50] + "...") if len(content) > 50 else content
                    return parsed_result # Return the successful result

            else:
                # --- Failure ---
                self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
                last_error = f"Model {model_cfg.name} failed" # Store generic error for final failure msg
                last_error_type = error_type

                if error_type == ErrorType.API_ERROR:
                    self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
                    # API error, likely wait won't help immediately with this model,
                    # but continue the loop to try other models in this attempt.
                elif error_type == ErrorType.CONTENT_ERROR:
                    # Content error (bad response format, etc.) - don't backoff model
                    logging.warning(f"{log_label}: 模型[{model_cfg.name}] 返回内容错误，尝试其他模型...")
                else: # Should not happen, but handle unknown error types
                     logging.error(f"{log_label}: 模型[{model_cfg.name}] 遇到未知错误类型: {error_type}")

                # Continue loop to try another model within the same attempt
                await asyncio.sleep(0.05) # Small delay before trying next model


            # End of inner loop (trying different models within one attempt)
            if attempt < self.global_retry_times - 1:
                 # If all models tried in this attempt failed, wait before next global attempt
                 wait_time = min(self.backoff_factor ** attempt, 30)
                 logging.warning(f"{log_label}: 第{attempt+1}轮尝试失败，等待 {wait_time:.1f} 秒后重试...")
                 await asyncio.sleep(wait_time)


        # All attempts failed for this record
        logging.error(f"{log_label}: 经过 {self.global_retry_times} 轮尝试后处理失败。最后错误: {last_error}")
        # Return a consistent error format
        final_error = last_error or "unknown_error"
        final_error_type = last_error_type or ErrorType.CONTENT_ERROR # Default to content error if type unknown
        return {"_error": f"all_attempts_failed: {final_error}", "_error_type": final_error_type}


    # --- Producer Coroutine ---
    async def _task_producer(self):
        """Loads shards and puts tasks onto the task_queue."""
        logging.info("任务生产者启动...")
        while True:
            # Check if task queue has space before loading more
            # This provides backpressure to shard loading
            if self.task_queue.full():
                 logging.debug(f"任务队列已满 ({self.task_queue.qsize()}/{self.task_queue_max_size})，等待...")
                 # Wait until queue has space
                 # Use a loop with sleep to avoid blocking indefinitely if queue never empties
                 while self.task_queue.full():
                     await asyncio.sleep(0.5)
                 logging.debug("任务队列有空间，继续加载...")


            # Load next shard if internal task list is empty
            if not self.task_pool.has_tasks():
                shard_params = self.task_manager.get_next_shard_params()
                if shard_params is None:
                    logging.info("所有分片已定义且已处理，生产者停止。")
                    break # All shards defined and iterated through

                shard_id, min_id, max_id = shard_params
                logging.info(f"加载分片 {shard_id}/{self.task_manager.total_shards} (范围: {min_id}-{max_id})...")
                try:
                     # Run blocking shard initialization in executor
                     loop = asyncio.get_running_loop()
                     loaded_count = await loop.run_in_executor(
                          self.io_executor,
                          self.task_pool.initialize_shard, shard_id, min_id, max_id
                     )
                     if loaded_count == 0 and not self.task_manager.are_all_shards_loaded():
                          logging.info(f"分片 {shard_id} 未加载任何任务，尝试下一个分片。")
                          continue # Skip to next shard loading iteration
                except Exception as e:
                     logging.error(f"加载分片 {shard_id} 时发生严重错误: {e}", exc_info=True)
                     # Decide how to handle: stop? skip? For now, log and stop producer.
                     break

            # Move tasks from internal list to queue
            processed_in_batch = 0
            while not self.task_queue.full():
                task = self.task_pool.get_single_task()
                if task is None:
                    break # No more tasks in the current shard's internal list
                await self.task_queue.put(task)
                processed_in_batch += 1

            if processed_in_batch == 0 and not self.task_pool.has_tasks() and self.task_manager.are_all_shards_loaded():
                 # Double check if really finished after attempting to move tasks
                 logging.info("确认所有任务已加载完毕，生产者结束。")
                 break

            # Optional small sleep if tasks were processed, to yield control
            if processed_in_batch > 0:
                 await asyncio.sleep(0.01)

        # Signal workers to stop by putting None for each worker
        logging.info(f"生产者完成，发送停止信号给 {self.max_workers} 个 Worker...")
        for _ in range(self.max_workers):
            await self.task_queue.put(None)


    # --- Worker Coroutine ---
    async def _worker(self, worker_id: int, session: aiohttp.ClientSession):
        """Gets tasks from queue, processes them, puts results to results_queue."""
        logging.info(f"Worker-{worker_id} 启动...")
        while True:
            task = await self.task_queue.get()
            if task is None:
                logging.info(f"Worker-{worker_id} 收到停止信号，退出。")
                self.task_queue.task_done()
                break # Exit loop

            record_id, row_data = task
            logging.debug(f"Worker-{worker_id} 开始处理记录 [{record_id}]")
            result = None
            try:
                result = await self._process_record_with_retries(record_id, row_data, session)
            except Exception as e:
                logging.error(f"Worker-{worker_id} 处理记录 [{record_id}] 时发生意外异常: {e}", exc_info=True)
                result = {"_error": f"worker_exception: {e}", "_error_type": ErrorType.SYSTEM_ERROR}

            # Put result (or error) onto the results queue
            await self.results_queue.put((record_id, result))
            self.task_queue.task_done() # Mark task as processed in input queue
            logging.debug(f"Worker-{worker_id} 完成处理记录 [{record_id}]")

            # Dynamic weight adjustment happens within process_record_with_retries? No, do it here.
            # Adjust weights periodically/randomly
            if random.random() < 0.01: # 1% chance per task
                 try:
                      self.adjust_model_weights()
                 except Exception as e:
                      logging.error(f"Worker-{worker_id}: 调整模型权重时出错: {e}")


    # --- Results Saver Coroutine ---
    async def _results_saver(self):
        """Gets results from queue, batches them, and saves them using executor."""
        logging.info("结果保存器启动...")
        while True:
            result_item = await self.results_queue.get()
            if result_item is None:
                logging.info("结果保存器收到停止信号。")
                # Process any remaining pending results before exiting
                if self.pending_results:
                     logging.info(f"保存最后 {len(self.pending_results)} 条挂起的结果...")
                     await self._save_results_batch()
                self.results_queue.task_done()
                break # Exit loop

            record_id, result = result_item

            # --- Critical Section for pending_results ---
            async with self.results_lock:
                 self.pending_results[record_id] = result
                 # Increment counters based on result type
                 if "_error" in result:
                      if result.get("_error_type") == ErrorType.SYSTEM_ERROR:
                           self.system_error_count += 1
                      else: # Assume CONTENT_ERROR or other non-system errors
                           self.content_error_count += 1
                 else:
                      self.processed_count += 1 # Count only non-error results

                 # Check if batch is ready to be saved
                 batch_ready = len(self.pending_results) >= self.result_save_batch_size
                 time_ready = (time.time() - self.last_save_trigger_time) >= self.save_interval

                 should_save = batch_ready or time_ready

                 if should_save:
                      await self._save_results_batch() # Call save within lock to avoid race conditions

            # --- End Critical Section ---

            # Log progress periodically outside the lock
            current_time = time.time()
            if current_time - self.last_progress_log_time >= self.progress_log_interval:
                 self.task_manager.log_progress(self.processed_count + self.content_error_count, # Include content errors in processed total for ETA
                                                self.task_queue.qsize(),
                                                self.results_queue.qsize())
                 self.last_progress_log_time = current_time
                 # Also monitor memory periodically
                 self.task_manager.monitor_memory_usage()


            self.results_queue.task_done() # Mark result processed in output queue

    async def _save_results_batch(self):
        """Saves the current batch of pending results. Must be called under results_lock."""
        if not self.pending_results:
             return

        results_to_save = dict(self.pending_results)
        self.pending_results.clear()
        self.last_save_trigger_time = time.time() # Reset save timer

        logging.info(f"准备保存 {len(results_to_save)} 条结果...")

        try:
            loop = asyncio.get_running_loop()
            # Run blocking save operation in I/O executor
            await loop.run_in_executor(
                self.io_executor,
                self.task_pool.update_task_results, results_to_save
            )
            logging.info(f"成功保存 {len(results_to_save)} 条结果。")
        except Exception as e:
            logging.error(f"保存结果批次时出错: {e}", exc_info=True)
            # Failed to save, put results back into pending for next attempt?
            # This could lead to infinite loops if saving always fails.
            # Alternative: Log failed IDs/data to a separate file.
            logging.error(f"以下记录ID未能保存: {list(results_to_save.keys())}")
            # For now, just log the error and lose the results for this batch.


    # --- Main Processing Orchestration ---
    async def _process_async(self):
        """Main async orchestration function."""
        logging.info("开始异步处理流程...")
        self.task_manager.start_time = time.time() # Reset start time

        # Initialize Task Manager (defines shards)
        if not self.task_manager.initialize():
            logging.error("任务管理器初始化失败，无法启动处理。")
            return

        # Create shared aiohttp session
        # Consider adding connection limits if needed
        async with aiohttp.ClientSession() as session:
            # Start producer, workers, and saver coroutines
            producer_task = asyncio.create_task(self._task_producer(), name="TaskProducer")
            saver_task = asyncio.create_task(self._results_saver(), name="ResultsSaver")
            worker_tasks = []
            for i in range(self.max_workers):
                worker_tasks.append(asyncio.create_task(self._worker(i + 1, session), name=f"Worker-{i+1}"))

            # Wait for producer to finish (meaning all tasks loaded or error)
            try:
                await producer_task
                logging.info("生产者已完成任务加载。")
            except Exception as e:
                 logging.error(f"生产者协程异常终止: {e}", exc_info=True)
                 # Cancel other tasks if producer fails critically
                 for task in worker_tasks: task.cancel()
                 saver_task.cancel()

            # Wait for all tasks in the queue to be processed by workers
            logging.info("等待任务队列处理完成...")
            try:
                 await self.task_queue.join()
                 logging.info("任务队列已处理完毕。")
            except asyncio.CancelledError:
                 logging.info("任务队列等待被取消。")


            # Workers should have exited upon receiving None. Wait for them.
            logging.info("等待 Worker 协程退出...")
            await asyncio.gather(*worker_tasks, return_exceptions=True) # Allow capturing worker errors if any
            logging.info("所有 Worker 已退出。")


            # Signal saver to stop and wait for it
            logging.info("发送停止信号给结果保存器...")
            await self.results_queue.put(None)
            logging.info("等待结果保存器完成...")
            try:
                 await saver_task # Wait for saver to finish final save
                 logging.info("结果保存器已完成。")
            except asyncio.CancelledError:
                 logging.info("结果保存器被取消。")


        # Final summary log is handled in finalize
        logging.info("异步处理流程结束。")


    def process_tasks(self):
        """Synchronous entry point to start the async processing."""
        logging.info(f"=== Universal AI Processor v{datetime.now().strftime('%Y%m%d')} ===")
        logging.info(f"配置文件: {self.config.get('config_path', 'N/A')}") # Store config path if needed

        # Get or create event loop
        try:
             loop = asyncio.get_event_loop()
             if loop.is_running():
                  logging.error("事件循环已在运行。无法在同一线程中启动新的处理。")
                  # In some environments (like Jupyter), need to handle existing loops differently
                  # For command-line, this usually indicates an issue.
                  return
        except RuntimeError:
             loop = asyncio.new_event_loop()
             asyncio.set_event_loop(loop)

        main_task = None
        try:
            main_task = loop.create_task(self._process_async())
            loop.run_until_complete(main_task)
        except KeyboardInterrupt:
            logging.warning("收到用户中断信号 (Ctrl+C)...")
            if main_task and not main_task.done():
                 logging.info("正在尝试优雅地取消任务...")
                 main_task.cancel()
                 # Allow time for cancellation and cleanup within _process_async's finally block
                 try:
                      loop.run_until_complete(main_task)
                 except asyncio.CancelledError:
                      logging.info("主任务已取消。")
                 except Exception as e:
                      logging.error(f"等待主任务取消时出错: {e}")
            logging.warning("程序已中断。可能存在未保存的结果。")
        except Exception as e:
            logging.error(f"处理过程中发生未捕获的严重错误: {e}", exc_info=True)
        finally:
            # Finalize Task Manager (logs summary, closes pool)
            # Calculate final processed count correctly
            final_processed = self.processed_count + self.content_error_count # Include content errors in total attempts
            self.task_manager.finalize(final_processed)

            # Shutdown I/O executor
            logging.info("正在关闭 I/O 线程池...")
            self.io_executor.shutdown(wait=True)
            logging.info("I/O 线程池已关闭。")

            # Close loop (optional, depends on context)
            # loop.close()
            logging.info("=== 处理结束 ===")


###############################################################################
# 命令行入口 (Modified: Added config path to Processor)
###############################################################################
def validate_config_file(config_path: str) -> bool:
    """Basic validation of the config file structure."""
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在: {config_path}")
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
             print("错误：配置文件根元素必须是字典。")
             return False
        if "models" not in config or not config["models"]:
            print("错误：配置文件缺少 'models' 部分或该部分为空。")
            return False
        if "channels" not in config or not config["channels"]:
             print("错误：配置文件缺少 'channels' 部分或该部分为空。")
             return False

        ds_cfg = config.get("datasource", {})
        ds_type = ds_cfg.get("type", "excel").lower()

        if ds_type not in ["mysql", "excel"]:
            print(f"错误：不支持的数据源类型: {ds_type} (仅支持 'mysql' 或 'excel')")
            return False

        if ds_type == "excel":
            excel_cfg = config.get("excel", {})
            if "input_path" not in excel_cfg or not excel_cfg["input_path"]:
                print("错误：使用Excel数据源时，'excel' 部分必须包含 'input_path'。")
                return False
            # Check if input file exists early
            # if not os.path.exists(excel_cfg["input_path"]):
            #     print(f"警告：Excel输入文件 '{excel_cfg['input_path']}' 不存在。")
                # Don't return False here, let the Processor handle it during init

        if ds_type == "mysql":
            if not MYSQL_AVAILABLE:
                print("错误：已配置MySQL数据源，但 'mysql-connector-python' 库未安装。")
                print("请运行: pip install mysql-connector-python")
                return False
            mysql_cfg = config.get("mysql", {})
            required_mysql = ["host", "user", "password", "database", "table_name"]
            missing = [r for r in required_mysql if r not in mysql_cfg or not mysql_cfg[r]]
            if missing:
                print(f"错误：MySQL配置缺少或为空的必需项: {', '.join(missing)}")
                return False

        # Add more checks as needed (e.g., prompt template, columns)

        print("配置文件基本验证通过。")
        return True
    except yaml.YAMLError as e:
        print(f"错误：配置文件 YAML 格式错误: {e}")
        return False
    except Exception as e:
        print(f"错误：验证配置文件时出错: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='通用AI批处理引擎 (异步版)')
    parser.add_argument('--config', '-c', default='./config.yaml', help='配置文件路径 (默认: ./config.yaml)')
    args = parser.parse_args()

    print(f"使用配置文件: {args.config}")

    # Validate config before initializing processor
    if not validate_config_file(args.config):
        sys.exit(1)

    try:
        # Pass config path for potential logging in init
        processor = UniversalAIProcessor(args.config)
        # Add config path to processor instance if needed later
        processor.config['config_path'] = args.config
        processor.process_tasks() # Start processing
    except (ValueError, ImportError, FileNotFoundError) as e:
         # Catch specific initialization errors
         logging.error(f"初始化处理器失败: {e}", exc_info=True)
         print(f"\n错误：初始化失败 - {e}")
         sys.exit(1)
    except KeyboardInterrupt:
        # Already handled in process_tasks, but good practice to have here too
        print("\n程序被用户中断。")
        sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
        # Catch unexpected errors during runtime after initialization
        logging.critical(f"运行时发生严重错误: {e}", exc_info=True)
        print(f"\n严重错误：{e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set higher default stack size for threads if needed (rarely necessary now with asyncio)
    # try:
    #      threading.stack_size(2*1024*1024) # 2MB stack size
    # except:
    #      print("无法设置线程堆栈大小。")
    main()