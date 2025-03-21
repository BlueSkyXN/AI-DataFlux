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
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from collections import defaultdict

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

    # 系统错误 - 不需要模型退避
    SYSTEM_ERROR = "system_error"  # 无可用模型等系统级错误

###############################################################################
# RWLock: 读写锁实现 - 用于模型调度器的锁竞争优化
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
# JsonValidator: 插件式JSON字段值验证器
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
# 令牌桶限流器：用于模型级请求限流
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
        self.lock = threading.Lock()

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
# 模型限流管理器：为每个模型维护独立的令牌桶
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
        检查指定模型是否可以处理新请求

        :param model_id: 模型ID
        :return: 如果可以处理返回True，否则返回False
        """
        model_id = str(model_id)
        with self.lock:
            if model_id not in self.limiters:
                return True
            return self.limiters[model_id].consume(1.0)

###############################################################################
# ModelDispatcher: 仅对API错误进行退避处理
###############################################################################
class ModelDispatcher:
    def __init__(self, models: List[Any], backoff_factor: int = 2):
        """
        使用改进的模型调度器:
        1. 读写锁分离 - 使用RWLock允许多个读操作并发
        2. 状态缓存 - 维护模型可用性缓存避免频繁锁操作
        3. 使用原子操作减少锁竞争

        仅对API_ERROR执行退避。

        :param models: 已解析的 ModelConfig 列表
        :param backoff_factor: 指数退避的基数
        """
        self.backoff_factor = backoff_factor

        # 模型状态记录表: model_id -> { fail_count, next_available_ts }
        self._model_state = {}
        for m in models:
            self._model_state[m.id] = {
                "fail_count": 0,
                "next_available_ts": 0,  # 0 表示随时可用
                "success_count": 0,
                "error_count": 0,
                "avg_response_time": 0
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
        with self._rwlock.write_lock():
            if model_id not in self._model_state:
                return
            state = self._model_state[model_id]

            # 更新成功/失败计数
            if success:
                state["success_count"] += 1
            else:
                state["error_count"] += 1

            # 计算平均响应时间（加权平均）
            total_calls = state["success_count"] + state["error_count"]
            if total_calls == 1:
                state["avg_response_time"] = response_time
            else:
                weight = min(0.1, 10.0 / total_calls)
                state["avg_response_time"] = state["avg_response_time"] * (1 - weight) + response_time * weight

    def get_model_success_rate(self, model_id: str) -> float:
        """获取模型的成功率"""
        with self._rwlock.read_lock():
            if model_id not in self._model_state:
                return 0.0
            state = self._model_state[model_id]
            total = state["success_count"] + state["error_count"]
            if total == 0:
                return 1.0
            return state["success_count"] / total

    def get_model_avg_response_time(self, model_id: str) -> float:
        """获取模型的平均响应时间"""
        with self._rwlock.read_lock():
            if model_id not in self._model_state:
                return 1.0
            return self._model_state[model_id]["avg_response_time"] or 1.0

    def is_model_available(self, model_id: str) -> bool:
        """判断某个模型当前是否可用 - 优先使用缓存，减少锁操作"""
        current_time = time.time()
        with self._rwlock.read_lock():
            cache_expired = (current_time - self._cache_last_update >= self._cache_ttl)
        if cache_expired:
            self._update_availability_cache()

        with self._rwlock.read_lock():
            if model_id in self._availability_cache:
                return self._availability_cache[model_id]
            if model_id in self._model_state:
                st = self._model_state[model_id]
                return (current_time >= st["next_available_ts"])
            return False

    def mark_model_success(self, model_id: str):
        """模型调用成功时，重置其失败计数"""
        with self._rwlock.write_lock():
            if model_id in self._model_state:
                self._model_state[model_id]["fail_count"] = 0
                self._model_state[model_id]["next_available_ts"] = 0
                self._availability_cache[model_id] = True

    def mark_model_failed(self, model_id: str, error_type: str = ErrorType.API_ERROR):
        """模型调用失败时的处理，只有 API_ERROR 才会导致退避"""
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
                    60
                )
            st["next_available_ts"] = time.time() + backoff_seconds
            self._availability_cache[model_id] = False
            logging.warning(
                f"模型[{model_id}] API调用失败，第{fail_count}次，进入退避 {backoff_seconds} 秒"
            )

    def get_available_models(self, exclude_model_ids: Set[str] = None) -> List[str]:
        """获取所有当前可用的模型ID"""
        if exclude_model_ids is None:
            exclude_model_ids = set()
        else:
            exclude_model_ids = set(exclude_model_ids)

        current_time = time.time()
        with self._rwlock.read_lock():
            cache_expired = current_time - self._cache_last_update >= self._cache_ttl
        if cache_expired:
            self._update_availability_cache()

        available_models = []
        with self._rwlock.read_lock():
            for model_id, is_available in self._availability_cache.items():
                if is_available and model_id not in exclude_model_ids:
                    available_models.append(model_id)
        return available_models

###############################################################################
# 抽象任务池基类
###############################################################################
class BaseTaskPool(ABC):
    """任务池抽象基类：定义数据源的通用接口"""

    def __init__(self, columns_to_extract: List[str], columns_to_write: Dict[str, str]):
        self.columns_to_extract = columns_to_extract
        self.columns_to_write = columns_to_write
        self.tasks = []  # 任务列表，每项为 (id, record_dict)
        self.lock = threading.Lock()

    @abstractmethod
    def get_total_task_count(self) -> int:
        """获取尚未处理的任务总数"""
        pass

    @abstractmethod
    def get_id_boundaries(self) -> Tuple[int, int]:
        """获取可用于分片的记录ID或索引范围，返回(最小ID, 最大ID)"""
        pass

    @abstractmethod
    def initialize_shard(self, shard_id: int, min_id: int, max_id: int) -> int:
        """初始化指定分片，返回加载的任务数"""
        pass

    @abstractmethod
    def get_task_batch(self, batch_size: int) -> List[Tuple[Any, Dict[str, Any]]]:
        """获取一批任务"""
        pass

    @abstractmethod
    def update_task_results(self, results: Dict[Any, Dict[str, Any]]):
        """批量更新任务处理结果"""
        pass

    @abstractmethod
    def reload_task_data(self, task_id: Any) -> Dict[str, Any]:
        """重新加载任务数据，用于系统错误重试"""
        pass

    def add_task_to_front(self, task_id: Any, record_dict: Dict[str, Any]):
        """将任务重新插回队列头部，用于系统错误重试"""
        with self.lock:
            self.tasks.insert(0, (task_id, record_dict))

    def has_tasks(self) -> bool:
        """检查是否还有待处理任务"""
        with self.lock:
            return len(self.tasks) > 0

    def get_remaining_count(self) -> int:
        """获取剩余任务数量"""
        with self.lock:
            return len(self.tasks)

###############################################################################
# MySQL任务池实现（不再使用processed字段，而根据输入/输出列是否为空来判断）
###############################################################################
class MySQLTaskPool(BaseTaskPool):
    """
    MySQL数据源任务池，支持分片加载
    
    逻辑：
      1. 只有当“所有输入列都非空” 且 “任意输出列为空” 时，才视为未处理任务。
      2. 如果输入列缺失或为空，则跳过不处理。
      3. 如果所有输出列也都已填充，则视为已处理。
    """

    def __init__(
        self,
        connection_config: Dict[str, Any],
        columns_to_extract: List[str],
        columns_to_write: Dict[str, str],
        table_name: str
    ):
        super().__init__(columns_to_extract, columns_to_write)

        if not MYSQL_AVAILABLE:
            raise ImportError("MySQL Connector库未安装，无法使用MySQL数据源")

        self.connection_config = connection_config
        self.table_name = table_name

        # 创建连接池（可自行封装或简化，这里保留原写法）
        self.pool = MySQLConnectionPool(
            connection_config=connection_config,
            min_size=5,
            max_size=20
        )

        # 分片状态
        self.current_shard_id = -1
        self.current_min_id = 0
        self.current_max_id = 0

    def execute_with_connection(self, callback):
        conn = None
        try:
            conn = self.pool.get_connection()
            if conn is None:
                raise Exception("无法获取数据库连接")
            return callback(conn)
        finally:
            if conn:
                self.pool.release_connection(conn)

    def _build_unprocessed_condition(self) -> str:
        """
        构造“未处理”条件：
          - 输入列均非空
          - 输出列至少有一个为空
        """
        # 输入列全部非空的条件
        input_conditions = []
        for col in self.columns_to_extract:
            c = f"`{col}` IS NOT NULL AND `{col}` <> ''"
            input_conditions.append(c)

        # 输出列至少有一个为空（要么是NULL，要么是''）
        output_conditions = []
        for _, out_col in self.columns_to_write.items():
            # 某个输出列为空的条件
            cond = f"`{out_col}` IS NULL OR `{out_col}` = ''"
            output_conditions.append(cond)
        # “至少一个”为空，需要把它们用 OR 连起来
        output_condition_str = "(" + " OR ".join(output_conditions) + ")"

        # 汇总
        unprocessed_where = "(" + " AND ".join(input_conditions) + ") AND " + output_condition_str
        return unprocessed_where

    def get_total_task_count(self) -> int:
        """获取尚未处理的任务数"""
        def _get_count(conn):
            try:
                cursor = conn.cursor()
                unprocessed_where = self._build_unprocessed_condition()
                sql = f"SELECT COUNT(*) FROM `{self.table_name}` WHERE {unprocessed_where}"
                cursor.execute(sql)
                result = cursor.fetchone()
                return result[0] if result else 0
            except Exception as e:
                logging.error(f"统计未处理任务总数失败: {e}")
                return 0
        return self.execute_with_connection(_get_count)

    def get_id_boundaries(self) -> Tuple[int, int]:
        """获取全表ID范围"""
        def _get_boundaries(conn):
            try:
                cursor = conn.cursor()
                cursor.execute(f"SELECT MIN(id), MAX(id) FROM `{self.table_name}`")
                result = cursor.fetchone()
                if result and result[0] is not None and result[1] is not None:
                    return (int(result[0]), int(result[1]))
                return (0, 0)
            except Exception as e:
                logging.error(f"获取ID范围失败: {e}")
                return (0, 0)
        return self.execute_with_connection(_get_boundaries)

    def initialize_shard(self, shard_id: int, min_id: int, max_id: int) -> int:
        """
        载入一个分片范围内的所有“未处理”记录
        """
        def _load_shard(conn):
            with self.lock:
                self.tasks = []
            self.current_shard_id = shard_id
            self.current_min_id = min_id
            self.current_max_id = max_id

            try:
                cursor = conn.cursor(dictionary=True)
                columns_str = ", ".join(f"`{col}`" for col in self.columns_to_extract)
                unprocessed_where = self._build_unprocessed_condition()

                # 在指定ID范围内筛选未处理
                sql = f"""
                    SELECT id, {columns_str}
                    FROM `{self.table_name}`
                    WHERE id BETWEEN %s AND %s
                      AND {unprocessed_where}
                    ORDER BY id
                """
                cursor.execute(sql, (min_id, max_id))
                rows = cursor.fetchall()

                with self.lock:
                    for row in rows:
                        record_id = row["id"]
                        record_dict = {}
                        for c in self.columns_to_extract:
                            record_dict[c] = row.get(c, "") or ""
                        # 添加到待处理列表
                        self.tasks.append((record_id, record_dict))

                loaded_count = len(self.tasks)
                logging.info(f"分片 {shard_id} (ID范围: {min_id}-{max_id}), 加载未处理任务数={loaded_count}")
                return loaded_count

            except Exception as e:
                logging.error(f"加载分片 {shard_id} 失败: {e}")
                return 0

        return self.execute_with_connection(_load_shard)

    def get_task_batch(self, batch_size: int) -> List[Tuple[Any, Dict[str, Any]]]:
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch

    def update_task_results(self, results: Dict[int, Dict[str, Any]]):
        """
        将结果写回输出列。
        只有没有 _error 的结果才更新。
        """
        if not results:
            return

        def _update(conn):
            try:
                cursor = conn.cursor()
                success_ids = []
                success_values = defaultdict(dict)

                # 过滤出成功处理的记录
                for record_id, row_result in results.items():
                    if "_error" not in row_result:
                        success_ids.append(record_id)
                        # 收集每个输出列的值
                        for alias, out_col in self.columns_to_write.items():
                            success_values[out_col][record_id] = row_result.get(alias, "")

                if not success_ids:
                    return

                # 尝试用批量更新（CASE WHEN）方式
                set_clauses = []
                for out_col, val_dict in success_values.items():
                    clause = f"`{out_col}` = CASE id "
                    for rid, val in val_dict.items():
                        safe_val = val.replace("'", "''") if isinstance(val, str) else str(val)
                        clause += f"WHEN {rid} THEN '{safe_val}' "
                    clause += f"ELSE `{out_col}` END"
                    set_clauses.append(clause)

                sql = f"""
                    UPDATE `{self.table_name}`
                    SET {', '.join(set_clauses)}
                    WHERE id IN ({', '.join(map(str, success_ids))})
                """
                cursor.execute(sql)
                conn.commit()
                logging.info(f"MySQL: 批量更新完成, 更新数量={len(success_ids)}")
            except Exception as e:
                conn.rollback()
                logging.error(f"MySQL: 批量更新失败, 将尝试单条更新: {e}")

                # 回退到单条更新
                try:
                    for record_id, row_result in results.items():
                        if "_error" not in row_result:
                            set_parts = []
                            params = []
                            for alias, out_col in self.columns_to_write.items():
                                set_parts.append(f"`{out_col}` = %s")
                                params.append(row_result.get(alias, ""))

                            sql_single = f"""
                                UPDATE `{self.table_name}`
                                SET {', '.join(set_parts)}
                                WHERE id = %s
                            """
                            params.append(record_id)
                            cursor.execute(sql_single, params)
                    conn.commit()
                    logging.info(f"MySQL: 单条更新模式完成, 更新数量={len(success_ids)}")
                except Exception as e2:
                    conn.rollback()
                    logging.error(f"MySQL: 单条更新也失败: {e2}")

        self.execute_with_connection(_update)

    def reload_task_data(self, record_id: int) -> Dict[str, Any]:
        """用于系统错误重试时，重新加载原始输入列"""
        def _reload(conn):
            record_dict = {}
            try:
                cursor = conn.cursor(dictionary=True)
                cols_str = ", ".join(f"`{c}`" for c in self.columns_to_extract)
                sql = f"SELECT {cols_str} FROM `{self.table_name}` WHERE id=%s"
                cursor.execute(sql, (record_id,))
                row = cursor.fetchone()
                if row:
                    for c in self.columns_to_extract:
                        record_dict[c] = row.get(c, "") or ""
            except Exception as e:
                logging.error(f"reload_task_data失败: {e}")
            return record_dict
        return self.execute_with_connection(_reload)

    def close(self):
        """关闭资源"""
        if hasattr(self, 'pool') and self.pool:
            self.pool.close_all()

###############################################################################
# MySQL连接池管理
###############################################################################
class MySQLConnectionPool:
    """线程安全的MySQL连接池管理器"""

    def __init__(self, connection_config: Dict[str, Any], min_size=5, max_size=20):
        self.config = connection_config
        self.min_size = min_size
        self.max_size = max_size
        self.pool = []
        self.in_use = set()
        self.lock = threading.Lock()

        # 初始化连接池
        for _ in range(min_size):
            self._add_connection()

    def _create_connection(self):
        return mysql.connector.connect(
            host=self.config["host"],
            port=self.config.get("port", 3306),
            user=self.config["user"],
            password=self.config["password"],
            database=self.config["database"],
            use_pure=True,
            autocommit=False,
            pool_reset_session=True,
            connection_timeout=10,
            get_warnings=True,
            raise_on_warnings=False
        )

    def _add_connection(self):
        try:
            conn = self._create_connection()
            self.pool.append(conn)
            return True
        except Exception as e:
            logging.error(f"创建数据库连接失败: {e}")
            return False

    def get_connection(self, timeout=5):
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                for i, conn in enumerate(self.pool):
                    if conn not in self.in_use:
                        try:
                            cursor = conn.cursor()
                            cursor.execute("SELECT 1")
                            cursor.fetchone()
                            cursor.close()
                            self.in_use.add(conn)
                            return conn
                        except Exception:
                            # 失效连接
                            try:
                                conn.close()
                            except:
                                pass
                            self.pool.pop(i)
                            new_conn = self._create_connection()
                            self.pool.append(new_conn)
                            self.in_use.add(new_conn)
                            return new_conn

                if len(self.pool) < self.max_size:
                    try:
                        new_conn = self._create_connection()
                        self.pool.append(new_conn)
                        self.in_use.add(new_conn)
                        return new_conn
                    except Exception as e:
                        logging.error(f"创建新连接失败: {e}")
            time.sleep(0.1)

        logging.error(f"获取数据库连接超时（{timeout}秒）")
        return None

    def release_connection(self, conn):
        if conn is None:
            return
        with self.lock:
            if conn in self.in_use:
                try:
                    conn.rollback()
                    self.in_use.remove(conn)
                except Exception as e:
                    logging.warning(f"重置连接状态失败: {e}")
                    try:
                        conn.close()
                        self.pool.remove(conn)
                        self.in_use.remove(conn)
                    except:
                        pass
                    if len(self.pool) < self.min_size:
                        self._add_connection()

    def close_all(self):
        with self.lock:
            for conn in self.pool:
                try:
                    conn.close()
                except:
                    pass
            self.pool.clear()
            self.in_use.clear()

###############################################################################
# Excel任务池实现（不使用processed列，而通过输入/输出列是否为空判断）
###############################################################################
class ExcelTaskPool(BaseTaskPool):
    """
    Excel数据源任务池，支持分片加载。

    逻辑：
      1. 只有当“所有输入列都非空” 且 “至少有一个输出列为空” 时，视为未处理。
      2. 若输入列为空或缺失则跳过。
      3. 若全部输出列都已填写，则视为已处理，跳过。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        columns_to_extract: List[str],
        columns_to_write: Dict[str, str],
        output_excel: str
    ):
        super().__init__(columns_to_extract, columns_to_write)
        self.df = df
        self.output_excel = output_excel

        self.current_shard_id = -1
        self.current_min_idx = 0
        self.current_max_idx = 0

        # 定期保存防止数据丢失
        self.last_save_time = time.time()
        self.save_interval = 300  # 默认5分钟保存一次

    def _filter_unprocessed_indices(self, min_idx: int, max_idx: int) -> List[int]:
        """
        返回 [min_idx, max_idx] 范围内的所有“未处理”行索引
        """
        sub_df = self.df.iloc[min_idx : max_idx + 1]

        # 输入列都非空
        cond_input = pd.Series([True] * len(sub_df), index=sub_df.index)
        for col in self.columns_to_extract:
            # 缺列就直接视为空
            if col not in sub_df.columns:
                cond_input &= False
            else:
                cond_input &= sub_df[col].notnull() & (sub_df[col].astype(str) != "")

        # 输出列至少有一个为空
        cond_output = pd.Series([False] * len(sub_df), index=sub_df.index)
        for alias, out_col in self.columns_to_write.items():
            if out_col not in sub_df.columns:
                # 该输出列不存在，等同于全部为空
                cond_output |= True
            else:
                # 任意一列为空 => 满足
                col_empty = sub_df[out_col].isnull() | (sub_df[out_col].astype(str) == "")
                cond_output |= col_empty

        # 组合条件
        final_cond = cond_input & cond_output
        return sub_df[final_cond].index.tolist()

    def get_total_task_count(self) -> int:
        """统计整表范围内的未处理行数"""
        unprocessed = self._filter_unprocessed_indices(0, len(self.df) - 1)
        return len(unprocessed)

    def get_id_boundaries(self) -> Tuple[int, int]:
        """获取整个DataFrame的索引范围"""
        return (0, len(self.df) - 1)

    def initialize_shard(self, shard_id: int, min_idx: int, max_idx: int) -> int:
        with self.lock:
            self.tasks = []
        self.current_shard_id = shard_id
        self.current_min_idx = min_idx
        self.current_max_idx = max_idx

        try:
            unprocessed_idx = self._filter_unprocessed_indices(min_idx, max_idx)
            with self.lock:
                for idx in unprocessed_idx:
                    record_dict = {}
                    for c in self.columns_to_extract:
                        val = ""
                        if c in self.df.columns:
                            cell_val = self.df.at[idx, c]
                            val = str(cell_val) if pd.notnull(cell_val) else ""
                        record_dict[c] = val
                    self.tasks.append((idx, record_dict))

            loaded_count = len(self.tasks)
            logging.info(f"加载分片 {shard_id} (索引范围: {min_idx}-{max_idx}), 未处理任务={loaded_count}")
            return loaded_count
        except Exception as e:
            logging.error(f"初始化分片 {shard_id} 失败: {e}")
            return 0

    def get_task_batch(self, batch_size: int) -> List[Tuple[Any, Dict[str, Any]]]:
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch

    def update_task_results(self, results: Dict[int, Dict[str, Any]]):
        if not results:
            return
        try:
            with self.lock:
                for idx, row_result in results.items():
                    if "_error" not in row_result:
                        for alias, col_name in self.columns_to_write.items():
                            self.df.at[idx, col_name] = row_result.get(alias, "")

                current_time = time.time()
                if current_time - self.last_save_time > self.save_interval:
                    self._save_excel()
                    self.last_save_time = current_time
        except Exception as e:
            logging.error(f"更新Excel记录失败: {e}")

    def _save_excel(self):
        try:
            self.df.to_excel(self.output_excel, index=False)
            logging.info(f"已保存数据到Excel文件: {self.output_excel}")
        except Exception as e:
            logging.error(f"保存Excel文件失败: {e}")

    def reload_task_data(self, idx: int) -> Dict[str, Any]:
        """重新加载指定行的输入列"""
        record_dict = {}
        try:
            if idx in self.df.index:
                for c in self.columns_to_extract:
                    if c in self.df.columns:
                        cell_val = self.df.at[idx, c]
                        record_dict[c] = str(cell_val) if pd.notnull(cell_val) else ""
            else:
                logging.warning(f"reload_task_data: 行索引={idx} 不存在")
        except Exception as e:
            logging.error(f"reload_task_data失败: {e}")
        return record_dict

    def close(self):
        """处理完毕后保存一次Excel"""
        self._save_excel()

###############################################################################
# 分片任务管理器（仅管理分片逻辑，不再维护任务进度表）
###############################################################################
class ShardedTaskManager:
    """分片任务管理器，负责大规模数据的分片加载与处理"""

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

        self.current_shard = 0
        self.total_shards = 0
        self.shard_boundaries = []

        # 性能统计
        self.total_estimated = 0
        self.total_processed = 0
        self.processing_metrics = {
            'avg_time_per_record': 0,
            'records_per_second': 0,
        }

        # 内存监控
        self.memory_tracker = {
            'last_check_time': time.time(),
            'check_interval': 60,
            'peak_memory': 0,
            'current_memory': 0
        }

    def calculate_optimal_shard_size(self) -> int:
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        record_size_mb = 0.01  # 假设每条记录平均占用10KB
        memory_based_size = int(available_mb * 0.3 / record_size_mb)
        if self.processing_metrics['records_per_second'] > 0:
            ideal_batch_duration = 600
            time_based_size = int(self.processing_metrics['records_per_second'] * ideal_batch_duration)
            calculated_size = min(memory_based_size, time_based_size)
        else:
            calculated_size = memory_based_size
        shard_size = max(self.min_shard_size, min(calculated_size, self.max_shard_size))
        logging.info(f"动态计算得到的分片大小: {shard_size}, 内存可用: {available_mb:.1f}MB")
        return shard_size

    def initialize(self) -> bool:
        try:
            self.total_estimated = self.task_pool.get_total_task_count()
            if self.total_estimated == 0:
                logging.info("没有需要处理的任务，已全部完成")
                return False

            min_id, max_id = self.task_pool.get_id_boundaries()
            logging.info(f"ID(或索引)范围: {min_id} - {max_id}, 预估未处理任务数: {self.total_estimated}")

            shard_size = self.calculate_optimal_shard_size()

            # 分片数
            total_range = max_id - min_id + 1
            if total_range <= 0:
                self.total_shards = 1
                self.shard_boundaries = [(min_id, max_id)]
            else:
                self.total_shards = max(1, (total_range + shard_size - 1) // shard_size)
                boundaries = []
                start = min_id
                for i in range(self.total_shards):
                    end = min(start + shard_size - 1, max_id)
                    boundaries.append((start, end))
                    start = end + 1
                self.shard_boundaries = boundaries

            logging.info(f"共分成 {self.total_shards} 个分片，每片约 {shard_size} 条记录")
            return True
        except Exception as e:
            logging.error(f"初始化分片任务管理器失败: {e}")
            return False

    def load_next_shard(self) -> bool:
        if self.current_shard >= self.total_shards:
            logging.info("所有分片已处理完毕")
            return False
        min_id, max_id = self.shard_boundaries[self.current_shard]
        loaded = self.task_pool.initialize_shard(self.current_shard, min_id, max_id)
        if loaded == 0:
            # 若当前分片无任务则尝试下一个
            self.current_shard += 1
            return self.load_next_shard()
        return True

    def update_processing_metrics(self, batch_size, processing_time):
        if processing_time > 0 and batch_size > 0:
            time_per_record = processing_time / batch_size
            if self.processing_metrics['avg_time_per_record'] == 0:
                self.processing_metrics['avg_time_per_record'] = time_per_record
            else:
                # 移动加权平均
                self.processing_metrics['avg_time_per_record'] = (
                    0.7 * self.processing_metrics['avg_time_per_record'] +
                    0.3 * time_per_record
                )
            self.processing_metrics['records_per_second'] = 1.0 / max(
                1e-9, self.processing_metrics['avg_time_per_record']
            )

    def monitor_memory_usage(self):
        current_time = time.time()
        if current_time - self.memory_tracker['last_check_time'] < self.memory_tracker['check_interval']:
            return
        try:
            process = psutil.Process()
            current_mem = process.memory_info().rss / (1024 * 1024)
            self.memory_tracker['current_memory'] = current_mem
            self.memory_tracker['peak_memory'] = max(self.memory_tracker['peak_memory'], current_mem)
            self.memory_tracker['last_check_time'] = current_time

            mem = psutil.virtual_memory()
            if mem.percent > 85 or current_mem > 1024:  # 大于1GB或系统使用率>85%
                gc.collect()
                logging.info(
                    f"内存使用较高，已触发GC: current={current_mem:.1f}MB, peak={self.memory_tracker['peak_memory']:.1f}MB"
                )
        except Exception as e:
            logging.warning(f"内存监控失败: {e}")

    def finalize(self):
        elapsed = 0  # 此处可自行记录处理总时长
        logging.info(f"任务处理结束，总处理记录: {self.total_processed}, 峰值内存: {self.memory_tracker['peak_memory']:.1f}MB")
        # 关闭资源
        self.task_pool.close()

###############################################################################
# 模型配置类
###############################################################################
class ModelConfig:
    def __init__(self, model_dict: Dict[str, Any], channels: Dict[str, Any]):
        self.id = model_dict.get("id")
        self.name = model_dict.get("name")
        self.model = model_dict.get("model")
        self.channel_id = str(model_dict.get("channel_id"))
        self.api_key = model_dict.get("api_key")
        self.timeout = model_dict.get("timeout", 600)
        self.weight = model_dict.get("weight", 1)
        self.base_weight = self.weight
        self.max_weight = self.weight * 2
        self.temperature = model_dict.get("temperature", 0.7)

        self.safe_rps = model_dict.get("safe_rps", max(0.5, min(self.weight / 10, 10)))

        if not self.id or not self.model or not self.channel_id:
            raise ValueError(f"模型配置缺少必填字段: id={self.id}, model={self.model}, channel_id={self.channel_id}")

        if self.channel_id not in channels:
            raise ValueError(f"channel_id={self.channel_id} 在channels中不存在")

        channel_cfg = channels[self.channel_id]
        self.channel_name = channel_cfg.get("name")
        self.base_url = channel_cfg.get("base_url")
        self.api_path = channel_cfg.get("api_path", "/v1/chat/completions")
        self.channel_timeout = channel_cfg.get("timeout", 600)
        self.channel_proxy = channel_cfg.get("proxy", "")

        self.final_timeout = min(self.timeout, self.channel_timeout)
        self.connect_timeout = 10
        self.read_timeout = self.final_timeout

###############################################################################
# 加载配置文件
###############################################################################
def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("配置文件格式错误，必须是字典类型")
            return config
    except FileNotFoundError:
        print(f"配置文件 {config_path} 不存在！")
        sys.exit(1)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        sys.exit(1)

###############################################################################
# 初始化日志
###############################################################################
def init_logging(log_config: Dict[str, Any]):
    level_str = log_config.get("level", "info").upper()
    level = getattr(logging, level_str, logging.INFO)

    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    if log_config.get("format") == "json":
        log_format = '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'

    output_type = log_config.get("output", "console")
    if output_type == "file":
        file_path = log_config.get("file_path", "./logs/universal_ai.log")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        except Exception as e:
            print(f"创建日志目录失败: {e}")
            sys.exit(1)
        logging.basicConfig(filename=file_path, level=level, format=log_format)
    else:
        logging.basicConfig(level=level, format=log_format)

###############################################################################
# 主处理类
###############################################################################
class UniversalAIProcessor:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        global_cfg = self.config.get("global", {})
        init_logging(global_cfg.get("log", {}))

        models_cfg = self.config.get("models", [])
        channels_cfg = self.config.get("channels", {})
        if not models_cfg:
            logging.error("配置文件中未找到 models 配置！")
            raise ValueError("缺少 models 配置")

        # 加载模型配置
        model_ids = set()
        self.models = []
        for m in models_cfg:
            mid = m.get("id")
            if mid in model_ids:
                logging.warning(f"发现重复的模型ID: {mid}, 将使用后者覆盖前者")
            model_ids.add(mid)
            self.models.append(ModelConfig(m, channels_cfg))

        # 数据源类型
        self.datasource_type = self.config.get("datasource", {}).get("type", "excel").lower()
        logging.info(f"使用数据源类型: {self.datasource_type}")

        # 并发相关
        concurrency_cfg = {}
        if "datasource" in self.config and "concurrency" in self.config["datasource"]:
            concurrency_cfg = self.config["datasource"]["concurrency"]
        elif "excel" in self.config and "concurrency" in self.config["excel"]:
            concurrency_cfg = self.config["excel"]["concurrency"]
        elif "mysql" in self.config and "concurrency" in self.config["mysql"]:
            concurrency_cfg = self.config["mysql"]["concurrency"]

        self.max_workers = concurrency_cfg.get("max_workers", 5)
        self.batch_size = concurrency_cfg.get("batch_size", 300)
        self.save_interval = concurrency_cfg.get("save_interval", 300)
        self.global_retry_times = concurrency_cfg.get("retry_times", 3)
        self.shard_size = concurrency_cfg.get("shard_size", 10000)
        self.min_shard_size = concurrency_cfg.get("min_shard_size", 1000)
        self.max_shard_size = concurrency_cfg.get("max_shard_size", 50000)
        backoff_factor = concurrency_cfg.get("backoff_factor", 2)

        # 提示词
        prompt_cfg = self.config.get("prompt", {})
        self.prompt_template = prompt_cfg.get("template", "")
        self.required_fields = prompt_cfg.get("required_fields", [])

        # JSON字段验证
        self.validator = JsonValidator()
        validation_cfg = self.config.get("validation", {})
        self.validator.configure(validation_cfg)

        # 字段配置
        self.columns_to_extract = self.config.get("columns_to_extract", [])
        self.columns_to_write = self.config.get("columns_to_write", {})

        # 初始化调度器
        self.dispatcher = ModelDispatcher(self.models, backoff_factor=backoff_factor)
        self.rate_limiter = ModelRateLimiter()
        self.rate_limiter.configure(models_cfg)

        # 构建模型Map
        self.model_map = {m.id: m for m in self.models}

        # 加权随机池
        self.models_pool = []
        for model_config in self.models:
            self.models_pool.extend([model_config] * model_config.weight)

        # 创建数据源任务池
        self.task_pool = self._create_task_pool()

        # 分片管理器
        self.task_manager = ShardedTaskManager(
            task_pool=self.task_pool,
            optimal_shard_size=self.shard_size,
            min_shard_size=self.min_shard_size,
            max_shard_size=self.max_shard_size
        )

        logging.info(f"共加载 {len(self.models)} 个模型，加权池大小: {len(self.models_pool)}")
        logging.info(f"批处理大小: {self.batch_size}, 保存间隔: {self.save_interval}")

    def _create_task_pool(self) -> BaseTaskPool:
        if self.datasource_type == "mysql":
            if not MYSQL_AVAILABLE:
                logging.error("MySQL Connector库未安装，无法使用MySQL数据源")
                raise ImportError("MySQL Connector库未安装")

            mysql_config = self.config.get("mysql", {})
            table_name = mysql_config.get("table_name")
            connection_config = {
                "host": mysql_config.get("host", "localhost"),
                "port": mysql_config.get("port", 3306),
                "user": mysql_config.get("user", "root"),
                "password": mysql_config.get("password", ""),
                "database": mysql_config.get("database", "")
            }
            return MySQLTaskPool(
                connection_config=connection_config,
                columns_to_extract=self.columns_to_extract,
                columns_to_write=self.columns_to_write,
                table_name=table_name
            )
        elif self.datasource_type == "excel":
            excel_config = self.config.get("excel", {})
            input_excel = excel_config.get("input_path")
            output_excel = excel_config.get("output_path") or (
                input_excel.replace(".xlsx", "_output.xlsx").replace(".xls", "_output.xls")
            )

            if not os.path.exists(input_excel):
                logging.error(f"Excel输入文件不存在: {input_excel}")
                raise FileNotFoundError(f"Excel文件不存在: {input_excel}")

            logging.info(f"读取Excel: {input_excel}")
            df = pd.read_excel(input_excel)

            # 确保要写回的列都存在（没有就创建空列）
            for _, out_col in self.columns_to_write.items():
                if out_col not in df.columns:
                    df[out_col] = ""

            return ExcelTaskPool(
                df=df,
                columns_to_extract=self.columns_to_extract,
                columns_to_write=self.columns_to_write,
                output_excel=output_excel
            )
        else:
            raise ValueError(f"不支持的数据源类型: {self.datasource_type}")

    def adjust_model_weights(self):
        """动态调整模型的权重"""
        adjusted_models = []
        for model_id, model in self.model_map.items():
            success_rate = self.dispatcher.get_model_success_rate(model_id)
            avg_response_time = self.dispatcher.get_model_avg_response_time(model_id)
            is_available = self.dispatcher.is_model_available(model_id)

            # 权重计算
            success_factor = success_rate ** 2
            speed_factor = 1.0 / max(0.1, avg_response_time)
            availability_factor = 1.0 if is_available else 0.1

            new_weight = int(model.base_weight * success_factor * speed_factor * availability_factor)
            new_weight = max(1, min(new_weight, model.max_weight))

            if new_weight != model.weight:
                logging.info(
                    f"调整模型[{model.name}]权重: {model.weight} -> {new_weight} "
                    f"(成功率={success_rate:.2f}, RT={avg_response_time:.2f}s, 可用={is_available})"
                )
                model.weight = new_weight
            adjusted_models.append(model)

        self.models_pool = []
        for model in adjusted_models:
            self.models_pool.extend([model] * model.weight)

    def get_available_model_randomly(self, exclude_model_ids=None) -> Optional[ModelConfig]:
        if exclude_model_ids is None:
            exclude_model_ids = set()
        available_model_ids = self.dispatcher.get_available_models(exclude_model_ids)
        # 限流检查
        available_model_ids = [m for m in available_model_ids if self.rate_limiter.can_process(m)]
        if not available_model_ids:
            return None
        pool = []
        for mid in available_model_ids:
            model = self.model_map[mid]
            pool.extend([model] * model.weight)
        if not pool:
            return None
        return random.choice(pool)

    def create_prompt(self, record_data: Dict[str, Any]) -> str:
        try:
            record_json_str = json.dumps(record_data, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logging.error(f"行数据无法序列化为JSON: {e}")
            return self.prompt_template
        if "{record_json}" not in self.prompt_template:
            logging.warning("提示词模板中不包含 {record_json} 占位符")
        return self.prompt_template.replace("{record_json}", record_json_str)

    def extract_json_from_response(self, content: str) -> Dict[str, Any]:
        def check_required(data_dict: Dict[str, Any], required: List[str]) -> bool:
            return all(k in data_dict for k in required)

        # 1) 整体解析
        try:
            data = json.loads(content)
            if isinstance(data, dict) and check_required(data, self.required_fields):
                if self.validator.enabled:
                    is_valid, errors = self.validator.validate(data)
                    if not is_valid:
                        return {
                            "_error": "invalid_field_values",
                            "_error_type": ErrorType.CONTENT_ERROR,
                            "_validation_errors": errors
                        }
                return data
        except json.JSONDecodeError:
            pass

        # 2) 正则提取
        pattern = r'(\{.*?\})'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                candidate = json.loads(match)
                if isinstance(candidate, dict) and check_required(candidate, self.required_fields):
                    if self.validator.enabled:
                        is_valid, errors = self.validator.validate(candidate)
                        if not is_valid:
                            continue
                    return candidate
            except:
                continue

        logging.warning(f"无法解析出包含必需字段{self.required_fields}的有效JSON")
        return {"_error": "invalid_json", "_error_type": ErrorType.CONTENT_ERROR}

    async def call_ai_api_async(self, model_cfg: ModelConfig, prompt: str) -> str:
        url = model_cfg.base_url.rstrip("/") + model_cfg.api_path
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model_cfg.api_key}"
        }
        payload = {
            "model": model_cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": model_cfg.temperature
        }
        proxy = model_cfg.channel_proxy or None
        timeout = aiohttp.ClientTimeout(
            connect=model_cfg.connect_timeout,
            total=model_cfg.read_timeout
        )
        start_time = time.time()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload, proxy=proxy) as resp:
                    if resp.status >= 400:
                        error_text = await resp.text()
                        raise aiohttp.ClientResponseError(
                            request_info=resp.request_info,
                            history=resp.history,
                            status=resp.status,
                            message=f"HTTP {resp.status}: {error_text}",
                            headers=resp.headers
                        )
                    data = await resp.json()
                    if "choices" not in data or not data["choices"]:
                        raise ValueError("AI返回不含choices字段")
                    content = data["choices"][0]["message"]["content"]
                    response_time = time.time() - start_time
                    self.dispatcher.update_model_metrics(model_cfg.id, response_time, True)
                    return content
        except aiohttp.ClientError as e:
            response_time = time.time() - start_time
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            raise
        except Exception as e:
            response_time = time.time() - start_time
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            raise

    async def process_one_record_async(self, record_id: Any, row_data: Dict[str, Any]) -> Dict[str, Any]:
        log_label = f"记录[{record_id}]"
        prompt = self.create_prompt(row_data)
        
        # 全局重试循环 - 使用配置文件中的重试次数
        for global_attempt in range(self.global_retry_times):
            used_model_ids = set()
            
            # 在当前轮次中尝试所有可用模型
            while True:
                model_cfg = self.get_available_model_randomly(used_model_ids)
                if not model_cfg:
                    # 当前轮次已尝试所有可用模型
                    break
                
                used_model_ids.add(model_cfg.id)
                try:
                    # 调用模型API
                    content = await self.call_ai_api_async(model_cfg, prompt)
                    parsed = self.extract_json_from_response(content)
                    
                    # 检查JSON解析结果
                    if "_error" in parsed and parsed.get("_error_type") == ErrorType.CONTENT_ERROR:
                        # 内容问题，继续尝试下一个模型
                        logging.warning(f"{log_label}: 模型[{model_cfg.name}]内容解析错误，尝试下一个模型")
                        continue
                    
                    # 调用成功
                    self.dispatcher.mark_model_success(model_cfg.id)
                    parsed["response_excerpt"] = (content[:100] + "...") if len(content) > 100 else content
                    parsed["used_model_id"] = model_cfg.id
                    parsed["used_model_name"] = model_cfg.name
                    logging.info(f"{log_label}: 模型[{model_cfg.name}] 调用成功")
                    return parsed
                    
                except aiohttp.ClientError as e:
                    logging.warning(f"{log_label}: 模型[{model_cfg.name}]网络异常(API_ERROR): {e}")
                    self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
                except Exception as e:
                    logging.warning(f"{log_label}: 模型[{model_cfg.name}]调用异常(API_ERROR): {e}")
                    self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
            
            # 当前轮次尝试完所有可用模型后，仍未成功
            if global_attempt < self.global_retry_times - 1:
                # 使用退避因子计算等待时间
                wait_time = self.backoff_factor ** (global_attempt + 1)
                # 限制最大等待时间，防止等待过长
                wait_time = min(wait_time, 60)
                
                logging.warning(
                    f"{log_label}: 第{global_attempt+1}轮所有可用模型均尝试失败，等待{wait_time}秒后开始下一轮重试"
                )
                await asyncio.sleep(wait_time)
                logging.info(f"{log_label}: 等待结束，开始第{global_attempt+2}轮重试")
        
        # 所有重试轮次都失败
        logging.error(f"{log_label}: 经过{self.global_retry_times}轮重试后所有模型均尝试失败!")
        return {"_error": "all_models_failed", "_error_type": ErrorType.CONTENT_ERROR}

    async def process_shard_async(self):
        if not self.task_manager.initialize():
            logging.info("无可处理分片")
            return

        async with aiohttp.ClientSession() as session:
            while True:
                if not self.task_manager.load_next_shard():
                    break
                while self.task_pool.has_tasks():
                    batch_size = min(self.batch_size, self.task_pool.get_remaining_count())
                    tasks_batch = self.task_pool.get_task_batch(batch_size)
                    if not tasks_batch:
                        break

                    batch_start = time.time()
                    coros = []
                    for rid, data in tasks_batch:
                        coros.append(self.process_one_record_async(rid, data))
                    results = await asyncio.gather(*coros, return_exceptions=True)

                    batch_results = {}
                    for (rid, _), result in zip(tasks_batch, results):
                        if isinstance(result, Exception):
                            logging.error(f"记录[{rid}]处理出现系统异常: {result}")
                            # 系统错误 => 放回队列头
                            self.task_pool.add_task_to_front(rid, self.task_pool.reload_task_data(rid))
                            continue
                        # 系统级错误 => 放回队列头
                        if result.get("_error_type") == ErrorType.SYSTEM_ERROR:
                            self.task_pool.add_task_to_front(rid, self.task_pool.reload_task_data(rid))
                            continue
                        batch_results[rid] = result

                    self.task_pool.update_task_results(batch_results)
                    batch_time = time.time() - batch_start
                    self.task_manager.update_processing_metrics(len(batch_results), batch_time)
                    self.task_manager.total_processed += len(batch_results)

                    # 动态调整模型权重
                    if random.random() < 0.1:
                        self.adjust_model_weights()

                    self.task_manager.monitor_memory_usage()

        self.task_manager.finalize()

    def process_tasks(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.process_shard_async())
        except KeyboardInterrupt:
            logging.info("处理被用户中断")
        except Exception as e:
            logging.error(f"执行过程中出现异常: {e}")
        finally:
            loop.close()

###############################################################################
# 命令行入口
###############################################################################
def validate_config_file(config_path: str) -> bool:
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if "models" not in config or not config["models"]:
            print("配置文件缺少 models 部分")
            return False
        ds_type = config.get("datasource", {}).get("type", "excel").lower()
        if ds_type not in ["mysql", "excel"]:
            print(f"不支持的数据源类型: {ds_type}")
            return False
        if ds_type == "excel":
            if "excel" not in config or "input_path" not in config["excel"]:
                print("Excel数据源缺少 input_path")
                return False
        if ds_type == "mysql":
            if not MYSQL_AVAILABLE:
                print("MySQL Connector库未安装，无法使用MySQL数据源")
                return False
            if "mysql" not in config:
                print("MySQL配置不存在")
                return False
            required_mysql = ["host", "user", "password", "database", "table_name"]
            for r in required_mysql:
                if r not in config["mysql"]:
                    print(f"MySQL配置缺少 {r}")
                    return False
        return True
    except Exception as e:
        print(f"配置文件验证失败: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='通用AI批处理引擎')
    parser.add_argument('--config', '-c', default='./config.yaml', help='配置文件路径')
    args = parser.parse_args()

    if not validate_config_file(args.config):
        sys.exit(1)

    try:
        processor = UniversalAIProcessor(args.config)
        processor.process_tasks()
    except KeyboardInterrupt:
        print("用户中断程序")
    except Exception as e:
        print(f"运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()