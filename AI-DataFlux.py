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
from typing import Dict, Any, Optional, List, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict

# 尝试导入MySQL相关库，如果不可用则标记为None
try:
    import mysql.connector
    from mysql.connector import Error, pooling
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

        # 读写锁：允许多线程同时读取模型状态
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
                new_cache[model_id] = current_time >= state["next_available_ts"]
            
            # 原子更新缓存
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
                # 使用加权平均计算新的平均响应时间
                weight = min(0.1, 10.0 / total_calls)  # 最新响应权重，随调用次数增加而减小
                state["avg_response_time"] = state["avg_response_time"] * (1 - weight) + response_time * weight
    
    def get_model_success_rate(self, model_id: str) -> float:
        """获取模型的成功率"""
        with self._rwlock.read_lock():
            if model_id not in self._model_state:
                return 0.0
                
            state = self._model_state[model_id]
            total = state["success_count"] + state["error_count"]
            
            if total == 0:
                return 1.0  # 无调用历史，默认完全可用
                
            return state["success_count"] / total
    
    def get_model_avg_response_time(self, model_id: str) -> float:
        """获取模型的平均响应时间"""
        with self._rwlock.read_lock():
            if model_id not in self._model_state:
                return 1.0  # 默认1秒
                
            return self._model_state[model_id]["avg_response_time"] or 1.0

    def is_model_available(self, model_id: str) -> bool:
        """判断某个模型当前是否可用 - 优先使用缓存，减少锁操作"""
        current_time = time.time()
        
        # 使用读锁检查缓存是否需要更新
        with self._rwlock.read_lock():
            cache_expired = current_time - self._cache_last_update >= self._cache_ttl
            
        if cache_expired:
            self._update_availability_cache()
        
        # 使用读锁获取模型可用性
        with self._rwlock.read_lock():
            if model_id in self._availability_cache:
                return self._availability_cache[model_id]
            
            # 如果缓存里没有，就读取状态
            if model_id in self._model_state:
                st = self._model_state[model_id]
                is_available = current_time >= st["next_available_ts"]
                return is_available
            
            return False  # 未知模型默认不可用

    def mark_model_success(self, model_id: str):
        """模型调用成功时，重置其失败计数"""
        with self._rwlock.write_lock():
            if model_id in self._model_state:
                self._model_state[model_id]["fail_count"] = 0
                self._model_state[model_id]["next_available_ts"] = 0
                # 更新缓存
                self._availability_cache[model_id] = True

    def mark_model_failed(self, model_id: str, error_type: str = ErrorType.API_ERROR):
        """
        模型调用失败时的处理
        :param model_id: 模型ID
        :param error_type: 错误类型，只有API_ERROR才会导致模型退避
        """
        # 如果是内容处理错误，不退避
        if error_type != ErrorType.API_ERROR:
            return
            
        with self._rwlock.write_lock():
            if model_id not in self._model_state:
                return
                
            st = self._model_state[model_id]
            st["fail_count"] += 1
            fail_count = st["fail_count"]
            
            # 使用更温和的退避算法（线性与指数的混合）
            if fail_count <= 3:
                # 初始阶段使用线性退避
                backoff_seconds = fail_count * 2
            else:
                # 超过3次失败使用受限指数退避
                backoff_seconds = min(
                    6 + (self.backoff_factor ** (fail_count - 3)),
                    60  # 最大退避60秒
                )
            
            st["next_available_ts"] = time.time() + backoff_seconds
            
            # 更新缓存
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
        
        # 检查缓存是否需要更新
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
# 进度跟踪类：用于记录任务处理进度和支持断点续传
###############################################################################
class ProgressTracker:
    """跟踪任务处理进度，支持断点续传"""
    
    def __init__(self, job_id: str = None):
        """
        初始化进度跟踪器
        
        :param job_id: 任务ID，如果为None则生成随机ID
        """
        self.job_id = job_id or f"job_{int(time.time())}_{random.randint(1000, 9999)}"
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.total_estimated = 0
        self.total_processed = 0
        self.current_shard = 0
        self.last_processed_id = 0
        self.status = "initializing"  # initializing, running, completed, failed
        self.error = None
        
        # 进度记录表是否存在的标志
        self.table_exists = False
        
        # 处理性能指标
        self.processing_rate = 0  # 每秒处理记录数
        self.estimated_completion_time = None  # 预计完成时间
    
    def init_progress_table(self, conn):
        """初始化进度记录表"""
        try:
            cursor = conn.cursor()
            
            # 检查表是否存在
            cursor.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = 'task_progress'
            """)
            
            if cursor.fetchone()[0] == 0:
                # 创建表
                cursor.execute("""
                    CREATE TABLE task_progress (
                        job_id VARCHAR(50) PRIMARY KEY,
                        current_shard INT,
                        last_processed_id BIGINT,
                        total_processed INT,
                        total_estimated INT,
                        status VARCHAR(20),
                        error TEXT,
                        start_time TIMESTAMP,
                        last_update_time TIMESTAMP,
                        processing_rate FLOAT
                    )
                """)
                conn.commit()
            
            self.table_exists = True
            
        except Exception as e:
            logging.warning(f"无法初始化进度表: {e}")
            self.table_exists = False
    
    def load_progress(self, conn, job_id: str = None):
        """
        从数据库加载进度
        
        :param conn: 数据库连接
        :param job_id: 要加载的任务ID，如果为None则加载最新的未完成任务
        :return: 是否成功加载
        """
        if not self.table_exists:
            return False
            
        try:
            cursor = conn.cursor(dictionary=True)
            
            if job_id:
                # 加载指定任务
                cursor.execute("""
                    SELECT * FROM task_progress
                    WHERE job_id = %s
                """, (job_id,))
            else:
                # 加载最新的未完成任务
                cursor.execute("""
                    SELECT * FROM task_progress
                    WHERE status NOT IN ('completed', 'failed')
                    ORDER BY last_update_time DESC
                    LIMIT 1
                """)
            
            result = cursor.fetchone()
            if not result:
                return False
            
            # 加载进度
            self.job_id = result["job_id"]
            self.current_shard = result["current_shard"]
            self.last_processed_id = result["last_processed_id"]
            self.total_processed = result["total_processed"]
            self.total_estimated = result["total_estimated"]
            self.status = result["status"]
            self.error = result["error"]
            self.start_time = result["start_time"].timestamp()
            self.last_update_time = result["last_update_time"].timestamp()
            self.processing_rate = result["processing_rate"] or 0
            
            logging.info(f"已加载任务进度: ID={self.job_id}, 当前分片={self.current_shard}, "
                        f"已处理={self.total_processed}/{self.total_estimated}")
            
            return True
            
        except Exception as e:
            logging.error(f"加载任务进度失败: {e}")
            return False
    
    def save_progress(self, conn):
        """
        保存进度到数据库
        
        :param conn: 数据库连接
        :return: 是否成功保存
        """
        if not self.table_exists:
            return False
        
        now = time.time()
        # 计算处理速率
        elapsed = now - self.last_update_time
        if elapsed > 0 and self.total_processed > 0:
            # 使用加权平均计算处理率，偏重于最新数据
            if self.processing_rate == 0:
                self.processing_rate = self.total_processed / (now - self.start_time)
            else:
                # 新旧比重: 30% 新, 70% 旧
                new_rate = self.total_processed / (now - self.start_time)
                self.processing_rate = 0.3 * new_rate + 0.7 * self.processing_rate
        
        # 计算预计完成时间
        if self.processing_rate > 0 and self.total_estimated > self.total_processed:
            remaining = self.total_estimated - self.total_processed
            seconds_left = remaining / self.processing_rate
            self.estimated_completion_time = now + seconds_left
        
        self.last_update_time = now
        
        try:
            cursor = conn.cursor()
            
            # 检查记录是否存在
            cursor.execute("SELECT COUNT(*) FROM task_progress WHERE job_id = %s", (self.job_id,))
            exists = cursor.fetchone()[0] > 0
            
            if exists:
                # 更新记录
                cursor.execute("""
                    UPDATE task_progress
                    SET 
                        current_shard = %s,
                        last_processed_id = %s,
                        total_processed = %s,
                        total_estimated = %s,
                        status = %s,
                        error = %s,
                        last_update_time = %s,
                        processing_rate = %s
                    WHERE job_id = %s
                """, (
                    self.current_shard,
                    self.last_processed_id,
                    self.total_processed,
                    self.total_estimated,
                    self.status,
                    self.error,
                    datetime.fromtimestamp(self.last_update_time),
                    self.processing_rate,
                    self.job_id
                ))
            else:
                # 插入新记录
                cursor.execute("""
                    INSERT INTO task_progress (
                        job_id, current_shard, last_processed_id,
                        total_processed, total_estimated, status,
                        error, start_time, last_update_time,
                        processing_rate
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.job_id,
                    self.current_shard,
                    self.last_processed_id,
                    self.total_processed,
                    self.total_estimated,
                    self.status,
                    self.error,
                    datetime.fromtimestamp(self.start_time),
                    datetime.fromtimestamp(self.last_update_time),
                    self.processing_rate
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logging.error(f"保存进度失败: {e}")
            try:
                conn.rollback()
            except:
                pass
            return False
    
    def update_status(self, conn, status: str, error: str = None):
        """
        更新任务状态
        
        :param conn: 数据库连接
        :param status: 新状态
        :param error: 错误信息
        """
        self.status = status
        self.error = error
        self.last_update_time = time.time()
        
        if conn and self.table_exists:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE task_progress
                    SET status = %s, error = %s, last_update_time = %s
                    WHERE job_id = %s
                """, (
                    status,
                    error,
                    datetime.fromtimestamp(self.last_update_time),
                    self.job_id
                ))
                conn.commit()
            except Exception as e:
                logging.error(f"更新任务状态失败: {e}")
                try:
                    conn.rollback()
                except:
                    pass
    
    def log_progress(self):
        """记录当前进度到日志"""
        elapsed = time.time() - self.start_time
        
        # 计算预计剩余时间
        if self.processing_rate > 0 and self.total_estimated > 0:
            remaining = self.total_estimated - self.total_processed
            time_left = remaining / self.processing_rate
            
            time_left_str = ""
            if time_left > 3600:
                time_left_str = f"{time_left/3600:.1f}小时"
            elif time_left > 60:
                time_left_str = f"{time_left/60:.1f}分钟"
            else:
                time_left_str = f"{time_left:.0f}秒"
            
            progress_msg = (
                f"进度: {self.total_processed}/{self.total_estimated} "
                f"({self.total_processed/self.total_estimated*100:.1f}%), "
                f"速率: {self.processing_rate:.1f}条/秒, 剩余时间: {time_left_str}"
            )
        else:
            progress_msg = (
                f"进度: {self.total_processed} 已处理, "
                f"速率: {self.total_processed/max(1, elapsed):.1f}条/秒, "
                f"运行时间: {elapsed:.0f}秒"
            )
        
        mem_usage = psutil.Process().memory_info().rss / (1024 * 1024)
        logging.info(f"{progress_msg}, 内存: {mem_usage:.1f}MB")

###############################################################################
# MySQL连接池管理
###############################################################################
class MySQLConnectionPool:
    """线程安全的MySQL连接池管理器"""
    
    def __init__(self, connection_config: Dict[str, Any], min_size=5, max_size=20):
        """
        初始化MySQL连接池
        
        :param connection_config: 连接配置
        :param min_size: 最小连接数
        :param max_size: 最大连接数
        """
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
        """创建新的数据库连接"""
        return mysql.connector.connect(
            host=self.config["host"],
            port=self.config.get("port", 3306),
            user=self.config["user"],
            password=self.config["password"],
            database=self.config["database"],
            use_pure=True,  # 使用纯Python实现，避免C扩展的问题
            autocommit=False,
            pool_reset_session=True,
            connection_timeout=10,
            get_warnings=True,
            raise_on_warnings=False
        )
    
    def _add_connection(self):
        """添加新连接到池中"""
        try:
            conn = self._create_connection()
            self.pool.append(conn)
            return True
        except Exception as e:
            logging.error(f"创建数据库连接失败: {e}")
            return False
    
    def get_connection(self, timeout=5):
        """
        获取连接
        
        :param timeout: 获取连接的超时时间（秒）
        :return: 数据库连接或None
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                # 检查是否有可用连接
                for i, conn in enumerate(self.pool):
                    if conn not in self.in_use:
                        try:
                            # 测试连接是否有效
                            cursor = conn.cursor()
                            cursor.execute("SELECT 1")
                            cursor.fetchone()
                            cursor.close()
                            
                            # 标记为使用中
                            self.in_use.add(conn)
                            return conn
                        except Exception:
                            # 连接已失效，关闭并创建新连接
                            try:
                                conn.close()
                            except:
                                pass
                            
                            # 从池中移除
                            self.pool.pop(i)
                            
                            # 创建新连接
                            new_conn = self._create_connection()
                            self.pool.append(new_conn)
                            self.in_use.add(new_conn)
                            return new_conn
                
                # 如果池中没有可用连接，但未达到最大连接数，创建新连接
                if len(self.pool) < self.max_size:
                    try:
                        new_conn = self._create_connection()
                        self.pool.append(new_conn)
                        self.in_use.add(new_conn)
                        return new_conn
                    except Exception as e:
                        logging.error(f"创建新连接失败: {e}")
                        # 继续等待
            
            # 短暂等待后重试
            time.sleep(0.1)
        
        # 超时
        logging.error(f"获取数据库连接超时（{timeout}秒）")
        return None
    
    def release_connection(self, conn):
        """
        释放连接回池中
        
        :param conn: 要释放的连接
        """
        if conn is None:
            return
            
        with self.lock:
            if conn in self.in_use:
                try:
                    # 重置连接状态
                    conn.rollback()
                    
                    # 标记为空闲
                    self.in_use.remove(conn)
                except Exception as e:
                    logging.warning(f"重置连接状态失败: {e}")
                    
                    # 连接可能已损坏，关闭并从池中移除
                    try:
                        conn.close()
                        self.pool.remove(conn)
                        self.in_use.remove(conn)
                    except:
                        pass
                    
                    # 确保连接池维持最小大小
                    if len(self.pool) < self.min_size:
                        self._add_connection()
    
    def close_all(self):
        """关闭所有连接"""
        with self.lock:
            for conn in self.pool:
                try:
                    conn.close()
                except:
                    pass
            
            self.pool.clear()
            self.in_use.clear()

###############################################################################
# 抽象任务池基类
###############################################################################
class BaseTaskPool(ABC):
    """任务池抽象基类：定义数据源的通用接口"""
    
    def __init__(self, columns_to_extract: List[str]):
        self.columns_to_extract = columns_to_extract
        self.tasks = []  # 任务列表，每项为 (id, record_dict)
        self.lock = threading.Lock()
    
    @abstractmethod
    def get_total_task_count(self) -> int:
        """获取未处理任务总数"""
        pass
    
    @abstractmethod
    def get_id_boundaries(self) -> Tuple[int, int]:
        """获取ID范围，返回(最小ID, 最大ID)"""
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
    
    @abstractmethod
    def reload_task_data(self, task_id: Any) -> Dict[str, Any]:
        """重新加载任务数据，用于系统错误重试"""
        pass

###############################################################################
# MySQL任务池实现
###############################################################################
class MySQLTaskPool(BaseTaskPool):
    """MySQL数据源任务池，支持分片加载"""
    
    def __init__(self, connection_config: Dict[str, Any], columns_to_extract: List[str], table_name: str):
        """
        初始化MySQL任务池
        
        :param connection_config: 数据库连接信息 (host, port, user, password, database)
        :param columns_to_extract: 需要提取的字段
        :param table_name: 需要操作的表名
        """
        super().__init__(columns_to_extract)
        
        if not MYSQL_AVAILABLE:
            raise ImportError("MySQL Connector库未安装，无法使用MySQL数据源")
        
        self.connection_config = connection_config
        self.table_name = table_name
        self.columns_to_write = {}  # 将在主处理类中设置
        
        # 创建连接池
        self.pool = MySQLConnectionPool(
            connection_config=connection_config,
            min_size=5,
            max_size=20
        )
        
        # 分片状态
        self.current_shard_id = -1
        self.current_min_id = 0
        self.current_max_id = 0
        
        # 进度跟踪
        self.progress_tracker = None
    
    def execute_with_connection(self, callback):
        """
        使用连接池执行操作
        
        :param callback: 回调函数，接收连接作为参数
        :return: 回调函数的返回值
        """
        conn = None
        try:
            conn = self.pool.get_connection()
            if conn is None:
                raise Exception("无法获取数据库连接")
            return callback(conn)
        finally:
            if conn:
                self.pool.release_connection(conn)
    
    def get_total_task_count(self) -> int:
        """获取未处理任务总数（使用EXPLAIN估计而非精确COUNT）"""
        def _get_count(conn):
            try:
                cursor = conn.cursor()
                
                # 使用估计查询
                cursor.execute(f"""
                    SELECT 
                        TABLE_ROWS
                    FROM 
                        information_schema.tables
                    WHERE 
                        table_schema = %s AND table_name = %s
                """, (self.connection_config["database"], self.table_name))
                
                result = cursor.fetchone()
                total_rows = result[0] if result else 0
                
                # 估计未处理比例
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as unprocessed,
                        (SELECT COUNT(*) FROM `{self.table_name}`) as total
                    FROM 
                        `{self.table_name}`
                    WHERE 
                        processed != 'yes' OR processed IS NULL
                    LIMIT 1000
                """)
                
                result = cursor.fetchone()
                if result and result[1] > 0:
                    sample_ratio = min(1.0, result[0] / result[1])
                    estimated_count = int(total_rows * sample_ratio)
                    return max(estimated_count, result[0])
                
                return total_rows
            except Exception as e:
                logging.error(f"估计任务总数失败: {e}")
                
                # 回退到简单COUNT
                try:
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM `{self.table_name}`
                        WHERE processed != 'yes' OR processed IS NULL
                    """)
                    result = cursor.fetchone()
                    return result[0] if result else 0
                except Exception as e2:
                    logging.error(f"COUNT查询失败: {e2}")
                    return 0
        
        return self.execute_with_connection(_get_count)
    
    def get_id_boundaries(self) -> Tuple[int, int]:
        """获取ID范围，返回(最小ID, 最大ID)"""
        def _get_boundaries(conn):
            try:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT 
                        MIN(id) as min_id,
                        MAX(id) as max_id
                    FROM 
                        `{self.table_name}`
                    WHERE 
                        processed != 'yes' OR processed IS NULL
                """)
                
                result = cursor.fetchone()
                if result and result[0] is not None and result[1] is not None:
                    return (int(result[0]), int(result[1]))
                
                # 如果没有未处理记录，则返回全表ID范围
                cursor.execute(f"""
                    SELECT 
                        MIN(id) as min_id,
                        MAX(id) as max_id
                    FROM 
                        `{self.table_name}`
                """)
                
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
        初始化指定分片
        
        :param shard_id: 分片ID
        :param min_id: 最小ID
        :param max_id: 最大ID
        :return: 加载的任务数
        """
        def _load_shard(conn):
            # 清空任务列表
            with self.lock:
                self.tasks = []
            
            # 记录分片信息
            self.current_shard_id = shard_id
            self.current_min_id = min_id
            self.current_max_id = max_id
            
            # 构建查询字段
            columns_str = ", ".join(f"`{col}`" for col in self.columns_to_extract)
            
            # 构建分片查询
            try:
                cursor = conn.cursor(dictionary=True)
                sql = f"""
                    SELECT 
                        id, {columns_str}, processed
                    FROM 
                        `{self.table_name}`
                    WHERE 
                        id BETWEEN %s AND %s
                        AND (processed != 'yes' OR processed IS NULL)
                    ORDER BY 
                        id
                """
                
                cursor.execute(sql, (min_id, max_id))
                rows = cursor.fetchall()
                
                # 加载到任务列表
                with self.lock:
                    for row in rows:
                        record_id = row["id"]
                        record_dict = {}
                        for col in self.columns_to_extract:
                            record_dict[col] = row.get(col, "")
                        
                        # 加入任务列表
                        self.tasks.append((record_id, record_dict))
                
                loaded_count = len(self.tasks)
                logging.info(f"加载分片 {shard_id} (ID范围: {min_id}-{max_id}), 共 {loaded_count} 个任务")
                
                return loaded_count
            except Exception as e:
                logging.error(f"加载分片 {shard_id} 失败: {e}")
                return 0
        
        return self.execute_with_connection(_load_shard)
    
    def get_task_batch(self, batch_size: int) -> List[Tuple[Any, Dict[str, Any]]]:
        """获取一批任务"""
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch
    
    def update_task_results(self, results: Dict[int, Dict[str, Any]]):
        """
        将一批结果写回数据库，使用优化的批量更新
        
        :param results: Dict[记录ID, 结果字典]
        """
        if not results:
            return
        
        def _update_results(conn):
            try:
                # 将结果分组 - 成功和失败
                success_ids = []
                success_values = {}
                
                for record_id, row_result in results.items():
                    if "_error" not in row_result:
                        success_ids.append(record_id)
                        
                        # 为每个字段收集值
                        for alias, col_name in self.columns_to_write.items():
                            if col_name not in success_values:
                                success_values[col_name] = {}
                            
                            success_values[col_name][record_id] = row_result.get(alias, "")
                
                if not success_ids:
                    return
                
                # 构建优化的批量更新SQL
                cursor = conn.cursor()
                
                # 使用CASE表达式构建批量更新
                set_clauses = []
                
                # 为每个输出字段构建CASE表达式
                for col_name, values in success_values.items():
                    clause = f"`{col_name}` = CASE id "
                    
                    for record_id, value in values.items():
                        # 转义值防止SQL注入
                        safe_value = value.replace("'", "''") if isinstance(value, str) else value
                        clause += f"WHEN {record_id} THEN '{safe_value}' "
                    
                    clause += "ELSE `{0}` END".format(col_name)
                    set_clauses.append(clause)
                
                # 添加processed标记
                processed_clause = "processed = CASE id "
                for record_id in success_ids:
                    processed_clause += f"WHEN {record_id} THEN 'yes' "
                processed_clause += "ELSE processed END"
                set_clauses.append(processed_clause)
                
                # 构建完整SQL
                sql = f"""
                    UPDATE `{self.table_name}`
                    SET {', '.join(set_clauses)}
                    WHERE id IN ({', '.join(map(str, success_ids))})
                """
                
                # 执行批量更新
                cursor.execute(sql)
                conn.commit()
                
                logging.info(f"成功批量更新 {len(success_ids)} 条记录")
                
            except Exception as e:
                conn.rollback()
                logging.error(f"批量更新失败: {e}")
                
                # 回退到单条更新
                try:
                    cursor = conn.cursor()
                    for record_id, row_result in results.items():
                        if "_error" not in row_result:
                            # 构建SET子句
                            set_parts = []
                            params = []
                            
                            for alias, col_name in self.columns_to_write.items():
                                set_parts.append(f"`{col_name}` = %s")
                                params.append(row_result.get(alias, ""))
                            
                            # 添加processed标记
                            set_parts.append("processed = %s")
                            params.append("yes")
                            
                            # 构建SQL
                            update_sql = f"""
                                UPDATE `{self.table_name}`
                                SET {', '.join(set_parts)}
                                WHERE id = %s
                            """
                            params.append(record_id)
                            
                            # 执行更新
                            cursor.execute(update_sql, params)
                    
                    conn.commit()
                    logging.info(f"使用单条更新模式完成 {len(success_ids)} 条记录更新")
                    
                except Exception as e2:
                    conn.rollback()
                    logging.error(f"单条更新也失败: {e2}")
        
        return self.execute_with_connection(_update_results)
    
    def reload_task_data(self, record_id: int) -> Dict[str, Any]:
        """重新加载特定记录数据"""
        def _reload(conn):
            record_dict = {}
            try:
                cursor = conn.cursor(dictionary=True)
                
                # 查询字段
                cols = ", ".join(f"`{col}`" for col in self.columns_to_extract)
                
                # 构建查询
                sql = f"SELECT {cols} FROM `{self.table_name}` WHERE id = %s"
                cursor.execute(sql, (record_id,))
                
                row = cursor.fetchone()
                if row:
                    for col in self.columns_to_extract:
                        record_dict[col] = row.get(col, "")
                else:
                    logging.warning(f"重新加载记录ID={record_id}时未找到对应记录")
            except Exception as e:
                logging.error(f"重新加载记录ID={record_id} 数据失败: {e}")
            
            return record_dict
        
        return self.execute_with_connection(_reload)
    
    def set_progress_tracker(self, tracker: ProgressTracker):
        """设置进度跟踪器"""
        self.progress_tracker = tracker
        
        # 初始化进度表
        def _init_table(conn):
            if self.progress_tracker:
                self.progress_tracker.init_progress_table(conn)
        
        self.execute_with_connection(_init_table)
    
    def close(self):
        """关闭资源"""
        if hasattr(self, 'pool') and self.pool:
            self.pool.close_all()

###############################################################################
# Excel任务池实现
###############################################################################
class ExcelTaskPool(BaseTaskPool):
    """Excel数据源任务池，支持分片加载"""
    
    def __init__(self, df: pd.DataFrame, columns_to_extract: List[str], output_excel: str):
        """
        初始化Excel任务池
        
        :param df: DataFrame对象，包含待处理的Excel数据
        :param columns_to_extract: 需要从DataFrame中提取的列
        :param output_excel: 输出Excel文件路径
        """
        super().__init__(columns_to_extract)
        self.df = df
        self.output_excel = output_excel
        self.columns_to_write = {}  # 将在主处理类中设置
        
        # 分片状态
        self.current_shard_id = -1
        self.current_min_idx = 0
        self.current_max_idx = 0
        
        # 进度跟踪
        self.progress_tracker = None
        
        # 定期保存防止数据丢失
        self.last_save_time = time.time()
        self.save_interval = 300  # 5分钟保存一次
    
    def get_total_task_count(self) -> int:
        """获取未处理任务总数"""
        try:
            return len(self.df[self.df.get("processed", "") != "yes"])
        except Exception as e:
            logging.error(f"计算任务总数失败: {e}")
            return 0
    
    def get_id_boundaries(self) -> Tuple[int, int]:
        """获取索引范围，返回(最小索引, 最大索引)"""
        try:
            # 获取未处理行的索引
            unprocessed = self.df[self.df.get("processed", "") != "yes"].index
            
            if len(unprocessed) > 0:
                return (int(unprocessed.min()), int(unprocessed.max()))
            
            # 如果没有未处理记录，返回全表范围
            return (0, len(self.df) - 1)
        except Exception as e:
            logging.error(f"获取索引范围失败: {e}")
            return (0, len(self.df) - 1)
    
    def initialize_shard(self, shard_id: int, min_idx: int, max_idx: int) -> int:
        """
        初始化指定分片
        
        :param shard_id: 分片ID
        :param min_idx: 最小索引
        :param max_idx: 最大索引
        :return: 加载的任务数
        """
        # 清空任务列表
        with self.lock:
            self.tasks = []
        
        # 记录分片信息
        self.current_shard_id = shard_id
        self.current_min_idx = min_idx
        self.current_max_idx = max_idx
        
        try:
            # 获取分片范围内的未处理行
            shard_df = self.df.loc[min_idx:max_idx]
            unprocessed = shard_df[shard_df.get("processed", "") != "yes"]
            
            # 加载到任务列表
            with self.lock:
                for idx, row in unprocessed.iterrows():
                    record_dict = {}
                    for col in self.columns_to_extract:
                        val = row.get(col, "")
                        record_dict[col] = str(val) if val is not None else ""
                    
                    # 加入任务列表
                    self.tasks.append((idx, record_dict))
            
            loaded_count = len(self.tasks)
            logging.info(f"加载分片 {shard_id} (索引范围: {min_idx}-{max_idx}), 共 {loaded_count} 个任务")
            
            return loaded_count
        except Exception as e:
            logging.error(f"加载分片 {shard_id} 失败: {e}")
            return 0
    
    def get_task_batch(self, batch_size: int) -> List[Tuple[Any, Dict[str, Any]]]:
        """获取一批任务"""
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch
    
    def update_task_results(self, results: Dict[int, Dict[str, Any]]):
        """
        将结果更新到DataFrame中，并写入Excel
        :param results: Dict[行索引, 结果字典]
        """
        if not results:
            return
            
        try:
            # 批量更新DataFrame
            with self.lock:
                for idx, result in results.items():
                    # 更新结果字段
                    for alias, col_name in self.columns_to_write.items():
                        self.df.at[idx, col_name] = result.get(alias, "")
                    
                    # 仅对成功处理的记录标记为已处理
                    if "_error" not in result:
                        self.df.at[idx, "processed"] = "yes"
                
                # 定期保存，防止数据丢失
                current_time = time.time()
                if current_time - self.last_save_time > self.save_interval:
                    self._save_excel()
                    self.last_save_time = current_time
            
            logging.info(f"成功将 {len(results)} 条记录更新到DataFrame")
        except Exception as e:
            logging.error(f"更新Excel记录失败: {e}")
    
    def _save_excel(self):
        """保存数据到Excel文件"""
        try:
            self.df.to_excel(self.output_excel, index=False)
            logging.info(f"已保存数据到Excel文件: {self.output_excel}")
        except Exception as e:
            logging.error(f"保存Excel文件失败: {e}")
    
    def reload_task_data(self, idx: int) -> Dict[str, Any]:
        """
        重新加载行数据，用于系统错误重试
        """
        record_dict = {}
        try:
            if idx in self.df.index:
                for col in self.columns_to_extract:
                    val = self.df.at[idx, col]
                    record_dict[col] = str(val) if val is not None else ""
            else:
                logging.warning(f"重新加载行索引={idx}时未找到对应记录")
        except Exception as e:
            logging.error(f"重新加载行索引={idx} 数据失败: {e}")
        return record_dict
    
    def set_progress_tracker(self, tracker: ProgressTracker):
        """设置进度跟踪器（Excel版本不支持数据库级进度跟踪）"""
        self.progress_tracker = tracker
    
    def close(self):
        """关闭资源，保存最终结果"""
        self._save_excel()

###############################################################################
# 分片任务管理器：负责分片加载与处理
###############################################################################
class ShardedTaskManager:
    """分片任务管理器，负责大规模数据的分片加载与处理"""
    
    def __init__(self, task_pool, optimal_shard_size=10000, min_shard_size=1000, max_shard_size=50000):
        """
        初始化分片任务管理器
        
        :param task_pool: 任务池对象
        :param optimal_shard_size: 最佳分片大小
        :param min_shard_size: 最小分片大小
        :param max_shard_size: 最大分片大小
        """
        self.task_pool = task_pool
        self.optimal_shard_size = optimal_shard_size
        self.min_shard_size = min_shard_size
        self.max_shard_size = max_shard_size
        
        # 分片状态
        self.current_shard = 0
        self.total_shards = 0
        self.shard_boundaries = []  # [(min_id, max_id), ...]
        
        # 处理统计
        self.total_estimated = 0
        self.total_processed = 0
        self.processing_metrics = {
            'avg_time_per_record': 0,
            'records_per_second': 0,
            'last_batch_size': 0,
            'last_batch_time': 0
        }
        
        # 进度跟踪
        self.progress_tracker = ProgressTracker()
        self.task_pool.set_progress_tracker(self.progress_tracker)
        
        # 记录内存使用
        self.memory_tracker = {
            'last_check_time': time.time(),
            'check_interval': 60,  # 60秒检查一次
            'peak_memory': 0,
            'current_memory': 0
        }
    
    def calculate_optimal_shard_size(self):
        """
        计算最佳分片大小，考虑系统资源和处理效率
        
        :return: 计算得到的分片大小
        """
        # 获取可用内存
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        
        # 估算每条记录内存占用（保守估计）
        record_size_mb = 0.01  # 假设每条记录平均占用10KB
        
        # 基于内存限制的分片大小
        memory_based_size = int(available_mb * 0.3 / record_size_mb)  # 只使用30%可用内存
        
        # 如果有处理速率数据，考虑处理效率
        if self.processing_metrics['records_per_second'] > 0:
            # 理想情况下一个分片处理时间为10分钟
            ideal_batch_duration = 600  # 秒
            time_based_size = int(self.processing_metrics['records_per_second'] * ideal_batch_duration)
            
            # 取两者的较小值
            calculated_size = min(memory_based_size, time_based_size)
        else:
            calculated_size = memory_based_size
        
        # 确保在合理范围内
        shard_size = max(self.min_shard_size, min(calculated_size, self.max_shard_size))
        
        logging.info(f"计算得到的分片大小: {shard_size}, "
                   f"内存可用: {available_mb:.1f}MB, "
                   f"处理速率: {self.processing_metrics['records_per_second']:.1f}条/秒")
        
        return shard_size
    
    def update_processing_metrics(self, batch_size, processing_time):
        """
        更新处理性能指标
        
        :param batch_size: 处理的批次大小
        :param processing_time: 处理时间(秒)
        """
        if processing_time > 0 and batch_size > 0:
            # 计算每条记录平均处理时间
            time_per_record = processing_time / batch_size
            
            # 更新移动平均
            if self.processing_metrics['avg_time_per_record'] == 0:
                self.processing_metrics['avg_time_per_record'] = time_per_record
            else:
                # 加权平均，新数据权重0.3
                self.processing_metrics['avg_time_per_record'] = (
                    0.7 * self.processing_metrics['avg_time_per_record'] + 
                    0.3 * time_per_record
                )
            
            # 更新处理速率
            self.processing_metrics['records_per_second'] = 1.0 / self.processing_metrics['avg_time_per_record']
            
            # 记录最近一批的信息
            self.processing_metrics['last_batch_size'] = batch_size
            self.processing_metrics['last_batch_time'] = processing_time
    
    def monitor_memory_usage(self):
        """监控内存使用情况"""
        current_time = time.time()
        
        # 达到检查间隔才更新
        if current_time - self.memory_tracker['last_check_time'] < self.memory_tracker['check_interval']:
            return
        
        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            self.memory_tracker['current_memory'] = current_memory
            self.memory_tracker['peak_memory'] = max(self.memory_tracker['peak_memory'], current_memory)
            self.memory_tracker['last_check_time'] = current_time
            
            # 内存压力过大时主动触发GC
            mem = psutil.virtual_memory()
            if mem.percent > 85 or current_memory > 1024:  # 系统内存使用率>85%或进程>1GB
                gc.collect()
                logging.info(f"内存使用较高，已触发GC: 当前={current_memory:.1f}MB, 峰值={self.memory_tracker['peak_memory']:.1f}MB")
            
        except Exception as e:
            logging.warning(f"监控内存使用失败: {e}")
    
    def initialize(self, resume_job_id=None):
        """
        初始化任务管理器
        
        :param resume_job_id: 要恢复的任务ID，如果为None则创建新任务
        :return: 是否成功初始化
        """
        try:
            # 检查是否需要恢复
            if resume_job_id and self.task_pool.__class__.__name__ == 'MySQLTaskPool':
                # 尝试从数据库加载进度
                def _load_progress(conn):
                    return self.progress_tracker.load_progress(conn, resume_job_id)
                
                resumed = self.task_pool.execute_with_connection(_load_progress)
                
                if resumed:
                    logging.info(f"已恢复任务 {resume_job_id}, "
                               f"当前分片={self.progress_tracker.current_shard}, "
                               f"已处理={self.progress_tracker.total_processed}")
                    
                    # 恢复状态
                    self.current_shard = self.progress_tracker.current_shard
                    self.total_processed = self.progress_tracker.total_processed
                    
                    # 更新进度状态
                    def _update_status(conn):
                        self.progress_tracker.update_status(conn, "running")
                    
                    self.task_pool.execute_with_connection(_update_status)
            
            # 获取任务总数估计
            self.total_estimated = self.task_pool.get_total_task_count()
            if self.total_estimated == 0:
                logging.warning("未发现需要处理的任务")
                return False
            
            # 更新进度跟踪器
            self.progress_tracker.total_estimated = self.total_estimated
            
            # 获取ID范围
            min_id, max_id = self.task_pool.get_id_boundaries()
            logging.info(f"任务ID范围: {min_id} - {max_id}, 估计任务总数: {self.total_estimated}")
            
            # 计算分片大小和数量
            shard_size = self.calculate_optimal_shard_size()
            
            # 防止分片过多
            if max_id - min_id + 1 > 1000000:
                # 对于超大范围，增加分片大小
                shard_size = max(shard_size, int((max_id - min_id + 1) / 100))
            
            # 计算分片数量
            self.total_shards = max(1, (max_id - min_id + shard_size - 1) // shard_size)
            
            # 生成分片边界
            self.shard_boundaries = []
            for i in range(self.total_shards):
                shard_min = min_id + i * shard_size
                shard_max = min(min_id + (i + 1) * shard_size - 1, max_id)
                self.shard_boundaries.append((shard_min, shard_max))
            
            logging.info(f"任务已分为 {self.total_shards} 个分片, 每片约 {shard_size} 条记录")
            
            # 保存初始进度
            if self.task_pool.__class__.__name__ == 'MySQLTaskPool':
                def _save_initial(conn):
                    self.progress_tracker.save_progress(conn)
                
                self.task_pool.execute_with_connection(_save_initial)
            
            return True
            
        except Exception as e:
            logging.error(f"初始化任务管理器失败: {e}")
            return False
    
    def load_next_shard(self):
        """
        加载下一个分片
        
        :return: 是否成功加载
        """
        # 检查是否还有分片
        if self.current_shard >= self.total_shards:
            logging.info("所有分片已处理完毕")
            return False
        
        # 获取当前分片边界
        min_id, max_id = self.shard_boundaries[self.current_shard]
        
        # 加载分片
        loaded_tasks = self.task_pool.initialize_shard(self.current_shard, min_id, max_id)
        
        if loaded_tasks == 0:
            logging.info(f"分片 {self.current_shard} 无需处理的任务，跳到下一个")
            self.current_shard += 1
            return self.load_next_shard()
        
        # 更新进度跟踪器的分片信息
        self.progress_tracker.current_shard = self.current_shard
        self.progress_tracker.last_processed_id = min_id
        
        # 保存进度
        if self.task_pool.__class__.__name__ == 'MySQLTaskPool':
            def _save_progress(conn):
                self.progress_tracker.save_progress(conn)
            
            self.task_pool.execute_with_connection(_save_progress)
        
        return True
    
    def update_progress(self, processed_count, last_id=None):
        """
        更新处理进度
        
        :param processed_count: 本次更新处理的记录数
        :param last_id: 最后处理的记录ID
        """
        # 更新总计数
        self.total_processed += processed_count
        
        # 更新进度跟踪器
        self.progress_tracker.total_processed = self.total_processed
        if last_id is not None:
            self.progress_tracker.last_processed_id = last_id
        
        # 记录进度
        self.progress_tracker.log_progress()
        
        # 保存进度
        if self.task_pool.__class__.__name__ == 'MySQLTaskPool':
            def _save_progress(conn):
                self.progress_tracker.save_progress(conn)
            
            self.task_pool.execute_with_connection(_save_progress)
    
    def finalize(self, status="completed", error=None):
        """
        完成任务处理
        
        :param status: 完成状态
        :param error: 错误信息
        """
        # 更新最终状态
        if self.task_pool.__class__.__name__ == 'MySQLTaskPool':
            def _update_final(conn):
                self.progress_tracker.update_status(conn, status, error)
            
            self.task_pool.execute_with_connection(_update_final)
        
        # 记录最终统计
        elapsed = time.time() - self.progress_tracker.start_time
        
        logging.info(
            f"任务处理完成，状态: {status}\n"
            f"总处理记录: {self.total_processed}\n"
            f"总用时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)\n"
            f"平均速率: {self.total_processed/elapsed:.1f}条/秒\n"
            f"峰值内存: {self.memory_tracker['peak_memory']:.1f}MB"
        )
        
        # 关闭资源
        self.task_pool.close()

###############################################################################
# 模型配置类：封装模型+通道信息
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
        self.base_weight = model_dict.get("weight", 1)  # 保存原始权重用于动态调整
        self.max_weight = model_dict.get("weight", 1) * 2  # 最大权重上限
        self.temperature = model_dict.get("temperature", 0.7)
        
        # 限流参数
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

        # 取模型自身和通道超时的最小值
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
# 初始化日志系统，允许JSON或文本格式，写文件或控制台
###############################################################################
def init_logging(log_config: Dict[str, Any]):
    level_str = log_config.get("level", "info").upper()
    level = getattr(logging, level_str, logging.INFO)

    # 默认文本格式
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    if log_config.get("format") == "json":
        # 简易 JSON 输出，可自行改造使用 python-json-logger 等库
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
        # 加载配置
        self.config = load_config(config_path)

        # 初始化日志
        global_cfg = self.config.get("global", {})
        init_logging(global_cfg.get("log", {}))

        # 解析模型&通道
        models_cfg = self.config.get("models", [])
        channels_cfg = self.config.get("channels", {})
        if not models_cfg:
            logging.error("配置文件中未找到 models 配置！")
            raise ValueError("缺少 models 配置")

        # 检查模型ID唯一性
        model_ids = set()
        for m in models_cfg:
            model_id = m.get("id")
            if model_id in model_ids:
                logging.warning(f"发现重复的模型ID: {model_id}, 将使用后面的配置")
            model_ids.add(model_id)

        self.models = []
        for m in models_cfg:
            self.models.append(ModelConfig(m, channels_cfg))

        # 确定数据源类型
        self.datasource_type = self.config.get("datasource", {}).get("type", "excel").lower()
        logging.info(f"使用数据源类型: {self.datasource_type}")

        # 从两个配置部分尝试获取并发配置
        concurrency_cfg = {}
        if "datasource" in self.config and "concurrency" in self.config["datasource"]:
            concurrency_cfg = self.config["datasource"]["concurrency"]
        elif "excel" in self.config and "concurrency" in self.config["excel"]:
            concurrency_cfg = self.config["excel"]["concurrency"]
        elif "mysql" in self.config and "concurrency" in self.config["mysql"]:
            concurrency_cfg = self.config["mysql"]["concurrency"]

        # 初始化调度器
        backoff_factor = concurrency_cfg.get("backoff_factor", 2)
        self.dispatcher = ModelDispatcher(self.models, backoff_factor=backoff_factor)
        
        # 初始化模型限流器
        self.rate_limiter = ModelRateLimiter()
        self.rate_limiter.configure(models_cfg)
        
        # 创建模型ID->对象映射
        self.model_map = {m.id: m for m in self.models}

        # 并发配置
        self.max_workers = concurrency_cfg.get("max_workers", 5)
        self.batch_size = concurrency_cfg.get("batch_size", 300)
        self.save_interval = concurrency_cfg.get("save_interval", 300)
        self.global_retry_times = concurrency_cfg.get("retry_times", 3)

        # 分片配置
        self.shard_size = concurrency_cfg.get("shard_size", 10000)
        self.min_shard_size = concurrency_cfg.get("min_shard_size", 1000)
        self.max_shard_size = concurrency_cfg.get("max_shard_size", 50000)
        
        # 提示词配置
        prompt_cfg = self.config.get("prompt", {})
        self.prompt_template = prompt_cfg.get("template", "")
        self.required_fields = prompt_cfg.get("required_fields", [])

        # 初始化JSON字段验证器
        self.validator = JsonValidator()
        validation_cfg = self.config.get("validation", {})
        self.validator.configure(validation_cfg)

        # 需要提取和写回的字段
        self.columns_to_extract = self.config.get("columns_to_extract", [])
        self.columns_to_write = self.config.get("columns_to_write", {})

        # 初始化任务池
        self.task_pool = self._create_task_pool()
        
        # 如果是MySQL任务池，设置写回字段
        if isinstance(self.task_pool, MySQLTaskPool):
            self.task_pool.columns_to_write = self.columns_to_write
        elif isinstance(self.task_pool, ExcelTaskPool):
            self.task_pool.columns_to_write = self.columns_to_write

        # 初始化锁
        self.lock = threading.Lock()
        
        # 构建加权随机池
        self.models_pool = []
        for model_config in self.models:
            self.models_pool.extend([model_config] * model_config.weight)

        logging.info(f"共加载 {len(self.models)} 个模型，加权池大小: {len(self.models_pool)}")
        logging.info(f"批处理大小: {self.batch_size}, 保存间隔: {self.save_interval}")
        
        # 初始化分片任务管理器
        self.task_manager = ShardedTaskManager(
            task_pool=self.task_pool,
            optimal_shard_size=self.shard_size,
            min_shard_size=self.min_shard_size,
            max_shard_size=self.max_shard_size
        )
    
    def _create_task_pool(self) -> BaseTaskPool:
        """根据配置创建适当的任务池"""
        if self.datasource_type == "mysql":
            if not MYSQL_AVAILABLE:
                logging.error("MySQL Connector库未安装，无法使用MySQL数据源")
                raise ImportError("MySQL Connector库未安装，无法使用MySQL数据源")
                
            mysql_config = self.config.get("mysql", {})
            if not mysql_config:
                logging.error("配置中未找到mysql部分")
                raise ValueError("配置中未找到mysql部分")
                
            table_name = mysql_config.get("table_name", "tasks")
            
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
                table_name=table_name
            )
        elif self.datasource_type == "excel":
            excel_config = self.config.get("excel", {})
            if not excel_config:
                logging.error("配置中未找excel部分")
                raise ValueError("配置中未找到excel部分")
                
            input_excel = excel_config.get("input_path")
            output_excel = excel_config.get("output_path")
            
            if not input_excel:
                logging.error("Excel输入路径未配置")
                raise ValueError("Excel输入路径未配置")
                
            if not output_excel:
                output_excel = input_excel.replace(".xlsx", "_output.xlsx").replace(".xls", "_output.xls")
                logging.info(f"未配置输出Excel，使用默认路径: {output_excel}")
            
            # 检查输入Excel是否存在
            if not os.path.exists(input_excel):
                logging.error(f"输入Excel文件不存在: {input_excel}")
                raise FileNotFoundError(f"Excel文件不存在: {input_excel}")
                
            # 读取Excel
            logging.info(f"开始读取Excel: {input_excel}")
            df = pd.read_excel(input_excel)
            
            # 如果没有 processed 列，则补一个
            if "processed" not in df.columns:
                df["processed"] = ""
                
            # 确保要写回的列都存在
            for alias, col_name in self.columns_to_write.items():
                if col_name not in df.columns:
                    df[col_name] = ""
            
            return ExcelTaskPool(
                df=df,
                columns_to_extract=self.columns_to_extract,
                output_excel=output_excel
            )
        else:
            raise ValueError(f"不支持的数据源类型: {self.datasource_type}")

    ###########################################################################
    # 调整模型权重
    ###########################################################################
    def adjust_model_weights(self):
        """动态调整模型权重，基于性能指标、成功率等因素"""
        adjusted_models = []
        
        for model_id, model in self.model_map.items():
            # 获取模型性能指标
            success_rate = self.dispatcher.get_model_success_rate(model_id)
            avg_response_time = self.dispatcher.get_model_avg_response_time(model_id)
            
            # 是否当前可用
            is_available = self.dispatcher.is_model_available(model_id)
            
            # 权重计算因子
            success_factor = success_rate ** 2  # 成功率的平方，放大差异
            speed_factor = 1.0 / max(0.1, avg_response_time)  # 响应时间越短，权重越高
            availability_factor = 1.0 if is_available else 0.1  # 不可用大幅降低权重
            
            # 计算新权重（保持原始权重作为基准）
            new_weight = int(model.base_weight * success_factor * speed_factor * availability_factor)
            
            # 限制权重范围
            new_weight = max(1, min(new_weight, model.max_weight))
            
            # 如果权重有变化，更新模型
            if new_weight != model.weight:
                logging.info(f"调整模型[{model.name}]权重: {model.weight} -> {new_weight} "
                           f"(成功率={success_rate:.2f}, 响应时间={avg_response_time:.2f}s, "
                           f"可用={is_available})")
                model.weight = new_weight
            
            # 记录已调整的模型
            adjusted_models.append(model)
        
        # 重建模型池
        self.models_pool = []
        for model in adjusted_models:
            self.models_pool.extend([model] * model.weight)

    ###########################################################################
    # 根据当前可用的"模型"信息，从加权池中随机挑选可用模型
    ###########################################################################
    def get_available_model_randomly(self, exclude_model_ids=None) -> Optional[ModelConfig]:
        if exclude_model_ids is None:
            exclude_model_ids = set()
        else:
            exclude_model_ids = set(exclude_model_ids)

        # 获取可用模型
        available_model_ids = self.dispatcher.get_available_models(exclude_model_ids)
        
        # 检查模型限流
        available_model_ids = [mid for mid in available_model_ids 
                              if self.rate_limiter.can_process(mid)]
        
        if not available_model_ids:
            return None

        # 构建可用模型的加权池
        available_pool = []
        for model_id in available_model_ids:
            model = self.model_map[model_id]
            available_pool.extend([model] * model.weight)

        if not available_pool:
            return None
        return random.choice(available_pool)

    ###########################################################################
    # 构造 prompt
    ###########################################################################
    def create_prompt(self, record_data: Dict[str, Any]) -> str:
        try:
            record_json_str = json.dumps(record_data, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logging.error(f"行数据无法序列化为JSON: {e}")
            return self.prompt_template
        if "{record_json}" not in self.prompt_template:
            logging.warning("提示词模板中不包含 {record_json} 占位符, 将无法注入JSON数据")
        return self.prompt_template.replace("{record_json}", record_json_str)

    ###########################################################################
    # 提取AI返回JSON，并检查必需字段
    ###########################################################################
    def extract_json_from_response(self, content: str) -> Dict[str, Any]:
        def contains_required_fields(data_dict: Dict[str, Any], required: List[str]) -> bool:
            return all(field in data_dict for field in required)

        # 1) 尝试整体解析
        try:
            data = json.loads(content)
            if isinstance(data, dict) and contains_required_fields(data, self.required_fields):
                if self.validator.enabled:
                    is_valid, errors = self.validator.validate(data)
                    if not is_valid:
                        logging.warning(f"字段值验证失败: {errors}")
                        return {
                            "_error": "invalid_field_values", 
                            "_error_type": ErrorType.CONTENT_ERROR, 
                            "_validation_errors": errors
                        }
                return data
        except json.JSONDecodeError:
            pass

        # 2) 正则提取所有形似JSON的子串
        pattern = r'(\{.*?\})'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                candidate = json.loads(match)
                if isinstance(candidate, dict) and contains_required_fields(candidate, self.required_fields):
                    if self.validator.enabled:
                        is_valid, errors = self.validator.validate(candidate)
                        if not is_valid:
                            logging.warning(f"字段值验证失败: {errors}")
                            continue
                    return candidate
            except Exception:
                continue

        logging.warning(f"无法解析出包含必需字段{self.required_fields}的有效JSON, content={content}")
        return {"_error": "invalid_json", "_error_type": ErrorType.CONTENT_ERROR}

    ###########################################################################
    # 异步调用AI接口
    ###########################################################################
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
        logging.info(
            f"调用AI接口: model_name={model_cfg.name}, channel_name={model_cfg.channel_name}, url={url}"
        )
        logging.debug(f"payload={json.dumps(payload, ensure_ascii=False)[:300]}...")
        
        proxy = model_cfg.channel_proxy if model_cfg.channel_proxy else None
        timeout = aiohttp.ClientTimeout(connect=model_cfg.connect_timeout, total=model_cfg.read_timeout)
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload, proxy=proxy) as resp:
                    if resp.status >= 400:
                        error_text = await resp.text()
                        logging.error(f"API错误: HTTP {resp.status}, {error_text}")
                        raise aiohttp.ClientResponseError(
                            request_info=resp.request_info,
                            history=resp.history,
                            status=resp.status,
                            message=f"HTTP {resp.status}: {error_text}",
                            headers=resp.headers
                        )
                    data = await resp.json()
                    if "choices" not in data or not data["choices"]:
                        logging.error(f"AI返回格式异常: {data}")
                        raise ValueError("AI返回不含choices字段")
                    
                    content = data["choices"][0]["message"]["content"]
                    logging.debug(f"AI返回 content={content[:300]}...")
                    
                    # 计算响应时间
                    response_time = time.time() - start_time
                    
                    # 更新模型指标
                    self.dispatcher.update_model_metrics(model_cfg.id, response_time, True)
                    
                    return content
        except aiohttp.ClientError as e:
            logging.error(f"AI请求异常 (API_ERROR): {str(e)}")
            
            # 计算响应时间并更新失败指标
            response_time = time.time() - start_time
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            
            raise
        except Exception as e:
            logging.error(f"处理AI响应异常 (API_ERROR): {str(e)}")
            
            # 计算响应时间并更新失败指标
            response_time = time.time() - start_time
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            
            raise

    ###########################################################################
    # 异步处理一条记录
    ###########################################################################
    async def process_one_record_async(self, session, record_id: Any, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        异步处理一条记录
        :param session: 共享的aiohttp会话
        :param record_id: 记录ID（可能是数据库ID或Excel行索引）
        :param row_data: 记录数据字典
        :return: 处理结果字典
        """
        task_id_str = str(record_id)
        if self.datasource_type == "excel":
            task_id_str = f"第 {record_id+2} 行"  # Excel行号从2开始(含标题行)
        
        logging.info(f"开始处理{task_id_str}的数据: {row_data}")
        prompt = self.create_prompt(row_data)

        used_model_ids = set()
        for attempt in range(self.global_retry_times):
            model_cfg = self.get_available_model_randomly(exclude_model_ids=used_model_ids)
            if not model_cfg:
                logging.error("当前无可用模型（全部退避或失败）")
                return {"_error": "no_available_model", "_error_type": ErrorType.SYSTEM_ERROR}

            used_model_ids.add(model_cfg.id)
            logging.info(
                f"{task_id_str}: 尝试使用模型[{model_cfg.name}] (第 {attempt+1} 次尝试)"
            )

            try:
                # 使用共享session调用AI API
                content = await self.call_ai_api_async(model_cfg, prompt)
                parsed_json = self.extract_json_from_response(content)
                
                if "_error" in parsed_json and parsed_json.get("_error_type") == ErrorType.CONTENT_ERROR:
                    # 内容问题，不退避，继续尝试下一个模型
                    if parsed_json.get("_error") == "invalid_field_values":
                        err_detail = parsed_json.get("_validation_errors", [])
                        logging.warning(f"模型[{model_cfg.name}]返回内容字段值验证失败: {err_detail}")
                    else:
                        logging.warning(f"模型[{model_cfg.name}]返回内容JSON无效或缺少必需字段")
                    continue

                self.dispatcher.mark_model_success(model_cfg.id)
                
                # 仅存储摘要或必要字段，减少内存占用
                parsed_json["response_excerpt"] = content[:100] + "..." if len(content) > 100 else content
                parsed_json["used_model_id"] = model_cfg.id
                parsed_json["used_model_name"] = model_cfg.name
                
                logging.info(f"{task_id_str}, 模型[{model_cfg.name}] 调用成功!")
                return parsed_json

            except aiohttp.ClientError as e:
                logging.warning(
                    f"{task_id_str}, 模型[{model_cfg.name}] 网络错误 (API_ERROR): {e}"
                )
                self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)

            except Exception as e:
                logging.warning(
                    f"{task_id_str}, 模型[{model_cfg.name}] 调用异常 (API_ERROR): {e}"
                )
                self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)

        logging.error(f"{task_id_str} 在不同模型中尝试 {self.global_retry_times} 次均失败!")
        return {"_error": "all_models_failed", "_error_type": ErrorType.CONTENT_ERROR}

    ###########################################################################
    # 核心流程：从数据源读取 => 并发处理 => 写回数据源
    ###########################################################################
    async def process_shard_async(self, resume_job_id=None):
        """
        异步处理整个分片
        
        :param resume_job_id: 要恢复的任务ID
        """
        # 初始化分片管理器
        if not self.task_manager.initialize(resume_job_id):
            logging.warning("初始化分片管理器失败或没有需要处理的任务")
            return
        
        # 创建aiohttp会话
        async with aiohttp.ClientSession() as session:
            # 加载第一个分片
            if not self.task_manager.load_next_shard():
                logging.warning("没有可加载的分片")
                return
            
            # 主处理循环
            while self.task_pool.has_tasks():
                # 动态计算批处理大小
                current_batch_size = min(self.batch_size, self.task_pool.get_remaining_count())
                
                # 获取一批任务
                task_batch = self.task_pool.get_task_batch(current_batch_size)
                if not task_batch:
                    # 当前分片任务已处理完，尝试加载下一个分片
                    if not self.task_manager.load_next_shard():
                        # 所有分片已处理完
                        break
                    continue
                
                # 批处理开始时间
                batch_start_time = time.time()
                
                # 创建并发任务
                tasks = []
                for record_id, record_dict in task_batch:
                    task = self.process_one_record_async(session, record_id, record_dict)
                    tasks.append(task)
                
                # 并发执行所有任务
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果
                batch_results = {}
                last_id = None
                system_retry_count = 0
                
                for (record_id, _), result in zip(task_batch, results):
                    last_id = record_id  # 记录处理的最后一个ID
                    
                    # 检查异常
                    if isinstance(result, Exception):
                        logging.error(f"处理记录 {record_id} 时发生异常: {result}")
                        # 系统错误，放回队列头部
                        self.task_pool.add_task_to_front(record_id, self.task_pool.reload_task_data(record_id))
                        system_retry_count += 1
                        continue
                    
                    # 检查系统错误（如无可用模型）
                    if "_error" in result and result.get("_error_type") == ErrorType.SYSTEM_ERROR:
                        logging.warning(f"记录 {record_id} 处理遇到系统错误: {result['_error']}")
                        # 放回队列头部
                        self.task_pool.add_task_to_front(record_id, self.task_pool.reload_task_data(record_id))
                        system_retry_count += 1
                        continue
                    
                    # 记录结果（包括内容错误）
                    batch_results[record_id] = result
                
                # 批量更新结果
                if batch_results:
                    self.task_pool.update_task_results(batch_results)
                
                # 计算批处理耗时
                batch_time = time.time() - batch_start_time
                
                # 更新处理指标
                self.task_manager.update_processing_metrics(len(batch_results), batch_time)
                
                # 更新进度
                self.task_manager.update_progress(len(batch_results), last_id)
                
                # 定期动态调整模型权重
                if random.random() < 0.1:  # 约10%的批次执行权重调整
                    self.adjust_model_weights()
                
                # 监控内存使用
                self.task_manager.monitor_memory_usage()
                
                # 如果系统错误较多，短暂等待
                if system_retry_count > 0:
                    await asyncio.sleep(min(1, system_retry_count * 0.2))
        
        # 处理完成
        self.task_manager.finalize("completed")
    
    def process_tasks(self, resume_job_id=None):
        """
        主处理函数：启动异步处理循环
        
        :param resume_job_id: 要恢复的任务ID
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.process_shard_async(resume_job_id))
        except KeyboardInterrupt:
            logging.info("处理被用户中断")
            self.task_manager.finalize("interrupted", "用户中断")
        except Exception as e:
            logging.error(f"处理任务时发生异常: {e}")
            self.task_manager.finalize("failed", str(e))
        finally:
            loop.close()

###############################################################################
# 入口
###############################################################################
def validate_config_file(config_path: str) -> bool:
    """
    验证配置文件是否存在并符合基本要求
    :param config_path: 配置文件路径
    :return: 配置是否有效
    """
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 检查必要部分是否存在
        required_sections = ["models", "channels", "prompt"]
        for section in required_sections:
            if section not in config:
                print(f"配置文件缺少必要部分: {section}")
                return False
            
        # 检查是否有至少一个模型
        if not config.get("models") or not isinstance(config.get("models"), list) or len(config.get("models")) == 0:
            print("配置文件未定义任何模型")
            return False
            
        # 检查数据源类型
        datasource_type = config.get("datasource", {}).get("type", "excel").lower()
        if datasource_type not in ["mysql", "excel"]:
            print(f"不支持的数据源类型: {datasource_type}")
            return False
            
        # 如果是Excel类型，检查Excel配置
        if datasource_type == "excel" and ("excel" not in config or "input_path" not in config.get("excel", {})):
            print("未找到Excel输入路径配置")
            return False
            
        # 如果是MySQL类型，检查MySQL配置
        if datasource_type == "mysql":
            if not MYSQL_AVAILABLE:
                print("MySQL Connector库未安装，无法使用MySQL数据源")
                return False
                
            if "mysql" not in config:
                print("未找到MySQL配置")
                return False
                
            mysql_cfg = config.get("mysql", {})
            required_mysql_fields = ["host", "user", "password", "database", "table_name"]
            for field in required_mysql_fields:
                if field not in mysql_cfg:
                    print(f"MySQL配置缺少必要字段: {field}")
                    return False
            
        return True
    except Exception as e:
        print(f"配置文件验证失败: {e}")
        return False

def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='AI-DataFlux：通用AI批处理引擎')
    parser.add_argument('--config', '-c', default='./config.yaml', help='配置文件路径')
    parser.add_argument('--resume', '-r', help='恢复指定的任务ID')
    
    args = parser.parse_args()
    
    # 验证配置文件
    if not validate_config_file(args.config):
        sys.exit(1)
    
    try:
        # 创建处理器并运行
        processor = UniversalAIProcessor(args.config)
        processor.process_tasks(args.resume)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()