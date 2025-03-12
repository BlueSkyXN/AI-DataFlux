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
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

# 尝试导入MySQL相关库，如果不可用则标记为None
try:
    import mysql.connector
    from mysql.connector import Error
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
                "next_available_ts": 0  # 0 表示随时可用
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
            backoff_seconds = self.backoff_factor ** (fail_count - 1)
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
# 抽象任务池基类
###############################################################################
class BaseTaskPool(ABC):
    """任务池抽象基类：定义数据源的通用接口"""
    
    def __init__(self, columns_to_extract: List[str]):
        self.columns_to_extract = columns_to_extract
        self.tasks = []  # 任务列表，每项为 (id, record_dict)
        self.lock = threading.Lock()
    
    @abstractmethod
    def initialize_tasks(self) -> int:
        """初始化任务列表，返回任务总数"""
        pass
    
    @abstractmethod
    def update_task_results(self, results: Dict[Any, Dict[str, Any]]):
        """批量更新任务处理结果"""
        pass
    
    def get_batch(self, batch_size: int) -> List[Tuple[Any, Dict[str, Any]]]:
        """获取一批任务"""
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch
    
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
    """MySQL数据源任务池"""
    
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
    
    def _connect_db(self):
        """
        建立并返回一个新的 MySQL 连接。
        注意在多线程环境下，每个线程都应该有自己独立的连接来避免冲突。
        """
        try:
            conn = mysql.connector.connect(
                host=self.connection_config["host"],
                port=self.connection_config.get("port", 3306),
                user=self.connection_config["user"],
                password=self.connection_config["password"],
                database=self.connection_config["database"]
            )
            return conn
        except Exception as e:
            logging.error(f"MySQL 连接失败: {e}")
            raise
    
    def initialize_tasks(self) -> int:
        """
        从数据库中加载尚未 processed='yes' 的记录，
        并存入 self.tasks 列表 (每条是 (id, {col1: val1, col2: val2, ...}) )
        返回待处理的任务总数。
        """
        self.tasks = []
        # 使用参数化查询减少SQL注入风险
        columns_str = ", ".join(f"`{col}`" for col in self.columns_to_extract)
        sql = f"SELECT id, {columns_str}, processed FROM `{self.table_name}` WHERE processed != 'yes' OR processed IS NULL"
        
        conn = None
        try:
            conn = self._connect_db()
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                for row in rows:
                    record_id = row["id"]
                    record_dict = {}
                    for col in self.columns_to_extract:
                        record_dict[col] = row.get(col, "")
                    # 组合成 (id, record_dict) 插入任务列表
                    self.tasks.append((record_id, record_dict))
        except Exception as e:
            logging.error(f"加载任务失败: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
        
        logging.info(f"从数据库加载任务，共 {len(self.tasks)} 个待处理任务")
        return len(self.tasks)
    
    def update_task_results(self, results: Dict[int, Dict[str, Any]]):
        """
        将一批结果写回数据库：
        - 对于成功记录，标记 processed='yes' 并更新 columns_to_write 对应字段
        - 对于内容错误记录，也可在此记录额外信息(如需要)，但这里示例只更新成功行
        """
        if not results:
            return
        
        conn = None
        try:
            conn = self._connect_db()
            conn.start_transaction()
            
            update_sql_parts = []
            update_params = []

            # 需要写入的字段
            # columns_to_write 形如: { "tag": "function_sub_category" }
            for record_id, row_result in results.items():
                if "_error" not in row_result:
                    # 构造 SET 语句
                    set_clauses = []
                    param_values = []
                    
                    for alias, col_name in self.columns_to_write.items():
                        set_clauses.append(f"`{col_name}` = %s")
                        param_values.append(row_result.get(alias, ""))

                    # processed='yes'
                    set_clauses.append("processed = %s")
                    param_values.append("yes")
                    
                    set_clause_str = ", ".join(set_clauses)
                    sql_single = f"UPDATE `{self.table_name}` SET {set_clause_str} WHERE id = %s"
                    param_values.append(record_id)

                    update_sql_parts.append(sql_single)
                    update_params.append(tuple(param_values))

            with conn.cursor() as cursor:
                for sql_part, param in zip(update_sql_parts, update_params):
                    cursor.execute(sql_part, param)
            
            conn.commit()
            logging.info(f"成功将 {len(update_params)} 条记录更新为 processed='yes'")
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"更新数据库记录失败: {e}")
        finally:
            if conn:
                conn.close()
    
    def reload_task_data(self, record_id: int) -> Dict[str, Any]:
        """
        当系统错误时，把该条重新放回队列，需要再次从数据库里加载原始数据
        """
        conn = None
        record_dict = {}
        try:
            conn = self._connect_db()
            with conn.cursor(dictionary=True) as cursor:
                # 增强SQL安全性：使用反引号包裹字段名和表名
                cols = ", ".join(f"`{col}`" for col in self.columns_to_extract)
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
        finally:
            if conn:
                conn.close()
        return record_dict

###############################################################################
# Excel任务池实现
###############################################################################
class ExcelTaskPool(BaseTaskPool):
    """Excel数据源任务池"""
    
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
    
    def initialize_tasks(self) -> int:
        """初始化任务列表，返回任务总数"""
        self.tasks = []
        for idx, row in self.df.iterrows():
            if row.get("processed", "") != "yes":
                record_dict = {}
                for col in self.columns_to_extract:
                    val = row.get(col, "")
                    record_dict[col] = str(val) if val is not None else ""
                self.tasks.append((idx, record_dict))
        
        logging.info(f"初始化任务池，共 {len(self.tasks)} 个待处理任务")
        return len(self.tasks)
    
    def update_task_results(self, results: Dict[int, Dict[str, Any]]):
        """
        将结果更新到DataFrame中，并写入Excel
        :param results: Dict[行索引, 结果字典]
        """
        if not results:
            return
            
        try:
            # 批量更新DataFrame
            for idx, result in results.items():
                # 更新结果字段
                for alias, col_name in self.columns_to_write.items():
                    self.df.at[idx, col_name] = result.get(alias, "")
                
                # 仅对成功处理的记录标记为已处理
                if "_error" not in result:
                    self.df.at[idx, "processed"] = "yes"
            
            # 写入Excel文件
            self.df.to_excel(self.output_excel, index=False)
            logging.info(f"成功将 {len(results)} 条记录更新到Excel")
        except Exception as e:
            logging.error(f"更新Excel记录失败: {e}")
    
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
        self.temperature = model_dict.get("temperature", 0.7)

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
        
        # 创建模型ID->对象映射
        self.model_map = {m.id: m for m in self.models}

        # 并发配置
        self.max_workers = concurrency_cfg.get("max_workers", 5)
        self.batch_size = concurrency_cfg.get("batch_size", 300)
        self.save_interval = concurrency_cfg.get("save_interval", 300)
        self.global_retry_times = concurrency_cfg.get("retry_times", 3)

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
                logging.error("配置中未找到excel部分")
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
    # 根据当前可用的"模型"信息，从加权池中随机挑选可用模型
    ###########################################################################
    def get_available_model_randomly(self, exclude_model_ids=None) -> Optional[ModelConfig]:
        if exclude_model_ids is None:
            exclude_model_ids = set()
        else:
            exclude_model_ids = set(exclude_model_ids)

        available_model_ids = self.dispatcher.get_available_models(exclude_model_ids)
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
                    return content
        except aiohttp.ClientError as e:
            logging.error(f"AI请求异常 (API_ERROR): {str(e)}")
            raise
        except Exception as e:
            logging.error(f"处理AI响应异常 (API_ERROR): {str(e)}")
            raise

    ###########################################################################
    # 异步处理一条记录
    ###########################################################################
    async def process_one_record_async(self, record_id: Any, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        异步处理一条记录
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
                parsed_json["full_response"] = content
                parsed_json["used_model_id"] = model_cfg.id
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
    # 同步包装
    ###########################################################################
    def process_one_record(self, record_id: Any, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        同步包装异步处理函数，用于线程池
        :param record_id: 记录ID（可能是数据库ID或Excel行索引）
        :param row_data: 记录数据字典
        :return: 处理结果字典
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_one_record_async(record_id, row_data))
        finally:
            loop.close()

    ###########################################################################
    # 核心流程：从数据源读取 => 并发处理 => 写回数据源
    ###########################################################################
    def process_tasks(self):
        """
        主处理函数：从数据源读取任务，多线程处理，再写回结果
        """
        # 初始化任务
        total_tasks = self.task_pool.initialize_tasks()
        if total_tasks == 0:
            logging.info("没有需要处理的任务.")
            return
        
        logging.info(f"本次需处理 {total_tasks} 条数据")
        
        futures_map = {}  # future -> record_id
        batch_results = {}  # record_id -> result
        
        processed_count = 0  # 已处理数量
        success_count = 0    # 成功数量
        system_retry_count = 0  # 系统错误重试数量
        content_error_count = 0  # 内容错误数量

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 主循环：直到任务池为空且所有已提交任务完成
            while self.task_pool.has_tasks() or futures_map:
                # 1. 若有空闲线程，则提交新任务
                available_workers = self.max_workers - len(futures_map)
                if self.task_pool.has_tasks() and available_workers > 0:
                    batch_size = min(available_workers, self.task_pool.get_remaining_count())
                    task_batch = self.task_pool.get_batch(batch_size)
                    for record_id, record_dict in task_batch:
                        future = executor.submit(self.process_one_record, record_id, record_dict)
                        futures_map[future] = record_id
                
                # 2. 收集已完成的 future
                done_futures = [f for f in futures_map if f.done()]
                for f in done_futures:
                    record_id = futures_map.pop(f)
                    try:
                        result = f.result()
                        processed_count += 1
                        
                        if "_error" in result and result.get("_error_type") == ErrorType.SYSTEM_ERROR:
                            # 比如无可用模型等
                            task_id_str = str(record_id)
                            if self.datasource_type == "excel":
                                task_id_str = f"第 {record_id+2} 行"
                                
                            logging.warning(f"{task_id_str} 遇到系统错误: {result['_error']}, 重新放回队列头部")
                            self.task_pool.add_task_to_front(record_id, self.task_pool.reload_task_data(record_id))
                            system_retry_count += 1
                            time.sleep(5)
                            continue
                        
                        if "_error" in result and result.get("_error_type") == ErrorType.CONTENT_ERROR:
                            task_id_str = str(record_id)
                            if self.datasource_type == "excel":
                                task_id_str = f"第 {record_id+2} 行"
                                
                            logging.warning(f"{task_id_str} 遇到内容错误: {result['_error']}")
                            content_error_count += 1
                            # 内容错误不标记为 processed='yes'，也不回到池
                            # 但你也可以选择写一些备注字段
                
                        else:
                            success_count += 1

                        batch_results[record_id] = result

                        # 满足 batch_size 或任务完成，就批量写回数据源
                        if len(batch_results) >= self.batch_size or (processed_count == total_tasks and batch_results):
                            # 使用锁保护批处理过程
                            with self.lock:
                                # 复制当前批次结果进行处理
                                current_results = batch_results.copy()
                                # 清空批次结果字典
                                batch_results = {}
                                
                                # 更新数据源
                                self.task_pool.update_task_results(current_results)
                                
                                # 每处理一定数量也做一次保存
                                if processed_count % self.save_interval == 0 or processed_count == total_tasks:
                                    logging.info(f"已处理 {processed_count}/{total_tasks} (含重试)，成功 {success_count}，内容错误 {content_error_count}，系统重试 {system_retry_count}")
                    except Exception as e:
                        task_id_str = str(record_id)
                        if self.datasource_type == "excel":
                            task_id_str = f"第 {record_id+2} 行"
                            
                        logging.error(f"处理{task_id_str}时发生异常: {e}")
                
                # 如果没有完成的任务，短暂休息
                if not done_futures:
                    time.sleep(0.1)

        # 处理剩余 batch_results (如果有的话)
        if batch_results:
            with self.lock:
                self.task_pool.update_task_results(batch_results)

        logging.info(f"处理完毕，共处理 {processed_count} 条，成功 {success_count} 条.")
        logging.info(f"内容错误 {content_error_count} 条，系统重试 {system_retry_count} 次.")

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

if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "./config.yaml"
    
    # 验证配置文件
    if not validate_config_file(config_file):
        sys.exit(1)
    
    try:
        # 创建处理器并运行
        processor = UniversalAIProcessor(config_file)
        processor.process_tasks()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        sys.exit(1)