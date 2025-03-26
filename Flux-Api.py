#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
import yaml
import json
import re
import aiohttp
import asyncio
import time
import logging
import random
import threading # 保留用于锁
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Literal, AsyncIterable
from contextlib import asynccontextmanager # 用于 lifespan

###############################################################################
# 错误类型枚举 - 区分需要退避的错误和内容处理错误
###############################################################################
class ErrorType:
    """错误类型枚举"""
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
        """初始化读写锁"""
        self._read_ready = threading.Condition(threading.Lock()) # 控制读线程的条件变量
        self._readers = 0 # 当前正在读取的线程数
        self._writers = 0 # 当前正在写入的线程数
        self._write_ready = threading.Condition(threading.Lock()) # 控制写线程的条件变量
        self._pending_writers = 0 # 等待写入的线程数

    def read_acquire(self):
        """获取读锁"""
        with self._read_ready:
            # 如果有写者或等待的写者，则等待
            while self._writers > 0 or self._pending_writers > 0:
                self._read_ready.wait()
            self._readers += 1 # 增加读者计数

    def read_release(self):
        """释放读锁"""
        with self._read_ready:
            self._readers -= 1 # 减少读者计数
            if self._readers == 0:
                # 如果没有读者了，通知所有等待的写者（可能只有一个会获得锁）
                self._read_ready.notify_all()

    def write_acquire(self):
        """获取写锁"""
        with self._write_ready:
            self._pending_writers += 1 # 增加等待写者计数

        with self._read_ready:
            # 等待直到没有读者和写者
            while self._readers > 0 or self._writers > 0:
                self._read_ready.wait()
            # 此时可以写入
            self._writers += 1 # 增加写者计数

        with self._write_ready:
            self._pending_writers -= 1 # 减少等待写者计数

    def write_release(self):
        """释放写锁"""
        with self._read_ready:
            self._writers -= 1 # 减少写者计数
            # 通知所有等待的线程（读者和写者）
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
        """返回读锁上下文管理器实例"""
        return self.ReadLock(self)

    def write_lock(self):
        """返回写锁上下文管理器实例"""
        return self.WriteLock(self)

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
        self.capacity = capacity # 桶的最大容量
        self.refill_rate = refill_rate # 每秒补充速率
        self.tokens = capacity # 当前令牌数，初始为满
        self.last_refill = time.time() # 上次补充令牌的时间
        self.lock = threading.Lock() # 保证线程安全

    def refill(self):
        """根据经过的时间补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        # 计算应补充的令牌数
        new_tokens = elapsed * self.refill_rate
        # 更新令牌数，不超过桶容量
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now # 更新上次补充时间

    def consume(self, tokens: float = 1.0) -> bool:
        """
        尝试消耗指定数量的令牌

        :param tokens: 要消耗的令牌数
        :return: 如果有足够令牌返回True，否则返回False
        """
        with self.lock: # 保证原子操作
            # 先补充令牌，确保基于最新状态判断
            self.refill()
            # 检查是否有足够的令牌
            if self.tokens >= tokens:
                self.tokens -= tokens # 消耗令牌
                return True # 成功消耗
            return False # 令牌不足

###############################################################################
# 模型限流管理器：为每个模型维护独立的令牌桶
###############################################################################
class ModelRateLimiter:
    """模型限流管理器"""
    def __init__(self):
        """初始化模型限流管理器"""
        self.limiters: Dict[str, TokenBucket] = {}  # 模型ID -> 令牌桶 映射
        self.lock = threading.Lock() # 保护 limiters 字典的访问

    def configure(self, models_config: List[Dict[str, Any]]):
        """从模型配置中配置限流器"""
        with self.lock:
            for model in models_config:
                model_id = str(model.get("id"))
                weight = model.get("weight", 1)
                # 基于权重估算基础RPS，设定上下限
                estimated_rps = max(0.5, min(weight / 10, 10))
                # 配置中指定的 safe_rps 优先
                safe_rps = model.get("safe_rps", estimated_rps)
                # 创建令牌桶，容量为安全RPS的2倍以允许突发
                self.limiters[model_id] = TokenBucket(
                    capacity=safe_rps * 2, refill_rate=safe_rps
                )
                # 使用 DEBUG 级别记录初始化日志，避免在 INFO 级别下输出过多
                logging.debug(f"为模型[{model_id}]配置限流: {safe_rps} RPS")

    def can_process(self, model_id: str) -> bool:
        """
        检查指定模型是否可以处理新请求 (即消耗一个令牌)

        :param model_id: 模型ID
        :return: 如果可以处理返回True，否则返回False
        """
        model_id = str(model_id)
        with self.lock:
            limiter = self.limiters.get(model_id)
            if limiter is None:
                # 如果模型没有配置限流器，默认允许（可能需要警告）
                logging.warning(f"模型 [{model_id}] 未找到限流器配置，默认允许请求。")
                return True
            # 尝试消耗一个令牌
            return limiter.consume(1.0)

###############################################################################
# ModelDispatcher: 模型调度器，处理退避和状态缓存
###############################################################################
class ModelDispatcher:
    """模型调度器，处理API错误退避和模型状态缓存"""
    def __init__(self, models: List['ModelConfig'], backoff_factor: int = 2):
        """
        初始化调度器

        :param models: 已解析的 ModelConfig 对象列表
        :param backoff_factor: 指数退避算法的基数
        """
        self.backoff_factor = backoff_factor
        # 存储每个模型的状态
        self._model_state: Dict[str, Dict[str, Any]] = {}
        for m in models:
            self._model_state[m.id] = {
                "fail_count": 0,            # 连续失败次数
                "next_available_ts": 0,     # 下次可用的时间戳 (0表示立即可用)
                "success_count": 0,         # 总成功次数 (用于统计)
                "error_count": 0,           # 总失败次数 (用于统计)
                "avg_response_time": 0      # 平均响应时间 (指数移动平均)
            }
        self._rwlock = RWLock() # 读写锁保护 _model_state 和缓存
        self._availability_cache: Dict[str, bool] = {} # 模型ID -> 是否可用 的缓存
        self._cache_last_update: float = 0 # 缓存上次更新时间
        self._cache_ttl: float = 0.5       # 缓存有效期（秒）
        self._update_availability_cache() # 初始化缓存

    def _update_availability_cache(self):
        """更新可用性缓存 (需要持有写锁调用)"""
        with self._rwlock.write_lock():
            current_time = time.time()
            new_cache = {}
            for model_id, state in self._model_state.items():
                # 如果当前时间大于等于下次可用时间戳，则模型可用
                new_cache[model_id] = (current_time >= state["next_available_ts"])
            self._availability_cache = new_cache
            self._cache_last_update = current_time
            logging.debug("模型可用性缓存已更新")

    def update_model_metrics(self, model_id: str, response_time: float, success: bool):
        """更新模型的性能指标 (成功/失败次数，平均响应时间)"""
        with self._rwlock.write_lock():
            state = self._model_state.get(model_id)
            if not state: return # 如果模型ID不存在，则忽略

            # 更新成功/失败计数
            if success: state["success_count"] += 1
            else: state["error_count"] += 1

            # 使用指数移动平均 (EMA) 更新平均响应时间
            total_calls = state["success_count"] + state["error_count"]
            if total_calls == 1:
                state["avg_response_time"] = response_time
            else:
                # alpha (权重因子)，可以调整，0.1表示新数据占10%权重
                weight = 0.1
                # 获取当前平均值，如果不存在则使用当前响应时间
                current_avg = state.get("avg_response_time", response_time)
                # EMA 计算公式
                state["avg_response_time"] = current_avg * (1 - weight) + response_time * weight

    def get_model_success_rate(self, model_id: str) -> float:
        """获取模型的成功率 (成功次数 / 总次数)"""
        with self._rwlock.read_lock():
            state = self._model_state.get(model_id)
            if not state: return 0.0 # 模型不存在
            total = state["success_count"] + state["error_count"]
            # 如果没有调用记录，视为100%成功
            return state["success_count"] / total if total > 0 else 1.0

    def get_model_avg_response_time(self, model_id: str) -> float:
        """获取模型的平均响应时间"""
        with self._rwlock.read_lock():
            state = self._model_state.get(model_id)
            # 如果模型不存在或从未记录响应时间，返回默认值1.0秒
            return (state.get("avg_response_time", 1.0) or 1.0) if state else 1.0

    def is_model_available(self, model_id: str) -> bool:
        """判断模型当前是否可用 (优先查缓存)"""
        current_time = time.time()
        # 检查缓存是否过期 (读锁内检查时间戳)
        with self._rwlock.read_lock():
            cache_expired = (current_time - self._cache_last_update >= self._cache_ttl)
            # 如果缓存未过期且在缓存中，直接返回结果
            if not cache_expired and model_id in self._availability_cache:
                return self._availability_cache[model_id]

        # 如果缓存过期，需要更新 (写锁)
        if cache_expired:
            self._update_availability_cache()
            # 更新后再次读取缓存 (读锁)
            with self._rwlock.read_lock():
                return self._availability_cache.get(model_id, False)

        # Fallback: 如果模型不在缓存中(理论上不应发生)，直接查状态
        with self._rwlock.read_lock():
            state = self._model_state.get(model_id)
            # 如果状态存在且当前时间 >= 下次可用时间，则可用
            return (current_time >= state["next_available_ts"]) if state else False

    def mark_model_success(self, model_id: str):
        """标记模型调用成功，重置失败计数和可用时间"""
        with self._rwlock.write_lock():
            state = self._model_state.get(model_id)
            if state:
                # 检查模型之前是否处于不可用状态
                was_unavailable = time.time() < state["next_available_ts"]
                # 重置失败计数和可用时间
                state["fail_count"] = 0
                state["next_available_ts"] = 0
                # 更新缓存
                self._availability_cache[model_id] = True
                # 如果模型刚恢复，打印日志
                if was_unavailable:
                    logging.info(f"模型[{model_id}] 调用成功，恢复可用。")

    def mark_model_failed(self, model_id: str, error_type: str = ErrorType.API_ERROR):
        """标记模型调用失败，仅对 API_ERROR 执行退避"""
        # 只处理 API 错误导致的失败，其他错误仅记录警告
        if error_type != ErrorType.API_ERROR:
            logging.warning(f"模型[{model_id}] 遇到内容或系统错误 ({error_type})，不执行退避。")
            return

        with self._rwlock.write_lock():
            state = self._model_state.get(model_id)
            if not state: return # 模型不存在

            # 增加失败计数
            state["fail_count"] += 1
            fail_count = state["fail_count"]

            # 计算退避时间 (前几次线性增长，之后指数增长，有上限)
            if fail_count <= 3:
                backoff_seconds = fail_count * 2 # 2s, 4s, 6s
            else:
                # 指数增长部分: factor^(n-3)，加上基础的6秒
                backoff_seconds = min(6 + (self.backoff_factor ** (fail_count - 3)), 60) # 最长退避60秒

            # 更新下次可用时间戳和缓存
            state["next_available_ts"] = time.time() + backoff_seconds
            self._availability_cache[model_id] = False
            logging.warning(f"模型[{model_id}] API调用失败，第{fail_count}次，进入退避 {backoff_seconds:.2f} 秒")

    def get_available_models(self, exclude_model_ids: Set[str] = None) -> List[str]:
        """获取所有当前可用的模型ID列表，排除指定ID"""
        exclude_ids = exclude_model_ids or set()
        available_models = []
        # 先获取所有模型的ID列表 (需要读锁保护 _model_state 的keys)
        with self._rwlock.read_lock():
            all_model_ids = list(self._model_state.keys())

        # 遍历所有模型ID，检查可用性 (is_model_available 会处理缓存和锁)
        for model_id in all_model_ids:
             if model_id not in exclude_ids and self.is_model_available(model_id):
                  available_models.append(model_id)
        return available_models

###############################################################################
# 模型配置类
###############################################################################
class ModelConfig:
    """存储单个模型的配置信息"""
    def __init__(self, model_dict: Dict[str, Any], channels: Dict[str, Any]):
        """
        从字典和通道配置初始化模型配置对象

        :param model_dict: 包含模型配置的字典
        :param channels: 包含所有通道配置的字典
        """
        self.id = str(model_dict.get("id")) # 模型唯一ID (转为字符串)
        self.name = model_dict.get("name")  # 模型别名 (可选)
        self.model = model_dict.get("model") # 模型在后端API的标识符
        self.channel_id = str(model_dict.get("channel_id")) # 所属通道ID (转为字符串)
        self.api_key = model_dict.get("api_key") # 访问该模型的API Key
        self.timeout = model_dict.get("timeout", 600) # 模型特定超时时间
        self.base_weight = model_dict.get("weight", 1) # 基础权重 (静态，用于选择)
        self.temperature = model_dict.get("temperature", 0.7) # 默认温度参数
        self.supports_json_schema = model_dict.get("supports_json_schema", False) # 是否支持JSON Schema格式输出
        # 基于权重的安全每秒请求数估算，优先使用配置值
        self.safe_rps = model_dict.get("safe_rps", max(0.5, min(self.base_weight / 10, 10)))

        # 基础验证
        if not self.id or not self.model or not self.channel_id:
            raise ValueError(f"模型配置缺少必填字段: id={self.id}, model={self.model}, channel_id={self.channel_id}")
        # 检查通道是否存在
        if self.channel_id not in channels:
            raise ValueError(f"模型 {self.id}: channel_id='{self.channel_id}' 在channels中不存在")

        # 从通道配置中获取信息
        channel_cfg = channels[self.channel_id]
        self.channel_name = channel_cfg.get("name") # 通道名称 (可选)
        self.base_url = channel_cfg.get("base_url") # 通道的基础URL
        self.api_path = channel_cfg.get("api_path", "/v1/chat/completions") # API路径
        self.channel_timeout = channel_cfg.get("timeout", 600) # 通道全局超时
        # 处理代理配置，确保是有效字符串或None
        proxy_setting = channel_cfg.get("proxy", "")
        self.channel_proxy = proxy_setting if isinstance(proxy_setting, str) and proxy_setting.strip() else None
        if not self.base_url:
             raise ValueError(f"通道 '{self.channel_id}' 缺少 base_url 配置")

        # 计算最终使用的超时时间 (取模型和通道中的较小值)
        self.final_timeout = min(self.timeout, self.channel_timeout)
        self.connect_timeout = 10 # 固定的连接超时时间
        self.read_timeout = self.final_timeout # 总读取/处理超时时间

###############################################################################
# OpenAI API 兼容的请求与响应模型 (Pydantic)
###############################################################################
class ChatMessage(BaseModel):
    """聊天消息结构"""
    role: str # 角色 (user, assistant, system)
    content: str # 消息内容
    name: Optional[str] = None # 参与者名称 (可选)

class ResponseFormat(BaseModel):
    """响应格式定义 (例如: text, json_object)"""
    type: str = "text"

class ChatCompletionRequest(BaseModel):
    """聊天补全请求体模型"""
    model: str # 请求的模型名称或ID
    messages: List[ChatMessage] # 消息列表
    temperature: Optional[float] = None # 温度 (可选，默认为模型配置)
    top_p: Optional[float] = 1.0 # Top-p采样 (可选)
    n: Optional[int] = 1 # 生成的选项数 (通常为1)
    max_tokens: Optional[int] = None # 最大生成token数 (可选)
    stream: Optional[bool] = False # 是否流式输出
    stop: Optional[Union[str, List[str]]] = None # 停止序列 (可选)
    presence_penalty: Optional[float] = 0 # 存在惩罚 (可选)
    frequency_penalty: Optional[float] = 0 # 频率惩罚 (可选)
    logit_bias: Optional[Dict[str, float]] = None # Logit偏差 (可选)
    user: Optional[str] = None # 用户标识符 (可选)
    response_format: Optional[ResponseFormat] = None # 期望的响应格式 (可选)
    class Config: extra = "allow" # 允许额外的字段

    @model_validator(mode='before')
    @classmethod
    def convert_stop_to_list(cls, values):
        """将字符串类型的stop转为列表 (Pydantic v2 验证器)"""
        if isinstance(values, dict) and "stop" in values and isinstance(values["stop"], str):
            values["stop"] = [values["stop"]]
        return values

class ChatCompletionResponseChoice(BaseModel):
    """聊天补全响应中的选项"""
    index: int # 选项索引 (通常为0)
    message: ChatMessage # 返回的消息
    finish_reason: Optional[str] = "stop" # 结束原因 (stop, length, etc.)

class ChatCompletionResponseUsage(BaseModel):
    """Token使用情况"""
    prompt_tokens: Optional[int] = None # 输入Token数 (可选)
    completion_tokens: Optional[int] = None # 输出Token数 (可选)
    total_tokens: Optional[int] = None # 总Token数 (可选)

class ChatCompletionResponse(BaseModel):
    """非流式聊天补全响应体模型"""
    id: str # 响应唯一ID
    object: str = "chat.completion" # 对象类型
    created: int # 创建时间戳
    model: str # 使用的模型名称 (应为用户请求的名称)
    choices: List[ChatCompletionResponseChoice] # 选项列表
    usage: Optional[ChatCompletionResponseUsage] = None # Token使用情况 (可选)

class ModelInfo(BaseModel):
    """管理接口返回的模型信息"""
    id: str # 内部模型ID
    name: Optional[str] = None # 模型别名
    model: str # 模型在后端的标识符
    weight: int # 基础权重 (静态)
    success_rate: float # 成功率
    avg_response_time: float # 平均响应时间
    available: bool # 当前是否可用
    channel: Optional[str] = None # 所属通道名称

class ModelsResponse(BaseModel):
    """管理接口 /admin/models 的响应体"""
    models: List[ModelInfo] # 模型信息列表
    total: int # 模型总数
    available: int # 可用模型数

class HealthResponse(BaseModel):
    """管理接口 /admin/health 的响应体"""
    status: Literal["healthy", "degraded", "unhealthy"] # 健康状态
    available_models: int # 可用模型数
    total_models: int # 总模型数
    uptime: float # 服务运行时间（秒）

###############################################################################
# API服务主类
###############################################################################
class FluxApiService:
    """OpenAI API 兼容的服务，管理模型池和请求处理"""
    def __init__(self, config_path: str):
        """初始化服务，加载配置、设置日志、初始化模型等"""
        self.start_time = time.time() # 记录启动时间
        self.load_config(config_path) # 加载配置
        self.setup_logging() # 设置日志
        self.initialize_models() # 初始化模型相关组件
        # 注意：后台任务（如动态权重调整）已移除
        logging.info(f"FluxApiService 初始化完成，加载了 {len(self.models)} 个模型")

    def load_config(self, config_path: str):
        """加载并验证YAML配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            if not isinstance(self.config, dict):
                 raise ValueError("配置文件内容不是有效的YAML字典")
        except FileNotFoundError:
            raise RuntimeError(f"配置文件未找到: {config_path}")
        except yaml.YAMLError as e:
            raise RuntimeError(f"配置文件YAML解析失败: {e}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件时发生未知错误: {e}")

    def setup_logging(self):
        """根据配置设置日志记录器"""
        # 检查配置是否已加载且为字典
        if not hasattr(self, 'config') or not isinstance(self.config, dict):
             logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
             logging.error("无法加载日志配置，使用默认配置。")
             return

        # 获取日志配置，使用默认值处理缺失项
        log_config = self.config.get("global", {}).get("log", {})
        level_str = log_config.get("level", "info").upper()
        level = getattr(logging, level_str, logging.INFO) # 将字符串级别转为 logging级别常量
        log_format = log_config.get("format_string", "%(asctime)s [%(levelname)s] %(message)s")

        # 如果配置了 JSON 格式
        if log_config.get("format") == "json":
            log_format = '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
            formatter = logging.Formatter(log_format)
        else:
             formatter = logging.Formatter(log_format)

        # 获取根 logger 并设置
        root_logger = logging.getLogger()
        # 确保根 logger 有级别，否则 handler 收不到消息
        if not root_logger.hasHandlers():
             root_logger.setLevel(level)

        # 移除已存在的 handlers 避免重复日志
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 根据配置选择输出目标 (控制台或文件)
        output_type = log_config.get("output", "console")
        handler: logging.Handler
        if output_type == "file":
            file_path = log_config.get("file_path", "./logs/flux_api.log")
            try:
                # 确保日志目录存在
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                handler = logging.FileHandler(file_path, encoding='utf-8')
            except Exception as e:
                # 如果创建文件失败，回退到控制台输出
                handler = logging.StreamHandler(sys.stdout)
                logging.error(f"无法创建日志文件 {file_path}: {e}。将输出到控制台。")
        else: # 默认输出到控制台
            handler = logging.StreamHandler(sys.stdout)

        # 设置 handler 的格式和级别，并添加到 root logger
        handler.setFormatter(formatter)
        handler.setLevel(level)
        root_logger.addHandler(handler)
        # 再次确保 root logger 级别正确
        root_logger.setLevel(level)

    def initialize_models(self):
        """初始化模型配置、调度器、限流器和静态权重池"""
        models_cfg = self.config.get("models", [])
        channels_cfg = self.config.get("channels", {})

        # 基本配置验证
        if not isinstance(models_cfg, list): raise ValueError("'models' 必须是一个列表")
        if not isinstance(channels_cfg, dict): raise ValueError("'channels' 必须是一个字典")
        if not models_cfg: raise ValueError("配置文件中未找到 'models' 或为空")
        if not channels_cfg: logging.warning("配置文件中未找到 'channels' 或为空")

        # 加载 ModelConfig 对象
        model_ids = set()
        self.models: List[ModelConfig] = []
        valid_models_cfg = [] # 用于配置限流器的原始字典列表
        for m_dict in models_cfg:
             if not isinstance(m_dict, dict): logging.warning(f"忽略无效的模型配置项: {m_dict}"); continue
             mid = str(m_dict.get("id"))
             if not mid: logging.warning(f"忽略缺少 'id' 的模型配置: {m_dict}"); continue
             # 检查重复ID
             if mid in model_ids: logging.debug(f"发现重复的模型ID: {mid} (将被忽略)."); continue # DEBUG级别
             try:
                 # 创建 ModelConfig 实例
                 model_config = ModelConfig(m_dict, channels_cfg)
                 self.models.append(model_config)
                 model_ids.add(mid)
                 valid_models_cfg.append(m_dict) # 保存有效配置用于限流器
             except ValueError as e:
                 logging.error(f"加载模型配置 {mid} 失败: {e}")
        if not self.models: raise RuntimeError("未能成功加载任何模型配置。请检查配置文件。")

        # 初始化调度器和限流器
        concurrency_cfg = self.config.get("global", {}).get("concurrency", {})
        backoff_factor = concurrency_cfg.get("backoff_factor", 2)
        self.dispatcher = ModelDispatcher(self.models, backoff_factor=backoff_factor)
        self.rate_limiter = ModelRateLimiter()
        self.rate_limiter.configure(valid_models_cfg) # 限流器配置日志为 DEBUG

        # 创建模型ID到ModelConfig对象的映射
        self.model_map: Dict[str, ModelConfig] = {m.id: m for m in self.models}
        # 创建模型名称/标识符到模型ID的映射
        self.model_name_to_id: Dict[str, str] = {}
        for m in self.models:
            self.model_name_to_id[m.id] = m.id # ID 映射
            if m.model: # 模型标识符映射
                 if m.model in self.model_name_to_id and self.model_name_to_id[m.model] != m.id:
                      logging.debug(f"模型标识符 '{m.model}' 被多个模型使用。") # DEBUG级别
                 self.model_name_to_id[m.model] = m.id
            if m.name and m.name != m.model: # 模型别名映射
                 if m.name in self.model_name_to_id and self.model_name_to_id[m.name] != m.id:
                      logging.debug(f"模型名称 '{m.name}' 被多个模型使用。") # DEBUG级别
                 self.model_name_to_id[m.name] = m.id

        # 构建静态权重池 (基于 base_weight)
        self.models_pool = []
        for model in self.models:
             if model.base_weight > 0: # 只包含权重大于0的模型
                 self.models_pool.extend([model] * model.base_weight)
        if not self.models_pool: logging.warning("模型池为空！所有模型的权重可能为0。")
        logging.debug(f"初始化静态模型池完成，大小: {len(self.models_pool)}")

    def resolve_model_id(self, model_name_or_id: str) -> Optional[str]:
        """将用户请求的模型名称/ID解析为内部配置的模型ID"""
        # 优先匹配内部映射
        if model_name_or_id in self.model_name_to_id:
            return self.model_name_to_id[model_name_or_id]
        # 处理通配符或默认情况
        if model_name_or_id.lower() in ["auto", "any", "default", "*", ""]:
            return None # 返回 None 表示需要随机选择
        # 未找到匹配
        logging.warning(f"无法解析请求的模型名称或ID: '{model_name_or_id}'")
        return None

    def get_available_model(self, requested_model_name: Optional[str] = None,
                            exclude_models: List[str] = None) -> Optional[ModelConfig]:
        """
        获取一个可用的模型。
        优先选择指定模型，否则根据静态权重从可用模型中随机选择。
        同时考虑调度器可用性和限流器限制。
        """
        exclude_set = set(exclude_models or [])
        target_model_id: Optional[str] = None
        use_random_selection = True

        # 1. 解析请求的模型名称
        if requested_model_name:
            resolved_id = self.resolve_model_id(requested_model_name)
            if resolved_id:
                target_model_id = resolved_id
                use_random_selection = False
            # 如果是 "auto" 或无法解析，则进行随机选择
            elif requested_model_name.lower() not in ["auto", "any", "default", "*", ""]:
                 logging.error(f"请求了未知的模型 '{requested_model_name}'，将尝试随机选择。")

        # 2. 如果指定了目标模型，检查其可用性
        if target_model_id and not use_random_selection:
            if target_model_id not in exclude_set:
                model = self.model_map.get(target_model_id)
                if model:
                    # 检查调度器可用性和限流器
                    is_available = self.dispatcher.is_model_available(target_model_id)
                    can_process = self.rate_limiter.can_process(target_model_id)
                    if is_available and can_process:
                         logging.debug(f"使用请求的可用模型: {model.name or target_model_id}")
                         return model
                    else:
                         # 指定模型不可用或被限流，转为随机选择
                         logging.warning(f"请求的模型 [{model.name or target_model_id}] 当前不可用/受限。尝试随机选择。")
                         exclude_set.add(target_model_id) # 从随机选择中也排除它
                         use_random_selection = True
                else:
                     # 理论上不应发生，如果 resolve_model_id 正常工作
                     logging.error(f"内部错误：解析得到的模型ID '{target_model_id}' 在 model_map 中不存在。")
                     use_random_selection = True # 回退到随机
            else:
                 # 指定的模型在排除列表中，转为随机选择
                 logging.warning(f"请求的模型 [{target_model_id}] 在排除列表中。尝试随机选择。")
                 use_random_selection = True

        # 3. 执行随机选择 (如果需要)
        if use_random_selection:
            current_pool = self.models_pool # 使用初始化时构建的静态池
            if not current_pool:
                logging.error("模型池为空，无法随机选择模型。")
                return None

            # 从池中过滤出当前可用且未被排除且能通过限流的模型
            eligible_models_in_pool = [
                 model for model in current_pool
                 if model.id not in exclude_set                     # 未被排除
                 and self.dispatcher.is_model_available(model.id) # 调度器可用
                 and self.rate_limiter.can_process(model.id)      # 限流器允许
            ]

            if not eligible_models_in_pool:
                logging.warning(f"没有符合条件的可用模型进行随机选择 (已排除: {exclude_set})。")
                return None

            # 从符合条件的模型池中随机选择一个 (已包含权重)
            chosen_model = random.choice(eligible_models_in_pool)
            logging.debug(f"随机选择了可用模型: {chosen_model.name or chosen_model.id} (基础权重: {chosen_model.base_weight})")
            return chosen_model

        return None # 理论上不应到达这里

    async def call_ai_api_async(self, model_cfg: ModelConfig, messages: List[ChatMessage],
                                temperature: Optional[float] = None, response_format: Optional[ResponseFormat] = None, stream: bool = False,
                                stop: Optional[List[str]] = None, max_tokens: Optional[int] = None, top_p: Optional[float] = None,
                                presence_penalty: Optional[float] = None, frequency_penalty: Optional[float] = None,
                                logit_bias: Optional[Dict[str, float]] = None, user: Optional[str] = None
                               ) -> Union[Tuple[str, Optional[int], Optional[int]], AsyncIterable[str]]:
        """
        异步调用后端AI API。

        :param model_cfg: 要使用的模型配置 (ModelConfig)
        :param messages: 聊天消息列表
        :param stream: 是否请求流式响应
        :param ...: 其他 OpenAI 兼容参数
        :return: 如果 stream=False，返回包含 (内容, 输入tokens, 输出tokens) 的元组；
                 如果 stream=True，返回一个异步迭代器 (AsyncIterable[str])，产生 SSE 格式的字符串。
        :raises HTTPException: 如果发生 API 调用错误 (如超时、连接错误、4xx/5xx 状态码) 或响应处理错误。
        """
        url = model_cfg.base_url.rstrip("/") + model_cfg.api_path
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {model_cfg.api_key}"}
        api_messages = [msg.model_dump(exclude_none=True) for msg in messages]
        payload: Dict[str, Any] = {"model": model_cfg.model, "messages": api_messages, "stream": stream}
        payload["temperature"] = temperature if temperature is not None else model_cfg.temperature
        # 添加其他可选参数
        if top_p is not None: payload["top_p"] = top_p
        if max_tokens is not None: payload["max_tokens"] = max_tokens
        if stop: payload["stop"] = stop
        if presence_penalty is not None: payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None: payload["frequency_penalty"] = frequency_penalty
        if logit_bias: payload["logit_bias"] = logit_bias
        if user: payload["user"] = user
        if response_format and model_cfg.supports_json_schema: payload["response_format"] = response_format.model_dump()

        proxy = model_cfg.channel_proxy or None # 如果配置为空字符串，则使用 None
        # 设置超时，包括 sock_read 以处理流式连接中的暂停
        timeout = aiohttp.ClientTimeout(
            connect=model_cfg.connect_timeout,    # 连接超时
            total=model_cfg.read_timeout,         # 总操作超时
            sock_read=60                           # 两次成功读取之间的最大间隔（秒）
        )

        logging.debug(f"发送请求至 [{model_cfg.id}]: URL={url}, 超时={timeout}, 代理={proxy}")
        start_time = time.time()
        response_status = -1
        response_text_preview = ""

        try:
            # 创建会话并发送 POST 请求
            # 注意: trust_env=False 可以禁用环境变量代理（如果需要精确控制）
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload, proxy=proxy) as resp:
                    response_status = resp.status
                    # 检查 HTTP 错误状态码 (4xx 或 5xx)
                    if response_status >= 400:
                        try: # 尝试读取错误响应体
                            error_text = await resp.text(); response_text_preview = error_text[:500]
                        except Exception as read_err: # 读取失败
                            error_text = f"(无法读取响应体: {read_err})"; response_text_preview = error_text
                        logging.error(f"模型 [{model_cfg.id}] API失败: HTTP {response_status}. URL: {url}. 预览: {response_text_preview}")
                        # 使用 aiohttp 的 raise_for_status() 抛出 ClientResponseError
                        resp.raise_for_status()

                    # 请求成功 (2xx)
                    response_time = time.time() - start_time
                    logging.info(f"模型 [{model_cfg.id}] API成功 (HTTP {response_status})，耗时: {response_time:.2f}s")

                    # 根据请求是流式还是非流式进行处理
                    if stream:
                        # 返回异步生成器对象，不需要 await
                        return self._handle_streaming_response(resp, model_cfg, start_time)
                    else:
                        # 处理非流式响应
                        content_type = resp.headers.get('Content-Type', '').lower()
                        response_text = await resp.text()
                        response_text_preview = response_text[:500] # 用于日志
                        try:
                            # 尝试解析 JSON
                            data = json.loads(response_text)
                            # 验证基本结构
                            if not isinstance(data, dict) or "choices" not in data or not data.get("choices"):
                                raise ValueError("响应缺少 'choices' 字段或为空")
                            choice = data["choices"][0]
                            if not choice.get("message") or "content" not in choice["message"]:
                                raise ValueError("响应 choice[0] 缺少 'message.content'")
                            content = choice["message"]["content"]
                            # 获取 token 使用情况 (如果存在)
                            usage = data.get("usage", {})
                            prompt_tokens = usage.get("prompt_tokens")
                            completion_tokens = usage.get("completion_tokens")
                            # 标记成功并更新指标
                            self.dispatcher.mark_model_success(model_cfg.id)
                            self.dispatcher.update_model_metrics(model_cfg.id, response_time, True)
                            # 返回包含内容和 token 数的元组
                            return content, prompt_tokens, completion_tokens
                        except json.JSONDecodeError as e:
                            # 如果 JSON 解析失败，检查是否是意外收到的事件流
                            if 'text/event-stream' in content_type or (response_text.strip().startswith("data:") and "[DONE]" in response_text):
                                logging.warning(f"模型 [{model_cfg.id}] 在非流请求中返回事件流，尝试提取内容。")
                                content = self._extract_content_from_event_stream(response_text)
                                if content:
                                     # 如果成功提取，也视为成功
                                     self.dispatcher.mark_model_success(model_cfg.id)
                                     self.dispatcher.update_model_metrics(model_cfg.id, response_time, True)
                                     return content, None, None # Token 数未知
                                else:
                                     # 提取失败
                                     raise ValueError("无法从意外的事件流中提取内容") from e
                            else:
                                # 真正的 JSON 解析错误
                                raise ValueError(f"响应无法解析为有效JSON") from e
                        except ValueError as e:
                            # 捕获上面抛出的结构验证错误
                            raise e

        # --- 异常处理 ---
        except aiohttp.ClientResponseError as e: # HTTP 4xx/5xx 错误
             response_time = time.time() - start_time
             self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
             self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
             # 将 aiohttp 错误包装为 FastAPI 的 HTTPException
             raise HTTPException(status_code=e.status, detail=f"模型API错误: {e.message}") from e
        except asyncio.TimeoutError as e: # 包括 total 和 sock_read 超时
            response_time = time.time() - start_time
            logging.error(f"模型 [{model_cfg.id}] API 调用超时 ({timeout}). URL: {url}")
            self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            raise HTTPException(status_code=408, detail="模型请求超时") from e
        except aiohttp.ClientConnectionError as e: # DNS 查询失败, 连接被拒绝等
             response_time = time.time() - start_time
             logging.error(f"模型 [{model_cfg.id}] API 连接错误: {e}. URL: {url}")
             self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
             self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
             raise HTTPException(status_code=503, detail=f"模型连接错误: {e}") from e
        except aiohttp.ClientError as e: # 其他 aiohttp 客户端错误 (如无效 payload)
             response_time = time.time() - start_time
             logging.error(f"模型 [{model_cfg.id}] AIOHTTP 客户端错误: {e}. URL: {url}")
             self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR) # 通常视为API问题
             self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
             raise HTTPException(status_code=500, detail=f"API客户端错误: {e}") from e
        except ValueError as e: # 来自非流式处理中的 JSON 解析或结构验证错误
            response_time = time.time() - start_time
            # 标记为内容错误，不触发模型退避
            self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.CONTENT_ERROR)
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            # 返回 500 错误给客户端
            raise HTTPException(status_code=500, detail=f"模型响应处理错误: {e}") from e
        except Exception as e: # 其他所有未预料到的异常
            response_time = time.time() - start_time
            logging.exception(f"模型 [{model_cfg.id}] API 调用时发生未知错误. URL: {url}", exc_info=e)
            # 假设是 API 问题进行退避
            self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            raise HTTPException(status_code=500, detail=f"调用模型时发生内部错误: {e}") from e

    async def _handle_streaming_response(self, response: aiohttp.ClientResponse, model_cfg: ModelConfig, start_time: float) -> AsyncIterable[str]:
        """
        处理流式响应，返回一个异步生成器，该生成器产生 SSE 格式的字符串。
        包含详细日志和错误处理。
        """
        buffer = "" # 用于存储不完整的行
        has_yielded = False # 是否成功产生过至少一个数据块
        chunk_count = 0 # 接收到的块计数
        last_activity_time = time.time() # 上次接收到活动的时间
        
        returned_successfully = False # 标记是否接收到了 [DONE]
        logging.debug(f"[{model_cfg.id}] 开始流式响应生成循环。")
        
        # 立即发送一个保持连接的初始消息
        try:
            # 发送一个空的SSE注释，保持连接活跃，不影响实际内容
            yield ": keeping connection alive\n\n"
            has_yielded = True  # 标记已经有输出，防止连接关闭时标记为早期失败
            logging.debug(f"[{model_cfg.id}] 已发送初始连接保持消息")
        except Exception as e:
            logging.warning(f"[{model_cfg.id}] 发送初始消息失败: {e}")
        try:
            # 异步迭代响应内容块
            async for chunk in response.content.iter_any():
                chunk_count += 1
                now = time.time()
                time_since_last = now - last_activity_time
                last_activity_time = now
                # DEBUG 日志记录每个块的接收情况
                logging.debug(f"[{model_cfg.id}] 收到流块 {chunk_count} (大小:{len(chunk)}字节, 距离上次:{time_since_last:.2f}秒)")

                if not chunk:
                    logging.debug(f"[{model_cfg.id}] 收到空块 {chunk_count}。")
                    continue # 跳过空块

                # 解码并添加到缓冲区，处理可能的解码错误
                try:
                    decoded_chunk = chunk.decode('utf-8')
                    buffer += decoded_chunk
                except UnicodeDecodeError:
                    buffer += chunk.decode('utf-8', errors='ignore') # 忽略错误字符继续
                    logging.warning(f"[{model_cfg.id}] 流包含无效UTF-8数据，已忽略。")

                # 按行分割处理缓冲区内容
                lines = buffer.split('\n')
                buffer = lines.pop() # 最后一部分可能不完整，留在缓冲区下次处理

                for line in lines:
                    line = line.strip()
                    if not line: continue # 跳过空行

                    # 检查是否是 SSE 数据行
                    if line.startswith('data:'):
                        data_content = line[len('data:'):].strip()
                        # 检查是否是结束标记
                        if data_content == '[DONE]':
                            logging.info(f"[{model_cfg.id}] 在流块 {chunk_count} 中收到 [DONE] 标记。")
                            returned_successfully = True
                            break # 结束内层行处理循环
                        elif data_content:
                            # 检查是否是 JSON 格式
                            if data_content.startswith("{") and data_content.endswith("}"):
                                # 产生符合 SSE 格式的输出
                                yield f"data: {data_content}\n\n"
                                has_yielded = True # 标记已成功产生数据
                            else:
                                # 记录非 JSON 数据块警告
                                logging.warning(f"[{model_cfg.id}] 流包含非JSON数据块: {data_content[:100]}...")
                    else:
                        # 记录非 'data:' 开头的行 (可能是注释或其他信息)
                        logging.debug(f"[{model_cfg.id}] 流包含非'data:'行: {line[:100]}...")
                # 如果收到了 [DONE]，则跳出外层块处理循环
                if returned_successfully:
                    break

            # 循环结束后记录日志
            logging.info(f"[{model_cfg.id}] 流式响应生成循环结束。总块数: {chunk_count}。是否收到[DONE]: {returned_successfully}")

            # 处理循环正常结束但缓冲区仍有数据的情况
            if not returned_successfully and buffer.strip():
                line = buffer.strip()
                if line.startswith('data:'):
                    data_content = line[len('data:'):].strip()
                    if data_content == '[DONE]':
                        returned_successfully = True; logging.info(f"[{model_cfg.id}] 在最终缓冲区中收到 [DONE]。")
                    elif data_content and data_content.startswith("{") and data_content.endswith("}"):
                          yield f"data: {data_content}\n\n"; has_yielded = True; returned_successfully = True; logging.info(f"[{model_cfg.id}] 成功处理最终缓冲区内容。")
                    elif data_content: logging.warning(f"[{model_cfg.id}] 流结束时缓冲区剩余无效数据: {data_content[:100]}...")
                elif buffer.strip(): logging.warning(f"[{model_cfg.id}] 流结束时缓冲区剩余非'data:'数据: {buffer[:100]}...")

        # --- 细化的异常处理 ---
        except aiohttp.ClientPayloadError as e: # 读取响应体时出错
             logging.error(f"[{model_cfg.id}] 流处理 ClientPayloadError: {e}")
             error_payload = {"error": {"message": f"流响应体错误: {e}", "type": "stream_error"}}
             try: yield f"data: {json.dumps(error_payload)}\n\n"
             except Exception: pass
             returned_successfully = False
        except aiohttp.ClientConnectionError as e: # 连接错误 (可能包含 "Connection closed")
             logging.error(f"[{model_cfg.id}] 流处理 ClientConnectionError: {e}")
             error_payload = {"error": {"message": f"流连接错误: {e}", "type": "stream_error"}}
             try: yield f"data: {json.dumps(error_payload)}\n\n"
             except Exception: pass
             returned_successfully = False
        except asyncio.TimeoutError as e: # 读取超时 (sock_read)
             logging.error(f"[{model_cfg.id}] 流处理 TimeoutError: {e}")
             error_payload = {"error": {"message": f"流读取超时: {e}", "type": "stream_error"}}
             try: yield f"data: {json.dumps(error_payload)}\n\n"
             except Exception: pass
             returned_successfully = False
        except aiohttp.ClientError as e: # 其他 aiohttp 客户端错误
            logging.error(f"[{model_cfg.id}] 流处理 ClientError: {e}")
            error_payload = {"error": {"message": f"流客户端错误: {e}", "type": "stream_error"}}
            try: yield f"data: {json.dumps(error_payload)}\n\n"
            except Exception: pass
            returned_successfully = False
        except Exception as e: # 未预料的错误
            logging.exception(f"[{model_cfg.id}] 处理流时发生未知错误", exc_info=e)
            error_payload = {"error": {"message": f"未知流处理错误: {e}", "type": "internal_stream_error"}}
            try: yield f"data: {json.dumps(error_payload)}\n\n"
            except Exception: pass
            returned_successfully = False
        finally:
             # --- 根据处理结果更新模型状态和指标 ---
             response_time = time.time() - start_time
             stream_fully_successful = returned_successfully        # 完全成功 = 收到 [DONE]
             stream_partially_successful = has_yielded and not returned_successfully # 部分成功 = 输出了数据但在 [DONE] 前出错
             stream_failed_early = not has_yielded                # 早期失败 = 未输出任何数据就出错

             if stream_fully_successful:
                 self.dispatcher.mark_model_success(model_cfg.id) # 标记完全成功
                 logging.info(f"[{model_cfg.id}] 流处理完全成功。耗时:{response_time:.2f}s")
             elif stream_partially_successful:
                 # 部分成功视为内容错误，不严厉惩罚模型
                 self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.CONTENT_ERROR)
                 logging.warning(f"[{model_cfg.id}] 流处理部分成功 (在[DONE]前出错)。耗时:{response_time:.2f}s")
             else: # stream_failed_early
                 # 早期失败视为 API 错误，触发退避
                 self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
                 logging.error(f"[{model_cfg.id}] 流处理失败 (未产生数据)。耗时:{response_time:.2f}s")

             # 更新指标，认为只要产生了数据就算某种程度上的成功 (用于计算成功率和平均响应时间)
             self.dispatcher.update_model_metrics(model_cfg.id, response_time, stream_fully_successful or stream_partially_successful)

             # 确保关闭响应连接
             if response and not response.closed:
                 response.close()

    def _extract_content_from_event_stream(self, event_stream_text: str) -> str:
        """从 (意外收到的) 事件流文本中提取内容"""
        full_content = []
        lines = event_stream_text.splitlines()
        for line in lines:
            line = line.strip()
            # 查找 data: 开头且非 [DONE] 的行
            if line.startswith('data:') and not line.endswith('[DONE]'):
                try:
                    data_str = line[len('data:'):].strip()
                    if data_str:
                        data = json.loads(data_str)
                        # 尝试按 OpenAI 流格式提取 delta content
                        if (isinstance(data, dict) and 'choices' in data and isinstance(data['choices'], list) and data['choices'] and
                            isinstance(data['choices'][0], dict) and 'delta' in data['choices'][0] and
                            isinstance(data['choices'][0]['delta'], dict) and 'content' in data['choices'][0]['delta']):
                            content_part = data['choices'][0]['delta']['content']
                            if isinstance(content_part, str):
                                full_content.append(content_part)
                except Exception:
                    # 忽略解析错误或结构不匹配的行
                    logging.debug(f"从意外事件流提取内容时忽略行: {line}")
                    continue
        return ''.join(full_content)

    async def create_chat_completion(self, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, AsyncIterable[str]]:
        """
        处理聊天补全请求的核心逻辑，包括模型选择、重试和结果包装。
        """
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_timestamp = int(time.time())
        start_time_total = time.time()
        max_retries = 3 # 最多重试轮次
        exclude_models: Set[str] = set() # 跟踪本轮已失败的模型

        for attempt in range(max_retries):
            logging.info(f"{request_id} (尝试 {attempt + 1}/{max_retries})")

            # 1. 选择一个可用的模型
            model_cfg = self.get_available_model(request.model, list(exclude_models))
            if not model_cfg:
                # 如果没有可用模型
                error_detail = f"无可用模型满足请求 (请求: {request.model or 'any'}, 已排除: {exclude_models})"
                logging.warning(f"{request_id}: {error_detail}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1)) # 等待一小会再试下一轮
                    continue
                else:
                    # 所有重试轮次都找不到可用模型
                    raise HTTPException(status_code=503, detail="服务当前不可用：无可用模型处理请求。")

            logging.info(f"{request_id}: 尝试使用模型 [{model_cfg.name or model_cfg.id}] (基础权重: {model_cfg.base_weight})")

            try:
                 # 2. 调用 AI API (流式或非流式)
                 # call_ai_api_async 现在会正确返回 Tuple 或 AsyncIterable
                 api_result = await self.call_ai_api_async(
                     model_cfg=model_cfg, messages=request.messages, temperature=request.temperature, response_format=request.response_format,
                     stream=request.stream, stop=request.stop, max_tokens=request.max_tokens, top_p=request.top_p,
                     presence_penalty=request.presence_penalty, frequency_penalty=request.frequency_penalty, logit_bias=request.logit_bias, user=request.user
                 )

                 # 3. 根据请求类型处理结果
                 if request.stream:
                     # 流式请求：期望得到异步迭代器
                     if isinstance(api_result, AsyncIterable):
                          logging.info(f"{request_id}: 从模型 [{model_cfg.name or model_cfg.id}] 返回流式响应。")
                          return api_result # 直接返回给 FastAPI 用于 StreamingResponse
                     else:
                          # 如果类型不匹配，是内部逻辑错误
                          logging.error(f"{request_id}: 内部错误 - 请求流式但 call_ai_api_async 返回了 {type(api_result)}")
                          raise HTTPException(status_code=500, detail="内部服务器错误：流处理意外失败。")
                 else:
                     # 非流式请求：期望得到元组 (content, prompt_tokens, completion_tokens)
                     if isinstance(api_result, tuple) and len(api_result) == 3:
                         content, prompt_tokens, completion_tokens = api_result
                         # 如果 API 未返回 token 数，进行估算
                         if prompt_tokens is None: prompt_tokens = len(''.join(msg.content for msg in request.messages if msg.content)) // 4
                         if completion_tokens is None: completion_tokens = len(content) // 4
                         total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
                         # 构建 OpenAI 格式的响应对象
                         response = ChatCompletionResponse(
                             id=request_id, created=created_timestamp, model=request.model, # 使用用户请求的 model 名称
                             choices=[ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content=content), finish_reason="stop")],
                             usage=ChatCompletionResponseUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
                         )
                         total_request_time = time.time() - start_time_total
                         logging.info(f"{request_id}: 非流式请求成功，模型 [{model_cfg.name or model_cfg.id}]，总耗时: {total_request_time:.2f}s")
                         return response
                     else:
                         # 如果类型不匹配，是内部逻辑错误
                         logging.error(f"{request_id}: 内部错误 - 非流式请求但 call_ai_api_async 返回了 {type(api_result)}")
                         raise HTTPException(status_code=500, detail="内部服务器错误：响应处理意外失败。")

            except HTTPException as e: # 捕获由 call_ai_api_async 抛出的已知 HTTP 错误
                 logging.warning(f"{request_id}: 模型 [{model_cfg.name or model_cfg.id}] 调用失败 (尝试 {attempt + 1}): HTTP {e.status_code} - {e.detail}")
                 exclude_models.add(model_cfg.id) # 本轮不再尝试此模型
                 if attempt == max_retries - 1:
                     # 如果是最后一轮尝试，将错误抛给上层
                     logging.error(f"{request_id}: 所有 {max_retries} 次尝试均失败。最后错误: {e.detail}")
                     raise e
                 else:
                      await asyncio.sleep(0.2 * (attempt + 1)) # 短暂等待后进入下一轮
                      continue
            except Exception as e: # 捕获其他意外错误
                 logging.exception(f"{request_id}: 处理模型 [{model_cfg.name or model_cfg.id}] 时发生未知错误 (尝试 {attempt + 1})", exc_info=e)
                 exclude_models.add(model_cfg.id) # 同样排除此模型
                 if attempt == max_retries - 1:
                     logging.error(f"{request_id}: 所有 {max_retries} 次尝试均因未知错误失败。")
                     # 抛出通用 500 错误
                     raise HTTPException(status_code=500, detail="处理请求时发生内部服务器错误。") from e
                 else:
                      await asyncio.sleep(0.2 * (attempt + 1))
                      continue

        # 如果循环正常结束但没有返回 (理论上不应发生，因为无可用模型会提前抛异常)
        logging.error(f"{request_id}: 未能成功处理请求，所有重试均已尝试。")
        raise HTTPException(status_code=500, detail="请求处理失败，请稍后重试。")

    def get_models_info(self) -> ModelsResponse:
        """获取所有模型的当前状态信息 (用于管理接口)"""
        models_info: List[ModelInfo] = []
        available_count = 0
        # 检查缓存是否需要更新
        with self.dispatcher._rwlock.read_lock():
            needs_update = (time.time() - self.dispatcher._cache_last_update >= self.dispatcher._cache_ttl)
        if needs_update:
            self.dispatcher._update_availability_cache() # 更新缓存

        # 读取模型状态和指标 (需要读锁保证一致性)
        with self.dispatcher._rwlock.read_lock():
            for model in self.models:
                model_id = model.id
                success_rate = self.dispatcher.get_model_success_rate(model_id)
                avg_response_time = self.dispatcher.get_model_avg_response_time(model_id)
                # 从缓存读取可用性
                is_available = self.dispatcher._availability_cache.get(model_id, False)
                if is_available:
                    available_count += 1
                # 添加到结果列表，报告静态基础权重
                models_info.append(ModelInfo(
                    id=model_id, name=model.name, model=model.model,
                    weight=model.base_weight, # 报告 base_weight
                    success_rate=success_rate, avg_response_time=avg_response_time,
                    available=is_available, channel=model.channel_name or model.channel_id
                ))
        return ModelsResponse(models=models_info, total=len(models_info), available=available_count)

    def get_health_info(self) -> HealthResponse:
        """获取服务的整体健康状态"""
        total_models = len(self.models)
        available_count = 0
        # 统计可用模型数量
        for model in self.models:
             if self.dispatcher.is_model_available(model.id):
                  available_count += 1
        # 判断健康状态
        status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        if total_models == 0: status = "unhealthy" # 没有加载模型
        elif available_count == 0: status = "unhealthy" # 有模型但都不可用
        elif available_count < total_models / 2: status = "degraded" # 可用模型少于一半
        # 计算运行时间
        uptime = time.time() - self.start_time
        return HealthResponse(status=status, available_models=available_count, total_models=total_models, uptime=uptime)

    def get_models_list(self) -> List[Dict[str, Any]]:
        """获取符合 OpenAI /v1/models 格式的模型列表 (基于配置)"""
        models_list = []
        seen_exposed_ids = set() # 用于去除重复暴露的模型名称/ID
        for model in self.models:
             # 决定对外暴露的ID (优先用 name, 其次 model, 最后 id)
             exposed_id = model.name or model.model or model.id
             # 只添加权重>0且未添加过的模型
             if model.base_weight > 0 and exposed_id not in seen_exposed_ids:
                  models_list.append({
                      "id": exposed_id,
                      "object": "model",
                      "created": int(self.start_time), # 使用服务启动时间作为创建时间
                      "owned_by": "flux-api" # 归属者标识
                  })
                  seen_exposed_ids.add(exposed_id)
        return models_list

###############################################################################
# FastAPI 应用定义与生命周期管理
###############################################################################
service: Optional[FluxApiService] = None # 全局服务实例变量

@asynccontextmanager
async def lifespan(app_instance: FastAPI): # 使用 lifespan 替代 on_event
    """FastAPI 应用的生命周期管理器"""
    global service # 声明 service 为全局变量
    config_path = os.environ.get("FLUX_API_CONFIG", "./config.yaml")
    print("--> 正在初始化 Flux API 服务...")
    try:
        # 在启动时创建服务实例
        service = FluxApiService(config_path)
        logging.info(f"Flux API Worker (PID: {os.getpid()}) 启动成功。监听地址: http://{os.environ.get('HOST', '0.0.0.0')}:{os.environ.get('PORT', '8787')}")
    except Exception as e:
         # 如果初始化失败，记录严重错误并设置 service 为 None
         logging.exception(f"Flux API Worker (PID: {os.getpid()}) 启动失败: {e}", exc_info=True)
         print(f"致命错误 (PID: {os.getpid()}): Worker 启动失败: {e}", file=sys.stderr)
         service = None
    print("--> Flux API 服务初始化完成。")

    yield # FastAPI 应用在此运行

    # --- 应用关闭时执行 ---
    pid = os.getpid()
    print(f"--> 正在关闭 Flux API Worker (PID: {pid})...")
    if service:
        logging.info(f"Worker (PID: {pid}) 正在关闭。")
        # 此处可以添加服务关闭时需要执行的清理代码 (如果需要)
    else:
        logging.info(f"Worker (PID: {pid}) 关闭 (未完全初始化或已关闭)。")
    print(f"--> Flux API Worker (PID: {pid}) 关闭完成。")

# 创建 FastAPI 应用实例并应用 lifespan 管理器
app = FastAPI(
    title="Flux API",
    description="OpenAI API 兼容网关，管理多模型、负载均衡和故障切换",
    version="1.1.2", # 更新版本号
    lifespan=lifespan
)

# 配置 CORS 中间件
allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins if o.strip()], # 允许的源列表
    allow_credentials=True, # 允许凭证
    allow_methods=["*"],    # 允许所有方法
    allow_headers=["*"],    # 允许所有头部
)

###############################################################################
# 中间件和 API 端点定义
###############################################################################

@app.middleware("http")
async def check_service_availability(request: Request, call_next):
    """中间件：检查服务是否已成功初始化"""
    global service
    # 放行管理接口、文档接口
    if request.url.path.startswith(("/admin", "/docs", "/openapi.json")):
         # 但如果服务未初始化，除了健康检查外，其他管理接口也应返回错误
         if service is None and not request.url.path.startswith("/admin/health"):
              return JSONResponse(status_code=503, content={"detail": "服务正在初始化或启动失败。"})
         return await call_next(request)
    # 对于 API 请求，服务必须已初始化
    if service is None:
        logging.error(f"PID {os.getpid()}: 收到 API 请求但服务未就绪。")
        return JSONResponse(status_code=503, content={"detail": "服务暂时不可用，请稍后重试。"})
    # 服务正常，继续处理请求
    response = await call_next(request)
    return response

# 定义响应类型联合，用于端点类型提示
ResponseType = Union[ChatCompletionResponse, StreamingResponse]

@app.post("/v1/chat/completions", response_model=None) # response_model=None 允许返回 Union 类型
async def create_chat_completion_endpoint(request: ChatCompletionRequest) -> ResponseType:
    """OpenAI 兼容的聊天补全接口"""
    global service
    assert service is not None # 中间件已保证服务可用
    try:
        # 调用核心处理逻辑
        result = await service.create_chat_completion(request)
        # 根据结果类型返回不同响应
        if isinstance(result, AsyncIterable): # 流式响应
             # 添加额外头部和配置，确保保持连接打开
             headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
                "X-Accel-Buffering": "no"  # 禁用Nginx缓冲
             }
             return StreamingResponse(
                content=result, 
                media_type="text/event-stream",
                headers=headers
             )
        elif isinstance(result, ChatCompletionResponse): # 非流式响应
             return result
        else: # 内部错误，类型不符预期
             logging.error(f"服务 create_chat_completion 返回了意外类型: {type(result)}")
             raise HTTPException(status_code=500, detail="内部服务器错误：响应格式不正确。")
    except HTTPException as e:
        # 透传已知的 HTTP 异常
        raise e
    except Exception as e:
        # 捕获其他所有未处理异常
        req_id = f"req-{uuid.uuid4().hex[:8]}" # 生成简短请求ID用于日志追踪
        logging.exception(f"{req_id}: 处理聊天补全请求时发生未捕获错误", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求时发生内部服务器错误 (ID: {req_id})")

@app.get("/v1/models")
async def list_models_endpoint():
    """OpenAI 兼容的模型列表接口"""
    global service
    assert service is not None
    try:
        # 调用服务方法获取模型列表
        models_data = service.get_models_list()
        return {"object": "list", "data": models_data} # 按 OpenAI 格式返回
    except Exception as e:
         logging.exception("获取模型列表时出错")
         raise HTTPException(status_code=500, detail="无法获取模型列表。")

# --- 管理接口 ---
@app.get("/admin/models", response_model=ModelsResponse)
async def get_admin_models():
    """获取所有模型详细状态的管理接口"""
    global service
    assert service is not None
    try:
        return service.get_models_info()
    except Exception as e:
         logging.exception("获取管理模型信息时出错")
         raise HTTPException(status_code=500, detail="无法获取模型状态信息。")

@app.get("/admin/health", response_model=HealthResponse)
async def health_check_endpoint():
    """健康检查接口"""
    global service # 允许在 service 未初始化时也能访问
    if not service:
        # 如果服务实例不存在，直接返回不健康
        return HealthResponse(status="unhealthy", available_models=0, total_models=0, uptime=0)
    try:
        # 调用服务方法获取健康信息
        return service.get_health_info()
    except Exception as e:
         # 如果健康检查本身出错，也返回不健康
         logging.exception("执行健康检查时出错")
         total_models=0; uptime=0
         try: # 尝试获取基本信息，即使出错
             total_models = len(service.models); uptime = time.time() - service.start_time
         except Exception: pass
         return HealthResponse(status="unhealthy", available_models=0, total_models=total_models, uptime=uptime)

@app.get("/")
async def root_endpoint():
    """API 根路径，提供基本信息和文档链接"""
    return {
        "name": "Flux API",
        "version": "1.1.2", # 与 FastAPI 实例中的版本一致
        "description": "OpenAI API 兼容网关，管理多模型、负载均衡和故障切换",
        "documentation": "/docs", # Swagger UI 文档路径
        "openai_compatible_endpoints": [
            {"path": "/v1/chat/completions", "method": "POST"},
            {"path": "/v1/models", "method": "GET"}
        ],
        "admin_endpoints": [
            {"path": "/admin/models", "method": "GET"},
            {"path": "/admin/health", "method": "GET"}
        ]
    }

###############################################################################
# 命令行入口
###############################################################################
def validate_config_file(config_path: str) -> bool:
    """验证配置文件是否存在及基本结构"""
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在: {config_path}", file=sys.stderr)
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
             print(f"错误：配置文件内容不是有效的YAML字典: {config_path}", file=sys.stderr)
             return False
        # 基本结构检查
        if "models" not in config:
            print(f"警告：配置文件缺少 'models' 部分: {config_path}", file=sys.stderr)
        elif not isinstance(config["models"], list):
             print(f"错误：配置文件中的 'models' 必须是一个列表: {config_path}", file=sys.stderr)
             return False
        if "channels" not in config:
             print(f"警告：配置文件缺少 'channels' 部分: {config_path}", file=sys.stderr)
        elif not isinstance(config["channels"], dict):
              print(f"错误：配置文件中的 'channels' 必须是一个字典: {config_path}", file=sys.stderr)
              return False
        return True
    except yaml.YAMLError as e:
        print(f"错误：配置文件YAML解析失败: {config_path}\n{e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"错误：读取或验证配置文件时发生未知错误: {config_path}\n{e}", file=sys.stderr)
        return False

def main():
    """主函数：解析命令行参数并启动 Uvicorn 服务"""
    parser = argparse.ArgumentParser(description="Flux API - OpenAI 兼容的多模型网关")
    parser.add_argument("--config", "-c", default="./config.yaml", help="配置文件路径 (默认: ./config.yaml)")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="监听的主机地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=int(os.environ.get("PORT", 8787)), help="监听的端口号 (默认: 8787)")
    parser.add_argument("--reload", action="store_true", help="启用开发模式下的自动重载 (需要安装 'watchfiles')")
    # 默认工作进程数为 1，以保证模型状态 (退避、限流等) 的一致性
    parser.add_argument("--workers", "-w", type=int, default=int(os.environ.get("WEB_CONCURRENCY", 1)), help="工作进程数 (默认: 1)")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"), help="Uvicorn 的日志级别 (debug, info, warning, error)")

    args = parser.parse_args()

    # 启动前验证配置文件
    if not validate_config_file(args.config):
        sys.exit(1)

    # 设置环境变量供应用内部使用 (例如 lifespan 函数)
    os.environ["FLUX_API_CONFIG"] = args.config
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)

    # 打印启动信息
    print(f"准备启动 Flux API (v1.1.2)...")
    print(f"  配置文件: {args.config}")
    print(f"  监听地址: http://{args.host}:{args.port}")
    # 如果启用了 reload，强制 workers 为 1
    actual_workers = 1 if args.reload else args.workers
    # 如果用户设置了多于1个 worker，给出警告
    if actual_workers != 1:
         print(f"警告: 工作进程数 ({actual_workers}) 大于 1。模型状态 (退避、限流等) 将在各进程间独立，这可能不是预期行为！建议使用 --workers 1 以保证状态一致性。")
    print(f"  工作进程: {actual_workers} {'(自动重载模式强制为 1)' if args.reload and args.workers > 1 else ''}")
    print(f"  日志级别: {args.log_level}")
    print(f"  自动重载: {'启用' if args.reload else '禁用'}")

    # 确定传递给 uvicorn 的应用字符串 (例如 'Flux-Api:app')
    module_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    app_string = f"{module_name}:app"

    # 启动 Uvicorn 服务器
    uvicorn.run(
        app_string,                      # 应用实例位置
        host=args.host,                  # 监听主机
        port=args.port,                  # 监听端口
        reload=args.reload,              # 是否启用自动重载
        workers=actual_workers,          # 工作进程数
        log_level=args.log_level.lower() # 日志级别
    )

if __name__ == "__main__":
    main()
