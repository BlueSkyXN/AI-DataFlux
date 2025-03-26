#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
# 1. 从 pydantic 导入 model_validator
from pydantic import BaseModel, Field, model_validator
import yaml
import json
import re
import aiohttp
import asyncio
import time
import logging
import random
import threading
import os
import sys
import uuid
from datetime import datetime
# 2. 从 typing 导入 AsyncIterable
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Literal, AsyncIterable

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
                # 如果没有为模型配置限流器，默认允许
                logging.warning(f"模型 [{model_id}] 未找到限流器配置，默认允许请求。")
                return True
            return self.limiters[model_id].consume(1.0)

###############################################################################
# ModelDispatcher: 仅对API错误进行退避处理
###############################################################################
class ModelDispatcher:
    def __init__(self, models: List['ModelConfig'], backoff_factor: int = 2): # Use forward reference for ModelConfig
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

            # 计算平均响应时间（指数移动平均）
            total_calls = state["success_count"] + state["error_count"]
            if total_calls == 1:
                state["avg_response_time"] = response_time
            else:
                # 使用更平滑的指数移动平均，避免早期波动过大
                weight = 0.1 # 或者根据需要调整 alpha 值
                state["avg_response_time"] = state["avg_response_time"] * (1 - weight) + response_time * weight

    def get_model_success_rate(self, model_id: str) -> float:
        """获取模型的成功率"""
        with self._rwlock.read_lock():
            if model_id not in self._model_state:
                return 0.0
            state = self._model_state[model_id]
            total = state["success_count"] + state["error_count"]
            if total == 0:
                # 如果没有调用记录，假设成功率为100%
                return 1.0
            return state["success_count"] / total

    def get_model_avg_response_time(self, model_id: str) -> float:
        """获取模型的平均响应时间"""
        with self._rwlock.read_lock():
            if model_id not in self._model_state:
                # 如果没有记录，返回一个默认值，例如1秒
                return 1.0
            # 如果平均响应时间为0（例如刚启动或从未成功），也返回默认值
            return self._model_state[model_id]["avg_response_time"] or 1.0

    def is_model_available(self, model_id: str) -> bool:
        """判断某个模型当前是否可用 - 优先使用缓存，减少锁操作"""
        current_time = time.time()
        # 读锁保护缓存读取
        with self._rwlock.read_lock():
            cache_expired = (current_time - self._cache_last_update >= self._cache_ttl)
            # 如果缓存未过期且模型在缓存中，直接返回缓存结果
            if not cache_expired and model_id in self._availability_cache:
                return self._availability_cache[model_id]

        # 如果缓存过期或模型不在缓存中，需要更新缓存（写锁）
        if cache_expired:
            self._update_availability_cache()
            # 更新后再次读取缓存（读锁）
            with self._rwlock.read_lock():
                 if model_id in self._availability_cache:
                     return self._availability_cache[model_id]

        # 如果模型不存在于状态中（理论上不应发生，除非配置动态变化），认为不可用
        with self._rwlock.read_lock():
            if model_id in self._model_state:
                 st = self._model_state[model_id]
                 return (current_time >= st["next_available_ts"])

        return False # Fallback: Model not tracked

    def mark_model_success(self, model_id: str):
        """模型调用成功时，重置其失败计数"""
        with self._rwlock.write_lock():
            if model_id in self._model_state:
                state = self._model_state[model_id]
                # 只有当模型之前处于失败状态时才打印恢复日志
                was_unavailable = time.time() < state["next_available_ts"]
                state["fail_count"] = 0
                state["next_available_ts"] = 0
                self._availability_cache[model_id] = True # 更新缓存
                if was_unavailable:
                     logging.info(f"模型[{model_id}] 调用成功，恢复可用。")


    def mark_model_failed(self, model_id: str, error_type: str = ErrorType.API_ERROR):
        """模型调用失败时的处理，只有 API_ERROR 才会导致退避"""
        if error_type != ErrorType.API_ERROR:
            # 对于非API错误，只记录日志，不进行退避
            logging.warning(f"模型[{model_id}] 遇到内容或系统错误 ({error_type})，不执行退避。")
            return

        with self._rwlock.write_lock():
            if model_id not in self._model_state:
                return
            st = self._model_state[model_id]
            st["fail_count"] += 1
            fail_count = st["fail_count"]

            # 更温和的退避算法（线性 + 指数的混合）
            if fail_count <= 3:
                backoff_seconds = fail_count * 2 # 2, 4, 6 seconds
            else:
                # 指数增长，但设置上限为60秒
                backoff_seconds = min(
                    6 + (self.backoff_factor ** (fail_count - 3)), # 6 + 2^1, 6 + 2^2, 6 + 2^3 ...
                    60
                )
            st["next_available_ts"] = time.time() + backoff_seconds
            self._availability_cache[model_id] = False # 更新缓存
            logging.warning(
                f"模型[{model_id}] API调用失败，第{fail_count}次，进入退避 {backoff_seconds:.2f} 秒"
            )

    def get_available_models(self, exclude_model_ids: Set[str] = None) -> List[str]:
        """获取所有当前可用的模型ID"""
        exclude_ids = exclude_model_ids or set()

        current_time = time.time()
        with self._rwlock.read_lock():
            cache_expired = current_time - self._cache_last_update >= self._cache_ttl

        if cache_expired:
            self._update_availability_cache()

        available_models = []
        with self._rwlock.read_lock():
            for model_id, is_available in self._availability_cache.items():
                if is_available and model_id not in exclude_ids:
                    available_models.append(model_id)
        return available_models

###############################################################################
# 模型配置类
###############################################################################
class ModelConfig:
    def __init__(self, model_dict: Dict[str, Any], channels: Dict[str, Any]):
        self.id = str(model_dict.get("id")) # Ensure ID is string
        self.name = model_dict.get("name")
        self.model = model_dict.get("model")
        self.channel_id = str(model_dict.get("channel_id"))
        self.api_key = model_dict.get("api_key")
        self.timeout = model_dict.get("timeout", 600)
        self.weight = model_dict.get("weight", 1)
        self.base_weight = self.weight # Store original weight
        self.max_weight = self.base_weight * 2 # Max dynamic weight
        self.temperature = model_dict.get("temperature", 0.7)

        # 添加支持JSON Schema的标志
        self.supports_json_schema = model_dict.get("supports_json_schema", False)

        self.safe_rps = model_dict.get("safe_rps", max(0.5, min(self.base_weight / 10, 10))) # Base RPS on base_weight

        if not self.id or not self.model or not self.channel_id:
            raise ValueError(f"模型配置缺少必填字段: id={self.id}, model={self.model}, channel_id={self.channel_id}")

        if self.channel_id not in channels:
            raise ValueError(f"模型 {self.id}: channel_id='{self.channel_id}' 在channels中不存在")

        channel_cfg = channels[self.channel_id]
        self.channel_name = channel_cfg.get("name")
        self.base_url = channel_cfg.get("base_url")
        self.api_path = channel_cfg.get("api_path", "/v1/chat/completions")
        self.channel_timeout = channel_cfg.get("timeout", 600)
        # Handle proxy configuration more robustly
        proxy_setting = channel_cfg.get("proxy", "")
        self.channel_proxy = proxy_setting if isinstance(proxy_setting, str) and proxy_setting.strip() else None


        if not self.base_url:
             raise ValueError(f"通道 '{self.channel_id}' 缺少 base_url 配置")

        self.final_timeout = min(self.timeout, self.channel_timeout)
        # Separate connect and total timeouts for aiohttp
        self.connect_timeout = 10 # Default connect timeout
        self.read_timeout = self.final_timeout # Total timeout includes connection, reading etc.

###############################################################################
# OpenAI API 兼容的请求与响应模型
###############################################################################
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ResponseFormat(BaseModel):
    type: str = "text"

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None # Let default handling happen later
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1 # Note: n > 1 is often not supported well by backends in streaming mode
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[ResponseFormat] = None

    class Config:
        extra = "allow"  # 允许附加字段

    # 3. 使用 Pydantic V2 风格的 model_validator
    @model_validator(mode='before')
    @classmethod
    def convert_stop_to_list(cls, values):
        """将stop字段转为列表"""
        # Ensure values is a dictionary before accessing keys
        if isinstance(values, dict) and "stop" in values and isinstance(values["stop"], str):
            values["stop"] = [values["stop"]]
        return values

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop" # Can be None in some cases

class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: Optional[int] = None # Make optional as estimation might fail or be disabled
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[ChatCompletionResponseUsage] = None # Make usage optional

class ModelInfo(BaseModel):
    id: str
    name: Optional[str] = None
    model: str
    weight: int
    success_rate: float
    avg_response_time: float
    available: bool
    channel: Optional[str] = None # Channel name might not be set

class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    total: int
    available: int

class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    available_models: int
    total_models: int
    uptime: float

###############################################################################
# API服务主类
###############################################################################
class FluxApiService:
    """OpenAI API 兼容的服务，内部使用模型池"""

    def __init__(self, config_path: str):
        """初始化API服务"""
        self.start_time = time.time()
        self.load_config(config_path)
        self.setup_logging()

        # 初始化模型和调度器
        self.initialize_models()

        # 启动后台任务
        self.start_background_tasks()

        logging.info(f"FluxApiService 初始化完成，加载了 {len(self.models)} 个模型")

    def load_config(self, config_path: str):
        """加载配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            if not isinstance(self.config, dict):
                 raise ValueError("配置文件内容不是有效的YAML字典")
        except FileNotFoundError:
            raise RuntimeError(f"配置文件未找到: {config_path}")
        except yaml.YAMLError as e:
            raise RuntimeError(f"配置文件解析失败: {e}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件时发生未知错误: {e}")


    def setup_logging(self):
        """设置日志"""
        # Ensure config is loaded and is a dictionary
        if not hasattr(self, 'config') or not isinstance(self.config, dict):
             logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
             logging.error("无法加载日志配置，使用默认配置。")
             return

        log_config = self.config.get("global", {}).get("log", {})
        level_str = log_config.get("level", "info").upper()
        level = getattr(logging, level_str, logging.INFO)

        log_format = log_config.get("format_string", "%(asctime)s [%(levelname)s] %(message)s")
        # Handle JSON format logging if specified
        if log_config.get("format") == "json":
            # Basic JSON formatting; consider using a dedicated library like python-json-logger for complex cases
            log_format = '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
            formatter = logging.Formatter(log_format)
        else:
             formatter = logging.Formatter(log_format)

        # Get root logger and remove existing handlers to avoid duplication
        root_logger = logging.getLogger()
        # Ensure root logger has a level set, otherwise handlers won't receive messages
        if not root_logger.hasHandlers():
             root_logger.setLevel(level) # Set level on root logger IF it has no handlers yet

        # Remove existing handlers added by basicConfig or previous runs
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        output_type = log_config.get("output", "console")
        handler: logging.Handler

        if output_type == "file":
            file_path = log_config.get("file_path", "./logs/flux_api.log")
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                handler = logging.FileHandler(file_path, encoding='utf-8')
            except Exception as e:
                handler = logging.StreamHandler(sys.stdout) # Fallback to console
                logging.error(f"无法创建日志文件 {file_path}: {e}。将输出到控制台。")
        else: # Default to console
            handler = logging.StreamHandler(sys.stdout)

        handler.setFormatter(formatter)
        handler.setLevel(level) # Set level on the specific handler
        root_logger.addHandler(handler)
        # Re-apply level to root logger in case it was reset or not set initially
        root_logger.setLevel(level)


    def initialize_models(self):
        """初始化模型配置、调度器和限流器"""
        models_cfg = self.config.get("models", [])
        channels_cfg = self.config.get("channels", {})

        if not isinstance(models_cfg, list):
             raise ValueError("配置文件中的 'models' 必须是一个列表")
        if not isinstance(channels_cfg, dict):
             raise ValueError("配置文件中的 'channels' 必须是一个字典")

        if not models_cfg:
            raise ValueError("配置文件中未找到 'models' 部分或为空")
        if not channels_cfg:
            logging.warning("配置文件中未找到 'channels' 部分或为空") # Channels might not be needed if models have full URLs?

        # 加载模型配置
        model_ids = set()
        self.models: List[ModelConfig] = []
        valid_models_cfg = [] # Store configs that successfully initialize ModelConfig
        for m_dict in models_cfg:
             if not isinstance(m_dict, dict):
                  logging.warning(f"忽略无效的模型配置项（非字典）: {m_dict}")
                  continue
             mid = str(m_dict.get("id")) # Ensure ID is string for lookup
             if not mid:
                  logging.warning(f"忽略缺少 'id' 的模型配置: {m_dict}")
                  continue

             if mid in model_ids:
                 logging.warning(f"发现重复的模型ID: {mid}, 将被忽略。请确保模型ID唯一。")
                 continue # Skip duplicate ID

             try:
                 model_config = ModelConfig(m_dict, channels_cfg)
                 self.models.append(model_config)
                 model_ids.add(mid)
                 valid_models_cfg.append(m_dict) # Add the original dict for rate limiter config
             except ValueError as e:
                 logging.error(f"加载模型配置 {mid} 失败: {e}")

        if not self.models:
             raise RuntimeError("未能成功加载任何模型配置。请检查配置文件。")

        # 初始化调度器和限流器
        concurrency_cfg = self.config.get("global", {}).get("concurrency", {}) # Check under 'global' section
        backoff_factor = concurrency_cfg.get("backoff_factor", 2)

        self.dispatcher = ModelDispatcher(self.models, backoff_factor=backoff_factor)
        self.rate_limiter = ModelRateLimiter()
        # Configure rate limiter using the validated model dictionaries
        self.rate_limiter.configure(valid_models_cfg)

        # 构建模型Map (ID to ModelConfig)
        self.model_map: Dict[str, ModelConfig] = {m.id: m for m in self.models}

        # 创建模型名称 -> 模型ID的映射 (Allow lookup by ID, name, or model identifier)
        self.model_name_to_id: Dict[str, str] = {}
        for m in self.models:
            # Always map the unique ID
            self.model_name_to_id[m.id] = m.id
            # Map the 'model' identifier if present
            if m.model:
                 if m.model in self.model_name_to_id and self.model_name_to_id[m.model] != m.id:
                      logging.warning(f"模型标识符 '{m.model}' 被多个模型 ({self.model_name_to_id[m.model]}, {m.id}) 使用。请求 '{m.model}' 将映射到模型ID '{m.id}'。")
                 self.model_name_to_id[m.model] = m.id
            # Map the 'name' if present and different from 'model'
            if m.name and m.name != m.model:
                 if m.name in self.model_name_to_id and self.model_name_to_id[m.name] != m.id:
                      logging.warning(f"模型名称 '{m.name}' 被多个模型 ({self.model_name_to_id[m.name]}, {m.id}) 使用。请求 '{m.name}' 将映射到模型ID '{m.id}'。")
                 self.model_name_to_id[m.name] = m.id

        # 初始化权重池 - 后续由 adjust_model_weights 动态构建
        self.models_pool = []
        self.adjust_model_weights() # Initial build of the pool

    def start_background_tasks(self):
        """启动后台任务"""
        self.should_stop = False

        # 定期调整模型权重
        self.weight_adjustment_thread = threading.Thread(
            target=self._weight_adjustment_loop,
            daemon=True, # Daemon threads exit when the main program exits
            name="WeightAdjustmentThread"
        )
        self.weight_adjustment_thread.start()

        logging.info("后台权重调整任务已启动")

    def _weight_adjustment_loop(self):
        """周期性调整模型权重的循环"""
        logging.info("权重调整循环开始")
        while not self.should_stop:
            try:
                # 等待一段时间再调整，避免启动时立即调整
                time.sleep(60)  # 每分钟调整一次
                if self.should_stop: break # Check again after sleep
                self.adjust_model_weights()
            except Exception as e:
                logging.error(f"调整模型权重时发生错误: {e}", exc_info=True)
                # Wait longer after an error before retrying
                time.sleep(30)
        logging.info("权重调整循环结束")


    def stop_background_tasks(self):
        """停止后台任务"""
        logging.info("正在停止后台任务...")
        self.should_stop = True
        if hasattr(self, 'weight_adjustment_thread') and self.weight_adjustment_thread.is_alive():
            # Don't wait indefinitely, give it a few seconds
            self.weight_adjustment_thread.join(timeout=5)
            if self.weight_adjustment_thread.is_alive():
                 logging.warning("权重调整线程未能及时停止。")
        logging.info("后台任务已停止")

    def adjust_model_weights(self):
        """动态调整模型的权重，根据成功率、响应时间和可用性，并重建模型池"""
        new_models_pool = []
        adjusted_count = 0
        logging.debug("开始调整模型权重...")

        with self.dispatcher._rwlock.read_lock(): # Need read lock to access metrics
             for model_id, model in self.model_map.items():
                 # 跳过基础权重为0的模型
                 if model.base_weight <= 0:
                     continue

                 success_rate = self.dispatcher.get_model_success_rate(model_id)
                 avg_response_time = self.dispatcher.get_model_avg_response_time(model_id)
                 is_available = self.dispatcher.is_model_available(model_id) # Use dispatcher's check

                 # --- 权重计算逻辑 ---
                 # 基础分 = 基础权重
                 # 可用性惩罚: 如果不可用，权重降为1（或0，如果希望完全禁用）
                 if not is_available:
                      new_weight = 1 # Keep a minimal chance if it recovers quickly
                      # new_weight = 0 # Or disable completely
                 else:
                      # 成功率因子 (指数增加奖励)
                      success_factor = success_rate ** 2 # e.g., 0.9 -> 0.81, 0.7 -> 0.49
                      # 速度因子 (响应时间越短越好，避免除以0)
                      # Normalize response time against a baseline, e.g., 1 second
                      baseline_rt = 1.0
                      speed_factor = baseline_rt / max(0.1, avg_response_time) # Cap minimum RT influence

                      # 综合计算新权重
                      calculated_weight = model.base_weight * success_factor * speed_factor
                      # 限制在新权重在 [1, max_weight] 之间
                      new_weight = int(round(max(1, min(calculated_weight, model.max_weight))))

                 # --- 更新权重和模型池 ---
                 if new_weight != model.weight:
                     logging.info(
                         f"调整模型[{model.name or model_id}]权重: {model.weight} -> {new_weight} "
                         f"(基础={model.base_weight}, 成功率={success_rate:.2f}, RT={avg_response_time:.2f}s, 可用={is_available})"
                     )
                     model.weight = new_weight
                     adjusted_count += 1

                 # 添加到新模型池（即使权重为1）
                 if new_weight > 0:
                     new_models_pool.extend([model] * new_weight) # Add model 'weight' times

        # --- 原子更新模型池 ---
        # This write operation should be quick
        self.models_pool = new_models_pool
        logging.debug(f"模型权重调整完成，{adjusted_count}个模型权重更新，新模型池大小: {len(self.models_pool)}")
        if not self.models_pool:
             logging.warning("调整权重后，模型池为空！所有模型可能都不可用或权重为0。")


    def resolve_model_id(self, model_name_or_id: str) -> Optional[str]:
        """将请求中的模型名称/ID解析为内部模型ID"""
        # 优先精确匹配内部ID, name, 或 model identifier
        if model_name_or_id in self.model_name_to_id:
            return self.model_name_to_id[model_name_or_id]

        # 处理通配符/默认情况 (选择随机可用模型)
        if model_name_or_id.lower() in ["auto", "any", "default", "*", ""]:
             # Let get_available_model handle random selection later
             return None # Indicate random selection is needed

        # 如果找不到匹配的模型
        logging.warning(f"无法解析请求的模型名称或ID: '{model_name_or_id}'")
        return None

    def get_available_model(self, requested_model_name: Optional[str] = None,
                            exclude_models: List[str] = None) -> Optional[ModelConfig]:
        """
        获取一个可用的模型。
        1. 如果指定了模型名，尝试使用该模型（如果可用且未被排除）。
        2. 否则，从可用且未被排除的模型中，根据当前权重随机选择一个。
        会同时考虑 dispatcher 的可用性 和 rate_limiter 的限制。
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
            else:
                 # resolve_model_id returned None, could be 'auto' or truly unknown
                 if requested_model_name.lower() not in ["auto", "any", "default", "*", ""]:
                      # It was an unknown specific model name
                      logging.error(f"请求了未知的模型 '{requested_model_name}'，将尝试随机选择。")
                 # Proceed with random selection

        # 2. 如果指定了目标模型，检查其可用性
        if target_model_id and not use_random_selection:
            if target_model_id not in exclude_set:
                model = self.model_map.get(target_model_id)
                if model:
                    # Check both dispatcher availability and rate limiter
                    is_available = self.dispatcher.is_model_available(target_model_id)
                    can_process = self.rate_limiter.can_process(target_model_id)
                    if is_available and can_process:
                         logging.debug(f"使用请求的可用模型: {model.name or target_model_id}")
                         return model
                    else:
                         logging.warning(f"请求的模型 [{model.name or target_model_id}] 当前不可用 (Available: {is_available}, RateLimitOK: {can_process}) 或已被排除。尝试随机选择。")
                         exclude_set.add(target_model_id) # Exclude it from random selection too
                         use_random_selection = True
                else:
                     logging.error(f"内部错误：解析得到的模型ID '{target_model_id}' 在 model_map 中不存在。")
                     use_random_selection = True # Fallback to random
            else:
                 logging.warning(f"请求的模型 [{target_model_id}] 在排除列表中。尝试随机选择。")
                 use_random_selection = True


        # 3. 如果需要随机选择 (或指定模型不可用)
        if use_random_selection:
            # 使用当前的模型池进行加权随机选择
            # Filter the pool based on dispatcher availability and exclude_set
            # This is potentially slow if the pool is huge and many checks fail.
            # Consider optimizing if this becomes a bottleneck.
            current_pool = self.models_pool # Get snapshot
            if not current_pool:
                 logging.error("模型池为空，无法随机选择模型。")
                 return None

            eligible_models = [
                 model for model in current_pool
                 if model.id not in exclude_set
                 and self.dispatcher.is_model_available(model.id) # Check dispatcher status
                 # Rate limit check happens *here* to filter the random choice
                 and self.rate_limiter.can_process(model.id)
            ]

            if not eligible_models:
                logging.warning(f"没有符合条件的可用模型进行随机选择 (排除: {exclude_set})。")
                # Maybe try again without rate limit check? Or just fail? For now, fail.
                return None

            # random.choice on the filtered (and already weighted) pool
            chosen_model = random.choice(eligible_models)
            logging.debug(f"随机选择了可用模型: {chosen_model.name or chosen_model.id}")
            return chosen_model

        return None # Should not be reached

    # This method might be redundant now as logic is in get_available_model
    # def get_available_model_randomly(self, exclude_model_ids: Set[str] = None) -> Optional[ModelConfig]:
    #     # ... (Implementation based on self.models_pool and checks) ...


    async def call_ai_api_async(self, model_cfg: ModelConfig, messages: List[ChatMessage],
                                temperature: Optional[float] = None,
                                response_format: Optional[ResponseFormat] = None,
                                stream: bool = False,
                                stop: Optional[List[str]] = None,
                                max_tokens: Optional[int] = None,
                                top_p: Optional[float] = None, # Added top_p
                                presence_penalty: Optional[float] = None, # Added presence_penalty
                                frequency_penalty: Optional[float] = None, # Added frequency_penalty
                                logit_bias: Optional[Dict[str, float]] = None, # Added logit_bias
                                user: Optional[str] = None # Added user
                               ) -> Union[str, AsyncIterable[str]]: # 4. Updated return type hint
        """调用AI API并获取响应, 包含更多OpenAI参数"""
        url = model_cfg.base_url.rstrip("/") + model_cfg.api_path
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model_cfg.api_key}"
        }

        # 转换消息格式
        api_messages = [msg.model_dump(exclude_none=True) for msg in messages]
        # api_messages = []
        # for msg in messages:
        #     msg_dict = {"role": msg.role, "content": msg.content}
        #     if msg.name:
        #         msg_dict["name"] = msg.name
        #     api_messages.append(msg_dict)

        # 构建有效载荷
        payload: Dict[str, Any] = {
            "model": model_cfg.model,
            "messages": api_messages,
            "stream": stream
        }

        # 使用请求中的 temperature (如果提供)，否则使用模型配置的
        payload_temp = temperature if temperature is not None else model_cfg.temperature
        # Ensure temperature is within valid range if necessary (some models might error)
        # payload_temp = max(0.0, min(payload_temp, 2.0)) # Example clamping
        payload["temperature"] = payload_temp

        # 添加其他可选参数 (从请求传递)
        if top_p is not None: payload["top_p"] = top_p
        if max_tokens is not None: payload["max_tokens"] = max_tokens
        if stop: payload["stop"] = stop
        if presence_penalty is not None: payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None: payload["frequency_penalty"] = frequency_penalty
        if logit_bias: payload["logit_bias"] = logit_bias
        if user: payload["user"] = user


        # 添加响应格式 (如果模型支持)
        if response_format and model_cfg.supports_json_schema:
             # Ensure we send what the backend expects, e.g., {"type": "json_object"}
             payload["response_format"] = response_format.model_dump()

        proxy = model_cfg.channel_proxy or None # Use None if empty string
        timeout = aiohttp.ClientTimeout(
            connect=model_cfg.connect_timeout,
            # sock_read=model_cfg.read_timeout, # Timeout for individual read operations
            total=model_cfg.read_timeout # Total duration timeout for the entire operation
        )

        logging.debug(f"向模型 [{model_cfg.id}] 发送请求: URL={url}, PayloadKeys={list(payload.keys())}, Timeout={timeout}, Proxy={proxy}")

        start_time = time.time()
        response_status = -1
        response_text_preview = ""

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload, proxy=proxy) as resp:
                    response_status = resp.status
                    # Check for HTTP errors first
                    if response_status >= 400:
                        # Try to read error body for better logging
                        try:
                             error_text = await resp.text()
                             response_text_preview = error_text[:500] # Log preview
                        except Exception as read_err:
                             error_text = f"(无法读取响应体: {read_err})"
                             response_text_preview = error_text

                        logging.error(
                             f"模型 [{model_cfg.id}] API 调用失败: HTTP {response_status}. URL: {url}. 响应预览: {response_text_preview}"
                        )
                        # Raise the original ClientResponseError for proper handling
                        resp.raise_for_status() # This will raise ClientResponseError


                    # --- Handle Successful Response (2xx) ---
                    response_time = time.time() - start_time
                    logging.info(f"模型 [{model_cfg.id}] API 调用成功 (HTTP {response_status})，耗时: {response_time:.2f}s")

                    # Mark success early for non-streaming? Or after processing? Let's do it after processing.

                    # A. 处理流式响应
                    if stream:
                        # Return the async generator directly
                        return self._handle_streaming_response(resp, model_cfg, start_time) # Pass start_time

                    # B. 处理普通 (非流式) 响应
                    else:
                        content_type = resp.headers.get('Content-Type', '').lower()
                        response_text = await resp.text()
                        response_text_preview = response_text[:500] # Log preview

                        try:
                            # Attempt to parse as JSON regardless of Content-Type header
                            data = json.loads(response_text)

                            # Validate OpenAI-like structure
                            if not isinstance(data, dict) or "choices" not in data or not isinstance(data["choices"], list) or not data["choices"]:
                                logging.error(f"模型 [{model_cfg.id}] 返回无效的 JSON 结构 (缺少 choices): {response_text_preview}")
                                raise ValueError("AI返回响应缺少 'choices' 字段或为空")

                            choice = data["choices"][0]
                            if "message" not in choice or "content" not in choice["message"]:
                                 # Handle cases like function calls or empty responses differently?
                                 # For now, assume content is expected.
                                 logging.error(f"模型 [{model_cfg.id}] 返回的 choice[0] 缺少 message.content: {response_text_preview}")
                                 raise ValueError("AI返回响应的 choice[0] 缺少 message.content")

                            content = choice["message"]["content"]

                            # Estimate usage if present in response, otherwise keep basic estimation
                            prompt_tokens = data.get("usage", {}).get("prompt_tokens")
                            completion_tokens = data.get("usage", {}).get("completion_tokens")

                            # --- Successfully processed non-stream response ---
                            response_time = time.time() - start_time # Recalculate precise time
                            self.dispatcher.mark_model_success(model_cfg.id) # Mark success
                            self.dispatcher.update_model_metrics(model_cfg.id, response_time, True) # Update metrics

                            # Return tuple: (content, prompt_tokens, completion_tokens)
                            return content, prompt_tokens, completion_tokens


                        except json.JSONDecodeError as e:
                            # If JSON parsing fails, check if it looks like an event stream
                            # (Some servers might send stream even if stream=False was requested)
                            if 'text/event-stream' in content_type or (response_text.strip().startswith("data:") and "[DONE]" in response_text):
                                logging.warning(f"模型 [{model_cfg.id}] 在非流式请求中返回了事件流格式，尝试提取内容。")
                                content = self._extract_content_from_event_stream(response_text)
                                if content:
                                     response_time = time.time() - start_time
                                     self.dispatcher.mark_model_success(model_cfg.id)
                                     self.dispatcher.update_model_metrics(model_cfg.id, response_time, True)
                                     # Return tuple with None for tokens if extracted from stream
                                     return content, None, None
                                else:
                                     logging.error(f"模型 [{model_cfg.id}] 返回的事件流无法提取有效内容: {response_text_preview}")
                                     raise ValueError("无法从意外的事件流中提取内容") from e
                            else:
                                # Genuine JSON decode error or unexpected format
                                logging.error(
                                    f"模型 [{model_cfg.id}] 响应无法解析为 JSON: {e}. "
                                    f"Content-Type: {content_type}. 响应预览: {response_text_preview}"
                                )
                                raise ValueError(f"响应无法解析为有效JSON") from e
                        except ValueError as e: # Catch validation errors from above
                             # Error already logged, just re-raise
                             raise e


        except aiohttp.ClientResponseError as e: # Handles 4xx/5xx from raise_for_status
             # Error already logged when status >= 400 was detected
             response_time = time.time() - start_time
             # Mark failed only for API errors (like 4xx, 5xx)
             self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
             self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
             # Re-raise the original exception to be handled by the retry loop
             raise HTTPException(status_code=e.status, detail=f"模型API错误: {e.message}") from e

        except asyncio.TimeoutError as e:
            response_time = time.time() - start_time
            logging.error(f"模型 [{model_cfg.id}] API 调用超时 ({model_cfg.read_timeout}s). URL: {url}")
            self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            raise HTTPException(status_code=408, detail="模型请求超时") from e # 408 Request Timeout

        except aiohttp.ClientError as e: # Other connection errors (DNS, Connection refused etc.)
             response_time = time.time() - start_time
             logging.error(f"模型 [{model_cfg.id}] API 连接错误: {e}. URL: {url}")
             self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
             self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
             raise HTTPException(status_code=503, detail=f"模型连接错误: {e}") from e # 503 Service Unavailable

        except ValueError as e: # Catch content processing errors (JSON decode, missing fields)
            response_time = time.time() - start_time
            # Don't back off the model for content errors, but record failure metric
            self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.CONTENT_ERROR)
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            raise HTTPException(status_code=500, detail=f"模型响应处理错误: {e}") from e

        except Exception as e: # Catch-all for unexpected errors during API call
            response_time = time.time() - start_time
            logging.exception(f"模型 [{model_cfg.id}] API 调用时发生未知错误. URL: {url}", exc_info=e) # Log stack trace
            # Treat unexpected errors as potential API issues for backoff
            self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            raise HTTPException(status_code=500, detail=f"调用模型时发生内部错误: {e}") from e


    async def _handle_streaming_response(self, response: aiohttp.ClientResponse, model_cfg: ModelConfig, start_time: float):
        """处理流式响应并包装为符合OpenAI SSE格式的异步生成器"""
        buffer = ""
        last_yield_time = time.time()
        has_yielded = False # Track if we successfully yielded anything

        async def generate() -> AsyncIterable[str]:
            nonlocal buffer, last_yield_time, has_yielded
            returned_successfully = False
            try:
                async for chunk in response.content.iter_any(): # Read whatever is available
                    if not chunk: continue # Skip empty chunks

                    try:
                         decoded_chunk = chunk.decode('utf-8')
                         buffer += decoded_chunk
                    except UnicodeDecodeError:
                         logging.warning(f"模型 [{model_cfg.id}] 流式响应包含无效UTF-8数据，尝试忽略。")
                         buffer += chunk.decode('utf-8', errors='ignore')


                    # Process lines separated by '\n\n' (standard SSE delimiter)
                    # Use splitlines() and handle different line endings potentially
                    lines = buffer.split('\n') # Simpler split, assumes \n is used

                    buffer = lines.pop() # Keep the last potentially incomplete line in buffer

                    for line in lines:
                         line = line.strip()
                         if not line: continue # Skip empty lines

                         # OpenAI SSE format is "data: {...}\n\n"
                         # We yield the JSON string part "{...}" directly
                         if line.startswith('data:'):
                             data_content = line[len('data:'):].strip()
                             if data_content == '[DONE]':
                                  logging.debug(f"模型 [{model_cfg.id}] 流结束 [DONE]")
                                  returned_successfully = True # Mark as successful stream end
                                  # Don't yield [DONE] itself, just stop
                                  break # Exit inner loop
                             elif data_content:
                                 # Basic check if it looks like JSON before yielding
                                 if data_content.startswith("{") and data_content.endswith("}"):
                                      yield f"data: {data_content}\n\n" # Yield in OpenAI SSE format
                                      has_yielded = True
                                      last_yield_time = time.time()
                                 else:
                                      logging.warning(f"模型 [{model_cfg.id}] 流包含非JSON数据块: {data_content[:100]}...")
                         else:
                              logging.debug(f"模型 [{model_cfg.id}] 流包含非'data:'行: {line[:100]}...")

                    if returned_successfully: # If [DONE] was received
                         break # Exit outer async for loop

                # --- End of stream ---
                # Process any remaining data in the buffer after the loop finishes
                if buffer.strip():
                     line = buffer.strip()
                     if line.startswith('data:'):
                         data_content = line[len('data:'):].strip()
                         if data_content == '[DONE]':
                             logging.debug(f"模型 [{model_cfg.id}] 流结束 (末尾) [DONE]")
                             returned_successfully = True
                         elif data_content and data_content.startswith("{") and data_content.endswith("}"):
                              yield f"data: {data_content}\n\n"
                              has_yielded = True
                              returned_successfully = True # If last part yielded, consider success
                         elif data_content: # Log if buffer remains but isn't valid/DONE
                              logging.warning(f"模型 [{model_cfg.id}] 流结束后缓冲区剩余无效数据: {data_content[:100]}...")
                     elif buffer.strip(): # Log if buffer remains but isn't data:
                          logging.warning(f"模型 [{model_cfg.id}] 流结束后缓冲区剩余非'data:'数据: {buffer[:100]}...")


            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logging.error(f"模型 [{model_cfg.id}] 处理流式响应时出错: {e}")
                # Yield an error message in SSE format?
                error_payload = {
                     "error": {
                         "message": f"流处理错误: {e}",
                         "type": "stream_error",
                         "param": None,
                         "code": None
                     }
                 }
                try:
                    yield f"data: {json.dumps(error_payload)}\n\n"
                except Exception: # pragma: no cover
                     pass # Ignore if yielding error fails
                # No success mark here
                returned_successfully = False # Ensure it's marked as failed

            except Exception as e:
                logging.exception(f"模型 [{model_cfg.id}] 处理流式响应时发生未知错误", exc_info=e)
                error_payload = {
                     "error": {
                         "message": f"未知流处理错误: {e}",
                         "type": "internal_stream_error",
                         "param": None,
                         "code": None
                     }
                 }
                try:
                    yield f"data: {json.dumps(error_payload)}\n\n"
                except Exception: # pragma: no cover
                     pass
                returned_successfully = False # Ensure it's marked as failed
            finally:
                 # --- Final status update for the model (after stream ends or fails) ---
                 response_time = time.time() - start_time
                 # Consider stream successful if it ended cleanly OR if it yielded at least some data before error
                 stream_success = returned_successfully or has_yielded

                 if stream_success:
                     # Check if [DONE] was actually seen, or just yielded data then errored
                     if returned_successfully:
                          self.dispatcher.mark_model_success(model_cfg.id)
                          logging.info(f"模型 [{model_cfg.id}] 流式响应处理成功，耗时: {response_time:.2f}s")
                     else:
                          # Yielded data but ended with error - still count as partial success? Or failure?
                          # Let's mark it as failed for metrics, but don't trigger backoff aggressively
                          self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.CONTENT_ERROR) # Treat as content error
                          logging.warning(f"模型 [{model_cfg.id}] 流式响应产生部分数据后出错，耗时: {response_time:.2f}s")
                 else:
                      # No data yielded or stream failed early
                      self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR) # Treat as API error if no data came
                      logging.error(f"模型 [{model_cfg.id}] 流式响应处理失败或未产生数据，耗时: {response_time:.2f}s")

                 # Update metrics regardless of success/failure
                 self.dispatcher.update_model_metrics(model_cfg.id, response_time, stream_success)
                 # Ensure connection is closed
                 if response and not response.closed:
                     response.close()


        return generate()


    def _extract_content_from_event_stream(self, event_stream_text: str) -> str:
        """从意外收到的事件流文本中提取内容 (用于非流式请求的后备)"""
        full_content = []
        lines = event_stream_text.splitlines() # Handles \n, \r\n
        for line in lines:
            line = line.strip()
            if line.startswith('data:') and not line.endswith('[DONE]'):
                try:
                    data_str = line[len('data:'):].strip()
                    if data_str:
                        data = json.loads(data_str)
                        # Check structure for delta content
                        if (isinstance(data, dict) and
                                'choices' in data and isinstance(data['choices'], list) and data['choices'] and
                                isinstance(data['choices'][0], dict) and 'delta' in data['choices'][0] and
                                isinstance(data['choices'][0]['delta'], dict) and 'content' in data['choices'][0]['delta']):
                            content_part = data['choices'][0]['delta']['content']
                            if isinstance(content_part, str):
                                 full_content.append(content_part)
                except json.JSONDecodeError:
                    # Ignore lines that look like data but aren't valid JSON
                    logging.debug(f"从事件流提取内容时忽略无效JSON行: {line}")
                    continue
                except Exception: # Catch other potential errors during extraction
                     logging.debug(f"从事件流提取内容时忽略错误行: {line}")
                     continue

        return ''.join(full_content)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest
        # 5. Updated return type hint for the service method
    ) -> Union[ChatCompletionResponse, AsyncIterable[str]]:
        """
        创建聊天完成 - 符合OpenAI API。
        处理模型选择、调用、重试和格式化响应。
        For streaming requests, returns an AsyncIterable[str].
        For non-streaming, returns a ChatCompletionResponse.
        """
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_timestamp = int(time.time())
        start_time_total = time.time() # Track total time for the request handling

        # 设置重试次数 (e.g., try up to 3 different models)
        max_retries = 3

        # 跟踪已尝试且失败的模型ID
        exclude_models: Set[str] = set()

        # 重试循环 - 每次尝试选择一个不同的可用模型
        for attempt in range(max_retries):
            logging.info(f"聊天完成请求 {request_id} (尝试 {attempt + 1}/{max_retries})")

            # 1. 选择模型
            model_cfg = self.get_available_model(request.model, list(exclude_models))

            if not model_cfg:
                error_detail = f"无可用模型满足请求 (模型: {request.model or 'any'}, 已排除: {exclude_models})"
                logging.warning(f"{request_id}: {error_detail}")

                # 如果不是最后一次尝试，等待一小段时间后再试 (e.g., maybe another model becomes available)
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1)) # Small incremental delay
                    continue
                else:
                    # All retries failed to find a model
                    raise HTTPException(status_code=503, detail="服务当前不可用：无可用模型处理请求")

            logging.info(f"{request_id}: 尝试使用模型 [{model_cfg.name or model_cfg.id}] (权重: {model_cfg.weight})")

            # 2. 调用模型API
            try:
                 api_result = await self.call_ai_api_async(
                     model_cfg=model_cfg,
                     messages=request.messages,
                     # Pass all relevant parameters from the request
                     temperature=request.temperature,
                     response_format=request.response_format,
                     stream=request.stream,
                     stop=request.stop,
                     max_tokens=request.max_tokens,
                     top_p=request.top_p,
                     presence_penalty=request.presence_penalty,
                     frequency_penalty=request.frequency_penalty,
                     logit_bias=request.logit_bias,
                     user=request.user
                 )

                 # --- 处理API调用结果 ---

                 # A. 如果是流式请求，直接返回异步生成器
                 if request.stream:
                     if isinstance(api_result, AsyncIterable):
                          logging.info(f"{request_id}: 模型 [{model_cfg.name or model_cfg.id}] 返回流式响应生成器")
                          # The finally block in _handle_streaming_response handles metrics/status
                          return api_result
                     else:
                          # Should not happen if stream=True was passed correctly
                          logging.error(f"{request_id}: 内部错误 - 请求流式但 call_ai_api_async 未返回 AsyncIterable")
                          raise HTTPException(status_code=500, detail="内部服务器错误：流处理失败")


                 # B. 如果是非流式请求，api_result 应该是 (content, prompt_tokens, completion_tokens)
                 elif isinstance(api_result, tuple) and len(api_result) == 3:
                     content, prompt_tokens, completion_tokens = api_result

                     # 基本的Token数估算 (如果API没返回)
                     if prompt_tokens is None:
                          message_text = ''.join(msg.content for msg in request.messages if msg.content)
                          prompt_tokens = len(message_text) // 4 # Rough estimate (adjust factor as needed)
                     if completion_tokens is None:
                          completion_tokens = len(content) // 4 # Rough estimate

                     total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

                     # 构建OpenAI格式的响应
                     response = ChatCompletionResponse(
                         id=request_id,
                         created=created_timestamp,
                         model=request.model,  # Use the model name requested by the user
                         choices=[
                             ChatCompletionResponseChoice(
                                 index=0,
                                 message=ChatMessage(role="assistant", content=content),
                                 # finish_reason might be available from API in future
                                 finish_reason="stop" # Assume stop for non-stream
                             )
                         ],
                         usage=ChatCompletionResponseUsage(
                             prompt_tokens=prompt_tokens,
                             completion_tokens=completion_tokens,
                             total_tokens=total_tokens
                         )
                     )

                     total_request_time = time.time() - start_time_total
                     logging.info(f"{request_id}: 模型 [{model_cfg.name or model_cfg.id}] 处理成功 (非流式)，总耗时: {total_request_time:.2f}s")
                     return response
                 else:
                     # Should not happen if stream=False was passed correctly
                     logging.error(f"{request_id}: 内部错误 - 非流式请求但 call_ai_api_async 返回类型错误: {type(api_result)}")
                     raise HTTPException(status_code=500, detail="内部服务器错误：响应处理失败")

            except HTTPException as e:
                 # API call failed with a specific HTTP status code (e.g., timeout, connection error, 4xx/5xx from model)
                 logging.warning(f"{request_id}: 模型 [{model_cfg.name or model_cfg.id}] 调用失败 (尝试 {attempt + 1}): HTTP {e.status_code} - {e.detail}")
                 exclude_models.add(model_cfg.id) # Don't retry this specific model instance

                 # If it's the last attempt, re-raise the exception
                 if attempt == max_retries - 1:
                     logging.error(f"{request_id}: 所有 {max_retries} 次尝试均失败。最后错误: {e.detail}")
                     raise e # Re-raise the last HTTPException
                 else:
                      # Wait briefly before next retry
                      await asyncio.sleep(0.2 * (attempt + 1))
                      continue # Go to next attempt

            except Exception as e:
                 # Unexpected error during call_ai_api_async or result processing
                 logging.exception(f"{request_id}: 处理模型 [{model_cfg.name or model_cfg.id}] 调用时发生未知错误 (尝试 {attempt + 1})", exc_info=e)
                 exclude_models.add(model_cfg.id) # Exclude model on unexpected errors too

                 # If it's the last attempt, raise a generic 500 error
                 if attempt == max_retries - 1:
                     logging.error(f"{request_id}: 所有 {max_retries} 次尝试均失败。最后遇到未知错误。")
                     raise HTTPException(status_code=500, detail=f"处理请求时发生内部错误") from e
                 else:
                      await asyncio.sleep(0.2 * (attempt + 1))
                      continue # Go to next attempt


        # Should theoretically not be reached if max_retries >= 1
        logging.error(f"{request_id}: 未能成功处理请求，所有重试均失败。")
        raise HTTPException(status_code=500, detail="请求处理失败，请稍后重试")


    def get_models_info(self) -> ModelsResponse: # Use the Pydantic model
        """获取所有模型及其状态（用于管理接口）"""
        models_info: List[ModelInfo] = []
        available_count = 0

        # Use dispatcher's lock for consistency when reading states
        with self.dispatcher._rwlock.read_lock():
             current_time = time.time()
             # Check cache validity once before the loop
             cache_expired = current_time - self.dispatcher._cache_last_update >= self.dispatcher._cache_ttl
             if cache_expired:
                  # Need to acquire write lock temporarily to update cache
                  # This might be slightly inefficient if called very frequently
                  # Consider a dedicated background task for cache refresh if needed
                  needs_update = True
             else:
                  needs_update = False

        # Temporarily acquire write lock ONLY if update is needed
        if needs_update:
             self.dispatcher._update_availability_cache()


        # Re-acquire read lock to get consistent data
        with self.dispatcher._rwlock.read_lock():
            for model in self.models:
                model_id = model.id
                # Read metrics (already under read lock)
                success_rate = self.dispatcher.get_model_success_rate(model_id)
                avg_response_time = self.dispatcher.get_model_avg_response_time(model_id)
                # Read availability from potentially updated cache
                is_available = self.dispatcher._availability_cache.get(model_id, False)


                if is_available:
                    available_count += 1

                models_info.append(ModelInfo(
                    id=model_id,
                    name=model.name,
                    model=model.model,
                    weight=model.weight, # Current dynamic weight
                    success_rate=success_rate,
                    avg_response_time=avg_response_time,
                    available=is_available,
                    channel=model.channel_name or model.channel_id
                ))

        return ModelsResponse(
            models=models_info,
            total=len(models_info),
            available=available_count
        )

    def get_health_info(self) -> HealthResponse: # Use the Pydantic model
        """获取服务健康状态（用于管理接口）"""
        total_models = len(self.models)
        available_count = 0

        # Use dispatcher's check which uses cache/locks internally
        for model in self.models:
             if self.dispatcher.is_model_available(model.id):
                  available_count += 1

        status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        if available_count == 0 and total_models > 0:
            status = "unhealthy"
        elif total_models > 0 and available_count < total_models / 2:
             # Consider degraded if less than half are available (adjust threshold as needed)
            status = "degraded"
        elif total_models == 0:
             status = "unhealthy" # No models loaded

        uptime = time.time() - self.start_time

        return HealthResponse(
            status=status,
            available_models=available_count,
            total_models=total_models,
            uptime=uptime
        )

    def get_models_list(self) -> List[Dict[str, Any]]:
        """获取符合OpenAI /v1/models 格式的模型列表"""
        models_list = []
        seen_model_names = set() # Track names to show only one entry per unique name/id

        # Iterate through configured models
        for model in self.models:
             # Determine the ID/Name to expose in the list
             # Prioritize name, then model identifier, then internal ID
             exposed_id = model.name or model.model or model.id

             # Only add if the model has >0 base weight and hasn't been added yet
             if model.base_weight > 0 and exposed_id not in seen_model_names:
                  # We list based on configuration, not runtime availability here,
                  # as per OpenAI spec (it lists all compatible models).
                  models_list.append({
                      "id": exposed_id,
                      "object": "model",
                      # Use a fixed creation time or relative time?
                      "created": int(self.start_time), # Or int(time.time()) - random.randint(80000, 90000)
                      "owned_by": "flux-api", # Or use channel name?
                      # Add capabilities if known?
                      # "capabilities": {"embeddings": False, "chat_completions": True}
                  })
                  seen_model_names.add(exposed_id)

        # Optionally add a generic "auto" model if desired
        # if "auto" not in seen_model_names:
        #      models_list.append({
        #           "id": "auto",
        #           "object": "model",
        #           "created": int(self.start_time),
        #           "owned_by": "flux-api"
        #      })

        return models_list

###############################################################################
# FastAPI 应用
###############################################################################
app = FastAPI(
    title="Flux API",
    description="OpenAI API 兼容的服务，使用模型池管理多模型、负载均衡、限流和故障退避",
    version="1.0.0"
)

# 添加CORS中间件
# TODO: Restrict allow_origins in production
allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务实例 (Global variable holding the service instance)
service: Optional[FluxApiService] = None

@app.on_event("startup")
async def startup_event():
    """服务启动时的初始化"""
    global service
    # 配置文件路径从环境变量或命令行参数获取 (handled in main)
    config_path = os.environ.get("FLUX_API_CONFIG", "./config.yaml")
    try:
        service = FluxApiService(config_path)
        logging.info(f"Flux API服务已成功启动。监听地址: http://{os.environ.get('HOST', '0.0.0.0')}:{os.environ.get('PORT', '8000')}")
    except Exception as e:
         # Log the error and prevent startup if initialization fails critically
         logging.exception(f"Flux API服务启动失败: {e}", exc_info=True)
         # Exit the application? Or let it run in a failed state?
         # For robustness, maybe let it run so admin endpoints might work.
         # Or: raise RuntimeError(f"Service failed to initialize: {e}") from e
         print(f"FATAL: Flux API服务启动失败: {e}", file=sys.stderr)
         # Perform a clean exit if initialization fails
         # This might require handling signals depending on how uvicorn runs
         # For simplicity here, just log and potentially the server won't be useful
         # A better approach might involve a state check in middleware or endpoints


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时的清理"""
    global service
    if service:
        logging.info("Flux API服务正在关闭...")
        service.stop_background_tasks()
        logging.info("Flux API服务已关闭")
    else:
        logging.info("Flux API服务关闭（未完全初始化或已关闭）")


# --- Middleware for Service Availability Check ---
@app.middleware("http")
async def check_service_availability(request: Request, call_next):
    global service
    # Allow health/admin checks even if service init failed partially
    if request.url.path.startswith("/admin") or request.url.path == "/docs" or request.url.path == "/openapi.json":
         return await call_next(request)

    # For API calls, ensure service is initialized
    if service is None:
        logging.error("接收到API请求，但服务未初始化或初始化失败。")
        return JSONResponse(
            status_code=503,
            content={"detail": "服务当前不可用，正在初始化或遇到启动错误。"}
        )
    # Optional: Add a check for overall health if needed
    # health = service.get_health_info()
    # if health['status'] == 'unhealthy':
    #     return JSONResponse(status_code=503, content={"detail": "服务当前不可用 (unhealthy)"})

    response = await call_next(request)
    return response

# --- OpenAI API 兼容接口 ---

# Define response model for the endpoint including StreamingResponse
ResponseType = Union[ChatCompletionResponse, StreamingResponse]

@app.post("/v1/chat/completions", response_model=None) # response_model=None to handle Union return
async def create_chat_completion_endpoint(
    request: ChatCompletionRequest,
    # authorization: Optional[str] = Header(None) # Authorization handled by upstream/proxy? Or needed here?
) -> ResponseType: # 6. Endpoint return type hint
    """
    创建聊天完成 - 符合OpenAI API规范。
    支持流式和非流式响应。
    """
    # Service availability checked by middleware
    global service
    if not service: # Should not happen if middleware is effective, but defense-in-depth
         raise HTTPException(status_code=503, detail="服务实例不可用")

    try:
        # Call the service method which returns either ChatCompletionResponse or AsyncIterable
        result = await service.create_chat_completion(request)

        if isinstance(result, AsyncIterable):
             # If it's an async iterable, wrap it in a StreamingResponse
             return StreamingResponse(
                 content=result, # The async generator yielding SSE formatted strings
                 media_type="text/event-stream"
             )
        elif isinstance(result, ChatCompletionResponse):
             # If it's a complete response object, return it directly
             return result
        else:
             # Should not happen based on service method's type hint
             logging.error(f"服务 create_chat_completion 返回了意外类型: {type(result)}")
             raise HTTPException(status_code=500, detail="内部服务器错误：响应格式不正确")

    except HTTPException as e:
         # Re-raise known HTTP exceptions from the service layer
         raise e
    except Exception as e:
        # Catch any unexpected errors during request processing
        request_id_for_log = f"req-{uuid.uuid4().hex[:8]}" # Generate a simple ID for logging correlation
        logging.exception(f"{request_id_for_log}: 处理聊天完成请求时发生未捕获错误: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"处理请求时发生内部服务器错误 (ID: {request_id_for_log})"
        )


@app.get("/v1/models")
async def list_models_endpoint():
    """列出此API代理支持的模型 - 符合OpenAI API规范"""
    # Service availability checked by middleware
    global service
    if not service: raise HTTPException(status_code=503, detail="服务实例不可用")

    try:
         models_data = service.get_models_list()
         return {"object": "list", "data": models_data}
    except Exception as e:
         logging.exception(f"获取模型列表时出错: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="无法获取模型列表")


# --- 管理接口 ---

@app.get("/admin/models", response_model=ModelsResponse)
async def get_admin_models():
    """获取所有配置模型及其当前状态 - 内部管理接口"""
    global service
    if not service: raise HTTPException(status_code=503, detail="服务实例不可用")

    try:
         return service.get_models_info()
    except Exception as e:
         logging.exception(f"获取管理模型信息时出错: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="无法获取模型状态信息")


@app.get("/admin/health", response_model=HealthResponse)
async def health_check_endpoint():
    """健康检查 - 内部管理接口"""
    global service
    # Allow health check even if service isn't fully ready, if possible
    if not service:
         # Return unhealthy if service object doesn't even exist
         return HealthResponse(
              status="unhealthy",
              available_models=0,
              total_models=0, # Cannot determine total if not initialized
              uptime=0
         )

    try:
        return service.get_health_info()
    except Exception as e:
         logging.exception(f"执行健康检查时出错: {e}", exc_info=True)
         # Return unhealthy if health check itself fails
         return HealthResponse(
              status="unhealthy",
              available_models=0,
              total_models=len(service.models) if service and hasattr(service, 'models') else 0,
              uptime=time.time() - service.start_time if service else 0,
              # Add error detail? Pydantic model doesn't support it directly
         )

@app.get("/")
async def root_endpoint():
    """API根路径，提供基本信息"""
    return {
        "name": "Flux API",
        "version": "1.0.0",
        "description": "OpenAI API 兼容的服务，使用模型池管理多模型",
        "documentation": "/docs",
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

        # Basic structure checks
        if "models" not in config:
            print(f"警告：配置文件缺少 'models' 部分: {config_path}", file=sys.stderr)
            # Allow running even without models? Or return False? Let's allow for now.
        elif not isinstance(config["models"], list):
             print(f"错误：配置文件中的 'models' 必须是一个列表: {config_path}", file=sys.stderr)
             return False

        if "channels" not in config:
             print(f"警告：配置文件缺少 'channels' 部分: {config_path}", file=sys.stderr)
        elif not isinstance(config["channels"], dict):
              print(f"错误：配置文件中的 'channels' 必须是一个字典: {config_path}", file=sys.stderr)
              return False

        # Add more validation logic here if needed (e.g., using Pydantic)

        return True
    except yaml.YAMLError as e:
        print(f"错误：配置文件YAML解析失败: {config_path}\n{e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"错误：读取或验证配置文件时发生未知错误: {config_path}\n{e}", file=sys.stderr)
        return False

def main():
    """主函数，解析参数并启动Uvicorn服务"""
    parser = argparse.ArgumentParser(description="Flux API - OpenAI Compatible Multi-Model Gateway")
    parser.add_argument("--config", "-c", default="./config.yaml", help="Path to the configuration file (config.yaml)")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=int(os.environ.get("PORT", 8787)), help="Port number to listen on (default: 8787)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development (requires 'watchfiles' package)")
    parser.add_argument("--workers", "-w", type=int, default=int(os.environ.get("WEB_CONCURRENCY", 300)), help="Number of worker processes (default: 300)")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"), help="Uvicorn log level (e.g., debug, info, warning, error)")

    args = parser.parse_args()

    # Validate config file before starting
    if not validate_config_file(args.config):
        sys.exit(1)

    # Set environment variables needed by the application startup
    os.environ["FLUX_API_CONFIG"] = args.config
    # Pass host/port for logging in startup message
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)

    print(f"准备启动 Flux API...")
    print(f"  配置文件: {args.config}")
    print(f"  监听地址: http://{args.host}:{args.port}")
    print(f"  工作进程: {args.workers}")
    print(f"  日志级别: {args.log_level}")
    print(f"  热重载: {'启用' if args.reload else '禁用'}")


    # Determine app string based on filename
    # This assumes the script is run as 'python flux_api.py' or similar
    module_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    app_string = f"{module_name}:app"

    # Start Uvicorn server
    uvicorn.run(
        app_string,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1, # Workers > 1 not compatible with reload
        log_level=args.log_level.lower()
        # Consider adding access log format options if needed
        # access_log=True,
        # use_colors=True,
    )

if __name__ == "__main__":
    main()