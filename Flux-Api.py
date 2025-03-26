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
import threading # Keep for Locks
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Literal, AsyncIterable

###############################################################################
# 错误类型枚举
###############################################################################
class ErrorType:
    API_ERROR = "api_error"
    CONTENT_ERROR = "content_error"
    SYSTEM_ERROR = "system_error"

###############################################################################
# RWLock
###############################################################################
class RWLock:
    """读写锁实现"""
    # ... (RWLock code - unchanged) ...
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers = 0
        self._write_ready = threading.Condition(threading.Lock())
        self._pending_writers = 0

    def read_acquire(self):
        with self._read_ready:
            while self._writers > 0 or self._pending_writers > 0: self._read_ready.wait()
            self._readers += 1
    def read_release(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0: self._read_ready.notify_all()
    def write_acquire(self):
        with self._write_ready: self._pending_writers += 1
        with self._read_ready:
            while self._readers > 0 or self._writers > 0: self._read_ready.wait()
            self._writers += 1
        with self._write_ready: self._pending_writers -= 1
    def write_release(self):
        with self._read_ready:
            self._writers -= 1
            self._read_ready.notify_all()
    class ReadLock:
        def __init__(self, rwlock): self.rwlock = rwlock
        def __enter__(self): self.rwlock.read_acquire(); return self
        def __exit__(self, exc_type, exc_val, exc_tb): self.rwlock.read_release()
    class WriteLock:
        def __init__(self, rwlock): self.rwlock = rwlock
        def __enter__(self): self.rwlock.write_acquire(); return self
        def __exit__(self, exc_type, exc_val, exc_tb): self.rwlock.write_release()
    def read_lock(self): return self.ReadLock(self)
    def write_lock(self): return self.WriteLock(self)

###############################################################################
# 令牌桶限流器
###############################################################################
class TokenBucket:
    """令牌桶限流器实现"""
    # ... (TokenBucket code - unchanged) ...
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    def refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now
    def consume(self, tokens: float = 1.0) -> bool:
        with self.lock:
            self.refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

###############################################################################
# 模型限流管理器
###############################################################################
class ModelRateLimiter:
    """模型限流管理器"""
    def __init__(self):
        self.limiters = {}
        self.lock = threading.Lock()

    def configure(self, models_config: List[Dict[str, Any]]):
        with self.lock:
            for model in models_config:
                model_id = str(model.get("id"))
                weight = model.get("weight", 1)
                estimated_rps = max(0.5, min(weight / 10, 10))
                safe_rps = model.get("safe_rps", estimated_rps)
                self.limiters[model_id] = TokenBucket(
                    capacity=safe_rps * 2, refill_rate=safe_rps
                )
                # --- MODIFICATION: Log level changed ---
                logging.debug(f"为模型[{model_id}]配置限流: {safe_rps} RPS") # Use DEBUG

    def can_process(self, model_id: str) -> bool:
        model_id = str(model_id)
        with self.lock:
            if model_id not in self.limiters:
                logging.warning(f"模型 [{model_id}] 未找到限流器配置，默认允许请求。")
                return True
            return self.limiters[model_id].consume(1.0)

###############################################################################
# ModelDispatcher
###############################################################################
class ModelDispatcher:
    """模型调度器，处理退避和状态缓存"""
    # ... (ModelDispatcher code - unchanged from previous version without dynamic weights) ...
    def __init__(self, models: List['ModelConfig'], backoff_factor: int = 2):
        self.backoff_factor = backoff_factor
        self._model_state = {}
        for m in models:
            self._model_state[m.id] = {
                "fail_count": 0, "next_available_ts": 0,
                "success_count": 0, "error_count": 0, "avg_response_time": 0
            }
        self._rwlock = RWLock()
        self._availability_cache = {}
        self._cache_last_update = 0
        self._cache_ttl = 0.5
        self._update_availability_cache()

    def _update_availability_cache(self):
        with self._rwlock.write_lock():
            current_time = time.time()
            new_cache = {}
            for model_id, state in self._model_state.items():
                new_cache[model_id] = (current_time >= state["next_available_ts"])
            self._availability_cache = new_cache
            self._cache_last_update = current_time

    def update_model_metrics(self, model_id: str, response_time: float, success: bool):
        with self._rwlock.write_lock():
            state = self._model_state.get(model_id)
            if not state: return
            if success: state["success_count"] += 1
            else: state["error_count"] += 1
            total_calls = state["success_count"] + state["error_count"]
            if total_calls == 1: state["avg_response_time"] = response_time
            else:
                weight = 0.1
                current_avg = state.get("avg_response_time", response_time)
                state["avg_response_time"] = current_avg * (1 - weight) + response_time * weight

    def get_model_success_rate(self, model_id: str) -> float:
        with self._rwlock.read_lock():
            state = self._model_state.get(model_id)
            if not state: return 0.0
            total = state["success_count"] + state["error_count"]
            return state["success_count"] / total if total > 0 else 1.0

    def get_model_avg_response_time(self, model_id: str) -> float:
        with self._rwlock.read_lock():
            state = self._model_state.get(model_id)
            return (state.get("avg_response_time", 1.0) or 1.0) if state else 1.0

    def is_model_available(self, model_id: str) -> bool:
        current_time = time.time()
        with self._rwlock.read_lock():
            cache_expired = (current_time - self._cache_last_update >= self._cache_ttl)
            if not cache_expired and model_id in self._availability_cache:
                return self._availability_cache[model_id]
        if cache_expired:
            self._update_availability_cache()
            with self._rwlock.read_lock():
                return self._availability_cache.get(model_id, False)
        with self._rwlock.read_lock():
            state = self._model_state.get(model_id)
            return (current_time >= state["next_available_ts"]) if state else False

    def mark_model_success(self, model_id: str):
        with self._rwlock.write_lock():
            state = self._model_state.get(model_id)
            if state:
                was_unavailable = time.time() < state["next_available_ts"]
                state["fail_count"] = 0
                state["next_available_ts"] = 0
                self._availability_cache[model_id] = True
                if was_unavailable: logging.info(f"模型[{model_id}] 调用成功，恢复可用。")

    def mark_model_failed(self, model_id: str, error_type: str = ErrorType.API_ERROR):
        if error_type != ErrorType.API_ERROR:
            logging.warning(f"模型[{model_id}] 遇到内容或系统错误 ({error_type})，不执行退避。")
            return
        with self._rwlock.write_lock():
            state = self._model_state.get(model_id)
            if not state: return
            state["fail_count"] += 1
            fail_count = state["fail_count"]
            if fail_count <= 3: backoff_seconds = fail_count * 2
            else: backoff_seconds = min(6 + (self.backoff_factor ** (fail_count - 3)), 60)
            state["next_available_ts"] = time.time() + backoff_seconds
            self._availability_cache[model_id] = False
            logging.warning(f"模型[{model_id}] API调用失败，第{fail_count}次，进入退避 {backoff_seconds:.2f} 秒")

    def get_available_models(self, exclude_model_ids: Set[str] = None) -> List[str]:
        exclude_ids = exclude_model_ids or set()
        available_models = []
        with self._rwlock.read_lock(): all_model_ids = list(self._model_state.keys())
        for model_id in all_model_ids:
             if model_id not in exclude_ids and self.is_model_available(model_id):
                  available_models.append(model_id)
        return available_models


###############################################################################
# 模型配置类
###############################################################################
class ModelConfig:
    """模型配置"""
    # ... (ModelConfig code - unchanged, uses base_weight) ...
    def __init__(self, model_dict: Dict[str, Any], channels: Dict[str, Any]):
        self.id = str(model_dict.get("id"))
        self.name = model_dict.get("name")
        self.model = model_dict.get("model")
        self.channel_id = str(model_dict.get("channel_id"))
        self.api_key = model_dict.get("api_key")
        self.timeout = model_dict.get("timeout", 600)
        self.base_weight = model_dict.get("weight", 1) # Static weight
        self.temperature = model_dict.get("temperature", 0.7)
        self.supports_json_schema = model_dict.get("supports_json_schema", False)
        self.safe_rps = model_dict.get("safe_rps", max(0.5, min(self.base_weight / 10, 10)))

        if not self.id or not self.model or not self.channel_id:
            raise ValueError(f"模型配置缺少必填字段: id={self.id}, model={self.model}, channel_id={self.channel_id}")
        if self.channel_id not in channels:
            raise ValueError(f"模型 {self.id}: channel_id='{self.channel_id}' 在channels中不存在")

        channel_cfg = channels[self.channel_id]
        self.channel_name = channel_cfg.get("name")
        self.base_url = channel_cfg.get("base_url")
        self.api_path = channel_cfg.get("api_path", "/v1/chat/completions")
        self.channel_timeout = channel_cfg.get("timeout", 600)
        proxy_setting = channel_cfg.get("proxy", "")
        self.channel_proxy = proxy_setting if isinstance(proxy_setting, str) and proxy_setting.strip() else None
        if not self.base_url: raise ValueError(f"通道 '{self.channel_id}' 缺少 base_url 配置")

        self.final_timeout = min(self.timeout, self.channel_timeout)
        self.connect_timeout = 10
        self.read_timeout = self.final_timeout

###############################################################################
# OpenAI API 兼容的请求与响应模型
###############################################################################
# ... (Pydantic models - ChatMessage, ResponseFormat, etc. - unchanged) ...
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ResponseFormat(BaseModel):
    type: str = "text"

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[ResponseFormat] = None
    class Config: extra = "allow"
    @model_validator(mode='before')
    @classmethod
    def convert_stop_to_list(cls, values):
        if isinstance(values, dict) and "stop" in values and isinstance(values["stop"], str):
            values["stop"] = [values["stop"]]
        return values

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[ChatCompletionResponseUsage] = None

class ModelInfo(BaseModel):
    id: str
    name: Optional[str] = None
    model: str
    weight: int # Represents base_weight now
    success_rate: float
    avg_response_time: float
    available: bool
    channel: Optional[str] = None

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
    """OpenAI API 兼容的服务"""
    def __init__(self, config_path: str):
        self.start_time = time.time()
        self.load_config(config_path)
        self.setup_logging()
        self.initialize_models()
        # --- MODIFICATION: Background tasks removed ---
        logging.info(f"FluxApiService 初始化完成，加载了 {len(self.models)} 个模型")

    def load_config(self, config_path: str):
        # ... (load_config - unchanged) ...
        try:
            with open(config_path, "r", encoding="utf-8") as f: self.config = yaml.safe_load(f)
            if not isinstance(self.config, dict): raise ValueError("配置文件内容不是有效的YAML字典")
        except FileNotFoundError: raise RuntimeError(f"配置文件未找到: {config_path}")
        except yaml.YAMLError as e: raise RuntimeError(f"配置文件解析失败: {e}")
        except Exception as e: raise RuntimeError(f"加载配置文件时发生未知错误: {e}")

    def setup_logging(self):
        # ... (setup_logging - unchanged) ...
        if not hasattr(self, 'config') or not isinstance(self.config, dict):
             logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
             logging.error("无法加载日志配置，使用默认配置。")
             return
        log_config = self.config.get("global", {}).get("log", {})
        level_str = log_config.get("level", "info").upper()
        level = getattr(logging, level_str, logging.INFO)
        log_format = log_config.get("format_string", "%(asctime)s [%(levelname)s] %(message)s")
        if log_config.get("format") == "json":
            log_format = '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
            formatter = logging.Formatter(log_format)
        else: formatter = logging.Formatter(log_format)
        root_logger = logging.getLogger()
        if not root_logger.hasHandlers(): root_logger.setLevel(level)
        for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
        output_type = log_config.get("output", "console")
        handler: logging.Handler
        if output_type == "file":
            file_path = log_config.get("file_path", "./logs/flux_api.log")
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                handler = logging.FileHandler(file_path, encoding='utf-8')
            except Exception as e:
                handler = logging.StreamHandler(sys.stdout); logging.error(f"无法创建日志文件 {file_path}: {e}。将输出到控制台。")
        else: handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter); handler.setLevel(level); root_logger.addHandler(handler); root_logger.setLevel(level)

    def initialize_models(self):
        """初始化模型配置、调度器和限流器"""
        # ... (initialize_models - unchanged from previous version without dynamic weights) ...
        models_cfg = self.config.get("models", [])
        channels_cfg = self.config.get("channels", {})
        if not isinstance(models_cfg, list): raise ValueError("'models' must be a list")
        if not isinstance(channels_cfg, dict): raise ValueError("'channels' must be a dictionary")
        if not models_cfg: raise ValueError("No 'models' found in config or empty")
        if not channels_cfg: logging.warning("No 'channels' found in config or empty")

        model_ids = set()
        self.models: List[ModelConfig] = []
        valid_models_cfg = []
        for m_dict in models_cfg:
             if not isinstance(m_dict, dict): logging.warning(f"Ignoring invalid model config item: {m_dict}"); continue
             mid = str(m_dict.get("id"))
             if not mid: logging.warning(f"Ignoring model config missing 'id': {m_dict}"); continue
             if mid in model_ids: logging.debug(f"Duplicate model ID found: {mid} (ignored)."); continue # Use DEBUG
             try:
                 model_config = ModelConfig(m_dict, channels_cfg)
                 self.models.append(model_config)
                 model_ids.add(mid)
                 valid_models_cfg.append(m_dict)
             except ValueError as e: logging.error(f"Failed to load model config {mid}: {e}")
        if not self.models: raise RuntimeError("No valid model configurations loaded.")

        concurrency_cfg = self.config.get("global", {}).get("concurrency", {})
        backoff_factor = concurrency_cfg.get("backoff_factor", 2)
        self.dispatcher = ModelDispatcher(self.models, backoff_factor=backoff_factor)
        self.rate_limiter = ModelRateLimiter()
        self.rate_limiter.configure(valid_models_cfg) # Uses DEBUG log level inside

        self.model_map: Dict[str, ModelConfig] = {m.id: m for m in self.models}
        self.model_name_to_id: Dict[str, str] = {}
        for m in self.models:
            self.model_name_to_id[m.id] = m.id
            if m.model:
                 if m.model in self.model_name_to_id and self.model_name_to_id[m.model] != m.id:
                      logging.debug(f"Model identifier '{m.model}' used by multiple models.") # Use DEBUG
                 self.model_name_to_id[m.model] = m.id
            if m.name and m.name != m.model:
                 if m.name in self.model_name_to_id and self.model_name_to_id[m.name] != m.id:
                      logging.debug(f"Model name '{m.name}' used by multiple models.") # Use DEBUG
                 self.model_name_to_id[m.name] = m.id

        # Build static weighted pool based on base_weight
        self.models_pool = []
        for model in self.models:
             if model.base_weight > 0: self.models_pool.extend([model] * model.base_weight)
        if not self.models_pool: logging.warning("Model pool is empty after initialization!")
        logging.debug(f"Initialized static model pool with size: {len(self.models_pool)}")

    def resolve_model_id(self, model_name_or_id: str) -> Optional[str]:
        # ... (resolve_model_id - unchanged) ...
        if model_name_or_id in self.model_name_to_id: return self.model_name_to_id[model_name_or_id]
        if model_name_or_id.lower() in ["auto", "any", "default", "*", ""]: return None
        logging.warning(f"Unable to resolve requested model name/ID: '{model_name_or_id}'")
        return None

    def get_available_model(self, requested_model_name: Optional[str] = None,
                            exclude_models: List[str] = None) -> Optional[ModelConfig]:
        """获取可用模型，使用静态权重池"""
        # ... (get_available_model - unchanged, uses static pool) ...
        exclude_set = set(exclude_models or [])
        target_model_id: Optional[str] = None
        use_random_selection = True

        if requested_model_name:
            resolved_id = self.resolve_model_id(requested_model_name)
            if resolved_id:
                target_model_id = resolved_id; use_random_selection = False
            elif requested_model_name.lower() not in ["auto", "any", "default", "*", ""]:
                 logging.error(f"Requested unknown model '{requested_model_name}', attempting random.")

        if target_model_id and not use_random_selection:
            if target_model_id not in exclude_set:
                model = self.model_map.get(target_model_id)
                if model:
                    is_available = self.dispatcher.is_model_available(target_model_id)
                    can_process = self.rate_limiter.can_process(target_model_id)
                    if is_available and can_process:
                         logging.debug(f"Using requested available model: {model.name or target_model_id}")
                         return model
                    else:
                         logging.warning(f"Requested model [{model.name or target_model_id}] unavailable/limited. Attempting random.")
                         exclude_set.add(target_model_id); use_random_selection = True
                else:
                     logging.error(f"Internal error: Resolved ID '{target_model_id}' not in map."); use_random_selection = True
            else:
                 logging.warning(f"Requested model [{target_model_id}] in exclude list. Attempting random."); use_random_selection = True

        if use_random_selection:
            current_pool = self.models_pool
            if not current_pool: logging.error("Model pool empty."); return None
            eligible_models_in_pool = [m for m in current_pool if m.id not in exclude_set and self.dispatcher.is_model_available(m.id) and self.rate_limiter.can_process(m.id)]
            if not eligible_models_in_pool: logging.warning(f"No eligible models available (Excluded: {exclude_set})."); return None
            chosen_model = random.choice(eligible_models_in_pool)
            logging.debug(f"Randomly selected model: {chosen_model.name or chosen_model.id} (BaseWeight: {chosen_model.base_weight})")
            return chosen_model
        return None


    async def call_ai_api_async(self, model_cfg: ModelConfig, messages: List[ChatMessage],
                                temperature: Optional[float] = None,
                                response_format: Optional[ResponseFormat] = None,
                                stream: bool = False,
                                stop: Optional[List[str]] = None,
                                max_tokens: Optional[int] = None,
                                top_p: Optional[float] = None,
                                presence_penalty: Optional[float] = None,
                                frequency_penalty: Optional[float] = None,
                                logit_bias: Optional[Dict[str, float]] = None,
                                user: Optional[str] = None
                               # --- MODIFICATION: Correct return type hint ---
                               ) -> Union[Tuple[str, Optional[int], Optional[int]], AsyncIterable[str]]:
        """调用AI API，处理流式和非流式"""
        url = model_cfg.base_url.rstrip("/") + model_cfg.api_path
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {model_cfg.api_key}"}
        api_messages = [msg.model_dump(exclude_none=True) for msg in messages]
        payload: Dict[str, Any] = {"model": model_cfg.model, "messages": api_messages, "stream": stream}
        payload_temp = temperature if temperature is not None else model_cfg.temperature
        payload["temperature"] = payload_temp
        if top_p is not None: payload["top_p"] = top_p
        if max_tokens is not None: payload["max_tokens"] = max_tokens
        if stop: payload["stop"] = stop
        if presence_penalty is not None: payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None: payload["frequency_penalty"] = frequency_penalty
        if logit_bias: payload["logit_bias"] = logit_bias
        if user: payload["user"] = user
        if response_format and model_cfg.supports_json_schema:
             payload["response_format"] = response_format.model_dump()

        proxy = model_cfg.channel_proxy or None
        timeout = aiohttp.ClientTimeout(connect=model_cfg.connect_timeout, total=model_cfg.read_timeout)
        logging.debug(f"Sending request to [{model_cfg.id}]: URL={url}, PayloadKeys={list(payload.keys())}, Timeout={timeout}, Proxy={proxy}")
        start_time = time.time()
        response_status = -1
        response_text_preview = ""

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload, proxy=proxy) as resp:
                    response_status = resp.status
                    if response_status >= 400:
                        try: error_text = await resp.text(); response_text_preview = error_text[:500]
                        except Exception as read_err: error_text = f"(无法读取响应体: {read_err})"; response_text_preview = error_text
                        logging.error(f"模型 [{model_cfg.id}] API失败: HTTP {response_status}. URL: {url}. 预览: {response_text_preview}")
                        resp.raise_for_status()

                    response_time = time.time() - start_time
                    logging.info(f"模型 [{model_cfg.id}] API成功 (HTTP {response_status})，耗时: {response_time:.2f}s")

                    # --- MODIFICATION: Correct handling of streaming vs non-streaming return ---
                    if stream:
                        # Await the coroutine that returns the async generator
                        return await self._handle_streaming_response(resp, model_cfg, start_time)
                    else:
                        # Non-streaming: Process and return tuple
                        content_type = resp.headers.get('Content-Type', '').lower()
                        response_text = await resp.text()
                        response_text_preview = response_text[:500]
                        try:
                            data = json.loads(response_text)
                            if not isinstance(data, dict) or "choices" not in data or not isinstance(data["choices"], list) or not data["choices"]:
                                raise ValueError("AI响应缺少 'choices' 字段或为空")
                            choice = data["choices"][0]
                            if "message" not in choice or "content" not in choice["message"]:
                                 raise ValueError("AI响应 choice[0] 缺少 message.content")
                            content = choice["message"]["content"]
                            prompt_tokens = data.get("usage", {}).get("prompt_tokens")
                            completion_tokens = data.get("usage", {}).get("completion_tokens")

                            self.dispatcher.mark_model_success(model_cfg.id)
                            self.dispatcher.update_model_metrics(model_cfg.id, response_time, True)
                            return content, prompt_tokens, completion_tokens
                        except json.JSONDecodeError as e:
                            if 'text/event-stream' in content_type or (response_text.strip().startswith("data:") and "[DONE]" in response_text):
                                logging.warning(f"模型 [{model_cfg.id}] 在非流请求中返回事件流，尝试提取。")
                                content = self._extract_content_from_event_stream(response_text)
                                if content:
                                     self.dispatcher.mark_model_success(model_cfg.id)
                                     self.dispatcher.update_model_metrics(model_cfg.id, response_time, True)
                                     return content, None, None
                                else:
                                     raise ValueError("无法从意外事件流中提取内容") from e
                            else:
                                raise ValueError(f"响应无法解析为有效JSON") from e
                        except ValueError as e: raise e

        except aiohttp.ClientResponseError as e:
             response_time = time.time() - start_time
             self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
             self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
             raise HTTPException(status_code=e.status, detail=f"模型API错误: {e.message}") from e
        except asyncio.TimeoutError as e:
            response_time = time.time() - start_time
            logging.error(f"模型 [{model_cfg.id}] API 调用超时 ({model_cfg.read_timeout}s). URL: {url}")
            self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            raise HTTPException(status_code=408, detail="模型请求超时") from e
        except aiohttp.ClientError as e:
             response_time = time.time() - start_time
             logging.error(f"模型 [{model_cfg.id}] API 连接错误: {e}. URL: {url}")
             self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR)
             self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
             raise HTTPException(status_code=503, detail=f"模型连接错误: {e}") from e
        except ValueError as e: # Catches JSON errors or validation errors from non-streaming path
            response_time = time.time() - start_time
            self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.CONTENT_ERROR) # Mark as content error
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            # Raise 500 for content processing issues on server side
            raise HTTPException(status_code=500, detail=f"模型响应处理错误: {e}") from e
        except Exception as e:
            response_time = time.time() - start_time
            logging.exception(f"模型 [{model_cfg.id}] API 调用时发生未知错误. URL: {url}", exc_info=e)
            self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR) # Assume API error for unknown
            self.dispatcher.update_model_metrics(model_cfg.id, response_time, False)
            raise HTTPException(status_code=500, detail=f"调用模型时发生内部错误: {e}") from e

    async def _handle_streaming_response(self, response: aiohttp.ClientResponse, model_cfg: ModelConfig, start_time: float) -> AsyncIterable[str]:
        """处理流式响应并返回异步生成器对象"""
        # ... (_handle_streaming_response - unchanged, returns generator object) ...
        buffer = ""
        has_yielded = False
        async def generate() -> AsyncIterable[str]: # This inner function is the generator
            nonlocal buffer, has_yielded
            returned_successfully = False
            try:
                async for chunk in response.content.iter_any():
                    if not chunk: continue
                    try: decoded_chunk = chunk.decode('utf-8'); buffer += decoded_chunk
                    except UnicodeDecodeError: buffer += chunk.decode('utf-8', errors='ignore'); logging.warning(f"[{model_cfg.id}] Stream contained invalid UTF-8.")
                    lines = buffer.split('\n'); buffer = lines.pop()
                    for line in lines:
                        line = line.strip()
                        if not line: continue
                        if line.startswith('data:'):
                            data_content = line[len('data:'):].strip()
                            if data_content == '[DONE]': returned_successfully = True; break
                            elif data_content:
                                if data_content.startswith("{") and data_content.endswith("}"):
                                    yield f"data: {data_content}\n\n"; has_yielded = True
                                else: logging.warning(f"[{model_cfg.id}] Stream non-JSON data: {data_content[:100]}...")
                        else: logging.debug(f"[{model_cfg.id}] Stream non-'data:' line: {line[:100]}...")
                    if returned_successfully: break
                if buffer.strip(): # Process remaining buffer
                     line = buffer.strip()
                     if line.startswith('data:'):
                         data_content = line[len('data:'):].strip()
                         if data_content == '[DONE]': returned_successfully = True
                         elif data_content and data_content.startswith("{") and data_content.endswith("}"):
                              yield f"data: {data_content}\n\n"; has_yielded = True; returned_successfully = True
                         elif data_content: logging.warning(f"[{model_cfg.id}] Stream ended with invalid buffer: {data_content[:100]}...")
                     elif buffer.strip(): logging.warning(f"[{model_cfg.id}] Stream ended with non-'data:' buffer: {buffer[:100]}...")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logging.error(f"[{model_cfg.id}] Stream processing error: {e}")
                try: yield f"data: {json.dumps({'error': {'message': f'Stream error: {e}', 'type': 'stream_error'}})}\n\n"
                except Exception: pass
                returned_successfully = False
            except Exception as e:
                logging.exception(f"[{model_cfg.id}] Unknown stream processing error", exc_info=e)
                try: yield f"data: {json.dumps({'error': {'message': f'Unknown stream error: {e}', 'type': 'internal_stream_error'}})}\n\n"
                except Exception: pass
                returned_successfully = False
            finally: # Final status update
                 response_time = time.time() - start_time
                 stream_success = returned_successfully or has_yielded
                 if stream_success:
                     if returned_successfully: self.dispatcher.mark_model_success(model_cfg.id); logging.info(f"[{model_cfg.id}] Stream success ({response_time:.2f}s)")
                     else: self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.CONTENT_ERROR); logging.warning(f"[{model_cfg.id}] Stream partial then error ({response_time:.2f}s)")
                 else: self.dispatcher.mark_model_failed(model_cfg.id, ErrorType.API_ERROR); logging.error(f"[{model_cfg.id}] Stream failed/no data ({response_time:.2f}s)")
                 self.dispatcher.update_model_metrics(model_cfg.id, response_time, stream_success)
                 if response and not response.closed: response.close()
        # Return the generator *object* created by calling the inner function
        return generate()

    def _extract_content_from_event_stream(self, event_stream_text: str) -> str:
        """从事件流文本中提取内容"""
        # ... (_extract_content_from_event_stream - unchanged) ...
        full_content = []
        lines = event_stream_text.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith('data:') and not line.endswith('[DONE]'):
                try:
                    data_str = line[len('data:'):].strip()
                    if data_str:
                        data = json.loads(data_str)
                        if (isinstance(data, dict) and 'choices' in data and isinstance(data['choices'], list) and data['choices'] and
                            isinstance(data['choices'][0], dict) and 'delta' in data['choices'][0] and
                            isinstance(data['choices'][0]['delta'], dict) and 'content' in data['choices'][0]['delta']):
                            content_part = data['choices'][0]['delta']['content']
                            if isinstance(content_part, str): full_content.append(content_part)
                except Exception: logging.debug(f"Ignoring line during stream extraction: {line}"); continue
        return ''.join(full_content)

    async def create_chat_completion(self, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, AsyncIterable[str]]:
        """创建聊天完成，处理重试和结果格式化"""
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_timestamp = int(time.time())
        start_time_total = time.time()
        max_retries = 3
        exclude_models: Set[str] = set()

        for attempt in range(max_retries):
            logging.info(f"{request_id} (Attempt {attempt + 1}/{max_retries})")
            model_cfg = self.get_available_model(request.model, list(exclude_models))
            if not model_cfg:
                error_detail = f"No available models (Requested: {request.model or 'any'}, Excluded: {exclude_models})"
                logging.warning(f"{request_id}: {error_detail}")
                if attempt < max_retries - 1: await asyncio.sleep(0.5 * (attempt + 1)); continue
                else: raise HTTPException(status_code=503, detail="Service Unavailable: No models available.")

            logging.info(f"{request_id}: Trying model [{model_cfg.name or model_cfg.id}] (BaseWeight: {model_cfg.base_weight})")

            try:
                 # This await now correctly gets either the tuple or the AsyncIterable
                 api_result = await self.call_ai_api_async(
                     model_cfg=model_cfg, messages=request.messages, temperature=request.temperature,
                     response_format=request.response_format, stream=request.stream, stop=request.stop,
                     max_tokens=request.max_tokens, top_p=request.top_p, presence_penalty=request.presence_penalty,
                     frequency_penalty=request.frequency_penalty, logit_bias=request.logit_bias, user=request.user
                 )

                 # --- Process result based on request.stream ---
                 if request.stream:
                     # Expect AsyncIterable for streaming
                     if isinstance(api_result, AsyncIterable):
                          logging.info(f"{request_id}: Returning stream from [{model_cfg.name or model_cfg.id}]")
                          return api_result
                     else:
                          # This *shouldn't* happen now if call_ai_api_async is correct
                          logging.error(f"{request_id}: Internal Error - Stream requested but call_ai_api_async returned {type(api_result)}")
                          raise HTTPException(status_code=500, detail="Internal Server Error: Stream processing failed unexpectedly.")
                 else:
                     # Expect Tuple for non-streaming
                     if isinstance(api_result, tuple) and len(api_result) == 3:
                         content, prompt_tokens, completion_tokens = api_result
                         # Estimate tokens if needed
                         if prompt_tokens is None: prompt_tokens = len(''.join(msg.content for msg in request.messages if msg.content)) // 4
                         if completion_tokens is None: completion_tokens = len(content) // 4
                         total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
                         # Build and return response object
                         response = ChatCompletionResponse(
                             id=request_id, created=created_timestamp, model=request.model,
                             choices=[ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content=content), finish_reason="stop")],
                             usage=ChatCompletionResponseUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
                         )
                         total_request_time = time.time() - start_time_total
                         logging.info(f"{request_id}: Success (non-stream) via [{model_cfg.name or model_cfg.id}], time: {total_request_time:.2f}s")
                         return response
                     else:
                         # This *shouldn't* happen now if call_ai_api_async is correct
                         logging.error(f"{request_id}: Internal Error - Non-stream request but call_ai_api_async returned {type(api_result)}")
                         raise HTTPException(status_code=500, detail="Internal Server Error: Response processing failed unexpectedly.")

            except HTTPException as e: # Handle errors raised by call_ai_api_async or here
                 logging.warning(f"{request_id}: Model [{model_cfg.name or model_cfg.id}] failed (Attempt {attempt + 1}): HTTP {e.status_code} - {e.detail}")
                 exclude_models.add(model_cfg.id)
                 if attempt == max_retries - 1:
                     logging.error(f"{request_id}: All {max_retries} attempts failed. Last error: {e.detail}")
                     raise e # Re-raise last HTTP error
                 else: await asyncio.sleep(0.2 * (attempt + 1)); continue
            except Exception as e: # Catch unexpected errors during the try block
                 logging.exception(f"{request_id}: Unexpected error processing model [{model_cfg.name or model_cfg.id}] (Attempt {attempt + 1})", exc_info=e)
                 exclude_models.add(model_cfg.id)
                 if attempt == max_retries - 1:
                     logging.error(f"{request_id}: All {max_retries} attempts failed with unexpected error.")
                     # Raise a generic 500 for safety
                     raise HTTPException(status_code=500, detail="Internal server error during request processing.") from e
                 else: await asyncio.sleep(0.2 * (attempt + 1)); continue

        # Should not be reached if max_retries >= 1
        logging.error(f"{request_id}: Failed after all retries.")
        raise HTTPException(status_code=500, detail="Request failed after multiple retries.")


    def get_models_info(self) -> ModelsResponse:
        """获取模型状态，报告静态基础权重"""
        # ... (get_models_info - unchanged, reports base_weight) ...
        models_info: List[ModelInfo] = []
        available_count = 0
        with self.dispatcher._rwlock.read_lock(): needs_update = (time.time() - self.dispatcher._cache_last_update >= self.dispatcher._cache_ttl)
        if needs_update: self.dispatcher._update_availability_cache() # Update cache if stale

        with self.dispatcher._rwlock.read_lock():
            for model in self.models:
                model_id = model.id
                success_rate = self.dispatcher.get_model_success_rate(model_id)
                avg_response_time = self.dispatcher.get_model_avg_response_time(model_id)
                is_available = self.dispatcher._availability_cache.get(model_id, False) # Read from potentially updated cache
                if is_available: available_count += 1
                models_info.append(ModelInfo(
                    id=model_id, name=model.name, model=model.model,
                    weight=model.base_weight, # Report static base_weight
                    success_rate=success_rate, avg_response_time=avg_response_time,
                    available=is_available, channel=model.channel_name or model.channel_id
                ))
        return ModelsResponse(models=models_info, total=len(models_info), available=available_count)


    def get_health_info(self) -> HealthResponse:
        # ... (get_health_info - unchanged) ...
        total_models = len(self.models); available_count = 0
        for model in self.models:
             if self.dispatcher.is_model_available(model.id): available_count += 1
        status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        if available_count == 0 and total_models > 0: status = "unhealthy"
        elif total_models > 0 and available_count < total_models / 2: status = "degraded"
        elif total_models == 0: status = "unhealthy"
        uptime = time.time() - self.start_time
        return HealthResponse(status=status, available_models=available_count, total_models=total_models, uptime=uptime)

    def get_models_list(self) -> List[Dict[str, Any]]:
        # ... (get_models_list - unchanged) ...
        models_list = []; seen_model_names = set()
        for model in self.models:
             exposed_id = model.name or model.model or model.id
             if model.base_weight > 0 and exposed_id not in seen_model_names:
                  models_list.append({"id": exposed_id, "object": "model", "created": int(self.start_time), "owned_by": "flux-api"})
                  seen_model_names.add(exposed_id)
        return models_list

###############################################################################
# FastAPI 应用
###############################################################################
app = FastAPI(
    title="Flux API",
    description="OpenAI API 兼容的服务，使用模型池管理多模型",
    version="1.1.0" # Bump version
)
allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware, allow_origins=[o.strip() for o in allowed_origins if o.strip()],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
service: Optional[FluxApiService] = None

# --- MODIFICATION: Use lifespan context manager ---
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- MODIFICATION: Declare global service ONCE at the beginning ---
    global service
    # --- MODIFICATION END ---

    # Startup phase
    config_path = os.environ.get("FLUX_API_CONFIG", "./config.yaml")
    print("--> Initializing Flux API Service...")
    try:
        service = FluxApiService(config_path)
        logging.info(f"Flux API Worker (PID: {os.getpid()}) 启动成功。监听地址: http://{os.environ.get('HOST', '0.0.0.0')}:{os.environ.get('PORT', '8787')}")
    except Exception as e:
         logging.exception(f"Flux API Worker (PID: {os.getpid()}) 启动失败: {e}", exc_info=True)
         print(f"FATAL (PID: {os.getpid()}): Flux API Worker 启动失败: {e}", file=sys.stderr)
         service = None # Assign None if init failed
    print("--> Flux API Service Initialization Complete.")

    yield # Application runs here

    # Shutdown phase
    # --- MODIFICATION: No need to re-declare global here ---
    # global service # REMOVE THIS LINE
    # --- MODIFICATION END ---
    pid = os.getpid()
    print(f"--> Shutting down Flux API Worker (PID: {pid})...")
    if service: # Check if service was successfully initialized
        logging.info(f"Flux API Worker (PID: {pid}) 正在关闭...")
        # No background tasks to stop
        logging.info(f"Flux API Worker (PID: {pid}) 已关闭")
    else:
        logging.info(f"Flux API Worker (PID: {pid}) 关闭 (未完全初始化或已关闭)")
    print(f"--> Flux API Worker (PID: {pid}) Shutdown Complete.")

# Apply the lifespan context manager to the app
# (Ensure FastAPI instantiation uses the lifespan function as before)
app = FastAPI(
    title="Flux API",
    description="OpenAI API 兼容的服务，使用模型池管理多模型",
    version="1.1.0",
    lifespan=lifespan # Use the new lifespan manager
)
# Re-add middleware if needed after recreating app instance
app.add_middleware(
    CORSMiddleware, allow_origins=[o.strip() for o in allowed_origins if o.strip()],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)



# --- Middleware for Service Availability Check ---
@app.middleware("http")
async def check_service_availability(request: Request, call_next):
    # ... (middleware - unchanged) ...
    global service
    if request.url.path.startswith(("/admin", "/docs", "/openapi.json")):
         if service is None and not request.url.path.startswith("/admin/health"):
              return JSONResponse(status_code=503, content={"detail": "Service in this worker failed initialization."})
         return await call_next(request)
    if service is None:
        logging.error(f"Worker (PID: {os.getpid()}) received API request but service not initialized.")
        return JSONResponse(status_code=503, content={"detail": "Service temporarily unavailable (worker initializing or failed)."})
    response = await call_next(request); return response

# --- OpenAI API 兼容接口 ---
ResponseType = Union[ChatCompletionResponse, StreamingResponse]

@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completion_endpoint(request: ChatCompletionRequest) -> ResponseType:
    """创建聊天完成 - 符合OpenAI API规范"""
    # ... (endpoint logic - unchanged, relies on corrected service methods) ...
    global service
    if not service: raise HTTPException(status_code=503, detail="Service instance unavailable in this worker.")
    try:
        # Call the service method which should now correctly return tuple or AsyncIterable
        result = await service.create_chat_completion(request)
        # Check the result type and return appropriately
        if isinstance(result, AsyncIterable):
             return StreamingResponse(content=result, media_type="text/event-stream")
        elif isinstance(result, ChatCompletionResponse):
             return result
        else: # Should not happen if service logic is correct
             logging.error(f"Service create_chat_completion returned unexpected type: {type(result)}")
             raise HTTPException(status_code=500, detail="Internal Server Error: Invalid response format.")
    except HTTPException as e: raise e # Re-raise known HTTP errors
    except Exception as e: # Catch unexpected errors
        req_id = f"req-{uuid.uuid4().hex[:8]}"
        logging.exception(f"{req_id}: Uncaught error processing chat completion", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error (ID: {req_id})")

@app.get("/v1/models")
async def list_models_endpoint():
    """列出支持的模型"""
    # ... (endpoint logic - unchanged) ...
    global service
    if not service: raise HTTPException(status_code=503, detail="Service instance unavailable.")
    try: return {"object": "list", "data": service.get_models_list()}
    except Exception as e: logging.exception("Error fetching models list"); raise HTTPException(status_code=500, detail="Failed to retrieve models list.")

# --- 管理接口 ---
@app.get("/admin/models", response_model=ModelsResponse)
async def get_admin_models():
    """获取模型状态"""
    # ... (endpoint logic - unchanged) ...
    global service
    if not service: raise HTTPException(status_code=503, detail="Service instance unavailable.")
    try: return service.get_models_info()
    except Exception as e: logging.exception("Error fetching admin models info"); raise HTTPException(status_code=500, detail="Failed to retrieve model status.")

@app.get("/admin/health", response_model=HealthResponse)
async def health_check_endpoint():
    """健康检查"""
    # ... (endpoint logic - unchanged) ...
    global service
    if not service: return HealthResponse(status="unhealthy", available_models=0, total_models=0, uptime=0)
    try: return service.get_health_info()
    except Exception as e:
         logging.exception("Error during health check")
         total_models=0; uptime=0
         try:
             if service and hasattr(service, 'models'): total_models = len(service.models)
             if service: uptime = time.time() - service.start_time
         except Exception: pass
         return HealthResponse(status="unhealthy", available_models=0, total_models=total_models, uptime=uptime)

@app.get("/")
async def root_endpoint():
    """API根路径"""
    # ... (endpoint logic - unchanged) ...
    return {
        "name": "Flux API", "version": "1.1.0", "description": "OpenAI API 兼容的服务，使用模型池管理多模型",
        "documentation": "/docs",
        "openai_compatible_endpoints": [{"path": "/v1/chat/completions", "method": "POST"}, {"path": "/v1/models", "method": "GET"}],
        "admin_endpoints": [{"path": "/admin/models", "method": "GET"}, {"path": "/admin/health", "method": "GET"}]
    }

###############################################################################
# 命令行入口
###############################################################################
def validate_config_file(config_path: str) -> bool:
    # ... (validate_config_file - unchanged) ...
    if not os.path.exists(config_path): print(f"错误：配置文件不存在: {config_path}", file=sys.stderr); return False
    try:
        with open(config_path, "r", encoding="utf-8") as f: config = yaml.safe_load(f)
        if not isinstance(config, dict): print(f"错误：配置文件内容不是有效的YAML字典: {config_path}", file=sys.stderr); return False
        if "models" not in config: print(f"警告：配置文件缺少 'models' 部分: {config_path}", file=sys.stderr)
        elif not isinstance(config["models"], list): print(f"错误：配置文件中的 'models' 必须是一个列表: {config_path}", file=sys.stderr); return False
        if "channels" not in config: print(f"警告：配置文件缺少 'channels' 部分: {config_path}", file=sys.stderr)
        elif not isinstance(config["channels"], dict): print(f"错误：配置文件中的 'channels' 必须是一个字典: {config_path}", file=sys.stderr); return False
        return True
    except yaml.YAMLError as e: print(f"错误：配置文件YAML解析失败: {config_path}\n{e}", file=sys.stderr); return False
    except Exception as e: print(f"错误：读取或验证配置文件时发生未知错误: {config_path}\n{e}", file=sys.stderr); return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Flux API - OpenAI Compatible Multi-Model Gateway")
    parser.add_argument("--config", "-c", default="./config.yaml", help="配置文件路径 (config.yaml)")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=int(os.environ.get("PORT", 8787)), help="监听端口 (默认: 8787)")
    parser.add_argument("--reload", action="store_true", help="开发模式下启用自动重载 (需要 'watchfiles')")
    # --- MODIFICATION: Default workers set to 1 ---
    parser.add_argument("--workers", "-w", type=int, default=int(os.environ.get("WEB_CONCURRENCY", 1)), help="工作进程数 (默认: 1 以保证状态一致性)")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"), help="Uvicorn 日志级别 (例如: debug, info, warning, error)")

    args = parser.parse_args()
    if not validate_config_file(args.config): sys.exit(1)

    os.environ["FLUX_API_CONFIG"] = args.config
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port) # Pass port for logging

    # Log arguments before starting Uvicorn
    print(f"准备启动 Flux API (v1.1.0)...")
    print(f"  配置文件: {args.config}")
    print(f"  监听地址: http://{args.host}:{args.port}")
    # Ensure workers=1 if reload is True
    actual_workers = 1 if args.reload else args.workers
    if actual_workers != 1:
         print(f"警告: 工作进程数 ({actual_workers}) 大于 1，模型状态 (退避, 限流) 将在进程间独立，可能不符合预期！建议使用 --workers 1。")
    print(f"  工作进程: {actual_workers} {'(热重载模式强制为 1)' if args.reload and args.workers > 1 else ''}")
    print(f"  日志级别: {args.log_level}")
    print(f"  热重载: {'启用' if args.reload else '禁用'}")

    module_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    app_string = f"{module_name}:app"

    uvicorn.run(
        app_string, host=args.host, port=args.port, reload=args.reload,
        workers=actual_workers, # Use potentially adjusted worker count
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main()