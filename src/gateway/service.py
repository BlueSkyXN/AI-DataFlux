"""
Flux API 核心服务模块

本模块实现 OpenAI API 兼容的核心服务逻辑，是 API 网关的业务层核心。
负责管理模型池、处理聊天补全请求、协调调度器和限流器。

核心功能:
    - 多模型管理：加载、验证、映射模型配置
    - 请求处理：支持流式和非流式聊天补全
    - 故障切换：自动重试其他可用模型
    - 状态追踪：健康检查、模型统计

类结构:
    FluxApiService
    ├── config: 配置数据
    ├── models: 模型配置列表 (ModelConfig)
    ├── dispatcher: 模型调度器 (ModelDispatcher)
    ├── rate_limiter: 限流器 (ModelRateLimiter)
    └── session_pool: HTTP 连接池 (SessionPool)

请求处理流程:
    1. 接收 ChatCompletionRequest
    2. 解析模型名称，映射到内部模型 ID
    3. 通过调度器选择可用模型（考虑权重、可用性、限流）
    4. 构建请求并调用上游 API
    5. 处理响应（流式 SSE 或 JSON）
    6. 更新模型指标（成功率、响应时间）
    7. 失败时自动切换到其他模型重试

使用示例:
    service = FluxApiService("config.yaml")
    await service.startup()  # 初始化异步资源

    response = await service.chat_completion(request)

    await service.shutdown()  # 清理资源

依赖模块:
    - ModelDispatcher: 模型调度和故障退避
    - ModelRateLimiter: 令牌桶限流
    - SessionPool: HTTP 连接复用
    - RoundRobinResolver: IP 池轮询
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterable, Union

import aiohttp
import yaml

from .dispatcher import ModelDispatcher, ModelConfig
from .limiter import ModelRateLimiter
from .resolver import RoundRobinResolver, build_ip_pools_from_channels
from .session import SessionPool
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseUsage,
    ChatMessage,
)
from ..models.errors import ErrorType


class FluxApiService:
    """
    Flux API 核心服务类

    OpenAI API 兼容的服务实现，提供多模型管理、自动故障切换、
    令牌桶限流、连接池复用等企业级功能。

    Attributes:
        config_path (str): 配置文件路径
        config (dict): 加载的配置数据
        models (list[ModelConfig]): 模型配置列表
        dispatcher (ModelDispatcher): 模型调度器
        rate_limiter (ModelRateLimiter): 限流器
        session_pool (SessionPool): HTTP 连接池
        start_time (float): 服务启动时间戳

    生命周期:
        1. __init__: 加载配置，初始化同步组件
        2. startup(): 初始化异步组件（连接池）
        3. chat_completion(): 处理请求
        4. shutdown(): 清理资源
    """

    def __init__(self, config_path: str):
        """
        初始化服务（同步部分）

        加载配置文件，初始化模型配置、调度器和限流器。
        异步资源（连接池）在 startup() 中初始化。

        Args:
            config_path: 配置文件路径（YAML 格式）

        Raises:
            ValueError: 配置文件无效或无可用模型
        """
        self.start_time = time.time()
        self.config_path = config_path

        # 加载配置文件
        self._load_config()

        # 初始化同步组件
        self._init_models()  # 模型配置
        self._init_dispatcher()  # 调度器
        self._init_rate_limiter()  # 限流器

        # Session 池在 startup() 中异步初始化
        self.session_pool: SessionPool | None = None

        logging.info("FluxApiService 初始化完成")

    def _load_config(self) -> None:
        """加载配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            # 验证配置根节点类型
            if not isinstance(self.config, dict):
                raise ValueError(
                    f"配置文件根节点必须是字典类型，"
                    f"实际类型为 {type(self.config).__name__}"
                )

            logging.info(f"配置文件 '{self.config_path}' 加载成功")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件 YAML 格式错误: {e}") from e
        except Exception as e:
            raise ValueError(f"无法加载配置文件: {e}") from e

        # 全局配置
        global_cfg = self.config.get("global", {})
        log_cfg = global_cfg.get("log", {})

        # 初始化日志
        level_str = log_cfg.get("level", "info").upper()
        level = getattr(logging, level_str, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 通道配置
        self.channels = self.config.get("channels", {})

        # 网关连接池配置
        gateway_cfg = self.config.get("gateway", {}) or {}
        self.gateway_max_connections = int(gateway_cfg.get("max_connections", 1000))
        self.gateway_max_connections_per_host = int(
            gateway_cfg.get("max_connections_per_host", 1000)
        )

    def _init_models(self) -> None:
        """初始化模型配置"""
        models_cfg = self.config.get("models", [])

        if not models_cfg:
            raise ValueError("配置中未找到模型定义")

        self.models: list[ModelConfig] = []

        for model_dict in models_cfg:
            try:
                model = ModelConfig(model_dict, self.channels)
                self.models.append(model)
                logging.info(f"加载模型: {model.id} ({model.model})")
            except Exception as e:
                logging.warning(f"加载模型失败: {e}")

        if not self.models:
            raise ValueError("没有成功加载任何模型")

        logging.info(f"共加载 {len(self.models)} 个模型")

        # 创建模型名称映射
        self.model_name_to_id: dict[str, str] = {}
        for m in self.models:
            self.model_name_to_id[m.id] = m.id  # ID 映射
            if m.model:  # 模型标识符映射
                if (
                    m.model in self.model_name_to_id
                    and self.model_name_to_id[m.model] != m.id
                ):
                    logging.debug(f"模型标识符 '{m.model}' 被多个模型使用")
                self.model_name_to_id[m.model] = m.id
            if m.name and m.name != m.model:  # 模型别名映射
                if (
                    m.name in self.model_name_to_id
                    and self.model_name_to_id[m.name] != m.id
                ):
                    logging.debug(f"模型名称 '{m.name}' 被多个模型使用")
                self.model_name_to_id[m.name] = m.id

        logging.info(f"模型名称映射已创建，共 {len(self.model_name_to_id)} 个映射")

        # 验证至少有一个模型具有有效权重
        if not any(model.weight > 0 for model in self.models):
            logging.warning("所有模型的权重都为 0 或负数，随机选择将无法工作")

    def _init_dispatcher(self) -> None:
        """初始化模型调度器"""
        self.dispatcher = ModelDispatcher(self.models)

    def _init_rate_limiter(self) -> None:
        """初始化限流器"""
        self.rate_limiter = ModelRateLimiter()
        self.rate_limiter.configure([m.to_dict() for m in self.models])

    async def startup(self) -> None:
        """启动服务 (异步初始化)"""
        # 构建 IP 池并创建自定义解析器
        ip_pools = build_ip_pools_from_channels(self.channels)
        resolver = RoundRobinResolver(ip_pools) if ip_pools else None

        if resolver:
            logging.info(f"已启用 IP 池轮询解析器，共 {len(ip_pools)} 个域名")

        self.session_pool = SessionPool(
            max_connections=self.gateway_max_connections,
            max_connections_per_host=self.gateway_max_connections_per_host,
            resolver=resolver,
        )
        logging.info("FluxApiService 启动完成")

    async def shutdown(self) -> None:
        """关闭服务"""
        if self.session_pool:
            await self.session_pool.close_all()
        logging.info("FluxApiService 已关闭")

    def resolve_model_id(self, model_name_or_id: str) -> str | None:
        """
        将用户请求的模型名称/ID解析为内部配置的模型ID

        Args:
            model_name_or_id: 用户请求的模型名称或ID

        Returns:
            内部模型ID，如果是 "auto" 或无法解析则返回 None
        """
        # 优先匹配内部映射
        if model_name_or_id in self.model_name_to_id:
            return self.model_name_to_id[model_name_or_id]

        # 处理通配符或默认情况
        if model_name_or_id.lower() in ["auto", "any", "default", "*", ""]:
            return None  # 返回 None 表示需要随机选择

        # 未找到匹配
        logging.warning(f"无法解析请求的模型名称或ID: '{model_name_or_id}'")
        return None

    def get_available_model(
        self,
        requested_model_name: str | None = None,
        exclude_models: list[str] | None = None,
    ) -> ModelConfig | None:
        """
        获取一个可用的模型

        优先选择指定模型，否则根据权重从可用模型中随机选择。
        同时考虑调度器可用性和限流器限制。

        Args:
            requested_model_name: 用户请求的模型名称/ID
            exclude_models: 要排除的模型ID列表

        Returns:
            可用的模型配置，如果没有可用模型返回 None
        """
        import random

        exclude_set = set(exclude_models or [])
        target_model_id: str | None = None
        use_random_selection = True

        # 1. 解析请求的模型名称
        if requested_model_name:
            resolved_id = self.resolve_model_id(requested_model_name)
            if resolved_id:
                target_model_id = resolved_id
                use_random_selection = False
            # 如果是 "auto" 或无法解析，则进行随机选择
            elif requested_model_name.lower() not in [
                "auto",
                "any",
                "default",
                "*",
                "",
            ]:
                logging.error(
                    f"请求了未知的模型 '{requested_model_name}'，将尝试随机选择"
                )

        # 2. 如果指定了目标模型，检查其可用性
        if target_model_id and not use_random_selection:
            if target_model_id not in exclude_set:
                model = self.dispatcher.get_model_config(target_model_id)
                if model:
                    # 检查调度器可用性和限流器
                    is_available = self.dispatcher.is_model_available(target_model_id)
                    can_process = self.rate_limiter.can_process(target_model_id)

                    if is_available and can_process:
                        logging.debug(
                            f"使用请求的可用模型: {model.name or target_model_id}"
                        )
                        return model
                    else:
                        # 指定模型不可用或被限流，转为随机选择
                        logging.warning(
                            f"请求的模型 [{model.name or target_model_id}] 当前不可用/受限。尝试随机选择"
                        )
                        exclude_set.add(target_model_id)
                        use_random_selection = True
            else:
                # 指定的模型在排除列表中，转为随机选择
                logging.warning(
                    f"请求的模型 [{target_model_id}] 在排除列表中。尝试随机选择"
                )
                use_random_selection = True

        # 3. 执行加权随机选择 (如果需要)
        if use_random_selection:
            # 从所有模型中过滤出符合条件的模型
            eligible_models = [
                model
                for model in self.models
                if model.weight > 0  # 权重必须大于0
                and model.id not in exclude_set  # 未被排除
                and self.dispatcher.is_model_available(model.id)  # 调度器可用
                and self.rate_limiter.can_process(model.id)  # 限流器允许
            ]

            if not eligible_models:
                logging.warning(f"没有符合条件的可用模型 (已排除: {exclude_set})")
                return None

            # 使用加权随机算法选择模型
            weights = [model.weight for model in eligible_models]
            chosen_model = random.choices(eligible_models, weights=weights, k=1)[0]
            logging.debug(
                f"加权随机选择了模型: {chosen_model.name or chosen_model.id} (权重: {chosen_model.weight}/{sum(weights)})"
            )
            return chosen_model

        return None

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncIterable[str]]:
        """
        处理聊天补全请求（支持流式和非流式）

        Args:
            request: 聊天补全请求

        Returns:
            非流式: ChatCompletionResponse
            流式: AsyncIterable[str] (SSE 格式)
        """
        start_time = time.time()
        tried_models: set[str] = set()
        last_error: Exception | None = None

        # 尝试多个模型
        max_retries = min(len(self.models), 3)

        for attempt in range(max_retries):
            # 选择模型 - 使用新的 get_available_model 方法
            model = self.get_available_model(
                requested_model_name=request.model,  # 传递用户请求的模型
                exclude_models=list(tried_models),
            )

            if not model:
                logging.warning("没有可用的模型")
                break

            tried_models.add(model.id)

            # 检查限流
            if not self.rate_limiter.acquire(model.id):
                logging.debug(f"模型 {model.id} 被限流，尝试下一个")
                continue

            try:
                # 调用 API (流式或非流式)
                response = await self._call_model_api(model, request)

                # 如果是流式响应，直接返回异步迭代器
                if isinstance(response, AsyncIterable):
                    logging.info(f"返回流式响应 (模型: {model.id})")
                    return response

                # 非流式响应，更新指标
                elapsed = time.time() - start_time
                self.dispatcher.update_model_metrics(model.id, elapsed, True)
                self.dispatcher.mark_model_success(model.id)

                return response

            except Exception as e:
                logging.warning(f"模型 {model.id} 调用失败: {e}")
                last_error = e

                elapsed = time.time() - start_time
                self.dispatcher.update_model_metrics(model.id, elapsed, False)
                self.dispatcher.mark_model_failed(model.id)

        # 所有模型都失败
        raise RuntimeError(f"所有模型调用失败: {last_error}") from last_error

    async def _call_model_api(
        self, model: ModelConfig, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncIterable[str]]:
        """
        调用模型 API (支持流式和非流式)

        Returns:
            非流式: ChatCompletionResponse
            流式: AsyncIterable[str]
        """
        if not self.session_pool:
            raise RuntimeError("SessionPool 未初始化")

        # 获取 Session
        session = await self.session_pool.get_or_create(
            ssl_verify=model.ssl_verify, proxy=model.proxy
        )

        # 构建请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model.api_key}",
        }

        payload: dict[str, Any] = {
            "model": model.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": (
                request.temperature
                if request.temperature is not None
                else model.temperature
            ),
            "stream": request.stream or False,
        }

        # 添加可选参数 (使用 is not None 以支持 0 和 False)
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            payload["response_format"] = request.response_format.model_dump()
        if request.stop is not None:
            payload["stop"] = request.stop
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.user is not None:
            payload["user"] = request.user
        if request.n is not None:
            payload["n"] = request.n

        # 高级参数 (如果模型支持)
        if model.supports_advanced_params:
            if request.presence_penalty is not None:
                payload["presence_penalty"] = request.presence_penalty
            if request.frequency_penalty is not None:
                payload["frequency_penalty"] = request.frequency_penalty
            if request.logit_bias is not None:
                payload["logit_bias"] = request.logit_bias

        # 发送请求
        # 流式请求需要更长的 sock_read 超时
        timeout = aiohttp.ClientTimeout(
            connect=model.connect_timeout,
            total=model.read_timeout,
            sock_read=300 if request.stream else model.read_timeout,
        )

        start_time = time.time()

        # 流式响应：手动管理响应生命周期，避免 async with 提前关闭连接
        if request.stream:
            resp = await session.post(
                model.api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
                proxy=model.proxy or None,
            )

            # 记录日志
            peer_ip = self._extract_peer_ip(resp, model)
            self._log_upstream_response(model, resp.status, peer_ip)

            if resp.status != 200:
                text = await resp.text()
                resp.close()
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=resp.status,
                    message=text[:500],
                    headers=resp.headers,
                )

            # 返回流式响应生成器，响应将在生成器 finally 中关闭
            return self._handle_streaming_response(resp, model, start_time)

        # 非流式响应：使用 async with 自动管理
        async with session.post(
            model.api_url,
            headers=headers,
            json=payload,
            timeout=timeout,
            proxy=model.proxy or None,
        ) as resp:
            peer_ip = self._extract_peer_ip(resp, model)
            self._log_upstream_response(model, resp.status, peer_ip)

            if resp.status != 200:
                text = await resp.text()
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=resp.status,
                    message=text[:500],
                    headers=resp.headers,
                )

            # 处理非流式响应
            data = await resp.json()

        # 解析响应
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("响应中没有 choices")

        choice = choices[0]
        message = choice.get("message", {})

        # 构建响应
        return ChatCompletionResponse(
            id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
            created=data.get("created", int(time.time())),
            model=request.model,  # 返回用户请求的模型名
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role=message.get("role", "assistant"),
                        content=message.get("content", ""),
                    ),
                    finish_reason=choice.get("finish_reason", "stop"),
                )
            ],
            usage=(
                ChatCompletionResponseUsage(
                    prompt_tokens=data.get("usage", {}).get("prompt_tokens"),
                    completion_tokens=data.get("usage", {}).get("completion_tokens"),
                    total_tokens=data.get("usage", {}).get("total_tokens"),
                )
                if data.get("usage")
                else None
            ),
        )

    def _extract_peer_ip(
        self, resp: aiohttp.ClientResponse, model: ModelConfig
    ) -> str | None:
        """从响应中提取对端 IP 地址"""
        if model.proxy:
            return None
        try:
            if resp.connection and resp.connection.transport:
                peername = resp.connection.transport.get_extra_info("peername")
                if peername:
                    return peername[0] if isinstance(peername, tuple) else str(peername)
        except Exception:
            pass
        return None

    def _log_upstream_response(
        self, model: ModelConfig, status: int, peer_ip: str | None
    ) -> None:
        """记录上游响应日志"""
        if peer_ip:
            logging.info(
                "上游响应 model=%s status=%s url=%s ip=%s",
                model.id,
                status,
                model.api_url,
                peer_ip,
            )
        else:
            logging.info(
                "上游响应 model=%s status=%s url=%s",
                model.id,
                status,
                model.api_url,
            )

    async def _handle_streaming_response(
        self, response: aiohttp.ClientResponse, model: ModelConfig, start_time: float
    ) -> AsyncIterable[str]:
        """
        处理流式响应，返回异步生成器产生 SSE 格式字符串

        Args:
            response: aiohttp 响应对象
            model: 模型配置
            start_time: 请求开始时间

        Yields:
            SSE 格式的字符串 (data: {...}\n\n)
        """
        buffer = ""
        has_business_output = False
        chunk_count = 0
        last_activity_time = time.time()
        returned_successfully = False

        logging.debug(f"[{model.id}] 开始流式响应处理")

        # 发送初始保活消息
        try:
            yield ": keeping connection alive\n\n"
            logging.debug(f"[{model.id}] 已发送初始连接保持消息")
        except Exception as e:
            logging.warning(f"[{model.id}] 发送初始消息失败: {e}")

        try:
            # 异步迭代响应内容块
            async for chunk in response.content.iter_any():
                chunk_count += 1
                now = time.time()
                time_since_last = now - last_activity_time
                last_activity_time = now

                logging.debug(
                    f"[{model.id}] 收到流块 {chunk_count} "
                    f"(大小:{len(chunk)}字节, 距离上次:{time_since_last:.2f}秒)"
                )

                if not chunk:
                    logging.debug(f"[{model.id}] 收到空块 {chunk_count}")
                    continue

                # 解码并添加到缓冲区
                try:
                    decoded_chunk = chunk.decode("utf-8")
                    buffer += decoded_chunk
                except UnicodeDecodeError:
                    buffer += chunk.decode("utf-8", errors="ignore")
                    logging.warning(f"[{model.id}] 流包含无效UTF-8数据，已忽略")

                # 按行分割处理缓冲区内容
                lines = buffer.split("\n")
                buffer = lines.pop()  # 最后一部分可能不完整

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # 检查是否是 SSE 数据行
                    if line.startswith("data:"):
                        data_content = line[len("data:") :].strip()

                        # 检查是否是结束标记
                        if data_content == "[DONE]":
                            logging.info(f"[{model.id}] 收到 [DONE] 标记")
                            # 转发 [DONE] 事件给客户端
                            yield "data: [DONE]\n\n"
                            returned_successfully = True
                            break

                        elif data_content:
                            # 检查是否是 JSON 格式
                            if data_content.startswith("{") and data_content.endswith(
                                "}"
                            ):
                                yield f"data: {data_content}\n\n"
                                has_business_output = True
                            else:
                                logging.warning(
                                    f"[{model.id}] 流包含非JSON数据块: {data_content[:100]}..."
                                )
                    else:
                        logging.debug(
                            f"[{model.id}] 流包含非'data:'行: {line[:100]}..."
                        )

                # 如果收到了 [DONE]，则跳出循环
                if returned_successfully:
                    break

            logging.info(
                f"[{model.id}] 流式响应处理结束。总块数: {chunk_count}。"
                f"是否收到[DONE]: {returned_successfully}"
            )

            # 处理循环结束后缓冲区剩余数据
            if not returned_successfully and buffer.strip():
                line = buffer.strip()
                if line.startswith("data:"):
                    data_content = line[len("data:") :].strip()
                    if data_content == "[DONE]":
                        # 转发 [DONE] 事件给客户端
                        yield "data: [DONE]\n\n"
                        returned_successfully = True
                        logging.info(f"[{model.id}] 在最终缓冲区中收到 [DONE]")
                    elif (
                        data_content
                        and data_content.startswith("{")
                        and data_content.endswith("}")
                    ):
                        yield f"data: {data_content}\n\n"
                        has_business_output = True
                        returned_successfully = True
                        logging.info(f"[{model.id}] 成功处理最终缓冲区内容")

            # 如果流正常结束但没有收到 [DONE]，也发送 [DONE] 保证客户端能正常结束
            if has_business_output and not returned_successfully:
                logging.warning(f"[{model.id}] 未收到 [DONE] 但流已结束，补发 [DONE]")
                yield "data: [DONE]\n\n"
                returned_successfully = True

        except aiohttp.ClientPayloadError as e:
            logging.error(f"[{model.id}] 流处理 ClientPayloadError: {e}")
            error_payload = {
                "error": {"message": "上游流响应体错误", "type": "stream_error"}
            }
            try:
                yield f"data: {json.dumps(error_payload)}\n\n"
            except Exception:
                pass
            returned_successfully = False

        except aiohttp.ClientConnectionError as e:
            logging.error(f"[{model.id}] 流处理 ClientConnectionError: {e}")
            error_payload = {
                "error": {"message": "上游流连接错误", "type": "stream_error"}
            }
            try:
                yield f"data: {json.dumps(error_payload)}\n\n"
            except Exception:
                pass
            returned_successfully = False

        except asyncio.TimeoutError as e:
            logging.error(f"[{model.id}] 流处理 TimeoutError: {e}")
            error_payload = {
                "error": {"message": "上游流读取超时", "type": "stream_error"}
            }
            try:
                yield f"data: {json.dumps(error_payload)}\n\n"
            except Exception:
                pass
            returned_successfully = False

        except aiohttp.ClientError as e:
            logging.error(f"[{model.id}] 流处理 ClientError: {e}")
            error_payload = {
                "error": {"message": "上游流客户端错误", "type": "stream_error"}
            }
            try:
                yield f"data: {json.dumps(error_payload)}\n\n"
            except Exception:
                pass
            returned_successfully = False

        except Exception as e:
            logging.exception(f"[{model.id}] 处理流时发生未知错误", exc_info=e)
            error_payload = {
                "error": {
                    "message": "内部流处理错误",
                    "type": "internal_stream_error",
                }
            }
            try:
                yield f"data: {json.dumps(error_payload)}\n\n"
            except Exception:
                pass
            returned_successfully = False

        finally:
            # 根据处理结果更新模型状态和指标
            response_time = time.time() - start_time
            stream_fully_successful = returned_successfully and has_business_output
            stream_partially_successful = has_business_output and not returned_successfully
            stream_no_business_output = returned_successfully and not has_business_output

            if stream_fully_successful:
                self.dispatcher.mark_model_success(model.id)
                logging.info(f"[{model.id}] 流处理完全成功。耗时:{response_time:.2f}s")
            elif stream_partially_successful or stream_no_business_output:
                self.dispatcher.mark_model_failed(model.id, ErrorType.CONTENT)
                if stream_no_business_output:
                    logging.warning(
                        f"[{model.id}] 流仅收到保活/结束信号，未产生业务输出。耗时:{response_time:.2f}s"
                    )
                else:
                    logging.warning(
                        f"[{model.id}] 流处理部分成功 (在[DONE]前出错)。耗时:{response_time:.2f}s"
                    )
            else:  # no data yielded
                self.dispatcher.mark_model_failed(model.id, ErrorType.API)
                logging.error(
                    f"[{model.id}] 流处理失败 (未产生数据)。耗时:{response_time:.2f}s"
                )

            # 更新指标
            self.dispatcher.update_model_metrics(
                model.id,
                response_time,
                stream_fully_successful,
            )

            # 确保关闭响应连接
            if response and not response.closed:
                response.close()

    def _extract_content_from_event_stream(self, event_stream_text: str) -> str:
        """
        从 (意外收到的) 事件流文本中提取内容

        Args:
            event_stream_text: SSE 格式的事件流文本

        Returns:
            提取的完整内容
        """
        full_content = []
        lines = event_stream_text.splitlines()

        for line in lines:
            line = line.strip()
            # 查找 data: 开头且非 [DONE] 的行
            if line.startswith("data:") and not line.endswith("[DONE]"):
                try:
                    data_str = line[len("data:") :].strip()
                    if data_str:
                        data = json.loads(data_str)
                        # 尝试按 OpenAI 流格式提取 delta content
                        if (
                            isinstance(data, dict)
                            and "choices" in data
                            and isinstance(data["choices"], list)
                            and data["choices"]
                            and isinstance(data["choices"][0], dict)
                            and "delta" in data["choices"][0]
                            and isinstance(data["choices"][0]["delta"], dict)
                            and "content" in data["choices"][0]["delta"]
                        ):
                            content_part = data["choices"][0]["delta"]["content"]
                            if isinstance(content_part, str):
                                full_content.append(content_part)
                except Exception:
                    # 忽略解析错误或结构不匹配的行
                    logging.debug(f"从意外事件流提取内容时忽略行: {line}")
                    continue

        return "".join(full_content)

    def get_uptime(self) -> float:
        """获取服务运行时间"""
        return time.time() - self.start_time

    def get_health_status(self) -> dict[str, Any]:
        """获取健康状态"""
        available = len(self.dispatcher.get_available_models())
        total = len(self.models)

        if available == 0:
            status = "unhealthy"
        elif available < total:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "available_models": available,
            "total_models": total,
            "uptime": self.get_uptime(),
        }

    def get_models_info(self) -> dict[str, Any]:
        """获取模型信息"""
        stats = self.dispatcher.get_all_model_stats()
        available = sum(1 for s in stats if s["available"])

        return {
            "models": stats,
            "total": len(stats),
            "available": available,
        }
