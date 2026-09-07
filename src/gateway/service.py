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

FluxApiService 方法清单:

    生命周期管理:
        ├── __init__(config_path: str)          加载配置，初始化同步组件
        ├── startup() -> None                   异步启动（创建连接池和解析器）
        └── shutdown() -> None                  关闭服务，释放所有资源

    配置初始化（内部方法）:
        ├── _load_config()                      加载 YAML 配置文件，初始化日志
        ├── _init_models()                      解析模型配置，创建名称映射
        ├── _init_dispatcher()                  初始化模型调度器
        └── _init_rate_limiter()                初始化令牌桶限流器

    请求处理（核心流程）:
        ├── chat_completion(request) -> Response | AsyncIterable
        │   处理聊天补全请求，自动重试最多 min(模型数, 3) 次
        │   输入: ChatCompletionRequest
        │   输出: ChatCompletionResponse（非流式）或 AsyncIterable[str]（流式 SSE）
        │   异常: RuntimeError — 所有模型调用失败
        ├── _call_model_api(model, request) -> Response | AsyncIterable
        │   调用单个模型的上游 API（核心 HTTP 调用逻辑）
        └── _handle_streaming_response(response, model, start_time) -> AsyncIterable
            流式 SSE 响应处理生成器（缓冲区拼接、[DONE] 检测、错误恢复）

    模型选择:
        ├── resolve_model_id(name_or_id) -> str | None
        │   将用户请求的模型名称解析为内部 ID
        └── get_available_model(requested_model_name, exclude) -> ModelConfig | None
            综合调度器 + 限流器选择可用模型

    辅助方法:
        ├── _extract_peer_ip(resp, model) -> str | None    提取对端 IP
        ├── _log_upstream_response(model, status, ip)      记录上游响应日志
        ├── _extract_content_from_event_stream(text) -> str 从意外 SSE 文本提取内容
        ├── get_uptime() -> float                          获取服务运行时间
        ├── get_health_status() -> dict                    获取健康状态
        └── get_models_info() -> dict                      获取模型统计信息

    关键变量:
        - config: 加载的 YAML 配置字典
        - models: ModelConfig 列表
        - model_name_to_id: 名称/别名 → 内部 ID 映射字典
        - dispatcher: ModelDispatcher 调度器实例
        - rate_limiter: ModelRateLimiter 限流器实例
        - session_pool: SessionPool HTTP 连接池实例

依赖模块:
    - dispatcher.ModelDispatcher: 模型调度和故障退避
    - dispatcher.ModelConfig: 模型配置封装
    - limiter.ModelRateLimiter: 令牌桶限流
    - session.SessionPool: HTTP 连接复用
    - resolver.RoundRobinResolver: IP 池轮询解析
    - schemas: 请求/响应 Pydantic 模型
    - models.errors.ErrorType: 错误类型枚举
"""

import asyncio
import logging
import json
import time
from typing import Any, AsyncIterable, Union

import aiohttp
from src.config import compile_gateway_config, init_logging, load_config

from .dispatcher import ModelDispatcher, ModelConfig
from .limiter import ModelRateLimiter
from .resolver import RoundRobinResolver, build_ip_pools_from_channels
from .session import SessionPool
from .affinity import ResponseAffinity
from .schemas import (
    ChatCompletionRequest,
    ResponsesRequest,
)

RETRYABLE_UPSTREAM_STATUSES = {429, 500, 502, 503, 504}


class GatewayAPIError(Exception):
    """可直接映射为 OpenAI error object 的网关错误。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
        upstream_error: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
            **(upstream_error or {}),
        }


class RetryableUpstreamError(GatewayAPIError):
    """仅表示响应开始前可安全切换到其他模型的错误。"""


class UpstreamStream:
    """即使流尚未开始迭代，也能关闭已经取得响应头的上游连接。"""

    def __init__(self, stream: AsyncIterable[bytes], response: aiohttp.ClientResponse):
        self.iterator = stream.__aiter__()
        self.response = response

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        return await self.iterator.__anext__()

    async def aclose(self) -> None:
        try:
            close = getattr(self.iterator, "aclose", None)
            if close is not None:
                await close()
        finally:
            self.response.close()


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
        self.affinity = ResponseAffinity(**self.config["gateway_affinity"])
        self.fallback_groups = self.config["fallback_groups"]
        self.max_attempts = self.config["gateway_retry"]["max_attempts_per_request"]

        # 初始化同步组件
        self._init_models()  # 模型配置
        self._init_dispatcher()  # 调度器
        self._init_rate_limiter()  # 限流器

        # Session 池在 startup() 中异步初始化
        self.session_pool: SessionPool | None = None

        logging.info("FluxApiService 初始化完成")

    def _load_config(self) -> None:
        """
        加载配置文件并初始化全局设置

        操作:
        1. 读取并解析 YAML 配置文件
        2. 初始化日志系统（级别从 global.log.level 读取）
        3. 提取通道配置 (self.channels)
        4. 提取网关连接池参数 (max_connections, max_connections_per_host)

        Raises:
            ValueError: YAML 格式错误或配置根节点不是字典
        """
        try:
            root_config = load_config(self.config_path)
            self.config = compile_gateway_config(root_config)
            logging.info(f"配置文件 '{self.config_path}' 加载成功")
        except Exception as exc:
            raise ValueError(f"无法加载配置文件: {exc}") from exc

        # 全局配置
        global_cfg = self.config.get("global", {})
        log_cfg = global_cfg.get("log", {})

        # 初始化日志
        init_logging(log_cfg)

        # 通道配置
        self.channels = self.config.get("channels", {})

        # 网关连接池配置
        gateway_cfg = self.config.get("gateway", {}) or {}
        self.gateway_max_connections = int(gateway_cfg.get("max_connections", 1000))
        self.gateway_max_connections_per_host = int(
            gateway_cfg.get("max_connections_per_host", 1000)
        )

    def _init_models(self) -> None:
        """
        初始化模型配置并创建名称映射

        操作:
        1. 遍历配置中的模型列表，为每个创建 ModelConfig 对象
        2. 创建 model_name_to_id 映射字典（ID → ID, model → ID, name → ID）
        3. 验证至少有一个有效权重的模型

        映射优先级: name > model > id（后注册的覆盖先注册的）

        Raises:
            ValueError: 未定义模型或所有模型加载失败
        """
        models_cfg = self.config.get("models", [])

        if not models_cfg:
            raise ValueError("配置中未找到模型定义")

        self.models: list[ModelConfig] = []

        for model_dict in models_cfg:
            model = ModelConfig(model_dict, self.channels)
            self.models.append(model)
            logging.info(f"加载模型: {model.id} ({model.model})")

        if not self.models:
            raise ValueError("没有成功加载任何模型")

        logging.info(f"共加载 {len(self.models)} 个模型")

        # 创建模型名称映射
        self.model_name_to_id: dict[str, str] = {}
        for m in self.models:
            self.model_name_to_id[m.id] = m.id  # ID 映射
            for alias in m.aliases:
                self.model_name_to_id[alias] = m.id
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
        """初始化模型调度器（ModelDispatcher 管理加权选择和退避）"""
        self.dispatcher = ModelDispatcher(self.models)

    def _init_rate_limiter(self) -> None:
        """初始化令牌桶限流器（每个模型独立的 TokenBucket，容量 = safe_rps × 2）"""
        self.rate_limiter = ModelRateLimiter()
        self.rate_limiter.configure([m.to_dict() for m in self.models])

    async def startup(self) -> None:
        """
        启动服务（异步初始化）

        操作:
        1. 从通道配置构建 IP 池映射
        2. 若有 IP 池，创建 RoundRobinResolver
        3. 创建 SessionPool（HTTP 连接池）
        """
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
        """关闭服务，释放 SessionPool 的所有连接和 DNS 解析器资源"""
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
        requires_json_schema: bool = False,
        required_capabilities: set[str] | None = None,
    ) -> ModelConfig | None:
        """
        获取一个可用的模型

        优先选择指定模型，否则根据权重从可用模型中随机选择。
        同时考虑调度器可用性和限流器限制。

        Args:
            requested_model_name: 用户请求的模型名称/ID
            exclude_models: 要排除的模型ID列表
            requires_json_schema: 请求是否需要 JSON 输出能力

        Returns:
            可用的模型配置，如果没有可用模型返回 None
        """
        import random

        required = set(required_capabilities or ())
        if requires_json_schema:
            required.add("json_schema")
        candidates = self.models
        if requested_model_name and requested_model_name.lower() not in {
            "auto",
            "any",
            "default",
            "*",
        }:
            target = self.resolve_model_id(requested_model_name)
            candidates = [model for model in self.models if model.id == target]
        eligible = [
            model
            for model in candidates
            if model.weight > 0
            and model.id not in (exclude_models or [])
            and model.supports(required)
            and self.dispatcher.is_model_available(model.id)
            and self.rate_limiter.can_process(model.id)
        ]
        return (
            random.choices(eligible, weights=[model.weight for model in eligible], k=1)[
                0
            ]
            if eligible
            else None
        )

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[Any, AsyncIterable[bytes]]:
        """代理 Chat Completions，不在网关内重建响应语义。"""
        return await self._proxy_request(request, endpoint="chat_completions")

    async def responses(
        self, request: ResponsesRequest
    ) -> Union[Any, AsyncIterable[bytes]]:
        """代理 Responses API，不与 Chat Completions 互相模拟。"""
        return await self._proxy_request(request, endpoint="responses")

    async def _proxy_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        endpoint: str,
    ) -> Union[Any, AsyncIterable[bytes]]:
        """统一处理能力路由和有边界的 failover。"""
        request_started_at = time.time()
        tried_models: set[str] = set()
        last_error: RetryableUpstreamError | None = None
        payload = self._build_upstream_payload_template(request, endpoint=endpoint)
        required_capabilities = self._required_capabilities(payload, endpoint=endpoint)
        requested_model = request.model.strip()
        group = payload.pop("fallback_group", None)
        auto = requested_model.lower() in {"auto", "any", "default", "*", ""}
        if group is not None and (
            not isinstance(group, str) or group not in self.fallback_groups or not auto
        ):
            raise GatewayAPIError(
                "fallback_group requires model=auto and a configured group",
                status_code=400,
                code="invalid_fallback_group",
                param="fallback_group",
            )
        if requested_model.lower() not in {"auto", "any", "default", "*", ""}:
            if self.resolve_model_id(requested_model) is None:
                raise GatewayAPIError(
                    f"The model '{requested_model}' does not exist",
                    status_code=404,
                    code="model_not_found",
                    param="model",
                )

        pinned = None
        previous = (
            payload.get("previous_response_id") if endpoint == "responses" else None
        )
        if previous is not None:
            if not isinstance(previous, str) or not previous:
                raise GatewayAPIError(
                    "previous_response_id must be a non-empty string",
                    status_code=400,
                    code="invalid_previous_response_id",
                    param="previous_response_id",
                )
            pinned = self.affinity.get(previous)
            if pinned is None:
                raise GatewayAPIError(
                    "Response affinity is missing, expired or ambiguous",
                    status_code=409,
                    code="response_affinity_lost",
                    param="previous_response_id",
                )
            if (not auto and self.resolve_model_id(requested_model) != pinned) or (
                group and pinned not in self.fallback_groups[group]
            ):
                raise GatewayAPIError(
                    "Requested route conflicts with response affinity",
                    status_code=409,
                    code="response_affinity_conflict",
                    param="model",
                )
        route_ids = (
            [pinned]
            if pinned
            else (
                list(self.fallback_groups[group])
                if group
                else (
                    [self.resolve_model_id(requested_model)]
                    if not auto
                    else [model.id for model in self.models]
                )
            )
        )
        capable_models = [
            model
            for model in self.models
            if model.id in route_ids
            and model.weight > 0
            and model.supports(required_capabilities)
        ]
        if not capable_models:
            raise GatewayAPIError(
                "Selected routes do not support the required capabilities",
                status_code=400,
                code="unsupported_capability",
                param="model",
            )
        max_retries = min(len(route_ids), self.max_attempts)
        for _attempt in range(max_retries):
            if group and not pinned:
                model = next(
                    (
                        candidate
                        for route_id in route_ids
                        if (
                            candidate := self.get_available_model(
                                route_id,
                                list(tried_models),
                                required_capabilities=required_capabilities,
                            )
                        )
                        is not None
                    ),
                    None,
                )
            else:
                model = self.get_available_model(
                    pinned or requested_model,
                    list(tried_models),
                    required_capabilities=required_capabilities,
                )
            if not model:
                break
            tried_models.add(model.id)
            if not self.rate_limiter.acquire(model.id):
                continue
            try:
                response = await self._call_model_api(
                    model,
                    payload,
                    endpoint=endpoint,
                )
                if isinstance(response, AsyncIterable):
                    return response
                if endpoint == "responses" and isinstance(response, dict):
                    self._remember_response(response, model.id)
                elapsed = time.time() - request_started_at
                self.dispatcher.update_model_metrics(model.id, elapsed, True)
                self.dispatcher.mark_model_success(model.id)
                return response
            except RetryableUpstreamError as exc:
                last_error = exc
                elapsed = time.time() - request_started_at
                self.dispatcher.update_model_metrics(model.id, elapsed, False)
                self.dispatcher.mark_model_failed(model.id)
                continue

        if last_error is not None:
            raise last_error
        capable_models = [
            model
            for model in self.models
            if model.weight > 0 and model.supports(required_capabilities)
        ]
        if not capable_models:
            missing = ", ".join(sorted(required_capabilities))
            raise GatewayAPIError(
                f"No available model supports the required capabilities: {missing}",
                status_code=400,
                code="unsupported_capability",
                param="model",
            )
        raise GatewayAPIError(
            "No capable model is currently available",
            status_code=503,
            error_type="server_error",
            code="model_unavailable",
            param="model",
        )

    def _remember_response(self, response: dict[str, Any], route_id: str) -> None:
        response_id = response.get("id")
        if isinstance(response_id, str) and response_id:
            try:
                self.affinity.remember(response_id, route_id)
            except ValueError as error:
                raise GatewayAPIError(
                    str(error), status_code=409, code="response_affinity_ambiguous"
                ) from error

    async def _call_model_api(
        self,
        model: ModelConfig,
        payload_template: dict[str, Any],
        *,
        endpoint: str,
    ) -> Union[Any, AsyncIterable[bytes]]:
        """调用一个上游；只将响应头到达前的安全错误标记为可重试。"""
        if not self.session_pool:
            raise GatewayAPIError(
                "Gateway session pool is not initialized",
                status_code=503,
                error_type="server_error",
                code="service_unavailable",
            )
        session = await self.session_pool.get_or_create(
            ssl_verify=model.ssl_verify, proxy=model.proxy
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model.api_key}",
        }
        payload = dict(payload_template)
        payload["model"] = model.model
        if endpoint == "chat_completions" and "temperature" not in payload:
            payload["temperature"] = model.temperature
        stream = bool(payload.get("stream", False))
        timeout = aiohttp.ClientTimeout(
            connect=model.connect_timeout,
            total=model.read_timeout,
            sock_read=300 if stream else model.read_timeout,
        )
        start_time = time.time()

        # session.post 返回前尚未向客户端开始响应，连接失败可安全 failover。
        try:
            resp = await session.post(
                model.api_url_for(endpoint),
                headers=headers,
                json=payload,
                timeout=timeout,
                proxy=model.proxy or None,
            )
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
            raise RetryableUpstreamError(
                "Unable to connect to upstream model",
                status_code=502,
                error_type="server_error",
                code="upstream_connection_error",
            ) from exc

        peer_ip = self._extract_peer_ip(resp, model)
        self._log_upstream_response(model, endpoint, resp.status, peer_ip)
        if not 200 <= resp.status < 300:
            try:
                error = await self._upstream_error(resp)
            finally:
                resp.close()
            if resp.status in RETRYABLE_UPSTREAM_STATUSES:
                raise RetryableUpstreamError(
                    error.get("message", "Retryable upstream error"),
                    status_code=resp.status,
                    upstream_error=error,
                )
            raise GatewayAPIError(
                error.get("message", "Upstream request failed"),
                status_code=resp.status,
                upstream_error=error,
            )

        if stream:
            return UpstreamStream(
                self._proxy_sse_response(resp, model, start_time, endpoint=endpoint),
                resp,
            )

        try:
            # 上游 JSON 是公共响应；不重建 choices，不丢弃 tool_calls/
            # logprobs/usage 扩展或 Responses 字段。
            return await resp.json()
        except Exception as exc:
            raise GatewayAPIError(
                "Upstream returned an invalid JSON response",
                status_code=502,
                error_type="server_error",
                code="upstream_response_error",
            ) from exc
        finally:
            resp.close()

    @staticmethod
    def _request_requires_json_schema(request: ChatCompletionRequest) -> bool:
        """仅严格 json_schema 需要显式能力；json_object 继续普通透传。"""
        payload = request.model_dump(mode="python", exclude_unset=True)
        response_format = payload.get("response_format")
        return (
            isinstance(response_format, dict)
            and str(response_format.get("type", "")).lower() == "json_schema"
        )

    @staticmethod
    def _build_upstream_payload_template(
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        endpoint: str,
    ) -> dict[str, Any]:
        """保留客户端提交的全部已知/未知字段，只在调用时替换 model。"""
        payload = request.model_dump(mode="python", exclude_unset=True)
        if endpoint == "chat_completions":
            payload.setdefault("stream", False)
        return payload

    def _build_upstream_payload(
        self, model: ModelConfig, request: ChatCompletionRequest
    ) -> dict[str, Any]:
        """保留旧的单元测试/内部调用入口。"""
        payload = self._build_upstream_payload_template(
            request,
            endpoint="chat_completions",
        )
        payload["model"] = model.model
        payload.setdefault("temperature", model.temperature)
        return payload

    @classmethod
    def _required_capabilities(
        cls,
        payload: dict[str, Any],
        *,
        endpoint: str,
    ) -> set[str]:
        """从请求中提取路由所需能力，不转换 Chat/Responses 语义。"""
        required = {endpoint}
        if payload.get("stream") is True:
            required.add("stream")
        if payload.get("tools") or payload.get("tool_choice") is not None:
            required.add("tools")
        if endpoint == "responses" and payload.get("previous_response_id"):
            required.add("previous_response_id")
        if payload.get("n", 1) not in (None, 1):
            required.add("n")
        if payload.get("logprobs") or payload.get("top_logprobs") is not None:
            required.add("logprobs")
        if cls._payload_requires_json_schema(payload, endpoint=endpoint):
            required.add("json_schema")
        content = (
            payload.get("messages")
            if endpoint == "chat_completions"
            else payload.get("input")
        )
        if cls._contains_multimodal_content(content):
            required.add("multimodal")
        return required

    @staticmethod
    def _payload_requires_json_schema(
        payload: dict[str, Any],
        *,
        endpoint: str,
    ) -> bool:
        response_format = payload.get("response_format")
        if (
            isinstance(response_format, dict)
            and str(response_format.get("type", "")).lower() == "json_schema"
        ):
            return True
        if endpoint == "responses":
            text = payload.get("text")
            if isinstance(text, dict):
                text_format = text.get("format")
                return (
                    isinstance(text_format, dict)
                    and str(text_format.get("type", "")).lower() == "json_schema"
                )
        return False

    @classmethod
    def _contains_multimodal_content(cls, value: Any) -> bool:
        if isinstance(value, dict):
            item_type = str(value.get("type", "")).lower()
            if item_type in {
                "image",
                "image_url",
                "input_image",
                "input_audio",
                "audio",
                "video",
            }:
                return True
            if "image_url" in value or "input_audio" in value:
                return True
            return any(
                cls._contains_multimodal_content(item) for item in value.values()
            )
        if isinstance(value, list):
            return any(cls._contains_multimodal_content(item) for item in value)
        return False

    @staticmethod
    async def _upstream_error(response: aiohttp.ClientResponse) -> dict[str, Any]:
        """保留上游 OpenAI error object；非标准错误包装成标准形式。"""
        try:
            data = await response.json()
        except Exception:
            try:
                message = (await response.text())[:1000]
            except Exception:
                message = "Upstream request failed"
            return {
                "message": message or "Upstream request failed",
                "type": "server_error",
                "param": None,
                "code": "upstream_error",
            }
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            return data["error"]
        return {
            "message": str(data)[:1000],
            "type": "server_error",
            "param": None,
            "code": "upstream_error",
        }

    def _extract_peer_ip(
        self, resp: aiohttp.ClientResponse, model: ModelConfig
    ) -> str | None:
        """
        从响应中提取对端 IP 地址（用于日志记录）

        通过 transport 的 peername 获取实际连接的服务器 IP，
        代理模式下返回 None（无法获取真实 IP）。

        Args:
            resp: aiohttp 响应对象
            model: 模型配置（用于检查是否使用代理）

        Returns:
            str | None: 对端 IP 地址，代理模式或获取失败返回 None
        """
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
        self,
        model: ModelConfig,
        endpoint: str,
        status: int,
        peer_ip: str | None,
    ) -> None:
        """
        记录上游 API 响应日志

        Args:
            model: 模型配置
            endpoint: canonical Gateway endpoint
            status: HTTP 状态码
            peer_ip: 对端 IP 地址（可选，代理模式下为 None）
        """
        upstream_url = model.api_url_for(endpoint)
        if peer_ip:
            logging.info(
                "上游响应 model=%s status=%s url=%s ip=%s",
                model.id,
                status,
                upstream_url,
                peer_ip,
            )
        else:
            logging.info(
                "上游响应 model=%s status=%s url=%s",
                model.id,
                status,
                upstream_url,
            )

    async def _proxy_sse_response(
        self,
        response: aiohttp.ClientResponse,
        model: ModelConfig,
        start_time: float,
        *,
        endpoint: str = "chat_completions",
    ) -> AsyncIterable[bytes]:
        """正常字节原样转发；异常流发出可识别错误，不切换后端。"""
        completed = False
        yielded = False
        failed = False
        buffer = b""
        sequence = -1
        try:
            async for chunk in response.content.iter_any():
                if not chunk:
                    continue
                buffer = (buffer + chunk).replace(b"\r\n", b"\n")
                while b"\n\n" in buffer:
                    frame, buffer = buffer.split(b"\n\n", 1)
                    data = b"\n".join(
                        line[5:].lstrip(b" ")
                        for line in frame.split(b"\n")
                        if line.startswith(b"data:")
                    )
                    if not data:
                        continue
                    if data == b"[DONE]":
                        completed = True
                        continue
                    event = json.loads(data)
                    if not isinstance(event, dict):
                        raise ValueError("SSE data must be an object")
                    if type(event.get("sequence_number")) is int:
                        sequence = max(sequence, event["sequence_number"])
                    event_type = event.get("type", "")
                    if endpoint == "responses" and isinstance(
                        event.get("response"), dict
                    ):
                        self._remember_response(event["response"], model.id)
                    if event_type in {
                        "response.completed",
                        "response.failed",
                        "response.incomplete",
                        "response.cancelled",
                    }:
                        completed = True
                    if (
                        event_type
                        in {"error", "response.failed", "response.incomplete"}
                        or "error" in event
                    ):
                        failed = True
                        completed = True
                if len(buffer) > 1024 * 1024:
                    raise ValueError("SSE frame exceeds buffer limit")
                yielded = True
                yield chunk
            if not completed:
                raise ValueError("upstream stream ended before terminal event")
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            OSError,
            ValueError,
            GatewayAPIError,
        ):
            failed = True
            error = {
                "message": "Upstream stream interrupted or invalid",
                "type": "server_error",
                "code": "upstream_stream_error",
                "param": None,
            }
            if endpoint == "responses":
                event = {**error, "type": "error", "sequence_number": sequence + 1}
                yield (
                    ("\n\n" if buffer else "")
                    + "event: error\ndata: "
                    + json.dumps(event)
                    + "\n\n"
                ).encode()
            else:
                yield (
                    ("\n\n" if buffer else "")
                    + "data: "
                    + json.dumps({"error": error})
                    + "\n\n"
                ).encode()
        finally:
            elapsed = time.time() - start_time
            success = completed and yielded and not failed
            self.dispatcher.update_model_metrics(model.id, elapsed, success)
            if success:
                self.dispatcher.mark_model_success(model.id)
            else:
                # 客户端已经收到响应头，失败只能结束当前流，不能切换模型。
                self.dispatcher.mark_model_failed(model.id)
            if not response.closed:
                response.close()

    def get_uptime(self) -> float:
        """
        获取服务运行时间

        Returns:
            float: 自服务启动以来经过的秒数
        """
        return time.time() - self.start_time

    def get_capabilities(self) -> dict[str, Any]:
        """返回每个配置模型的有效 capability 和 endpoint 信息。"""
        data = []
        all_capabilities: set[str] = set()
        for model in self.models:
            capabilities = sorted(model.capabilities)
            all_capabilities.update(capabilities)
            data.append(
                {
                    "id": model.id,
                    "name": model.name,
                    "model": model.model,
                    "channel_id": model.channel_id,
                    "capabilities": capabilities,
                    "endpoints": {
                        "chat_completions": (
                            model.api_url_for("chat_completions")
                            if "chat_completions" in model.capabilities
                            else None
                        ),
                        "responses": (
                            model.api_url_for("responses")
                            if "responses" in model.capabilities
                            else None
                        ),
                    },
                }
            )
        return {
            "object": "list",
            "capabilities": sorted(all_capabilities),
            "data": data,
        }

    def get_health_status(self) -> dict[str, Any]:
        """
        获取服务健康状态

        Returns:
            dict: 包含以下字段:
                - status: "healthy" | "degraded" | "unhealthy"
                - available_models: 可用模型数
                - total_models: 模型总数
                - uptime: 运行时间（秒）
        """
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
        """
        获取所有模型的详细信息（用于 /admin/models 接口）

        Returns:
            dict: 包含以下字段:
                - models: 模型统计列表（来自 dispatcher.get_all_model_stats）
                - total: 模型总数
                - available: 当前可用模型数
        """
        stats = self.dispatcher.get_all_model_stats()
        available = sum(1 for s in stats if s["available"])

        return {
            "models": stats,
            "total": len(stats),
            "available": available,
        }
