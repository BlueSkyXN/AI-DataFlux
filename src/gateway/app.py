"""
FastAPI 应用创建和路由定义模块

本模块负责创建 FastAPI 应用实例、注册 API 路由、管理应用生命周期。
是 API 网关的 HTTP 层入口。

核心功能:
    - 创建并配置 FastAPI 应用实例
    - 注册 OpenAI 兼容的 API 路由
    - 管理 FluxApiService 的生命周期（启动/关闭）
    - 提供健康检查和管理端点

API 路由:
    POST /v1/chat/completions - 聊天补全（支持流式/非流式）
    POST /v1/responses        - Responses API（支持流式/非流式）
    GET  /v1/models          - 列出可用模型（OpenAI 兼容格式）
    GET  /admin/models       - 管理接口：模型详情和统计
    GET  /admin/capabilities - 管理接口：有效能力与上游 endpoint
    GET  /admin/health       - 健康检查
    GET  /                   - 根路径，返回网关信息

使用方式:
    # 直接运行
    python -m src.gateway.app --config config.yaml --port 8787

    # 通过 CLI
    python cli.py gateway --config config.yaml --port 8787

    # 程序化使用
    from src.gateway.app import create_app, run_server
    app = create_app("config.yaml")
    run_server("config.yaml", port=8787)

架构说明:
    本模块使用全局变量 _service 存储 FluxApiService 实例，
    通过 lifespan 上下文管理器确保服务的正确启动和关闭。

    请求处理流程:
    1. FastAPI 接收 HTTP 请求
    2. 路由处理函数调用 get_service() 获取服务实例
    3. 服务实例处理请求（模型选择、API 调用等）
    4. 返回响应（JSON 或 SSE 流）

函数清单:
    - get_service() -> FluxApiService
        获取全局服务实例，未初始化时抛出 RuntimeError

    - lifespan(app: FastAPI) -> AsyncGenerator
        应用生命周期管理器，负责 startup/shutdown

    - create_app(config_path: str) -> FastAPI
        工厂函数：创建 FastAPI 应用，初始化服务，注册路由
        输入: config_path — YAML 配置文件路径
        输出: 配置完成的 FastAPI 实例

    - _register_routes(app: FastAPI) -> None
        注册所有 API 路由和 HTTP 中间件（内部函数）

    - run_server(config_path, host, port, workers, reload) -> None
        便捷函数：创建应用并启动 uvicorn 服务器
        输入: config_path, host="0.0.0.0", port=8787, workers=1, reload=False

    - main() -> None
        命令行入口，解析参数并调用 run_server

关键变量:
    - _service (FluxApiService | None): 全局服务单例，lifespan 管理其生命周期

依赖模块:
    - FastAPI: Web 框架
    - uvicorn: ASGI 服务器
    - FluxApiService: 核心服务逻辑 (service.py)
    - schemas: 请求/响应数据模型 (schemas.py)
"""

import argparse
import inspect
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, AsyncIterable, Awaitable, Callable, Union

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src import __version__
from src.config import (
    load_config,
    make_token_checker,
    require_gateway_config,
    resolve_access_token,
)

from .service import FluxApiService, GatewayAPIError, UpstreamStream
from .schemas import (
    ChatCompletionRequest,
    HealthResponse,
    ModelsResponse,
    ModelInfo,
    ResponsesRequest,
)

# 全局服务实例，在应用启动时初始化
_service: FluxApiService | None = None
IncomingTokenChecker = Callable[[str], bool | Awaitable[bool]]


def allow_all_incoming_tokens(_token: str) -> bool:
    """默认兼容策略；主程序可在 create_app 时注入实际 token 检查器。"""
    return True


def _extract_incoming_bearer_token(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


async def _check_incoming_token(
    checker: IncomingTokenChecker,
    token: str,
) -> bool:
    result = checker(token)
    if inspect.isawaitable(result):
        return bool(await result)
    return bool(result)


def get_service() -> FluxApiService:
    """
    获取全局 FluxApiService 实例

    Returns:
        FluxApiService: 已初始化的服务实例

    Raises:
        RuntimeError: 如果服务尚未初始化
    """
    if _service is None:
        raise RuntimeError("FluxApiService 未初始化")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    应用生命周期管理器

    使用 FastAPI 的 lifespan 机制管理服务的启动和关闭，
    确保资源的正确初始化和清理。

    Args:
        app: FastAPI 应用实例

    Yields:
        None: 控制权交给应用运行

    生命周期:
        1. 启动阶段: 调用 _service.startup() 初始化连接池等资源
        2. 运行阶段: yield，应用正常处理请求
        3. 关闭阶段: 调用 _service.shutdown() 释放资源
    """
    global _service

    # 启动阶段：初始化异步资源
    if _service:
        await _service.startup()
        logging.info("FluxApiService 启动完成")

    yield  # 应用运行阶段

    # 关闭阶段：清理资源
    if _service:
        await _service.shutdown()
        logging.info("FluxApiService 已关闭")


def create_app(
    config_path: str,
    incoming_token_checker: IncomingTokenChecker | None = None,
) -> FastAPI:
    """
    创建并配置 FastAPI 应用实例

    初始化 FluxApiService 服务，创建 FastAPI 应用，注册所有路由。
    这是创建网关应用的主要工厂函数。

    Args:
        config_path: 配置文件路径（YAML 格式）

    Returns:
        FastAPI: 配置完成的 FastAPI 应用实例

    配置加载:
        配置文件应包含 models、channels、gateway 等配置节，
        详见 config-example.yaml
    """
    global _service

    # 初始化核心服务（同步部分）
    _service = FluxApiService(config_path)

    # 创建 FastAPI 应用
    app = FastAPI(
        title="Flux API Gateway",
        description="OpenAI API 兼容的多模型网关",
        version=__version__,
        lifespan=lifespan,  # 使用生命周期管理器
    )
    if incoming_token_checker is None:
        merged_config = load_config(config_path)
        access_token = resolve_access_token(
            merged_config, host="127.0.0.1", allow_generate_loopback=True
        )
        incoming_token_checker = make_token_checker(access_token)
        app.state.access_token = access_token
    else:
        app.state.access_token = None
    app.state.incoming_token_checker = incoming_token_checker

    # 注册所有 API 路由
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """
    注册所有 API 路由和中间件

    包括:
    - HTTP 中间件：服务可用性检查
    - POST /v1/chat/completions：聊天补全
    - GET /v1/models：模型列表
    - GET /admin/models：管理接口
    - GET /admin/health：健康检查
    - GET /：根路径

    Args:
        app: FastAPI 应用实例
    """

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError):
        locations = [
            ".".join(str(part) for part in item["loc"] if part != "body")
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Invalid request fields: " + ", ".join(locations),
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "param": locations[0] if locations else None,
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, error: StarletteHTTPException):
        return JSONResponse(
            status_code=error.status_code,
            headers=error.headers,
            content={
                "error": {
                    "message": str(error.detail),
                    "type": "invalid_request_error",
                    "code": "http_error",
                    "param": None,
                }
            },
        )

    @app.middleware("http")
    async def check_service_availability(request: Request, call_next):
        """
        HTTP 中间件：检查服务是否已初始化

        在所有请求处理前检查 _service 是否可用，
        避免在服务未就绪时处理请求。
        """
        if _service is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Service not initialized",
                        "type": "server_error",
                        "code": "service_unavailable",
                        "param": None,
                    }
                },
            )
        if request.url.path.startswith(("/v1/", "/admin/")):
            token = _extract_incoming_bearer_token(request)
            checker = request.app.state.incoming_token_checker
            if not await _check_incoming_token(checker, token):
                return JSONResponse(
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                    content={
                        "error": {
                            "message": "Invalid authentication credentials",
                            "type": "authentication_error",
                            "param": None,
                            "code": "invalid_api_key",
                        }
                    },
                )
        return await call_next(request)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completion(
        request: ChatCompletionRequest,
    ) -> Union[dict, StreamingResponse, JSONResponse]:
        """
        聊天补全端点（OpenAI 兼容，支持流式和非流式）

        处理流程:
            1. 解析请求中的模型名称
            2. 选择可用模型（指定模型或加权随机）
            3. 调用上游 API
            4. 返回响应（JSON 或 SSE 流）

        Args:
            request: 聊天补全请求，包含 model、messages 等字段

        Returns:
            - 非流式 (stream=False): ChatCompletionResponse JSON
            - 流式 (stream=True): StreamingResponse (text/event-stream)

        Raises:
            HTTPException: 500 - 所有模型调用失败
        """
        try:
            service = get_service()
            result = await service.chat_completion(request)

            # 流式响应：包装为 SSE StreamingResponse
            if isinstance(result, AsyncIterable):
                return StreamingResponse(
                    content=result,
                    background=(
                        BackgroundTask(result.aclose)
                        if isinstance(result, UpstreamStream)
                        else None
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲（用于反向代理场景）
                    },
                )

            # 非流式响应：直接返回 JSON
            return result

        except GatewayAPIError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.error},
            )
        except Exception:
            logging.exception("聊天补全请求失败")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Internal server error",
                        "type": "server_error",
                        "param": None,
                        "code": "internal_error",
                    }
                },
            )

    @app.post("/v1/responses", response_model=None)
    async def responses(
        request: ResponsesRequest,
    ) -> Union[dict, StreamingResponse, JSONResponse]:
        """OpenAI Responses API 透明代理；不模拟 Chat Completions 语义。"""
        try:
            service = get_service()
            result = await service.responses(request)
            if isinstance(result, AsyncIterable):
                return StreamingResponse(
                    content=result,
                    background=(
                        BackgroundTask(result.aclose)
                        if isinstance(result, UpstreamStream)
                        else None
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            return result
        except GatewayAPIError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.error},
            )
        except Exception:
            logging.exception("Responses 请求失败")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Internal server error",
                        "type": "server_error",
                        "param": None,
                        "code": "internal_error",
                    }
                },
            )

    @app.get("/v1/models")
    async def list_models():
        """
        列出可用模型（OpenAI 兼容格式）

        返回当前配置中权重大于 0 的所有模型，
        格式与 OpenAI /v1/models 接口兼容。

        去重逻辑:
            同一个 exposed_id (name > model > id) 只返回一次

        Returns:
            dict: {"object": "list", "data": [模型列表]}
        """
        service = get_service()

        # 获取模型列表并按 exposed_id 去重
        models_list = []
        seen_exposed_ids = set()

        for model in service.models:
            # 决定对外暴露的ID：优先用 name，其次 model，最后 id
            exposed_id = model.name or model.model or model.id

            # 只添加权重 > 0 且未添加过的模型
            if model.weight > 0 and exposed_id not in seen_exposed_ids:
                models_list.append(
                    {
                        "id": exposed_id,
                        "object": "model",
                        "created": int(service.start_time),  # 使用服务启动时间
                        "owned_by": "flux-api",
                    }
                )
                seen_exposed_ids.add(exposed_id)

        return {
            "object": "list",
            "data": models_list,
        }

    @app.get("/admin/models", response_model=ModelsResponse)
    async def admin_models() -> ModelsResponse:
        """
        管理接口：获取所有模型的详细信息

        返回每个模型的配置和运行时统计，包括：
        - 模型 ID、名称、权重
        - 成功率和平均响应时间
        - 当前可用状态
        - 所属通道

        Returns:
            ModelsResponse: 模型详情列表
        """
        service = get_service()
        info = service.get_models_info()

        return ModelsResponse(
            models=[
                ModelInfo(
                    id=m["id"],
                    name=m.get("name"),
                    model=m["model"],
                    weight=m["weight"],
                    success_rate=m["success_rate"],
                    avg_response_time=m["avg_response_time"],
                    available=m["available"],
                    channel=m.get("channel"),
                )
                for m in info["models"]
            ],
            total=info["total"],
            available=info["available"],
        )

    @app.get("/admin/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """
        健康检查端点

        返回服务健康状态，用于负载均衡器和监控系统。

        状态定义:
            - healthy: 所有模型可用
            - degraded: 部分模型可用
            - unhealthy: 无可用模型

        Returns:
            HealthResponse: 健康状态信息
        """
        service = get_service()
        health = service.get_health_status()

        return HealthResponse(
            status=health["status"],
            available_models=health["available_models"],
            total_models=health["total_models"],
            uptime=health["uptime"],
        )

    @app.get("/admin/capabilities")
    async def admin_capabilities():
        """列出模型/通道合并后的有效 capability。"""
        return get_service().get_capabilities()

    @app.get("/")
    async def root():
        """
        根路径端点

        返回网关基本信息，可用于快速验证服务是否运行。

        Returns:
            dict: 网关名称、版本和状态
        """
        return {
            "name": "Flux API Gateway",
            "version": __version__,
            "status": "running",
        }


def run_server(
    config_path: str,
    host: str = "127.0.0.1",
    port: int = 8787,
    workers: int = 1,
    reload: bool = False,
    incoming_token_checker: IncomingTokenChecker | None = None,
) -> None:
    """
    启动 uvicorn 服务器运行网关应用

    这是一个便捷函数，创建应用并使用 uvicorn 运行。
    适用于生产环境和开发环境。

    Args:
        config_path: 配置文件路径（YAML 格式）
        host: 监听地址（默认 0.0.0.0 接受所有来源）
        port: 监听端口（默认 8787）
        workers: 当前固定为 1，Responses affinity 不跨进程共享
        reload: 是否启用热重载（开发模式使用）

    注意:
        - workers > 1 会被明确拒绝
        - reload 会重建 affinity，旧 previous_response_id 可能被明确拒绝
    """
    root_config = load_config(config_path)
    require_gateway_config(root_config)
    if workers != 1:
        raise ValueError("AI-DataFlux 4.0 仅支持 gateway workers=1")

    merged_config = root_config
    access_token = resolve_access_token(
        merged_config, host=host, allow_generate_loopback=True
    )
    if access_token.generated:
        # Share the generated value with factory workers without writing it to logs.
        os.environ["DATAFLUX_TOKEN"] = access_token.value
    checker = incoming_token_checker or make_token_checker(access_token)

    if workers > 1 or reload:
        if incoming_token_checker is not None:
            raise ValueError("多 worker/reload 模式不支持进程内自定义 token checker")
        os.environ["DATAFLUX_CONFIG"] = str(Path(config_path).resolve())
        os.environ["DATAFLUX_GATEWAY_HOST"] = host
        uvicorn.run(
            "src.gateway.app:create_app_from_env",
            factory=True,
            host=host,
            port=port,
            workers=workers,
            reload=reload,
        )
        return

    app = create_app(config_path, incoming_token_checker=checker)
    uvicorn.run(app, host=host, port=port)


def create_app_from_env() -> FastAPI:
    """Uvicorn factory used when reload or multiple workers are requested."""

    config_path = os.environ.get("DATAFLUX_CONFIG")
    if not config_path:
        raise RuntimeError("DATAFLUX_CONFIG is required")
    host = os.environ.get("DATAFLUX_GATEWAY_HOST", "127.0.0.1")
    merged_config = load_config(config_path)
    access_token = resolve_access_token(
        merged_config, host=host, allow_generate_loopback=True
    )
    return create_app(
        config_path,
        incoming_token_checker=make_token_checker(access_token),
    )


def main() -> None:
    """
    命令行入口函数

    解析命令行参数并启动网关服务器。
    当直接运行本模块时调用 (python -m src.gateway.app)。

    支持的参数:
        --config, -c: 配置文件路径
        --host: 监听地址
        --port, -p: 监听端口
        --workers, -w: 工作进程数
        --reload: 启用热重载
    """
    parser = argparse.ArgumentParser(description="Flux API Gateway")
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址 (默认: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8787,
        help="监听端口 (默认: 8787)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="工作进程数 (默认: 1)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="启用自动重载",
    )

    args = parser.parse_args()

    run_server(
        config_path=args.config,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
