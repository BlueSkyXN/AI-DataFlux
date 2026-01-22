"""
FastAPI 应用创建和路由定义
"""

import argparse
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, AsyncIterable, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .service import FluxApiService
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    ModelsResponse,
    ModelInfo,
)


# 全局服务实例
_service: FluxApiService | None = None


def get_service() -> FluxApiService:
    """获取服务实例"""
    if _service is None:
        raise RuntimeError("FluxApiService 未初始化")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    global _service

    # 启动
    if _service:
        await _service.startup()
        logging.info("FluxApiService 启动完成")

    yield

    # 关闭
    if _service:
        await _service.shutdown()
        logging.info("FluxApiService 已关闭")


def create_app(config_path: str) -> FastAPI:
    """
    创建 FastAPI 应用

    Args:
        config_path: 配置文件路径

    Returns:
        FastAPI 应用实例
    """
    global _service

    # 初始化服务
    _service = FluxApiService(config_path)

    # 创建应用
    app = FastAPI(
        title="Flux API Gateway",
        description="OpenAI API 兼容的多模型网关",
        version="2.0.0",
        lifespan=lifespan,
    )

    # 注册路由
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """注册路由"""

    @app.middleware("http")
    async def check_service_availability(request: Request, call_next):
        """检查服务可用性"""
        if _service is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Service not initialized"},
            )
        return await call_next(request)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completion(
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, StreamingResponse]:
        """
        聊天补全端点（支持流式和非流式）

        返回:
            - 非流式: ChatCompletionResponse
            - 流式: StreamingResponse (text/event-stream)
        """
        try:
            service = get_service()
            result = await service.chat_completion(request)

            # 如果是流式响应，包装为 StreamingResponse
            if isinstance(result, AsyncIterable):
                return StreamingResponse(
                    content=result,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
                    },
                )

            # 非流式响应，直接返回
            return result

        except Exception as e:
            logging.error(f"聊天补全请求失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/models")
    async def list_models():
        """列出可用模型 (OpenAI 兼容)"""
        service = get_service()

        # 获取模型列表并去重
        models_list = []
        seen_exposed_ids = set()

        for model in service.models:
            # 决定对外暴露的ID (优先用 name, 其次 model, 最后 id)
            exposed_id = model.name or model.model or model.id

            # 只添加权重>0且未添加过的模型
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
        """管理接口: 获取模型详情"""
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
        """健康检查"""
        service = get_service()
        health = service.get_health_status()

        return HealthResponse(
            status=health["status"],
            available_models=health["available_models"],
            total_models=health["total_models"],
            uptime=health["uptime"],
        )

    @app.get("/")
    async def root():
        """根路径"""
        return {
            "name": "Flux API Gateway",
            "version": "2.0.0",
            "status": "running",
        }


def run_server(
    config_path: str,
    host: str = "0.0.0.0",
    port: int = 8787,
    workers: int = 1,
    reload: bool = False,
) -> None:
    """
    运行服务器

    Args:
        config_path: 配置文件路径
        host: 监听地址
        port: 监听端口
        workers: 工作进程数
        reload: 是否自动重载
    """
    # 创建应用
    app = create_app(config_path)

    # 运行服务器
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


def main() -> None:
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Flux API Gateway")
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听地址 (默认: 0.0.0.0)",
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
