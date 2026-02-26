"""
API 网关模块 (src.gateway)

本模块实现 OpenAI 兼容的 API 网关，提供多模型管理、负载均衡、
自动故障切换、令牌桶限流等企业级功能。

模块架构:
    ┌─────────────────────────────────────────────────────────────┐
    │                     FastAPI Application                      │
    │                        (app.py)                              │
    │  ┌─────────────────────────────────────────────────────────┐│
    │  │                   FluxApiService                        ││
    │  │                    (service.py)                         ││
    │  │  ┌───────────┐  ┌───────────┐  ┌────────────────────┐  ││
    │  │  │ Dispatcher│  │  Limiter  │  │    SessionPool     │  ││
    │  │  │(调度器)   │  │ (限流器)  │  │  (HTTP 连接池)     │  ││
    │  │  └───────────┘  └───────────┘  └────────────────────┘  ││
    │  │        │              │               │                 ││
    │  │        ▼              ▼               ▼                 ││
    │  │  加权随机选择   令牌桶限流    按配置复用连接            ││
    │  │  指数退避恢复   平滑流量      IP 池轮询                 ││
    │  └─────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────┘

核心组件:
    - FluxApiService: 核心服务类，协调所有组件
    - ModelDispatcher: 模型调度器，实现加权随机选择和故障退避
    - ModelRateLimiter: 令牌桶限流器，保护上游 API
    - SessionPool: HTTP 连接池，按 (ssl_verify, proxy) 复用连接
    - RoundRobinResolver: IP 池轮询解析器，实现多 IP 负载均衡

API 端点:
    POST /v1/chat/completions - OpenAI 兼容的聊天补全接口
    GET  /v1/models          - 获取可用模型列表
    GET  /admin/models       - 管理接口：获取模型详情
    GET  /admin/health       - 健康检查端点

使用示例:
    from src.gateway import create_app, run_server

    # 方式一：直接运行服务器
    run_server(config_path="config.yaml", port=8787)

    # 方式二：获取 FastAPI 应用实例（用于测试等）
    app = create_app("config.yaml")

本模块导出清单:
    - create_app(config_path: str) -> FastAPI
        创建并配置 FastAPI 应用实例
        输入: config_path — YAML 配置文件路径
        输出: 配置完成的 FastAPI 应用实例

    - run_server(config_path, host, port, workers, reload) -> None
        启动 uvicorn 服务器运行网关
        输入: config_path, host="0.0.0.0", port=8787, workers=1, reload=False
        输出: 无（阻塞运行直到进程终止）

    - FluxApiService(config_path: str)
        核心服务类（从 service 模块重新导出）
        详见 src/gateway/service.py
"""

from .app import create_app, run_server
from .service import FluxApiService

__all__ = [
    "create_app",
    "run_server",
    "FluxApiService",
]
