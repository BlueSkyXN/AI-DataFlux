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
"""

from .app import create_app, run_server
from .service import FluxApiService

__all__ = [
    "create_app",
    "run_server",
    "FluxApiService",
]
