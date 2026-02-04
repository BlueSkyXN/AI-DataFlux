"""
Control Server 主模块

提供 AI-DataFlux 的 Web GUI 控制面板。

功能:
    - 配置文件读写 API
    - 进程管理 API (Gateway/Process 启动/停止)
    - 状态查询 API
    - WebSocket 日志流
    - 静态文件服务 (web/dist/)

启动方式:
    python cli.py gui  # 启动控制面板，自动打开浏览器

端口:
    - Control Server: 8790 (默认)
    - Gateway: 8787 (由 ProcessManager 管理)
"""

import asyncio
import logging
import os
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config_api import read_config, write_config, get_project_root
from .process_manager import get_process_manager, ProcessManager


# 项目根目录
PROJECT_ROOT = get_project_root()
WEB_DIST_DIR = os.path.join(PROJECT_ROOT, "web", "dist")


# ========== Pydantic Models ==========


class ConfigWriteRequest(BaseModel):
    """配置文件写入请求"""

    path: str
    content: str


class GatewayStartRequest(BaseModel):
    """Gateway 启动请求"""

    config_path: str = "config.yaml"
    port: int = 8787
    workers: int = 1


class ProcessStartRequest(BaseModel):
    """Process 启动请求"""

    config_path: str = "config.yaml"


# ========== WebSocket Manager ==========


class LogConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.active_connections: dict[str, List[WebSocket]] = {
            "gateway": [],
            "process": [],
        }

    async def connect(self, websocket: WebSocket, target: str):
        """接受连接"""
        await websocket.accept()
        if target in self.active_connections:
            self.active_connections[target].append(websocket)

    def disconnect(self, websocket: WebSocket, target: str):
        """断开连接"""
        if target in self.active_connections:
            if websocket in self.active_connections[target]:
                self.active_connections[target].remove(websocket)

    async def broadcast(self, target: str, message: str):
        """广播消息"""
        if target not in self.active_connections:
            return
        disconnected = []
        for connection in self.active_connections[target]:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn, target)


ws_manager = LogConnectionManager()


# ========== Lifespan ==========


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    manager = get_process_manager()

    # 注册日志回调
    def log_callback(name: str, line: str):
        # 在事件循环中广播
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(ws_manager.broadcast(name, line))
            # 添加异常处理回调，避免未处理的异常警告
            task.add_done_callback(
                lambda t: t.exception() if t.done() and not t.cancelled() else None
            )
        except RuntimeError:
            pass

    manager.add_log_callback("gateway", log_callback)
    manager.add_log_callback("process", log_callback)

    # 打开浏览器 (在服务启动后)
    app_state = app.state
    if getattr(app_state, "open_browser", False):
        port = getattr(app_state, "port", 8790)
        url = f"http://127.0.0.1:{port}"
        logging.info(f"Opening browser: {url}")
        webbrowser.open(url)

    yield

    # 关闭时清理进程
    manager.remove_log_callback("gateway", log_callback)
    manager.remove_log_callback("process", log_callback)
    manager.shutdown()


# ========== FastAPI App ==========


def create_control_app() -> FastAPI:
    """创建 FastAPI 应用"""

    app = FastAPI(
        title="AI-DataFlux Control Panel",
        description="Web GUI for managing AI-DataFlux Gateway and Process",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS (开发态需要)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ========== Config API ==========

    @app.get("/api/config")
    async def api_get_config(path: str = Query(default="config.yaml")):
        """读取配置文件"""
        content = read_config(path)
        return {"path": path, "content": content}

    @app.put("/api/config")
    async def api_put_config(request: ConfigWriteRequest):
        """写入配置文件"""
        result = write_config(request.path, request.content)
        return result

    # ========== Gateway API ==========

    @app.post("/api/gateway/start")
    async def api_gateway_start(request: GatewayStartRequest = None):
        """启动 Gateway"""
        if request is None:
            request = GatewayStartRequest()
        manager = get_process_manager()
        result = manager.start_gateway(
            config_path=request.config_path,
            port=request.port,
            workers=request.workers,
        )
        return result

    @app.post("/api/gateway/stop")
    async def api_gateway_stop():
        """停止 Gateway"""
        manager = get_process_manager()
        result = manager.stop_gateway()
        return result

    # ========== Process API ==========

    @app.post("/api/process/start")
    async def api_process_start(request: ProcessStartRequest = None):
        """启动 Process"""
        if request is None:
            request = ProcessStartRequest()
        manager = get_process_manager()
        result = manager.start_process(config_path=request.config_path)
        return result

    @app.post("/api/process/stop")
    async def api_process_stop():
        """停止 Process"""
        manager = get_process_manager()
        result = manager.stop_process()
        return result

    # ========== Status API ==========

    @app.get("/api/status")
    async def api_status():
        """获取所有进程状态"""
        manager = get_process_manager()
        return manager.get_all_status()

    # ========== WebSocket Logs ==========

    @app.websocket("/api/logs")
    async def websocket_logs(
        websocket: WebSocket, target: str = Query(default="gateway")
    ):
        """WebSocket 日志流"""
        if target not in ("gateway", "process"):
            await websocket.close(code=1008)  # Policy Violation
            return

        await ws_manager.connect(websocket, target)

        # 发送历史日志
        manager = get_process_manager()
        history = manager.get_log_buffer(target)
        for line in history:
            try:
                await websocket.send_text(line)
            except Exception:
                break

        try:
            while True:
                # 保持连接，等待客户端消息 (心跳)
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, target)
        except Exception:
            ws_manager.disconnect(websocket, target)

    # ========== Static Files ==========

    # 检查 web/dist 是否存在
    if os.path.isdir(WEB_DIST_DIR):
        # 挂载静态文件
        app.mount(
            "/assets",
            StaticFiles(directory=os.path.join(WEB_DIST_DIR, "assets")),
            name="assets",
        )

        @app.get("/")
        async def serve_index():
            """服务前端首页"""
            index_path = os.path.join(WEB_DIST_DIR, "index.html")
            if os.path.isfile(index_path):
                return FileResponse(index_path)
            return {"message": "AI-DataFlux Control Panel", "status": "running"}

        @app.get("/{path:path}")
        async def serve_static(path: str):
            """服务静态文件或 SPA fallback"""
            file_path = os.path.join(WEB_DIST_DIR, path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            # SPA fallback: 返回 index.html
            index_path = os.path.join(WEB_DIST_DIR, "index.html")
            if os.path.isfile(index_path):
                return FileResponse(index_path)
            raise HTTPException(404, "Not found")
    else:
        # 没有前端，显示 API 信息

        @app.get("/")
        async def serve_api_info():
            """API 信息"""
            return {
                "message": "AI-DataFlux Control Panel API",
                "status": "running",
                "frontend": "not built",
                "hint": "Run 'cd web && npm install && npm run build' to build frontend",
            }

    return app


def run_control_server(
    host: str = "127.0.0.1",
    port: int = 8790,
    open_browser: bool = True,
) -> None:
    """
    启动 Control Server

    Args:
        host: 监听地址 (默认 127.0.0.1，仅本地访问)
        port: 监听端口 (默认 8790)
        open_browser: 是否自动打开浏览器
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 创建应用
    app = create_control_app()

    # 设置应用状态 (用于 lifespan 中打开浏览器)
    app.state.open_browser = open_browser
    app.state.port = port

    logging.info(f"Starting Control Server on http://{host}:{port}")

    # 启动服务
    uvicorn.run(app, host=host, port=port, log_level="info")
