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
from typing import AsyncGenerator, List

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config_api import read_config, write_config
from .process_manager import get_process_manager
from .runtime import find_web_dist_dir, get_project_root


# 项目根目录
PROJECT_ROOT = str(get_project_root())
WEB_DIST_PATH = find_web_dist_dir(get_project_root())
WEB_DIST_DIR = str(WEB_DIST_PATH) if WEB_DIST_PATH else ""


# ========== Pydantic Models ==========


class ConfigWriteRequest(BaseModel):
    """配置文件写入请求"""

    path: str
    content: str


class ConfigValidateRequest(BaseModel):
    """配置文件校验请求"""

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


async def _open_browser_when_ready(
    url: str,
    host: str,
    port: int,
    timeout_seconds: float = 5.0,
    interval_seconds: float = 0.2,
) -> None:
    """
    等待服务端口可连接后再打开浏览器，减少 Connection Refused 的概率。
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while True:
        try:
            _reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            break
        except OSError:
            if loop.time() >= deadline:
                break
            await asyncio.sleep(interval_seconds)

    await asyncio.to_thread(webbrowser.open, url)


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

    # 打开浏览器 (等待端口就绪，避免偶现 Connection Refused)
    app_state = app.state
    if getattr(app_state, "open_browser", False):
        port = getattr(app_state, "port", 8790)
        url = f"http://127.0.0.1:{port}"
        logging.info(f"Opening browser: {url}")
        browser_task = asyncio.create_task(
            _open_browser_when_ready(url=url, host="127.0.0.1", port=port)
        )
        browser_task.add_done_callback(
            lambda t: t.exception() if t.done() and not t.cancelled() else None
        )
        app_state._browser_task = browser_task

    yield

    # 关闭时清理进程
    browser_task = getattr(app_state, "_browser_task", None)
    if browser_task and not browser_task.done():
        browser_task.cancel()
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

    # CORS (仅开发态需要；生产态同源不需要)
    cors_env = os.environ.get("DATAFLUX_GUI_CORS_ORIGINS", "").strip()
    if cors_env:
        cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
    elif not os.path.isdir(WEB_DIST_DIR):
        # 默认仅放开 Vite dev server 的常见 origin（可通过环境变量覆盖）
        cors_origins = [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ]
    else:
        cors_origins = []

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 防止 localhost CSRF：所有写操作要求 application/json
    @app.middleware("http")
    async def require_json_for_api_writes(request: Request, call_next):
        if request.url.path.startswith("/api/") and request.method in (
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
        ):
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                return JSONResponse(
                    status_code=415,
                    content={"detail": "Content-Type must be application/json"},
                )
        return await call_next(request)

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

    @app.post("/api/config/validate")
    async def api_validate_config(request: ConfigValidateRequest):
        """校验 YAML 配置内容（仅语法/解析）"""
        import yaml

        try:
            yaml.safe_load(request.content)
            return {"valid": True}
        except yaml.YAMLError as e:
            return {"valid": False, "error": str(e)}

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
        return await manager.get_all_status_async()

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
                logging.debug(f"Failed to send history log to WebSocket: {target}")
                break

        try:
            while True:
                # 保持连接，等待客户端消息 (心跳)
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            logging.debug(f"WebSocket client disconnected: {target}")
            ws_manager.disconnect(websocket, target)
        except Exception:
            logging.exception(f"WebSocket error for {target}")
            ws_manager.disconnect(websocket, target)

    # ========== Static Files ==========

    # 检查 web/dist 是否存在
    if os.path.isdir(WEB_DIST_DIR):
        assets_dir = os.path.join(WEB_DIST_DIR, "assets")
        # 挂载静态文件
        if os.path.isdir(assets_dir):
            app.mount(
                "/assets",
                StaticFiles(directory=assets_dir),
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
            # 防止 path traversal：仅允许访问 WEB_DIST_DIR 内的文件
            base_dir = os.path.realpath(WEB_DIST_DIR)
            file_path = os.path.realpath(os.path.join(base_dir, path))
            try:
                common = os.path.commonpath([base_dir, file_path])
                if common != base_dir:
                    raise HTTPException(404, "Not found")
            except ValueError:
                # Windows 上不同驱动器会抛出 ValueError
                raise HTTPException(404, "Not found")
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
