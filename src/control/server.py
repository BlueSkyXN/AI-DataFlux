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

鉴权函数:
    get_control_auth_token() -> str
        获取控制面板鉴权 Token（优先环境变量 DATAFLUX_CONTROL_TOKEN，缺省自动生成）
    _extract_bearer_token(auth_header) -> str
        从 Authorization 头提取 Bearer token
    _is_authorized_token(token) -> bool
        校验 token 是否与控制面板 token 匹配（常量时间比较）
    _is_authorized_from_candidates(*tokens) -> bool
        多来源 token 校验，任一合法即通过
    _mask_token(token) -> str
        Token 脱敏（用于日志输出）
    _decode_base64url_token(encoded) -> str
        解码 base64url 格式的 token
    _extract_ws_token(protocol_header) -> tuple[str, str]
        从 Sec-WebSocket-Protocol 头提取 token（支持 b64 和 raw 两种格式）

Pydantic 请求模型:
    ConfigWriteRequest    - 配置文件写入请求 (path, content)
    ConfigValidateRequest - 配置 YAML 语法校验请求 (content)
    GatewayStartRequest   - Gateway 启动请求 (config_path, port, workers)
    ProcessStartRequest   - Process 启动请求 (config_path)
    FeishuTestConnectionRequest - 飞书连接测试请求 (app_id, app_secret)

类:
    LogConnectionManager
        WebSocket 连接管理器，管理 gateway/process 两个频道的连接池
        方法: connect(), disconnect(), broadcast()

辅助函数:
    _test_feishu_connection(app_id, app_secret) -> dict
        异步测试飞书 API 连接，获取 tenant_access_token
    _open_browser_when_ready(url, host, port, ...) -> None
        等待服务端口就绪后打开浏览器（避免 Connection Refused）

生命周期:
    lifespan(app) -> AsyncGenerator
        FastAPI 应用生命周期管理：注册日志回调、打开浏览器、关闭时清理进程

应用工厂:
    create_control_app() -> FastAPI
        创建 FastAPI 应用实例，注册所有路由和中间件

    REST API 路由:
        GET  /api/config              - 读取配置文件
        PUT  /api/config              - 写入配置文件
        POST /api/config/validate     - 校验 YAML 语法
        POST /api/feishu/test_connection - 测试飞书连接
        POST /api/gateway/start       - 启动 Gateway
        POST /api/gateway/stop        - 停止 Gateway
        POST /api/process/start       - 启动 Process
        POST /api/process/stop        - 停止 Process
        GET  /api/status              - 获取所有进程状态

    WebSocket 路由:
        WS   /api/logs?target=        - 实时日志流

    静态文件:
        GET  /                        - 前端首页 (index.html)
        GET  /{path}                  - 静态文件 / SPA fallback

入口函数:
    run_control_server(host, port, open_browser) -> None
        启动 uvicorn 服务器
        输入: host (str), port (int), open_browser (bool)

关键变量:
    PROJECT_ROOT: str - 项目根目录
    WEB_DIST_DIR: str - 前端构建产物目录
    _CONTROL_AUTH_TOKEN: str | None - 缓存的鉴权 Token
    _CONTROL_AUTH_TOKEN_SOURCE: str - Token 来源 ("env" | "generated")
    _BASE64URL_PATTERN: re.Pattern - base64url 字符合法性正则
    ws_manager: LogConnectionManager - 全局 WebSocket 连接管理器

模块依赖:
    - fastapi, uvicorn, pydantic: Web 框架与数据校验
    - asyncio, secrets, webbrowser: 异步、安全、浏览器
    - .config_api: 配置文件读写
    - .process_manager: 进程管理
    - .runtime: 项目根目录与前端资源查找
"""

import asyncio
import base64
import binascii
import logging
import os
import re
import secrets
import webbrowser
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, List
from urllib.parse import quote

import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
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
_CONTROL_AUTH_TOKEN: str | None = None
_CONTROL_AUTH_TOKEN_SOURCE = "env"
_BASE64URL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def get_control_auth_token() -> str:
    """
    获取控制面板鉴权 Token

    优先使用环境变量 DATAFLUX_CONTROL_TOKEN，
    未设置时自动生成 32 字节的 URL-safe 随机 token。
    生成后缓存到模块级变量，整个进程生命周期内保持不变。

    Returns:
        str: 鉴权 Token
    """
    global _CONTROL_AUTH_TOKEN, _CONTROL_AUTH_TOKEN_SOURCE

    if _CONTROL_AUTH_TOKEN is None:
        token = os.environ.get("DATAFLUX_CONTROL_TOKEN", "").strip()
        if token:
            _CONTROL_AUTH_TOKEN = token
            _CONTROL_AUTH_TOKEN_SOURCE = "env"
        else:
            _CONTROL_AUTH_TOKEN = secrets.token_urlsafe(32)
            _CONTROL_AUTH_TOKEN_SOURCE = "generated"

    return _CONTROL_AUTH_TOKEN


def _extract_bearer_token(auth_header: str) -> str:
    """
    从 Authorization 头提取 Bearer token

    Args:
        auth_header: Authorization 头的值 (例如 "Bearer xxx")

    Returns:
        str: 提取的 token 值，格式不匹配时返回空字符串
    """
    prefix = "Bearer "
    if not auth_header.startswith(prefix):
        return ""
    return auth_header[len(prefix) :].strip()


def _is_authorized_token(token: str) -> bool:
    """
    校验控制面板 Token（使用 secrets.compare_digest 防时序攻击）

    Args:
        token: 待校验的 token

    Returns:
        bool: token 是否与控制面板 token 匹配
    """
    if not token:
        return False
    return secrets.compare_digest(token, get_control_auth_token())


def _is_authorized_from_candidates(*tokens: str) -> bool:
    """
    多来源 token 校验，任一合法即通过

    用于 WebSocket 场景：浏览器 WebSocket 无法设置自定义 Header，
    因此同时支持 Authorization 头和 Sec-WebSocket-Protocol 头传递 token。

    Args:
        *tokens: 多个待校验的 token

    Returns:
        bool: 任一 token 合法则返回 True
    """
    return any(_is_authorized_token(token) for token in tokens)


def _mask_token(token: str) -> str:
    """
    返回脱敏后的 token 文本（用于日志安全输出）

    Args:
        token: 原始 token

    Returns:
        str: 脱敏后的文本 (例如 "abc...xyz")，短 token 返回 "***"
    """
    if not token:
        return "***"
    if len(token) <= 6:
        return "***"
    return f"{token[:3]}...{token[-3:]}"


def _decode_base64url_token(encoded: str) -> str:
    """
    解码 base64url 格式的 token

    先校验字符合法性（仅允许 base64url 字符集），
    再补齐 padding 后解码为 UTF-8 字符串。

    Args:
        encoded: base64url 编码的字符串

    Returns:
        str: 解码后的 token 值，失败时返回空字符串
    """
    if not encoded or not _BASE64URL_PATTERN.fullmatch(encoded):
        return ""
    try:
        padding = "=" * (-len(encoded) % 4)
        decoded = base64.b64decode(
            encoded + padding,
            altchars=b"-_",
            validate=True,
        )
        return decoded.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return ""


def _extract_ws_token(protocol_header: str) -> tuple[str, str]:
    """
    从 Sec-WebSocket-Protocol 头提取鉴权 token

    支持两种格式（优先新格式）：
    - 新格式: dataflux-token-b64.<base64url编码的token> (解决特殊字符问题)
    - 旧格式: dataflux-token.<原始token> (向后兼容)

    Args:
        protocol_header: Sec-WebSocket-Protocol 头的值（可能包含多个逗号分隔的协议）

    Returns:
        tuple[str, str]: (匹配的协议名, 解码后的 token)，未匹配时返回 ("", "")
    """
    for protocol in (p.strip() for p in protocol_header.split(",") if p.strip()):
        if protocol.startswith("dataflux-token-b64."):
            encoded_token = protocol[len("dataflux-token-b64.") :]
            decoded_token = _decode_base64url_token(encoded_token)
            if decoded_token:
                return protocol, decoded_token
            continue
        if protocol.startswith("dataflux-token."):
            raw_token = protocol[len("dataflux-token.") :]
            if raw_token:
                return protocol, raw_token
    return "", ""


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


class FeishuTestConnectionRequest(BaseModel):
    """飞书连接测试请求"""

    app_id: str
    app_secret: str


# ========== WebSocket Manager ==========


class LogConnectionManager:
    """
    WebSocket 连接管理器

    管理 gateway 和 process 两个频道的 WebSocket 连接池。
    支持连接注册、断开和消息广播，自动清理断开的连接。

    Attributes:
        active_connections: 按频道分组的活跃 WebSocket 连接列表
    """

    def __init__(self):
        self.active_connections: dict[str, List[WebSocket]] = {
            "gateway": [],
            "process": [],
        }

    async def connect(
        self,
        websocket: WebSocket,
        target: str,
        subprotocol: str | None = None,
    ):
        """
        接受 WebSocket 连接并加入指定频道

        Args:
            websocket: WebSocket 连接对象
            target: 目标频道 ("gateway" | "process")
            subprotocol: 响应的子协议名（用于 token 协商）
        """
        await websocket.accept(subprotocol=subprotocol)
        if target in self.active_connections:
            self.active_connections[target].append(websocket)

    def disconnect(self, websocket: WebSocket, target: str):
        """
        从指定频道移除 WebSocket 连接

        Args:
            websocket: 要移除的连接
            target: 目标频道
        """
        if target in self.active_connections:
            if websocket in self.active_connections[target]:
                self.active_connections[target].remove(websocket)

    async def broadcast(self, target: str, message: str):
        """
        向指定频道的所有连接广播消息

        发送失败的连接自动从池中移除。

        Args:
            target: 目标频道 ("gateway" | "process")
            message: 要发送的文本消息
        """
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


async def _test_feishu_connection(app_id: str, app_secret: str) -> dict[str, Any]:
    """
    测试飞书 API 连接

    通过获取 tenant_access_token 验证飞书应用凭证是否有效。
    使用懒加载避免未使用飞书功能时引入额外依赖。

    Args:
        app_id: 飞书应用 ID
        app_secret: 飞书应用密钥

    Returns:
        dict: {"success": bool, "message": str, "token_preview": str (成功时)}
    """
    # 懒加载，避免未使用飞书功能时引入额外依赖
    from src.data.feishu.client import FeishuAPIError, FeishuClient

    client = FeishuClient(
        app_id=app_id,
        app_secret=app_secret,
        max_retries=0,  # 连接测试走快速失败
    )
    try:
        token = await client.ensure_token()
        return {
            "success": True,
            "message": "Connected successfully",
            "token_preview": f"{token[:5]}...{token[-5:]}" if token else "",
        }
    except FeishuAPIError as e:
        return {
            "success": False,
            "message": f"Feishu API Error: {e.msg} (Code: {e.code})",
        }
    except Exception as e:  # pragma: no cover - 兜底保护
        return {"success": False, "message": f"Connection failed: {e}"}
    finally:
        try:
            await client.close()
        except Exception:
            pass


async def _open_browser_when_ready(
    url: str,
    host: str,
    port: int,
    timeout_seconds: float = 5.0,
    interval_seconds: float = 0.2,
) -> None:
    """
    等待服务端口可连接后再打开浏览器

    通过循环尝试 TCP 连接来检测服务就绪，减少 Connection Refused 的概率。

    Args:
        url: 要打开的完整 URL（包含 token fragment）
        host: 服务监听地址
        port: 服务监听端口
        timeout_seconds: 最大等待时间 (秒)
        interval_seconds: 重试间隔 (秒)
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
        control_token = getattr(app_state, "control_auth_token", "")
        token_fragment = f"#token={quote(control_token)}" if control_token else ""
        browser_url = f"http://127.0.0.1:{port}{token_fragment}"
        logging.info(f"Opening browser: http://127.0.0.1:{port}")
        browser_task = asyncio.create_task(
            _open_browser_when_ready(url=browser_url, host="127.0.0.1", port=port)
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
    """
    创建 FastAPI 应用实例

    注册所有 REST API 路由、WebSocket 端点、中间件和静态文件服务。
    包含鉴权中间件（所有 /api/ 路径要求 Bearer token）和
    CSRF 防护（写操作要求 Content-Type: application/json）。

    Returns:
        FastAPI: 配置完成的应用实例
    """

    app = FastAPI(
        title="AI-DataFlux Control Panel",
        description="Web GUI for managing AI-DataFlux Gateway and Process",
        version="1.0.0",
        lifespan=lifespan,
    )
    control_auth_token = get_control_auth_token()

    # CORS 配置策略:
    # 1. 环境变量 DATAFLUX_GUI_CORS_ORIGINS 可显式指定允许的 origins
    # 2. 无前端构建产物时，默认允许 Vite dev server 的 origin (开发模式)
    # 3. 有前端构建产物时，同源访问不需要 CORS (生产模式)
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

    # 鉴权 + CSRF 防护中间件:
    # - 所有 /api/ 路径要求有效的 Bearer token (防止未授权访问)
    # - 写操作要求 Content-Type: application/json (防止 localhost CSRF 攻击)
    @app.middleware("http")
    async def require_json_for_api_writes(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            if request.method == "OPTIONS":
                return await call_next(request)

            auth_token = _extract_bearer_token(request.headers.get("authorization", ""))
            if not (
                auth_token and secrets.compare_digest(auth_token, control_auth_token)
            ):
                return JSONResponse(
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                    content={"detail": "Unauthorized"},
                )

            if request.method in ("POST", "PUT", "PATCH", "DELETE"):
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

    @app.post("/api/feishu/test_connection")
    async def api_feishu_test_connection(request: FeishuTestConnectionRequest):
        """测试飞书连接（获取 tenant_access_token）"""
        return await _test_feishu_connection(
            app_id=request.app_id,
            app_secret=request.app_secret,
        )

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
        websocket: WebSocket,
        target: str = Query(default="gateway"),
    ):
        """WebSocket 日志流"""
        auth_header_token = _extract_bearer_token(
            websocket.headers.get("authorization", "")
        )
        protocol_header = websocket.headers.get("sec-websocket-protocol", "")
        ws_subprotocol, ws_protocol_token = _extract_ws_token(protocol_header)

        if not _is_authorized_from_candidates(auth_header_token, ws_protocol_token):
            await websocket.close(code=1008)  # Policy Violation
            return

        if target not in ("gateway", "process"):
            await websocket.close(code=1008)  # Policy Violation
            return

        await ws_manager.connect(websocket, target, ws_subprotocol or None)

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
    control_auth_token = get_control_auth_token()

    # 设置应用状态 (用于 lifespan 中打开浏览器)
    app.state.open_browser = open_browser
    app.state.port = port
    app.state.control_auth_token = control_auth_token

    if _CONTROL_AUTH_TOKEN_SOURCE == "env":
        logging.info("Control API 鉴权已启用 (token 来源: DATAFLUX_CONTROL_TOKEN)")
    else:
        if not open_browser:
            raise RuntimeError(
                "未配置 DATAFLUX_CONTROL_TOKEN 且禁用了自动打开浏览器，"
                "请显式设置 DATAFLUX_CONTROL_TOKEN 后重试"
            )
        logging.warning(
            "未配置 DATAFLUX_CONTROL_TOKEN，已生成临时控制面 token（掩码）: %s",
            _mask_token(control_auth_token),
        )

    logging.info(f"Starting Control Server on http://{host}:{port}")

    # 启动服务
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
