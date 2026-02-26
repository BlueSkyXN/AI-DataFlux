"""
进程管理器模块

管理 Gateway 和 Process 两个子进程的生命周期。

状态机 (三态):
    STOPPED  ──start──▶  RUNNING  ──进程退出──▶  EXITED
       ▲                    │                      │
       └────────stop────────┘                      │
       └───────────────────────────────────────────┘

功能:
    - start: 启动子进程
    - stop: 停止子进程 (杀进程树)
    - 状态查询
    - 日志流采集

数据类:
    ManagedProcess
        被管理的进程信息数据类，记录进程名称、状态、PID、启动时间等
        属性: name, status, pid, start_time, exit_code, command, config_path, port
        方法: to_dict() -> dict

函数清单:
    kill_tree(pid) -> None
        杀掉进程及其所有子进程（优先使用 psutil 进程树清理）
        输入: pid (int) - 进程 ID
        设计: 先 SIGTERM 等待 3 秒，仍存活则 SIGKILL

类: ProcessManager
    管理 Gateway 和 Process 两个子进程的完整生命周期

    公开方法:
        start_gateway(config_path, port, workers) -> dict
            启动 Gateway 子进程，幂等设计（已运行则直接返回）
        stop_gateway() -> dict
            停止 Gateway 子进程
        start_process(config_path) -> dict
            启动 Process 子进程，自动传递 --progress-file 参数
        stop_process() -> dict
            停止 Process 子进程
        get_status(name) -> dict
            获取指定进程状态
        get_all_status() -> dict
            获取所有进程状态（同步版本，含 Gateway 健康检查和 Process 进度）
        get_all_status_async() -> dict
            获取所有进程状态（异步版本，避免阻塞 event loop）
        add_log_callback(name, callback) -> None
            注册日志行回调（用于 WebSocket 推送）
        remove_log_callback(name, callback) -> None
            移除日志行回调
        get_log_buffer(name) -> List[str]
            获取日志环形缓冲区内容
        shutdown() -> None
            关闭所有托管进程

    内部方法:
        _build_subprocess_cmd(subcommand, args) -> list[str]
            构建子进程启动命令（区分源码 / 打包环境）
        _check_process_status(name) -> None
            轮询进程退出状态并更新 ManagedProcess
        _stop_process(name) -> dict
            停止指定进程的通用实现
        _probe_gateway_health_sync(timeout) -> Optional[dict]
            同步探测 Gateway /admin/health 端点
        _gateway_health_cache_ttl_seconds() -> float
            根据上次探测结果返回缓存 TTL
        _get_gateway_health_cached() -> Optional[dict]
            异步获取 Gateway 健康状态（带缓存）
        _read_process_progress() -> Optional[dict]
            读取 Process 进度文件（.dataflux_progress.json）
        _start_log_reader(name, proc) -> None
            启动异步日志读取任务，逐行读取 stdout 并分发回调

    get_process_manager() -> ProcessManager
        获取全局单例 ProcessManager 实例

关键变量:
    PROJECT_ROOT: str - 项目根目录
    PROGRESS_FILE: str - 进度文件名 (.dataflux_progress.json)
    PROGRESS_TIMEOUT_SECONDS: int - 进度文件超时阈值 (15 秒)
    LOG_BUFFER_SIZE: int - 日志环形缓冲区容量 (1000 行)
    PSUTIL_AVAILABLE: bool - psutil 库是否可用
    _manager: Optional[ProcessManager] - 全局单例实例

模块依赖:
    - asyncio, subprocess, os, sys, time: 进程与异步管理
    - psutil (可选): 进程树清理
    - .runtime: 项目根目录与打包环境检测
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .runtime import get_project_root, is_packaged

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# 项目根目录
PROJECT_ROOT = str(get_project_root())

# 进度文件路径
PROGRESS_FILE = ".dataflux_progress.json"

# 进度文件超时时间（秒）- 超过此时间未更新视为过期
PROGRESS_TIMEOUT_SECONDS = 15

# 日志缓冲区大小（行数）
LOG_BUFFER_SIZE = 1000


@dataclass
class ManagedProcess:
    """
    被管理的进程信息

    记录单个子进程 (Gateway 或 Process) 的运行状态和元数据。
    由 ProcessManager 创建和维护。

    Attributes:
        name: 进程名称 ("gateway" | "process")
        status: 当前状态 ("stopped" | "running" | "exited")
        pid: 系统进程 ID (停止时为 None)
        start_time: 启动时间戳 (Unix 时间)
        exit_code: 退出码 (仅 exited 状态有值)
        command: 启动命令列表
        config_path: 使用的配置文件路径
        port: 监听端口 (仅 Gateway 使用)
    """

    name: str  # "gateway" | "process"
    status: str = "stopped"  # "stopped" | "running" | "exited"
    pid: Optional[int] = None
    start_time: Optional[float] = None
    exit_code: Optional[int] = None
    command: List[str] = field(default_factory=list)
    config_path: Optional[str] = None
    port: Optional[int] = None  # 仅 Gateway

    def to_dict(self) -> dict:
        """转换为可序列化的字典（用于 API 响应）"""
        return {
            "status": self.status,
            "pid": self.pid,
            "start_time": self.start_time,
            "exit_code": self.exit_code,
            "config_path": self.config_path,
            "port": self.port,
        }


def kill_tree(pid: int) -> None:
    """
    杀掉进程及其所有子进程

    Args:
        pid: 进程 ID
    """
    if not PSUTIL_AVAILABLE:
        # 回退到简单的 kill
        try:
            os.kill(pid, 15)  # SIGTERM
        except OSError:
            pass
        return

    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # 先发 SIGTERM
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass

        # 等待 3 秒
        gone, alive = psutil.wait_procs(children + [parent], timeout=3)

        # 还活着的进程强制 kill
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


class ProcessManager:
    """
    进程管理器

    管理 Gateway 和 Process 两个子进程的生命周期。
    """

    def __init__(self):
        """
        初始化进程管理器

        创建 gateway 和 process 两个子进程的管理结构，包括：
        - 进程状态对象 (ManagedProcess)
        - Popen 句柄引用
        - 日志回调列表和环形缓冲区
        - Gateway 健康检查缓存
        """
        # 进程状态对象：记录每个子进程的运行元数据
        self.processes: Dict[str, ManagedProcess] = {
            "gateway": ManagedProcess(name="gateway"),
            "process": ManagedProcess(name="process"),
        }
        # Popen 句柄：持有子进程引用，用于轮询退出状态和关闭 stdout
        self._popen: Dict[str, Optional[subprocess.Popen]] = {
            "gateway": None,
            "process": None,
        }
        # 日志回调列表：每当读取到新日志行时通知所有注册的回调
        self._log_callbacks: Dict[str, List[Callable[[str, str], None]]] = {
            "gateway": [],
            "process": [],
        }
        # 日志环形缓冲区：保留最近 N 行日志，供新连接的 WebSocket 客户端回放历史
        self._log_buffer: Dict[str, List[str]] = {
            "gateway": [],
            "process": [],
        }
        self._log_buffer_size = LOG_BUFFER_SIZE
        # 异步日志读取任务：每个进程一个，持续从 stdout 读取日志行
        self._read_tasks: Dict[str, Optional[asyncio.Task]] = {
            "gateway": None,
            "process": None,
        }
        # Gateway 健康检查缓存：避免频繁 HTTP 探测
        self._gateway_health_cache: Optional[dict] = None
        self._gateway_health_cache_time: float = 0.0
        self._gateway_health_cache_ttl_ok_seconds: float = 2.0  # 探测成功时缓存 2 秒
        self._gateway_health_cache_ttl_fail_seconds: float = 5.0  # 探测失败时缓存 5 秒
        self._gateway_health_probe_timeout_seconds: float = 0.8  # 单次探测超时

    def _build_subprocess_cmd(self, subcommand: str, args: list[str]) -> list[str]:
        """
        构建子进程启动命令

        - 源码运行：python cli.py <subcommand> ...
        - 打包运行：<exe> <subcommand> ...
        """
        if is_packaged():
            return [sys.executable, subcommand, *args]
        return [sys.executable, "cli.py", subcommand, *args]

    def _check_process_status(self, name: str) -> None:
        """检查并更新进程状态"""
        proc = self._popen.get(name)
        managed = self.processes[name]

        if proc is None:
            return

        # 检查进程是否退出
        ret = proc.poll()
        if ret is not None:
            managed.status = "exited"
            managed.exit_code = ret
            self._popen[name] = None
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass

            # 取消日志读取任务
            if self._read_tasks.get(name):
                self._read_tasks[name].cancel()
                self._read_tasks[name] = None

    def start_gateway(
        self,
        config_path: str = "config.yaml",
        port: int = 8787,
        workers: int = 1,
    ) -> dict:
        """
        启动 Gateway

        Args:
            config_path: 配置文件路径
            port: 监听端口
            workers: 工作进程数

        Returns:
            dict: 进程状态
        """
        self._check_process_status("gateway")
        managed = self.processes["gateway"]

        # 幂等: 已运行则直接返回
        if managed.status == "running":
            return managed.to_dict()

        # 构建命令
        cmd = self._build_subprocess_cmd(
            "gateway",
            [
                "--config",
                config_path,
                "--port",
                str(port),
                "--workers",
                str(workers),
            ],
        )

        # 启动进程
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=PROJECT_ROOT,
        )

        self._popen["gateway"] = proc
        managed.status = "running"
        managed.pid = proc.pid
        managed.start_time = time.time()
        managed.exit_code = None
        managed.command = cmd
        managed.config_path = config_path
        managed.port = port

        # 启动日志读取任务
        self._start_log_reader("gateway", proc)

        logging.info(f"Gateway started: PID={proc.pid}, port={port}")
        return managed.to_dict()

    def stop_gateway(self) -> dict:
        """
        停止 Gateway

        Returns:
            dict: 进程状态
        """
        return self._stop_process("gateway")

    def start_process(self, config_path: str = "config.yaml") -> dict:
        """
        启动 Process

        Args:
            config_path: 配置文件路径

        Returns:
            dict: 进程状态
        """
        self._check_process_status("process")
        managed = self.processes["process"]

        # 幂等: 已运行则直接返回
        if managed.status == "running":
            return managed.to_dict()

        # 构建命令 (带 --progress-file 参数)
        progress_file = os.path.join(PROJECT_ROOT, PROGRESS_FILE)
        cmd = self._build_subprocess_cmd(
            "process",
            [
                "--config",
                config_path,
                "--progress-file",
                progress_file,
            ],
        )

        # 启动进程
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=PROJECT_ROOT,
        )

        self._popen["process"] = proc
        managed.status = "running"
        managed.pid = proc.pid
        managed.start_time = time.time()
        managed.exit_code = None
        managed.command = cmd
        managed.config_path = config_path

        # 启动日志读取任务
        self._start_log_reader("process", proc)

        logging.info(f"Process started: PID={proc.pid}")
        return managed.to_dict()

    def stop_process(self) -> dict:
        """
        停止 Process

        Returns:
            dict: 进程状态
        """
        return self._stop_process("process")

    def _stop_process(self, name: str) -> dict:
        """
        停止指定进程

        Args:
            name: "gateway" | "process"

        Returns:
            dict: 进程状态
        """
        self._check_process_status(name)
        managed = self.processes[name]

        # 幂等: 已停止则直接返回
        if managed.status == "stopped":
            managed.pid = None
            managed.exit_code = None
            return managed.to_dict()

        # 已退出：保留 exited + exit_code（便于前端展示）
        if managed.status == "exited":
            return managed.to_dict()

        proc = self._popen.get(name)
        if proc and proc.pid:
            kill_tree(proc.pid)

            # 等待进程退出
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass

        self._popen[name] = None
        managed.status = "stopped"
        managed.pid = None
        managed.exit_code = None

        # 取消日志读取任务
        if self._read_tasks.get(name):
            self._read_tasks[name].cancel()
            self._read_tasks[name] = None

        logging.info(f"{name.capitalize()} stopped")
        return managed.to_dict()

    def get_status(self, name: str) -> dict:
        """
        获取进程状态

        Args:
            name: "gateway" | "process"

        Returns:
            dict: 进程状态
        """
        self._check_process_status(name)
        return self.processes[name].to_dict()

    def get_all_status(self) -> dict:
        """
        获取所有进程状态

        Returns:
            dict: {"gateway": ..., "process": ...}
        """
        self._check_process_status("gateway")
        self._check_process_status("process")

        result = {
            "gateway": {
                "managed": self.processes["gateway"].to_dict(),
            },
            "process": {
                "managed": self.processes["process"].to_dict(),
            },
        }

        # 添加 Gateway 健康检查（同步版本：可能阻塞；Server 侧优先用 get_all_status_async）
        now = time.time()
        if (
            self._gateway_health_cache_time
            and (now - self._gateway_health_cache_time)
            < self._gateway_health_cache_ttl_seconds()
        ):
            gateway_health = self._gateway_health_cache
        else:
            gateway_health = self._probe_gateway_health_sync(
                self._gateway_health_probe_timeout_seconds
            )
            self._gateway_health_cache = gateway_health
            self._gateway_health_cache_time = now

        if gateway_health:
            result["gateway"]["health"] = gateway_health

        # 添加 Process 进度信息
        process_progress = self._read_process_progress()
        if process_progress:
            result["process"]["progress"] = process_progress

        # 添加工作目录
        result["working_directory"] = PROJECT_ROOT

        return result

    async def get_all_status_async(self) -> dict:
        """
        获取所有进程状态（异步版本，避免阻塞 event loop）

        Returns:
            dict: {"gateway": ..., "process": ...}
        """
        self._check_process_status("gateway")
        self._check_process_status("process")

        result = {
            "gateway": {
                "managed": self.processes["gateway"].to_dict(),
            },
            "process": {
                "managed": self.processes["process"].to_dict(),
            },
        }

        gateway_health = await self._get_gateway_health_cached()
        if gateway_health:
            result["gateway"]["health"] = gateway_health

        process_progress = self._read_process_progress()
        if process_progress:
            result["process"]["progress"] = process_progress

        # 添加工作目录
        result["working_directory"] = PROJECT_ROOT

        return result

    def _probe_gateway_health_sync(self, timeout: float) -> Optional[dict]:
        """
        探测 Gateway 健康状态

        Returns:
            dict | None: 健康状态或 None
        """
        import urllib.request
        import urllib.error

        # 确定要探测的端口
        managed = self.processes["gateway"]
        port = managed.port or 8787

        try:
            url = f"http://127.0.0.1:{port}/admin/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            return None

    def _gateway_health_cache_ttl_seconds(self) -> float:
        """
        根据上次探测结果返回缓存 TTL

        设计意图：探测失败时使用更长的 TTL，减少对不可用服务的无效探测。

        Returns:
            float: 缓存有效期 (秒)
        """
        if self._gateway_health_cache is None:
            return self._gateway_health_cache_ttl_fail_seconds
        return self._gateway_health_cache_ttl_ok_seconds

    async def _get_gateway_health_cached(self) -> Optional[dict]:
        """
        异步获取 Gateway 健康状态（带缓存）

        缓存未过期时直接返回缓存结果，否则通过 asyncio.to_thread
        在线程池中执行同步探测，避免阻塞 event loop。

        Returns:
            dict | None: 健康状态数据，或 None (不可达)
        """
        now = time.time()
        # 缓存未过期，直接返回
        if (
            self._gateway_health_cache_time
            and (now - self._gateway_health_cache_time)
            < self._gateway_health_cache_ttl_seconds()
        ):
            return self._gateway_health_cache

        # 在线程池中执行同步 HTTP 探测，避免阻塞异步事件循环
        health = await asyncio.to_thread(
            self._probe_gateway_health_sync, self._gateway_health_probe_timeout_seconds
        )
        self._gateway_health_cache = health
        self._gateway_health_cache_time = now
        return health

    def _read_process_progress(self) -> Optional[dict]:
        """
        读取 Process 进度文件

        Returns:
            dict | None: 进度数据或 None (文件不存在或超时)
        """
        progress_path = os.path.join(PROJECT_ROOT, PROGRESS_FILE)
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 检查时间戳，超时视为过期
            if time.time() - data.get("ts", 0) > PROGRESS_TIMEOUT_SECONDS:
                return None

            return data
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _start_log_reader(self, name: str, proc: subprocess.Popen) -> None:
        """启动异步日志读取任务"""

        async def read_logs():
            loop = asyncio.get_running_loop()
            try:
                while True:
                    line = await loop.run_in_executor(None, proc.stdout.readline)
                    if not line:
                        break
                    line = line.rstrip("\n")

                    # 添加到缓冲区
                    buf = self._log_buffer[name]
                    buf.append(line)
                    if len(buf) > self._log_buffer_size:
                        buf.pop(0)

                    # 通知所有回调
                    for callback in self._log_callbacks[name]:
                        try:
                            callback(name, line)
                        except Exception:
                            pass
            except asyncio.CancelledError:
                pass
            except Exception:
                logging.exception(f"Log reader for {name} ended with error")

        # 创建任务 (需要在事件循环中)
        try:
            loop = asyncio.get_running_loop()
            self._read_tasks[name] = loop.create_task(read_logs())
        except RuntimeError:
            # 没有运行中的事件循环，延迟创建
            pass

    def add_log_callback(self, name: str, callback: Callable[[str, str], None]) -> None:
        """
        添加日志回调

        Args:
            name: "gateway" | "process"
            callback: 回调函数 (name, line) -> None
        """
        if name in self._log_callbacks:
            self._log_callbacks[name].append(callback)

    def remove_log_callback(
        self, name: str, callback: Callable[[str, str], None]
    ) -> None:
        """
        移除日志回调

        Args:
            name: "gateway" | "process"
            callback: 回调函数
        """
        if name in self._log_callbacks and callback in self._log_callbacks[name]:
            self._log_callbacks[name].remove(callback)

    def get_log_buffer(self, name: str) -> List[str]:
        """
        获取日志缓冲区内容

        Args:
            name: "gateway" | "process"

        Returns:
            List[str]: 日志行列表
        """
        return list(self._log_buffer.get(name, []))

    def shutdown(self) -> None:
        """关闭所有进程"""
        logging.info("Shutting down all managed processes...")
        self.stop_gateway()
        self.stop_process()


# 全局实例
_manager: Optional[ProcessManager] = None


def get_process_manager() -> ProcessManager:
    """获取全局 ProcessManager 实例"""
    global _manager
    if _manager is None:
        _manager = ProcessManager()
    return _manager
