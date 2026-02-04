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
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# 项目根目录
PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())

# 进度文件路径
PROGRESS_FILE = ".dataflux_progress.json"


@dataclass
class ManagedProcess:
    """被管理的进程信息"""

    name: str  # "gateway" | "process"
    status: str = "stopped"  # "stopped" | "running" | "exited"
    pid: Optional[int] = None
    start_time: Optional[float] = None
    exit_code: Optional[int] = None
    command: List[str] = field(default_factory=list)
    config_path: Optional[str] = None
    port: Optional[int] = None  # 仅 Gateway

    def to_dict(self) -> dict:
        """转换为字典"""
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
        self.processes: Dict[str, ManagedProcess] = {
            "gateway": ManagedProcess(name="gateway"),
            "process": ManagedProcess(name="process"),
        }
        self._popen: Dict[str, Optional[subprocess.Popen]] = {
            "gateway": None,
            "process": None,
        }
        self._log_callbacks: Dict[str, List[Callable[[str, str], None]]] = {
            "gateway": [],
            "process": [],
        }
        # 日志环形缓冲区 (最近 1000 行)
        self._log_buffer: Dict[str, List[str]] = {
            "gateway": [],
            "process": [],
        }
        self._log_buffer_size = 1000
        self._read_tasks: Dict[str, Optional[asyncio.Task]] = {
            "gateway": None,
            "process": None,
        }

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
        cmd = [
            sys.executable,
            "cli.py",
            "gateway",
            "--config",
            config_path,
            "--port",
            str(port),
            "--workers",
            str(workers),
        ]

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
        cmd = [
            sys.executable,
            "cli.py",
            "process",
            "--config",
            config_path,
            "--progress-file",
            progress_file,
        ]

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
        if managed.status in ("stopped", "exited"):
            managed.status = "stopped"
            managed.pid = None
            managed.exit_code = None
            return managed.to_dict()

        proc = self._popen.get(name)
        if proc and proc.pid:
            kill_tree(proc.pid)

            # 等待进程退出
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
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

        # 添加 Gateway 健康检查 (如果未 managed 但外部可能运行)
        gateway_health = self._probe_gateway_health()
        if gateway_health:
            result["gateway"]["health"] = gateway_health

        # 添加 Process 进度信息
        process_progress = self._read_process_progress()
        if process_progress:
            result["process"]["progress"] = process_progress

        return result

    def _probe_gateway_health(self) -> Optional[dict]:
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
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            return None

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

            # 检查时间戳，超过 15 秒视为过期
            if time.time() - data.get("ts", 0) > 15:
                return None

            return data
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _start_log_reader(self, name: str, proc: subprocess.Popen) -> None:
        """启动异步日志读取任务"""

        async def read_logs():
            loop = asyncio.get_event_loop()
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
            except Exception as e:
                logging.debug(f"Log reader for {name} ended: {e}")

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

        # 清理进度文件
        progress_path = os.path.join(PROJECT_ROOT, PROGRESS_FILE)
        if os.path.exists(progress_path):
            try:
                os.remove(progress_path)
            except OSError:
                pass


# 全局实例
_manager: Optional[ProcessManager] = None


def get_process_manager() -> ProcessManager:
    """获取全局 ProcessManager 实例"""
    global _manager
    if _manager is None:
        _manager = ProcessManager()
    return _manager
