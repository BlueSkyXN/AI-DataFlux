"""
AI-DataFlux Control Server

本模块提供 AI-DataFlux 的 Web GUI 控制面板功能，包括：
- 配置文件编辑 (config.yaml)
- Gateway 和 Process 进程管理 (启动/停止)
- 实时日志查看

架构:
    python cli.py gui
           │
           ▼
    ┌─────────────────────────────────┐
    │  Control Server (FastAPI)       │
    │  127.0.0.1:8790                 │
    │                                 │
    │  ┌───────────┐ ┌──────────────┐ │
    │  │ ConfigAPI │ │ProcessManager│ │
    │  └───────────┘ └──────────────┘ │
    │         serve web/dist/         │
    └─────────────────────────────────┘
              │ subprocess
              ▼
       ┌──────────────┐
       │ Gateway/     │
       │ Process      │
       └──────────────┘

使用方式:
    python cli.py gui  # 启动控制面板，自动打开浏览器
"""

from .server import run_control_server

__all__ = ["run_control_server"]
