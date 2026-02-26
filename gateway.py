#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flux API 网关快速启动入口模块

本模块提供 API 网关的便捷启动方式，是 cli.py gateway 子命令的精简替代方案。
适合在只需要启动网关服务而不需要其他 CLI 功能的场景下使用。

功能说明:
    启动 OpenAI 兼容的 API 网关服务，支持：
    - 多模型加权负载均衡
    - 自动故障检测与切换
    - 令牌桶限流保护
    - IP 池 DNS 轮询
    - HTTP 连接复用

运行方式:
    # 使用默认配置和端口
    python gateway.py

    # 指定配置文件
    python gateway.py --config my_config.yaml

    # 指定端口
    python gateway.py --port 8000

    # 完整参数
    python gateway.py --config config.yaml --port 8787 --host 0.0.0.0

API 端点:
    POST /v1/chat/completions - OpenAI 兼容的聊天补全接口
    GET  /health             - 健康检查端点
    GET  /v1/models          - 获取可用模型列表

架构位置:
    本模块是网关服务的直接入口，调用 src.gateway.app.main 函数启动服务。
    网关独立于数据处理引擎运行，通过 HTTP API 接收请求。

依赖模块:
    - src.gateway.app: FastAPI 应用和服务器配置
    - src.gateway.app.main: 网关主入口函数 (解析命令行参数并启动 uvicorn 服务)

入口逻辑:
    __main__ 块调用 src.gateway.app.main()，该函数:
        输入: 无 (通过 sys.argv 获取 --config, --port, --host, --workers, --reload 参数)
        输出: 退出码 (0=成功, None 时转为 0)

作者: AI-DataFlux Team
版本: 参见 src/__init__.py
"""

import sys

from src.gateway.app import main


if __name__ == "__main__":
    # 调用网关主入口函数，返回退出码
    sys.exit(main() or 0)
