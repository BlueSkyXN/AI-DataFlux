"""
AI 客户端模块

本模块提供与 AI API 通信的客户端实现，采用抽象基类设计，
支持扩展不同的 AI 服务提供商。

模块结构:
    - BaseAIClient: 抽象基类，定义客户端接口
    - FluxAIClient: Flux/OpenAI 兼容 API 的具体实现

设计模式:
    采用策略模式 (Strategy Pattern)，不同的 AI 服务商实现
    同一接口，可在运行时切换。

使用示例:
    from src.core.clients import FluxAIClient
    
    client = FluxAIClient("http://localhost:8787/v1/chat/completions")
    response = await client.call(session, messages, model="gpt-4")
"""

from .base import BaseAIClient
from .flux_client import FluxAIClient

__all__ = ["BaseAIClient", "FluxAIClient"]
