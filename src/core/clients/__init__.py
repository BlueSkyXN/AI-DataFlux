"""
AI 客户端模块

本模块提供与 AI API 通信的客户端实现，采用抽象基类设计，
支持扩展不同的 AI 服务提供商。

模块结构:
    - BaseAIClient: 抽象基类，定义客户端接口
    - FluxAIClient: Flux/OpenAI 兼容 API 的具体实现

类/函数清单:
    BaseAIClient (抽象基类):
        - call(session, messages, model, temperature, **kwargs) -> str
          异步调用 AI API，返回生成文本
          输入: aiohttp.ClientSession, 消息列表, 模型名, 温度系数
          输出: AI 响应文本字符串

    FluxAIClient (具体实现):
        - __init__(api_url, timeout=600) -> None
          初始化客户端，自动补全 /v1/chat/completions 路径
          输入: API 端点 URL, 超时秒数
        - call(session, messages, model, temperature, use_json_schema, **kwargs) -> str
          调用 Flux/OpenAI 兼容 API
          输入: aiohttp.ClientSession, 消息列表, 模型名, 温度系数, 是否启用 JSON Schema
          输出: AI 响应文本字符串

关键变量:
    - api_url: 完整的 API 端点 URL (FluxAIClient 自动补全路径)
    - request_timeout: aiohttp.ClientTimeout 超时配置 (connect=20s, total=600s)

模块依赖:
    - aiohttp: 异步 HTTP 客户端
    - abc: 抽象基类支持

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
