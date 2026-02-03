"""
AI 客户端抽象基类

本模块定义 AI 客户端的抽象接口，所有具体的 AI 服务客户端
都应继承此基类并实现 call 方法。

设计目的:
    - 统一不同 AI 服务的调用接口
    - 便于单元测试 (可注入 Mock 实现)
    - 支持未来扩展其他 AI 服务商

扩展指南:
    1. 继承 BaseAIClient
    2. 实现 call 方法
    3. 在 __init__.py 中导出新类
    
    class NewAIClient(BaseAIClient):
        async def call(self, session, messages, model, **kwargs) -> str:
            # 实现具体的 API 调用逻辑
            pass
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import aiohttp


class BaseAIClient(ABC):
    """
    AI 客户端抽象基类
    
    定义与 AI 服务交互的标准接口。所有具体客户端实现
    必须继承此类并实现 call 方法。
    
    接口契约:
        - call 方法是异步的，使用 aiohttp 进行 HTTP 通信
        - 返回值是 AI 响应的原始文本内容
        - 网络/API 错误应抛出 aiohttp 相关异常
        - 超时应抛出 TimeoutError
    """

    @abstractmethod
    async def call(
        self,
        session: aiohttp.ClientSession,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float | None = 0.7,
        **kwargs
    ) -> str:
        """
        调用 AI API

        Args:
            session: aiohttp Session，用于发送 HTTP 请求
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            model: 模型名称 (如 "gpt-4", "claude-3-opus")
            temperature: 温度系数，控制输出随机性 (0.0-2.0)；None 表示不发送该参数
            **kwargs: 其他 API 特定参数

        Returns:
            API 响应的内容字符串 (通常是 AI 生成的文本)
            
        Raises:
            aiohttp.ClientResponseError: API 返回非 200 状态码
            aiohttp.ClientError: 网络连接错误
            TimeoutError: 请求超时
        """
        pass
