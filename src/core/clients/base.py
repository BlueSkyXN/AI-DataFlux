from abc import ABC, abstractmethod
from typing import Any, Dict, List

import aiohttp

class BaseAIClient(ABC):
    """AI 客户端抽象基类"""

    @abstractmethod
    async def call(
        self,
        session: aiohttp.ClientSession,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        调用 AI API

        Args:
            session: aiohttp Session
            messages: 消息列表 [{"role": "user", "content": "..."}]
            model: 模型名称
            temperature: 温度系数
            **kwargs: 其他参数

        Returns:
            API 响应的内容字符串
        """
        pass
