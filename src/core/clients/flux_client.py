"""
Flux AI 客户端实现

本模块实现与 Flux API Gateway (OpenAI 兼容) 的通信客户端。
Flux Gateway 是本项目的 API 网关组件，提供多模型路由和负载均衡。

通信协议:
    - 使用 OpenAI Chat Completions API 格式
    - POST /v1/chat/completions
    - Content-Type: application/json

请求格式:
    {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "temperature": 0.7,
        "stream": false,
        "response_format": {"type": "json_object"}  // 可选
    }

响应格式:
    {
        "choices": [
            {
                "message": {
                    "content": "AI 生成的内容"
                }
            }
        ]
    }

超时配置:
    - connect: 20 秒 (连接建立超时)
    - total: 600 秒 (总超时，包含响应等待)

错误处理:
    - HTTP 非 200: 抛出 ClientResponseError
    - 响应结构无效: 抛出 ClientResponseError
    - 连接超时: 抛出 TimeoutError
    - 网络错误: 抛出 ClientError

使用示例:
    client = FluxAIClient("http://localhost:8787")
    async with aiohttp.ClientSession() as session:
        response = await client.call(
            session,
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            use_json_schema=True
        )
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List

import aiohttp

from .base import BaseAIClient


class FluxAIClient(BaseAIClient):
    """
    Flux API (OpenAI 兼容) 客户端
    
    与 Flux Gateway 通信的具体实现。自动处理 URL 格式化、
    超时管理、错误转换等细节。
    
    特性:
        - 自动补全 /v1/chat/completions 路径
        - 支持 JSON Schema 模式 (response_format)
        - 详细的请求/响应日志
        - 统一的异常处理
    
    Attributes:
        api_url: 完整的 API 端点 URL
        request_timeout: aiohttp 超时配置
    """

    def __init__(self, api_url: str, timeout: int = 600):
        """
        初始化客户端

        Args:
            api_url: API 端点 URL (可以是基础 URL 或完整路径)
            timeout: 总超时时间（秒），默认 600 秒 (10 分钟)
        """
        self.api_url = api_url
        # 确保 URL 指向 chat/completions
        if "/v1/chat/completions" not in self.api_url:
            self.api_url = self.api_url.rstrip("/") + "/v1/chat/completions"

        # 配置超时: 连接 20 秒，总计 timeout 秒
        self.request_timeout = aiohttp.ClientTimeout(connect=20, total=timeout)

    async def call(
        self,
        session: aiohttp.ClientSession,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        use_json_schema: bool = False,
        **kwargs
    ) -> str:
        """
        调用 Flux/OpenAI 兼容 API
        
        发送 Chat Completions 请求并解析响应。
        
        Args:
            session: aiohttp 客户端会话
            messages: 消息列表 [{"role": "...", "content": "..."}]
            model: 模型名称 (如 "auto", "gpt-4", "claude-3")
            temperature: 温度系数 (0.0-2.0)
            use_json_schema: 是否强制 JSON 输出格式
            **kwargs: 其他 OpenAI API 参数 (如 max_tokens)
            
        Returns:
            AI 生成的文本内容
            
        Raises:
            aiohttp.ClientResponseError: HTTP 错误或响应格式无效
            TimeoutError: 请求超时
            aiohttp.ClientError: 网络连接错误
        """

        headers = {"Content-Type": "application/json"}

        # 构建请求体
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,  # 不使用流式响应
        }

        # 启用 JSON 模式
        if use_json_schema:
            payload["response_format"] = {"type": "json_object"}

        # 合并其他参数
        payload.update(kwargs)

        logging.debug(f"向 Flux API ({self.api_url}) 发送请求...")
        start_time = time.time()

        try:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.request_timeout,
            ) as resp:
                resp_status = resp.status
                response_text = await resp.text()
                elapsed = time.time() - start_time

                logging.debug(f"Flux API 响应状态: {resp_status} in {elapsed:.2f}s")

                if resp_status == 200:
                    # 解析响应
                    try:
                        data = json.loads(response_text)
                        choices = data.get("choices")

                        # 验证响应结构
                        if choices and isinstance(choices, list) and choices[0]:
                            message = choices[0].get("message")
                            content = message.get("content") if message else None

                            if content is not None and isinstance(content, str):
                                return content

                        raise ValueError("响应结构无效或缺少内容")

                    except (json.JSONDecodeError, ValueError) as e:
                        logging.error(f"Flux API 返回 200 OK 但响应无效: {e}")
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp_status,
                            message=f"Invalid 200 OK response: {e}",
                            headers=resp.headers,
                        ) from e
                else:
                    # HTTP 错误
                    logging.warning(f"Flux API 调用失败: HTTP {resp_status}")
                    raise aiohttp.ClientResponseError(
                        resp.request_info,
                        resp.history,
                        status=resp_status,
                        message=f"Flux API Error: {response_text[:500]}",
                        headers=resp.headers,
                    )

        except asyncio.TimeoutError as e:
            logging.error(f"调用 Flux API 超时 (>{self.request_timeout.total}s)")
            raise TimeoutError(
                f"Flux API call timed out after {time.time() - start_time:.2f}s"
            ) from e

        except aiohttp.ClientError as e:
            logging.error(f"调用 Flux API 时网络/客户端错误: {e}")
            raise
