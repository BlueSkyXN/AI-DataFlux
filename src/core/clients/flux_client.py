import asyncio
import json
import logging
import time
from typing import Any, Dict, List

import aiohttp

from .base import BaseAIClient

class FluxAIClient(BaseAIClient):
    """Flux API (OpenAI 兼容) 客户端"""

    def __init__(self, api_url: str, timeout: int = 600):
        """
        初始化客户端

        Args:
            api_url: API 端点 URL
            timeout: 总超时时间（秒）
        """
        self.api_url = api_url
        # 确保 URL 指向 chat/completions
        if "/v1/chat/completions" not in self.api_url:
            self.api_url = self.api_url.rstrip("/") + "/v1/chat/completions"

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
        """调用 Flux/OpenAI 兼容 API"""

        headers = {"Content-Type": "application/json"}

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

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
                    try:
                        data = json.loads(response_text)
                        choices = data.get("choices")

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
