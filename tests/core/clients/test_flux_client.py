"""
Flux AI 客户端单元测试

被测模块: src/core/clients/flux_client.py (FluxAIClient)

测试 src/core/clients/flux_client.py 的 FluxAIClient 类功能，包括：
- API 调用成功场景
- HTTP 错误处理
- 超时处理
- 响应解析
- JSON 模式支持

测试类/函数清单:
    TestFluxAIClient                   Flux AI 客户端测试
        test_call_success              验证成功调用返回内容并传递正确 model
        test_call_api_error            验证 500 错误抛出 ClientResponseError
        test_call_invalid_json         验证 200 但 JSON 无效时抛出异常
        test_call_timeout              验证超时抛出 TimeoutError
        test_url_normalization         验证 URL 自动补全 /v1/chat/completions
"""

import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

aiohttp = pytest.importorskip("aiohttp")
flux_client_module = importlib.import_module("src.core.clients.flux_client")
FluxAIClient = flux_client_module.FluxAIClient


class TestFluxAIClient:
    """Flux AI 客户端测试"""

    @pytest.fixture
    def client(self):
        return FluxAIClient("http://test.api/v1/chat/completions")

    @pytest.fixture
    def mock_session(self):
        session = AsyncMock(spec=aiohttp.ClientSession)
        return session

    @pytest.mark.asyncio
    async def test_call_success(self, client, mock_session):
        # 模拟成功响应
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = (
            '{"choices": [{"message": {"content": "Hello"}}]}'
        )

        # 配置 session.post 上下文管理器
        mock_session.post.return_value.__aenter__.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        result = await client.call(mock_session, messages, model="gpt-4")

        assert result == "Hello"
        mock_session.post.assert_called_once()
        args, kwargs = mock_session.post.call_args
        assert kwargs["json"]["model"] == "gpt-4"
        assert kwargs["json"]["messages"] == messages

    @pytest.mark.asyncio
    async def test_call_api_error(self, client, mock_session):
        # 模拟 500 错误
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"
        mock_response.request_info = MagicMock()
        mock_response.history = []
        mock_response.headers = {}

        mock_session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(aiohttp.ClientResponseError):
            await client.call(mock_session, [], model="test")

    @pytest.mark.asyncio
    async def test_call_invalid_json(self, client, mock_session):
        # 模拟 200 但 JSON 无效
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = "Invalid JSON"
        mock_response.request_info = MagicMock()

        mock_session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(aiohttp.ClientResponseError) as exc:
            await client.call(mock_session, [], model="test")

        assert "Invalid 200 OK response" in str(exc.value)

    @pytest.mark.asyncio
    async def test_call_timeout(self, client, mock_session):
        # 模拟超时
        mock_session.post.side_effect = asyncio.TimeoutError()

        with pytest.raises(TimeoutError) as exc:
            await client.call(mock_session, [], model="test")

        assert "timed out" in str(exc.value)

    def test_url_normalization(self):
        """测试 URL 自动补全 /v1/chat/completions 路径"""
        c1 = FluxAIClient("http://api.com")
        assert c1.api_url == "http://api.com/v1/chat/completions"

        c2 = FluxAIClient("http://api.com/")
        assert c2.api_url == "http://api.com/v1/chat/completions"

        c3 = FluxAIClient("http://api.com/v1/chat/completions")
        assert c3.api_url == "http://api.com/v1/chat/completions"
