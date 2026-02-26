"""
飞书客户端异步行为测试

被测模块: src/data/feishu/client.py (FeishuClient)

测试 FeishuClient 的异步逻辑，包括：
1. Token 获取与刷新
2. HTTP 请求重试（429, 5xx, 网络错误）
3. 业务层错误处理（频控、Token 失效）
4. Bitable/Sheet 批量操作的自动分块

使用 pytest-asyncio 和 aioresponses/unittest.mock 进行测试。

测试类/函数清单:
    TestFeishuClientAsync                      飞书客户端异步测试
        test_ensure_token_success              验证成功获取 Token 并缓存
        test_ensure_token_use_cache            验证使用缓存的 Token（不发起请求）
        test_ensure_token_refresh_early        验证 Token 快过期时提前刷新（<300s）
        test_request_success                   验证常规请求成功及 Authorization 头
        test_request_429_retry                 验证 429 限流后自动重试成功
        test_request_429_exhausted             验证 429 重试耗尽抛 FeishuRateLimitError
        test_request_token_expired_refresh     验证 Token 失效（99991661）自动刷新并重试
        test_request_business_rate_limit       验证业务层频控（99991400）自动重试
        test_batch_create_chunking             验证批量创建遇 TooLargeRequest 自动分块
        test_sheet_read_chunking               验证 Sheet 读取遇 TooLargeResponse 自动分块
    test_session_recreated_across_event_loops   验证跨 event loop 时旧会话被关闭并重建
"""

import time
import asyncio
import pytest
from unittest.mock import patch
from src.data.feishu.client import (
    FeishuClient,
    FeishuAPIError,
    FeishuRateLimitError,
    TOKEN_URL,
)


# 模拟 HTTP 响应对象，用于替代 aiohttp.ClientResponse
class MockResponse:
    def __init__(self, status=200, data=None, headers=None, text_content=""):
        self.status = status
        self._data = data or {}
        self.headers = headers or {}
        self._text = text_content

    async def json(self):
        return self._data

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def client():
    """创建一个测试用的 FeishuClient"""
    return FeishuClient(
        app_id="cli_test",
        app_secret="test_secret",
        max_retries=2,  # 减少重试次数以加快测试
        qps_limit=0,  # 禁用本地限流以免拖慢测试
    )


@pytest.mark.asyncio
class TestFeishuClientAsync:
    """飞书客户端异步行为测试"""

    async def test_ensure_token_success(self, client):
        """测试成功获取 Token"""
        mock_resp = MockResponse(
            data={"code": 0, "tenant_access_token": "fake_token_123", "expire": 7200}
        )

        with patch("aiohttp.ClientSession.post", return_value=mock_resp):
            token = await client.ensure_token()

        assert token == "fake_token_123"
        assert client._token == "fake_token_123"
        assert client._token_expires_at > time.time()

    async def test_ensure_token_use_cache(self, client):
        """测试使用缓存的 Token"""
        client._token = "cached_token"
        client._token_expires_at = time.time() + 3600  # 1小时后过期

        # 即使 post 抛出异常，也不应该被调用
        with patch(
            "aiohttp.ClientSession.post", side_effect=Exception("Should not be called")
        ):
            token = await client.ensure_token()

        assert token == "cached_token"

    async def test_ensure_token_refresh_early(self, client):
        """测试 Token 快过期时提前刷新"""
        client._token = "old_token"
        client._token_expires_at = time.time() + 100  # 剩余 100s < 300s

        mock_resp = MockResponse(
            data={"code": 0, "tenant_access_token": "new_token", "expire": 7200}
        )

        with patch("aiohttp.ClientSession.post", return_value=mock_resp):
            token = await client.ensure_token()

        assert token == "new_token"

    async def test_request_success(self, client):
        """测试常规成功请求"""
        client._token = "valid_token"
        client._token_expires_at = time.time() + 3600

        mock_resp = MockResponse(data={"code": 0, "data": {"key": "value"}})

        with patch("aiohttp.ClientSession.request", return_value=mock_resp) as mock_req:
            data = await client._request("GET", "http://test.com")

        assert data == {"key": "value"}
        # 验证 Authorization 头
        call_kwargs = mock_req.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer valid_token"

    async def test_request_429_retry(self, client):
        """测试 429 限流重试"""
        client._token = "valid_token"
        client._token_expires_at = time.time() + 3600

        # 第一次 429，第二次 200
        responses = [
            MockResponse(status=429, headers={"x-ogw-ratelimit-reset": "0.1"}),
            MockResponse(status=200, data={"code": 0, "data": "success"}),
        ]

        # 注意：Mock side_effect 也可以是 iterator
        with patch("aiohttp.ClientSession.request", side_effect=responses):
            data = await client._request("GET", "http://test.com")

        assert data == "success"

    async def test_request_429_exhausted(self, client):
        """测试 429 重试耗尽"""
        client._token = "valid_token"
        client._token_expires_at = time.time() + 3600

        # 总是 429
        mock_resp = MockResponse(status=429, headers={"x-ogw-ratelimit-reset": "0.1"})

        with patch("aiohttp.ClientSession.request", return_value=mock_resp):
            with pytest.raises(FeishuRateLimitError):
                await client._request("GET", "http://test.com")

    async def test_request_token_expired_refresh(self, client):
        """测试 Token 失效自动刷新 (code 99991661)"""
        client._token = "invalid_token"
        client._token_expires_at = time.time() + 3600

        # 1. 业务请求返回 Token 失效
        resp_token_invalid = MockResponse(
            data={"code": 99991661, "msg": "Token invalid"}
        )
        # 2. Token 刷新请求 (ensure_token) -> 在 _request 内部逻辑中，
        #    ensure_token 会被调用，这里我们需要 mock ensure_token 里的 post 或者直接 mock ensure_token
        #    为了完整集成测试，我们让 ensure_token 发起 post 请求
        resp_token_refresh = MockResponse(
            data={"code": 0, "tenant_access_token": "new_token", "expire": 7200}
        )
        # 3. 业务请求重试成功
        resp_success = MockResponse(data={"code": 0, "data": "success"})

        # 包装 mock_request 使其适配 async context manager
        class AsyncContextManagerMock:
            def __init__(self, *args, **kwargs):
                # 如果调用的是 session.post(url, ...)，args[0] 是 url
                # 如果调用的是 session.request(method, url, ...)，args[0] 是 method, args[1] 是 url
                self.args = args
                self.kwargs = kwargs

            async def __aenter__(self):
                # 调用 mock_request 获取响应
                return await self.mock_request(*self.args, **self.kwargs)

            async def __aexit__(self, *args):
                pass

            async def mock_request(self, *args, **kwargs):
                # 解析参数
                if len(args) == 1:  # session.post(url)
                    url = args[0]
                elif len(args) >= 2:  # session.request(method, url)
                    url = args[1]
                else:
                    # 可能通过 kwargs 传参
                    url = kwargs.get("url", "")

                if url == TOKEN_URL:
                    return resp_token_refresh

                # 检查 Header
                auth = kwargs.get("headers", {}).get("Authorization")
                if auth == "Bearer invalid_token":
                    return resp_token_invalid
                elif auth == "Bearer new_token":
                    return resp_success
                else:
                    return MockResponse(status=400)  # Should not happen

        def side_effect(*args, **kwargs):
            return AsyncContextManagerMock(*args, **kwargs)

        with patch("aiohttp.ClientSession.request", side_effect=side_effect), patch(
            "aiohttp.ClientSession.post", side_effect=side_effect
        ):

            data = await client._request("GET", "http://test.com")

        assert data == "success"
        assert client._token == "new_token"

    async def test_request_business_rate_limit(self, client):
        """测试业务层频控 (code 99991400)"""
        client._token = "valid"
        client._token_expires_at = time.time() + 3600

        # 1. 频控
        resp_limit = MockResponse(data={"code": 99991400, "msg": "Business Limit"})
        # 2. 成功
        resp_success = MockResponse(data={"code": 0, "data": "ok"})

        with patch(
            "aiohttp.ClientSession.request", side_effect=[resp_limit, resp_success]
        ):
            data = await client._request("GET", "http://test.com")

        assert data == "ok"

    async def test_batch_create_chunking(self, client):
        """测试批量创建自动分块 (TooLargeRequest)"""
        client._token = "valid"
        client._token_expires_at = time.time() + 3600

        # 构造一批数据，假设一次发不过去
        records = [{"fields": {"id": i}} for i in range(10)]

        # 1. 第一次请求 (10条) -> 失败 (TooLarge)
        # 2. 第二次请求 (5条) -> 成功
        # 3. 第三次请求 (5条) -> 成功

        # 这里的难点是，client.bitable_batch_create 调用的是 self._request
        # 我们需要 mock self._request 来模拟不同情况

        with patch.object(client, "_request") as mock_req:
            # Side effect 函数
            def side_effect(method, url, json_data=None, **kwargs):
                recs = json_data.get("records", [])
                if len(recs) == 10:
                    raise FeishuAPIError(code=90227, msg="TooLargeRequest")
                return {"records": recs}

            mock_req.side_effect = side_effect

            res = await client.bitable_batch_create("token", "table", records)

        assert len(res) == 10
        # 应该调用了 3 次: 10(fail), 5(ok), 5(ok)
        assert mock_req.call_count == 3

    async def test_sheet_read_chunking(self, client):
        """测试 Sheet 读取自动分块"""
        client._token = "valid"
        client._token_expires_at = time.time() + 3600

        # 读取 A1:A10
        # 1. 1-10 -> TooLarge
        # 2. 1-5 -> OK
        # 3. 6-10 -> OK

        with patch.object(client, "_request") as mock_req:

            def side_effect(method, url, params=None, **kwargs):
                # 从 URL 解析 range
                # url 格式 .../values/range_str
                range_str = url.split("/values/")[-1]

                if "A1:A10" in range_str:
                    raise FeishuAPIError(code=90221, msg="TooLargeResponse")
                elif "A1:A5" in range_str:
                    return {"valueRange": {"values": [["1"]] * 5}}
                elif "A6:A10" in range_str:
                    return {"valueRange": {"values": [["2"]] * 5}}
                return {}

            mock_req.side_effect = side_effect

            res = await client.sheet_read_range("token", "Sheet1!A1:A10")

        assert len(res) == 10
        assert res[0] == ["1"]
        assert res[5] == ["2"]
        assert mock_req.call_count == 3


def test_session_recreated_across_event_loops():
    """
    回归测试：同一 FeishuClient 在多个 asyncio.run() 调用间不应复用旧会话

    修复前：第二次复用旧 loop 创建的 ClientSession，易触发 RuntimeError。
    修复后：检测 loop 切换后关闭旧会话并重建。
    """
    client = FeishuClient(
        app_id="cli_test",
        app_secret="test_secret",
    )

    async def _get_session():
        return await client._get_session()

    session1 = asyncio.run(_get_session())
    session2 = asyncio.run(_get_session())

    try:
        assert session1 is not session2
        assert session1.closed is True
        assert session2.closed is False
    finally:
        asyncio.run(client.close())
