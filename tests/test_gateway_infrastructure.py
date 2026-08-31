from __future__ import annotations

import socket
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.gateway.resolver import RoundRobinResolver, build_ip_pools_from_channels
from src.gateway.session import SessionPool
from src.gateway.dispatcher import ModelConfig, ModelDispatcher


class _DefaultResolver:
    def __init__(self):
        self.resolve = AsyncMock(
            return_value=[
                {
                    "hostname": "fallback.test",
                    "host": "127.0.0.1",
                    "port": 443,
                    "family": socket.AF_INET,
                    "proto": socket.IPPROTO_TCP,
                    "flags": 0,
                }
            ]
        )
        self.close = AsyncMock()


@pytest.mark.asyncio
async def test_round_robin_resolver_rotation_family_fallback_and_close(monkeypatch):
    fallback = _DefaultResolver()
    monkeypatch.setattr(
        "src.gateway.resolver.aiohttp.DefaultResolver", lambda: fallback
    )
    resolver = RoundRobinResolver(
        {
            "api.test": ["10.0.0.1", "10.0.0.2", "2001:db8::1"],
            "empty.test": [],
        }
    )

    first = await resolver.resolve("api.test", 8443, socket.AF_UNSPEC)
    second = await resolver.resolve("api.test", 8443, socket.AF_UNSPEC)
    ipv6 = await resolver.resolve("api.test", 0, socket.AF_INET6)
    missing = await resolver.resolve("missing.test", 443, socket.AF_INET)
    empty = await resolver.resolve("empty.test", 443, socket.AF_INET)

    assert [item["host"] for item in first] == [
        "10.0.0.1",
        "10.0.0.2",
        "2001:db8::1",
    ]
    assert [item["host"] for item in second] == [
        "10.0.0.2",
        "2001:db8::1",
        "10.0.0.1",
    ]
    assert [item["host"] for item in ipv6] == ["2001:db8::1"]
    assert ipv6[0]["port"] == 443
    assert missing == empty == fallback.resolve.return_value
    assert resolver.get_stats()["counters"]["api.test"] == 0
    await resolver.close()
    fallback.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_round_robin_resolver_falls_back_when_family_has_no_match(
    monkeypatch,
):
    fallback = _DefaultResolver()
    monkeypatch.setattr(
        "src.gateway.resolver.aiohttp.DefaultResolver", lambda: fallback
    )
    resolver = RoundRobinResolver({"ipv4.test": ["10.0.0.1"]})

    result = await resolver.resolve("ipv4.test", 443, socket.AF_INET6)

    assert result == fallback.resolve.return_value
    fallback.resolve.assert_awaited_once()


def test_build_ip_pools_filters_invalid_proxy_and_deduplicates():
    pools = build_ip_pools_from_channels(
        {
            "first": {
                "base_url": "https://api.example.test/v1",
                "ip_pool": ["10.0.0.1", "invalid"],
            },
            "second": {
                "base_url": "https://api.example.test",
                "ip_pool": ["10.0.0.1", "10.0.0.2"],
            },
            "proxy": {
                "base_url": "https://ignored.test",
                "proxy": "http://proxy.test",
                "ip_pool": ["10.0.0.3"],
            },
            "missing": {"ip_pool": ["10.0.0.4"]},
            "none": {"base_url": "https://none.test"},
        }
    )

    assert pools == {"api.example.test": ["10.0.0.1", "10.0.0.2"]}


class _Session:
    def __init__(self, connector=None):
        self.connector = connector
        self.closed = False
        self.close = AsyncMock(side_effect=self._close)

    async def _close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_session_pool_reuses_replaces_closes_and_reports_stats(monkeypatch):
    connectors = []
    sessions = []

    def connector_factory(**kwargs):
        connector = MagicMock(kwargs=kwargs)
        connectors.append(connector)
        return connector

    def session_factory(*, connector):
        session = _Session(connector)
        sessions.append(session)
        return session

    resolver = MagicMock()
    resolver.close = AsyncMock()
    resolver.get_stats.return_value = {"configured_hosts": ["api.test"]}
    monkeypatch.setattr("src.gateway.session.aiohttp.TCPConnector", connector_factory)
    monkeypatch.setattr("src.gateway.session.aiohttp.ClientSession", session_factory)
    pool = SessionPool(10, 2, resolver=resolver)

    first = await pool.get_or_create(ssl_verify=False, proxy="http://proxy")
    assert await pool.get_or_create(False, "http://proxy") is first
    first.closed = True
    replacement = await pool.get_or_create(False, "http://proxy")

    assert replacement is not first
    assert connectors[0].kwargs["ssl"] is False
    assert connectors[0].kwargs["use_dns_cache"] is False
    stats = pool.get_stats()
    assert stats["total_sessions"] == 1
    assert stats["resolver_stats"] == {"configured_hosts": ["api.test"]}

    await pool.close_all()
    replacement.close.assert_awaited_once()
    resolver.close.assert_awaited_once()
    with pytest.raises(RuntimeError, match="关闭"):
        await pool.get_or_create()


@pytest.mark.asyncio
async def test_session_pool_tolerates_close_failures(monkeypatch):
    session = _Session()
    session.close = AsyncMock(side_effect=RuntimeError("close failed"))
    resolver = MagicMock()
    resolver.close = AsyncMock(side_effect=RuntimeError("resolver close failed"))
    pool = SessionPool(resolver=resolver)
    pool.sessions[(True, "")] = session

    await pool.close_all()

    assert pool.get_stats()["total_sessions"] == 0


def _model_config(model_id="model-a", *, capabilities=None, endpoints=None):
    channels = {
        "channel": {
            "name": "Channel",
            "base_url": "https://api.example.test",
            "endpoints": endpoints
            or {
                "chat_completions": "/v1/chat/completions",
                "responses": "/v1/responses",
            },
            "timeout": 30,
        }
    }
    return ModelConfig(
        {
            "id": model_id,
            "name": model_id,
            "model": f"upstream-{model_id}",
            "channel_id": "channel",
            "weight": 2,
            "capabilities": capabilities or ["chat_completions", "responses"],
        },
        channels,
    )


def test_model_config_and_dispatcher_metrics_backoff_and_selection(monkeypatch):
    first = _model_config("first")
    second = _model_config("second")
    dispatcher = ModelDispatcher([first, second])

    dispatcher.update_model_metrics("first", 2.0, True)
    dispatcher.update_model_metrics("first", 4.0, False)
    dispatcher.update_model_metrics("missing", 1.0, True)
    assert dispatcher.get_model_success_rate("first") == 0.5
    assert dispatcher.get_model_success_rate("missing") == 0.0
    assert dispatcher.get_model_avg_response_time("first") == pytest.approx(2.2)
    assert dispatcher.get_model_avg_response_time("missing") == 1.0

    dispatcher.mark_model_failed("first", "content_error")
    assert dispatcher.is_model_available("first") is True
    dispatcher.mark_model_failed("first")
    assert dispatcher.is_model_available("first") is False
    assert dispatcher.get_available_models({"second"}) == []
    dispatcher.mark_model_success("first")
    assert dispatcher.is_model_available("first") is True

    monkeypatch.setattr(
        "src.gateway.dispatcher.random.choices", lambda models, **_kwargs: [models[-1]]
    )
    assert dispatcher.select_model().id == "second"
    assert dispatcher.select_model({"first", "second"}) is None
    assert dispatcher.get_model_config("first") is first
    assert {item["id"] for item in dispatcher.get_all_model_stats()} == {
        "first",
        "second",
    }


@pytest.mark.parametrize(
    ("model", "channels", "message"),
    [
        (
            {"id": "x", "model": "x", "channel_id": "c", "capabilities": []},
            {"c": {"base_url": "https://x", "endpoints": {}}},
            "capabilities",
        ),
        (
            {
                "id": "x",
                "model": "x",
                "channel_id": "missing",
                "capabilities": ["chat_completions"],
            },
            {},
            "不存在",
        ),
        (
            {
                "id": "x",
                "model": "x",
                "channel_id": "c",
                "capabilities": ["responses"],
            },
            {
                "c": {
                    "base_url": "https://x",
                    "endpoints": {"chat_completions": "/v1/chat/completions"},
                }
            },
            "responses",
        ),
    ],
)
def test_model_config_rejects_invalid_contract(model, channels, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig(model, channels)
