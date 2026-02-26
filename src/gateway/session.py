"""
HTTP Session 连接池管理模块

本模块实现 HTTP 连接的高效复用，按 (ssl_verify, proxy) 组合管理
ClientSession 实例，避免重复创建连接，提高请求效率。

核心功能:
    - 连接复用：相同配置的请求共享同一个 ClientSession
    - 资源管理：统一管理所有 Session 的生命周期
    - 自定义解析器支持：可配置 IP 池轮询解析器
    - 统计信息：提供连接池状态查询

设计说明:
    为什么按 (ssl_verify, proxy) 分组？
    - aiohttp 的 ClientSession 与 TCPConnector 绑定
    - TCPConnector 的 ssl 和代理设置在创建时固定
    - 不同的 ssl_verify 或 proxy 需要不同的 Connector

    示例：
    - Session 1: (ssl_verify=True, proxy="")
    - Session 2: (ssl_verify=False, proxy="")
    - Session 3: (ssl_verify=True, proxy="http://proxy:8080")

使用示例:
    # 创建连接池
    pool = SessionPool(
        max_connections=1000,
        max_connections_per_host=100,
        resolver=custom_resolver,
    )

    # 获取或创建 Session
    session = await pool.get_or_create(ssl_verify=True, proxy="")

    # 使用 Session 发送请求
    async with session.get(url) as resp:
        data = await resp.json()

    # 关闭连接池
    await pool.close_all()

类与方法清单:

    SessionPool:
        HTTP Session 连接池（按 ssl_verify × proxy 分组复用）
        ├── __init__(max_connections=1000, max_connections_per_host=1000, resolver=None)
        │   输入: max_connections — 总最大连接数
        │         max_connections_per_host — 每主机最大连接数
        │         resolver — 自定义 DNS 解析器（可选）
        ├── get_or_create(ssl_verify=True, proxy="") -> aiohttp.ClientSession
        │   获取或创建 Session（双重检查锁定模式）
        │   输入: ssl_verify — 是否验证 SSL; proxy — 代理地址
        │   输出: 可用的 ClientSession 实例
        │   异常: RuntimeError — 连接池已关闭
        ├── close_all() -> None
        │   关闭所有 Session 和解析器，释放连接资源
        └── get_stats() -> dict
            获取连接池统计（Session 数量、配置、解析器状态）
        关键变量: sessions — (ssl_verify, proxy) 到 ClientSession 的映射
                  _create_lock — 异步锁，防止并发创建重复 Session

依赖模块:
    - aiohttp: 异步 HTTP 客户端和连接器
    - aiohttp.abc.AbstractResolver: 自定义 DNS 解析器接口

注意事项:
    - Session 不要手动关闭，由连接池统一管理
    - 使用自定义解析器时会禁用 DNS 缓存以确保轮询生效
    - 关闭连接池后不能再获取 Session
"""

import asyncio
import logging
from typing import Any

import aiohttp
from aiohttp.abc import AbstractResolver


class SessionPool:
    """
    HTTP Session 连接池

    按 (ssl_verify, proxy) 组合管理 ClientSession 实例，
    相同配置的请求复用同一个 Session，提高连接效率。

    Attributes:
        sessions (dict): Session 缓存字典，键为 (ssl_verify, proxy) 元组
        max_connections (int): 总最大连接数
        max_connections_per_host (int): 每个主机最大连接数

    生命周期:
        1. 创建 SessionPool 实例
        2. 通过 get_or_create() 获取 Session
        3. 使用 Session 发送请求
        4. 调用 close_all() 关闭所有连接
    """

    def __init__(
        self,
        max_connections: int = 1000,
        max_connections_per_host: int = 1000,
        resolver: AbstractResolver | None = None,
    ):
        """
        初始化连接池

        Args:
            max_connections: 总最大连接数（所有 Session 共享此限制）
            max_connections_per_host: 每个主机最大连接数
            resolver: 自定义 DNS 解析器（用于 IP 池轮询等场景）
        """
        self.sessions: dict[tuple[bool, str], aiohttp.ClientSession] = {}
        self._closed = False
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self._resolver = resolver
        self._create_lock = asyncio.Lock()

    async def get_or_create(
        self, ssl_verify: bool = True, proxy: str = ""
    ) -> aiohttp.ClientSession:
        """
        获取或创建 ClientSession

        根据 ssl_verify 和 proxy 的组合查找或创建 Session。
        相同配置的请求会复用同一个 Session。

        Args:
            ssl_verify: 是否验证 SSL 证书
            proxy: 代理地址（空字符串表示不使用代理）

        Returns:
            aiohttp.ClientSession: 可用的 Session 实例

        Raises:
            RuntimeError: 如果连接池已关闭
        """
        if self._closed:
            raise RuntimeError("SessionPool 已关闭")

        key = (ssl_verify, proxy)

        existing = self.sessions.get(key)
        if existing is not None:
            if not existing.closed:
                return existing
            self.sessions.pop(key, None)

        async with self._create_lock:
            if self._closed:
                raise RuntimeError("SessionPool 已关闭")

            existing = self.sessions.get(key)
            if existing is not None:
                if not existing.closed:
                    return existing
                self.sessions.pop(key, None)

            # 创建新的 Session
            # 注意：如果有自定义解析器，禁用 DNS 缓存以确保每次新连接都走解析器轮询
            connector = aiohttp.TCPConnector(
                ssl=ssl_verify if ssl_verify else False,
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host,
                resolver=self._resolver,
                use_dns_cache=self._resolver is None,  # 有自定义解析器时禁用缓存
                ttl_dns_cache=(
                    10 if self._resolver is None else 0
                ),  # 无自定义解析器时使用 10 秒 TTL
            )

            session = aiohttp.ClientSession(connector=connector)
            self.sessions[key] = session

            logging.debug(
                f"创建新的 ClientSession: ssl_verify={ssl_verify}, proxy={proxy or 'None'}"
            )
            return session

    async def close_all(self) -> None:
        """
        关闭所有 Session

        释放所有连接资源，包括 Session 和自定义解析器。
        关闭后不能再获取 Session。
        """
        async with self._create_lock:
            self._closed = True
            sessions_to_close = list(self.sessions.items())
            self.sessions.clear()

        # 关闭所有 Session
        for key, session in sessions_to_close:
            try:
                await session.close()
                logging.debug(f"已关闭 ClientSession: {key}")
            except Exception as e:
                logging.warning(f"关闭 ClientSession 时出错: {e}")

        # 关闭自定义解析器
        if self._resolver:
            try:
                await self._resolver.close()
                logging.debug("已关闭自定义 DNS 解析器")
            except Exception as e:
                logging.warning(f"关闭 DNS 解析器时出错: {e}")

        logging.info("SessionPool 已关闭所有连接")

    def get_stats(self) -> dict[str, Any]:
        """
        获取连接池统计信息

        Returns:
            dict: 包含以下信息：
                - total_sessions: Session 总数
                - sessions: 各 Session 的配置信息
                - has_custom_resolver: 是否使用自定义解析器
                - resolver_stats: 解析器统计（如有）
        """
        stats: dict[str, Any] = {
            "total_sessions": len(self.sessions),
            "sessions": [
                {
                    "ssl_verify": key[0],
                    "proxy": key[1] or None,
                }
                for key in self.sessions.keys()
            ],
            "has_custom_resolver": self._resolver is not None,
        }

        # 添加解析器统计信息（如果解析器支持）
        if self._resolver and hasattr(self._resolver, "get_stats"):
            stats["resolver_stats"] = self._resolver.get_stats()

        return stats
