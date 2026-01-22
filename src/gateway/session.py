"""
HTTP Session 连接池管理

按 (ssl_verify, proxy) 组合复用 HTTP 连接，避免重复创建 ClientSession。
"""

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
        sessions: Session 缓存字典
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
            max_connections: 总最大连接数
            max_connections_per_host: 每个主机最大连接数
            resolver: 自定义 DNS 解析器 (用于 IP 池轮询等场景)
        """
        self.sessions: dict[tuple[bool, str], aiohttp.ClientSession] = {}
        self._closed = False
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self._resolver = resolver

    async def get_or_create(
        self, ssl_verify: bool = True, proxy: str = ""
    ) -> aiohttp.ClientSession:
        """
        获取或创建 ClientSession

        Args:
            ssl_verify: 是否验证 SSL 证书
            proxy: 代理地址 (空字符串表示不使用代理)

        Returns:
            ClientSession 实例
        """
        if self._closed:
            raise RuntimeError("SessionPool 已关闭")

        key = (ssl_verify, proxy)

        if key not in self.sessions:
            # 创建新的 Session
            # 如果有自定义解析器，禁用 DNS 缓存以确保每次新连接都走解析器轮询
            connector = aiohttp.TCPConnector(
                ssl=ssl_verify if ssl_verify else False,
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host,
                resolver=self._resolver,
                use_dns_cache=self._resolver is None,  # 有自定义解析器时禁用缓存
                ttl_dns_cache=10 if self._resolver is None else 0,
            )

            session = aiohttp.ClientSession(connector=connector)
            self.sessions[key] = session

            logging.debug(
                f"创建新的 ClientSession: ssl_verify={ssl_verify}, proxy={proxy or 'None'}"
            )

        return self.sessions[key]

    async def close_all(self) -> None:
        """关闭所有 Session"""
        self._closed = True

        for key, session in self.sessions.items():
            try:
                await session.close()
                logging.debug(f"已关闭 ClientSession: {key}")
            except Exception as e:
                logging.warning(f"关闭 ClientSession 时出错: {e}")

        self.sessions.clear()

        # 关闭自定义解析器
        if self._resolver:
            try:
                await self._resolver.close()
                logging.debug("已关闭自定义 DNS 解析器")
            except Exception as e:
                logging.warning(f"关闭 DNS 解析器时出错: {e}")

        logging.info("SessionPool 已关闭所有连接")

    def get_stats(self) -> dict[str, Any]:
        """获取连接池统计信息"""
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

        # 添加解析器统计信息
        if self._resolver and hasattr(self._resolver, "get_stats"):
            stats["resolver_stats"] = self._resolver.get_stats()

        return stats
