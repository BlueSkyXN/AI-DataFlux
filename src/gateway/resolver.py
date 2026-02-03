"""
自定义 DNS 解析器模块

本模块实现 IP 池轮询 DNS 解析，支持多 IP 负载均衡和故障回退。
当配置了 IP 池时，按轮询顺序返回 IP 列表，避免单点故障。

核心功能:
    - IP 池轮询：每次解析返回轮换排序的 IP 列表
    - 故障回退：aiohttp 会按返回顺序尝试连接
    - 线程安全：使用锁保护轮询计数器
    - 无缝降级：未配置 IP 池的域名使用系统默认解析

工作原理:
    假设域名配置了 IP 池 [ip1, ip2, ip3]:
    - 第 1 次解析: 返回 [ip1, ip2, ip3]
    - 第 2 次解析: 返回 [ip2, ip3, ip1]
    - 第 3 次解析: 返回 [ip3, ip1, ip2]
    - 第 4 次解析: 返回 [ip1, ip2, ip3]（循环）

    aiohttp 会按返回顺序尝试连接，第一个失败时自动尝试下一个。

配置方式:
    在 channels 中配置 ip_pool:

    channels:
      "1":
        base_url: "https://api.example.com"
        ip_pool:
          - "1.2.3.4"
          - "1.2.3.5"
          - "1.2.3.6"

使用示例:
    # 从通道配置构建 IP 池
    ip_pools = build_ip_pools_from_channels(channels)

    # 创建解析器
    resolver = RoundRobinResolver(ip_pools)

    # 创建使用自定义解析器的连接器
    connector = aiohttp.TCPConnector(resolver=resolver)
    session = aiohttp.ClientSession(connector=connector)

注意事项:
    - 配置了代理的通道不应使用 IP 池（代理模式下无效）
    - IP 池中的地址必须是有效的 IPv4 或 IPv6 地址
    - 使用自定义解析器时应禁用 DNS 缓存以确保轮询生效
"""

import logging
import socket
from collections import defaultdict
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult


class RoundRobinResolver(AbstractResolver):
    """
    轮询 DNS 解析器

    对配置了 IP 池的域名进行轮询解析，实现多 IP 均匀使用。
    未配置 IP 池的域名使用系统默认解析。

    特性:
        - 按域名轮询 IP 地址，每次新连接使用下一个 IP
        - 支持故障回退：aiohttp 会按返回的 IP 列表顺序尝试连接
        - 线程安全的轮询计数器
        - 支持 IPv4 和 IPv6 地址

    Attributes:
        _ip_pools (dict): 域名到 IP 列表的映射
        _default_port (int): 默认端口号（通常为 443）
        _counters (dict): 域名到轮询计数器的映射

    Example:
        ip_pools = {
            "api.example.com": ["1.2.3.4", "1.2.3.5", "1.2.3.6"],
        }
        resolver = RoundRobinResolver(ip_pools)
        connector = aiohttp.TCPConnector(resolver=resolver)
    """

    def __init__(
        self,
        ip_pools: dict[str, list[str]] | None = None,
        default_port: int = 443,
    ):
        """
        初始化解析器

        Args:
            ip_pools: 域名到 IP 列表的映射
                     如 {"api.example.com": ["1.2.3.4", "1.2.3.5"]}
            default_port: 默认端口号（用于无端口的 IP）
        """
        self._ip_pools: dict[str, list[str]] = ip_pools or {}
        self._default_port = default_port
        self._counters: dict[str, int] = defaultdict(int)
        self._lock = Lock()

        # 延迟创建默认解析器（需要在事件循环中创建）
        self._default_resolver: aiohttp.DefaultResolver | None = None

        # 记录初始化信息
        if self._ip_pools:
            logging.info(
                f"RoundRobinResolver 初始化: {len(self._ip_pools)} 个域名配置了 IP 池"
            )
            for host, ips in self._ip_pools.items():
                logging.info(f"  - {host}: {len(ips)} 个 IP")

    def _get_default_resolver(self) -> aiohttp.DefaultResolver:
        """
        获取或创建默认解析器

        延迟初始化，确保在事件循环中创建。

        Returns:
            aiohttp.DefaultResolver: 默认 DNS 解析器
        """
        if self._default_resolver is None:
            self._default_resolver = aiohttp.DefaultResolver()
        return self._default_resolver

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        """
        解析域名

        对配置了 IP 池的域名返回轮询排序的 IP 列表，
        未配置的域名使用默认解析器。

        Args:
            host: 要解析的域名
            port: 端口号（0 表示使用默认端口）
            family: 地址族（AF_INET 或 AF_INET6）

        Returns:
            list[ResolveResult]: 按轮询顺序排列的解析结果列表

        轮询逻辑:
            IP 列表 [a, b, c]，计数器从 0 开始:
            - 计数器=0: 返回 [a, b, c]
            - 计数器=1: 返回 [b, c, a]
            - 计数器=2: 返回 [c, a, b]
        """
        # 检查是否有配置的 IP 池
        if host not in self._ip_pools:
            # 使用默认解析器
            return await self._get_default_resolver().resolve(host, port, family)

        ip_list = self._ip_pools[host]
        if not ip_list:
            # IP 列表为空，回退到默认解析
            logging.warning(f"域名 {host} 的 IP 池为空，使用默认解析")
            return await self._get_default_resolver().resolve(host, port, family)

        # 获取轮询起始位置并更新计数器
        with self._lock:
            start_index = self._counters[host]
            self._counters[host] = (start_index + 1) % len(ip_list)

        # 按轮询顺序构建 IP 列表
        # 例如: [ip1, ip2, ip3] 且 start_index=1 -> [ip2, ip3, ip1]
        rotated_ips = ip_list[start_index:] + ip_list[:start_index]

        # 构建 ResolveResult 列表
        results: list[ResolveResult] = []
        effective_port = port if port > 0 else self._default_port

        for ip in rotated_ips:
            # 检测 IP 类型（IPv4 或 IPv6）
            try:
                socket.inet_pton(socket.AF_INET6, ip)
                ip_family = socket.AF_INET6
            except socket.error:
                ip_family = socket.AF_INET

            # 如果请求特定地址族，跳过不匹配的 IP
            if family != socket.AF_UNSPEC and family != ip_family:
                continue

            results.append(
                ResolveResult(
                    hostname=host,
                    host=ip,
                    port=effective_port,
                    family=ip_family,
                    proto=socket.IPPROTO_TCP,
                    flags=socket.AI_NUMERICHOST,
                )
            )

        if not results:
            # 没有匹配的 IP，回退到默认解析
            logging.warning(
                f"域名 {host} 的 IP 池中没有匹配 family={family} 的地址，使用默认解析"
            )
            return await self._get_default_resolver().resolve(host, port, family)

        logging.debug(
            f"RoundRobinResolver: {host} -> {[r['host'] for r in results]} "
            f"(起始索引: {start_index})"
        )

        return results

    async def close(self) -> None:
        """
        关闭解析器

        清理默认解析器资源。
        """
        if self._default_resolver is not None:
            await self._default_resolver.close()

    def get_stats(self) -> dict[str, Any]:
        """
        获取解析器统计信息

        Returns:
            dict: 包含配置的域名列表和当前计数器状态
        """
        with self._lock:
            return {
                "configured_hosts": list(self._ip_pools.keys()),
                "counters": dict(self._counters),
            }


def build_ip_pools_from_channels(channels: dict[str, Any]) -> dict[str, list[str]]:
    """
    从通道配置构建 IP 池映射

    解析每个通道的 base_url 和 ip_pool 配置，
    生成域名到 IP 列表的映射字典。

    Args:
        channels: 通道配置字典

    Returns:
        dict: 域名到 IP 列表的映射

    配置示例:
        channels = {
            "1": {
                "base_url": "https://api.example.com",
                "ip_pool": ["1.2.3.4", "1.2.3.5"],
            }
        }
        # 返回: {"api.example.com": ["1.2.3.4", "1.2.3.5"]}

    注意:
        - 配置了 proxy 的通道会跳过 ip_pool（代理模式下无效）
        - 无效的 IP 地址会被过滤并记录警告
        - 多个通道可以为同一域名配置 IP，会自动合并
    """
    ip_pools: dict[str, list[str]] = {}

    for channel_id, channel_config in channels.items():
        ip_pool = channel_config.get("ip_pool")

        # 跳过未配置 ip_pool 或配置无效的通道
        if not ip_pool or not isinstance(ip_pool, list):
            continue

        # 检查是否配置了代理
        proxy = channel_config.get("proxy", "")
        if proxy:
            logging.warning(
                f"通道 {channel_id} 同时配置了 proxy 和 ip_pool，"
                f"ip_pool 将被忽略 (代理模式下 IP 池无效)"
            )
            continue

        base_url = channel_config.get("base_url", "")
        if not base_url:
            continue

        # 从 URL 提取域名
        try:
            parsed = urlparse(base_url)
            host = parsed.hostname
            if host:
                # 过滤有效的 IP 地址
                valid_ips = []
                for ip in ip_pool:
                    if _is_valid_ip(ip):
                        valid_ips.append(ip)
                    else:
                        logging.warning(
                            f"通道 {channel_id} 的 ip_pool 中包含无效 IP: {ip}"
                        )

                if valid_ips:
                    if host in ip_pools:
                        # 合并多个通道对同一域名的 IP 配置（去重）
                        existing = set(ip_pools[host])
                        for ip in valid_ips:
                            if ip not in existing:
                                ip_pools[host].append(ip)
                                existing.add(ip)
                    else:
                        ip_pools[host] = valid_ips

                    logging.info(
                        f"通道 {channel_id}: 域名 {host} 配置了 {len(valid_ips)} 个 IP"
                    )
        except Exception as e:
            logging.warning(f"解析通道 {channel_id} 的 base_url 失败: {e}")

    return ip_pools


def _is_valid_ip(ip_str: str) -> bool:
    """
    检查字符串是否为有效的 IPv4 或 IPv6 地址

    Args:
        ip_str: 要检查的字符串

    Returns:
        bool: 如果是有效的 IP 地址返回 True
    """
    # 尝试解析为 IPv4
    try:
        socket.inet_pton(socket.AF_INET, ip_str)
        return True
    except socket.error:
        pass

    # 尝试解析为 IPv6
    try:
        socket.inet_pton(socket.AF_INET6, ip_str)
        return True
    except socket.error:
        pass

    return False
