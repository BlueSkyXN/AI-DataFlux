"""
HTTP Session 连接池管理

按 (ssl_verify, proxy) 组合复用 HTTP 连接，避免重复创建 ClientSession。
"""

import logging
from typing import Any

import aiohttp


class SessionPool:
    """
    HTTP Session 连接池
    
    按 (ssl_verify, proxy) 组合管理 ClientSession 实例，
    相同配置的请求复用同一个 Session，提高连接效率。
    
    Attributes:
        sessions: Session 缓存字典
    """
    
    def __init__(self):
        """初始化连接池"""
        self.sessions: dict[tuple[bool, str], aiohttp.ClientSession] = {}
        self._closed = False
    
    async def get_or_create(
        self, 
        ssl_verify: bool = True, 
        proxy: str = ""
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
            connector = aiohttp.TCPConnector(
                ssl=ssl_verify if ssl_verify else False,
                limit=100,
                limit_per_host=30,
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
        logging.info("SessionPool 已关闭所有连接")
    
    def get_stats(self) -> dict[str, Any]:
        """获取连接池统计信息"""
        return {
            "total_sessions": len(self.sessions),
            "sessions": [
                {
                    "ssl_verify": key[0],
                    "proxy": key[1] or None,
                }
                for key in self.sessions.keys()
            ],
        }
