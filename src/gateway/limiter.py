"""
限流组件

包含读写锁、令牌桶和模型限流管理器。
"""

import threading
import time
from typing import Any


class RWLock:
    """
    读写锁

    允许多个读操作并发，但写操作需要独占。
    用于保护模型配置等需要频繁读取但偶尔更新的数据。
    """

    def __init__(self):
        """初始化读写锁"""
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers = 0
        self._write_ready = threading.Condition(threading.Lock())
        self._pending_writers = 0

    def read_acquire(self) -> None:
        """获取读锁"""
        with self._read_ready:
            while self._writers > 0 or self._pending_writers > 0:
                self._read_ready.wait()
            self._readers += 1

    def read_release(self) -> None:
        """释放读锁"""
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def write_acquire(self) -> None:
        """获取写锁"""
        with self._write_ready:
            self._pending_writers += 1

        with self._read_ready:
            while self._readers > 0 or self._writers > 0:
                self._read_ready.wait()
            self._writers += 1

        with self._write_ready:
            self._pending_writers -= 1

    def write_release(self) -> None:
        """释放写锁"""
        with self._read_ready:
            self._writers -= 1
            self._read_ready.notify_all()

    class ReadLock:
        """读锁上下文管理器"""

        def __init__(self, rwlock: "RWLock"):
            self.rwlock = rwlock

        def __enter__(self) -> "RWLock.ReadLock":
            self.rwlock.read_acquire()
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            self.rwlock.read_release()

    class WriteLock:
        """写锁上下文管理器"""

        def __init__(self, rwlock: "RWLock"):
            self.rwlock = rwlock

        def __enter__(self) -> "RWLock.WriteLock":
            self.rwlock.write_acquire()
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            self.rwlock.write_release()

    def read_lock(self) -> "RWLock.ReadLock":
        """返回读锁上下文管理器"""
        return self.ReadLock(self)

    def write_lock(self) -> "RWLock.WriteLock":
        """返回写锁上下文管理器"""
        return self.WriteLock(self)


class TokenBucket:
    """
    令牌桶限流器

    实现基于令牌桶算法的限流，支持平滑的请求速率控制。

    Attributes:
        capacity: 桶容量（最大令牌数）
        refill_rate: 每秒补充的令牌数
        tokens: 当前令牌数
    """

    def __init__(self, capacity: float, refill_rate: float):
        """
        初始化令牌桶

        Args:
            capacity: 桶容量（最大令牌数）
            refill_rate: 每秒补充的令牌数
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def refill(self) -> None:
        """根据经过的时间补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        """
        尝试消耗指定数量的令牌

        Args:
            tokens: 要消耗的令牌数

        Returns:
            如果有足够令牌返回 True，否则返回 False
        """
        with self.lock:
            self.refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_tokens(self) -> float:
        """获取当前令牌数"""
        with self.lock:
            self.refill()
            return self.tokens


class ModelRateLimiter:
    """
    模型限流管理器

    为每个模型维护独立的令牌桶，实现模型级别的请求限流。
    """

    def __init__(self):
        """初始化模型限流管理器"""
        self.limiters: dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    def configure(self, models_config: list[dict[str, Any]]) -> None:
        """
        从模型配置中配置限流器

        Args:
            models_config: 模型配置列表，每个配置需包含 id 和 safe_rps
        """
        with self.lock:
            for model_cfg in models_config:
                model_id = str(model_cfg.get("id", ""))
                safe_rps = float(model_cfg.get("safe_rps", 5.0))

                if model_id and safe_rps > 0:
                    # 容量为 safe_rps * 2，允许短时间突发
                    self.limiters[model_id] = TokenBucket(
                        capacity=safe_rps * 2, refill_rate=safe_rps
                    )

    def can_process(self, model_id: str) -> bool:
        """
        检查模型是否可以处理请求（不消耗令牌）

        用于模型选择阶段的预检查。

        Args:
            model_id: 模型 ID

        Returns:
            是否有足够令牌处理请求
        """
        with self.lock:
            limiter = self.limiters.get(model_id)

        if limiter is None:
            return True  # 未配置限流器，默认允许

        # 只检查不消耗
        return limiter.get_tokens() >= 1.0

    def acquire(self, model_id: str) -> bool:
        """
        尝试获取模型的请求许可（消耗令牌）

        Args:
            model_id: 模型 ID

        Returns:
            是否获取成功
        """
        with self.lock:
            limiter = self.limiters.get(model_id)

        if limiter is None:
            return True  # 未配置限流器，默认允许

        return limiter.consume()

    def get_status(self, model_id: str) -> dict[str, Any] | None:
        """
        获取模型限流器状态

        Args:
            model_id: 模型 ID

        Returns:
            限流器状态字典或 None
        """
        with self.lock:
            limiter = self.limiters.get(model_id)

        if limiter is None:
            return None

        return {
            "model_id": model_id,
            "capacity": limiter.capacity,
            "refill_rate": limiter.refill_rate,
            "current_tokens": limiter.get_tokens(),
        }
