"""
限流组件模块

本模块提供 API 网关的并发控制和限流功能，包括：
- RWLock: 读写锁，用于保护共享数据
- TokenBucket: 令牌桶限流算法实现
- ModelRateLimiter: 模型级别的限流管理器

核心概念:

读写锁 (RWLock):
    允许多个读操作并发执行，但写操作需要独占访问。
    适用于读多写少的场景，如模型状态查询。
    
令牌桶 (TokenBucket):
    经典的流量整形算法，具有以下特点：
    - 以固定速率向桶中添加令牌
    - 每个请求消耗一个令牌
    - 桶有容量上限，允许短时间突发流量
    - 桶空时拒绝请求，实现限流

限流策略:
    每个模型有独立的令牌桶，配置参数：
    - capacity = safe_rps * 2（允许 2 秒的突发流量）
    - refill_rate = safe_rps（每秒补充令牌数）
    
    例如 safe_rps=5 时：
    - 容量 10，最多积累 10 个令牌
    - 每秒补充 5 个令牌
    - 短时间可处理 10 个请求，持续速率 5 QPS

使用示例:
    # 读写锁
    rwlock = RWLock()
    with rwlock.read_lock():
        # 读取操作
        pass
    with rwlock.write_lock():
        # 写入操作
        pass
    
    # 限流器
    limiter = ModelRateLimiter()
    limiter.configure([{"id": "model-1", "safe_rps": 5}])
    
    if limiter.acquire("model-1"):
        # 处理请求
        pass
    else:
        # 被限流，稍后重试
        pass
"""

import threading
import time
from typing import Any


class RWLock:
    """
    读写锁
    
    允许多个读操作并发执行，但写操作需要独占访问。
    适用于保护模型配置等读多写少的数据结构。
    
    特性:
    - 多个读者可同时持有读锁
    - 写者需要等待所有读者释放
    - 有待处理的写者时，新读者需要等待（防止写者饥饿）
    
    使用方式:
        rwlock = RWLock()
        
        # 方式一：使用上下文管理器
        with rwlock.read_lock():
            # 读操作
            pass
        
        with rwlock.write_lock():
            # 写操作
            pass
        
        # 方式二：手动获取/释放
        rwlock.read_acquire()
        try:
            # 读操作
            pass
        finally:
            rwlock.read_release()
    """

    def __init__(self):
        """初始化读写锁"""
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0          # 当前持有读锁的数量
        self._writers = 0          # 当前持有写锁的数量（0 或 1）
        self._write_ready = threading.Condition(threading.Lock())
        self._pending_writers = 0  # 等待写锁的数量

    def read_acquire(self) -> None:
        """
        获取读锁
        
        等待直到没有写者且没有待处理的写者。
        """
        with self._read_ready:
            # 如果有写者或待处理的写者，等待
            while self._writers > 0 or self._pending_writers > 0:
                self._read_ready.wait()
            self._readers += 1

    def read_release(self) -> None:
        """
        释放读锁
        
        当最后一个读者释放时，通知所有等待者。
        """
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def write_acquire(self) -> None:
        """
        获取写锁
        
        标记为待处理写者，然后等待所有读者和其他写者释放。
        """
        # 标记为待处理写者
        with self._write_ready:
            self._pending_writers += 1

        # 等待读者和其他写者释放
        with self._read_ready:
            while self._readers > 0 or self._writers > 0:
                self._read_ready.wait()
            self._writers += 1

        # 取消待处理状态
        with self._write_ready:
            self._pending_writers -= 1

    def write_release(self) -> None:
        """
        释放写锁
        
        释放后通知所有等待者（包括读者和写者）。
        """
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
        """
        返回读锁上下文管理器
        
        Returns:
            ReadLock: 可用于 with 语句的读锁
        """
        return self.ReadLock(self)

    def write_lock(self) -> "RWLock.WriteLock":
        """
        返回写锁上下文管理器
        
        Returns:
            WriteLock: 可用于 with 语句的写锁
        """
        return self.WriteLock(self)


class TokenBucket:
    """
    令牌桶限流器
    
    实现基于令牌桶算法的限流，支持平滑的请求速率控制。
    令牌以固定速率补充，每个请求消耗一个令牌，桶满时停止补充。

    Attributes:
        capacity (float): 桶容量（最大令牌数）
        refill_rate (float): 每秒补充的令牌数
        tokens (float): 当前令牌数
        last_refill (float): 上次补充的时间戳
    
    算法特点:
        - 允许突发流量（最多 capacity 个请求）
        - 持续速率受 refill_rate 限制
        - 线程安全（使用互斥锁）
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
        self.tokens = capacity  # 初始满桶
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def refill(self) -> None:
        """
        根据经过的时间补充令牌
        
        计算自上次补充以来应该添加的令牌数，
        但不超过桶容量。
        """
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        """
        尝试消耗指定数量的令牌

        Args:
            tokens: 要消耗的令牌数（默认 1）

        Returns:
            bool: 如果有足够令牌返回 True，否则返回 False
        """
        with self.lock:
            self.refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_tokens(self) -> float:
        """
        获取当前令牌数
        
        会先执行补充计算，返回实时的令牌数。
        
        Returns:
            float: 当前可用令牌数
        """
        with self.lock:
            self.refill()
            return self.tokens


class ModelRateLimiter:
    """
    模型限流管理器
    
    为每个模型维护独立的令牌桶，实现模型级别的请求限流。
    支持在模型选择阶段预检查（不消耗令牌）和实际请求时获取许可（消耗令牌）。

    使用流程:
        1. 配置: configure([{"id": "model-1", "safe_rps": 5}, ...])
        2. 预检查: can_process("model-1")  # 不消耗令牌
        3. 获取许可: acquire("model-1")     # 消耗令牌
    
    Attributes:
        limiters (dict): 模型 ID 到 TokenBucket 的映射
    """

    def __init__(self):
        """初始化模型限流管理器"""
        self.limiters: dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    def configure(self, models_config: list[dict[str, Any]]) -> None:
        """
        从模型配置中配置限流器
        
        为每个模型创建独立的令牌桶，容量为 safe_rps * 2，
        允许短时间的突发流量。

        Args:
            models_config: 模型配置列表，每个配置需包含 id 和 safe_rps
        """
        with self.lock:
            for model_cfg in models_config:
                model_id = str(model_cfg.get("id", ""))
                safe_rps = float(model_cfg.get("safe_rps", 5.0))

                if model_id and safe_rps > 0:
                    # 容量为 safe_rps * 2，允许 2 秒的突发流量
                    self.limiters[model_id] = TokenBucket(
                        capacity=safe_rps * 2, refill_rate=safe_rps
                    )

    def can_process(self, model_id: str) -> bool:
        """
        检查模型是否可以处理请求（不消耗令牌）
        
        用于模型选择阶段的预检查，避免选择已被限流的模型。

        Args:
            model_id: 模型 ID

        Returns:
            bool: 是否有足够令牌处理请求
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
        
        在实际发送请求前调用，消耗一个令牌。

        Args:
            model_id: 模型 ID

        Returns:
            bool: 是否获取成功
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
            dict: 限流器状态，包含容量、补充速率、当前令牌数
            None: 如果模型未配置限流器
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
