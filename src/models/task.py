"""
任务元数据定义

本模块定义任务处理过程中的内部状态数据结构，完全独立于业务数据。
采用 dataclass 实现，确保类型安全和代码简洁。

设计目标:
    1. 状态分离: 元数据不包含业务数据，避免内存泄漏
    2. 重试追踪: 按错误类型独立统计重试次数
    3. 错误历史: 保留最近 N 条错误记录用于调试
    4. 生命周期管理: 支持重置和清理操作

数据结构:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        TaskMetadata                              │
    ├─────────────────────────────────────────────────────────────────┤
    │  record_id: Any              # 任务唯一标识                      │
    │  created_at: float           # 创建时间戳                        │
    │  last_retry_at: float?       # 最后重试时间                      │
    │  retry_counts: Dict          # 按错误类型的重试计数              │
    │    ├─ API: int                                                   │
    │    ├─ CONTENT: int                                               │
    │    └─ SYSTEM: int                                                │
    │  error_history: List         # 错误历史记录                      │
    │    └─ ErrorRecord[]                                              │
    └─────────────────────────────────────────────────────────────────┘

与 TaskStateManager 的关系:
    - TaskStateManager 管理 TaskMetadata 的生命周期
    - 任务开始时创建 TaskMetadata
    - 任务完成或失败后移除 TaskMetadata
    - 重试时保留 TaskMetadata，从数据源重新加载业务数据

使用示例:
    metadata = TaskMetadata(record_id=123)
    
    # 记录错误
    metadata.add_error(ErrorType.API, "连接超时")
    metadata.increment_retry(ErrorType.API)
    
    # 查询状态
    print(metadata.get_retry_count(ErrorType.API))  # 1
    print(metadata.has_errors)  # True
    print(metadata.last_error.message)  # "连接超时"
"""

from dataclasses import dataclass, field
from time import time
from typing import Any

from .errors import ErrorType


@dataclass
class ErrorRecord:
    """
    错误记录

    记录单次错误的详细信息，用于调试和问题追踪。
    保存在 TaskMetadata.error_history 中。

    Attributes:
        timestamp: 错误发生时间戳 (Unix 时间)
        error_type: 错误类型 (API/CONTENT/SYSTEM)
        message: 错误消息文本
    """

    timestamp: float
    error_type: ErrorType
    message: str


@dataclass
class TaskMetadata:
    """
    任务元数据

    管理任务的内部状态和重试信息，完全独立于业务数据。
    重试时通过 reload_task_data() 从数据源重新加载原始数据，
    避免将旧数据保存在内存中导致内存泄漏。

    Attributes:
        record_id: 任务唯一标识 (可以是任意类型)
        created_at: 任务创建时间 (Unix 时间戳)
        last_retry_at: 最后一次重试时间 (None 表示未重试过)
        retry_counts: 按错误类型统计的重试次数字典
        error_history: 错误历史记录列表 (保留最近 N 条)

    Example:
        >>> metadata = TaskMetadata(record_id=123)
        >>> metadata.increment_retry(ErrorType.API)
        1
        >>> metadata.get_retry_count(ErrorType.API)
        1
        >>> metadata.total_retries
        1
    """

    record_id: Any
    created_at: float = field(default_factory=time)
    last_retry_at: float | None = None
    retry_counts: dict[ErrorType, int] = field(
        default_factory=lambda: {
            ErrorType.API: 0,
            ErrorType.CONTENT: 0,
            ErrorType.SYSTEM: 0,
        }
    )
    error_history: list[ErrorRecord] = field(default_factory=list)

    # 最大保留的错误历史条数 (防止内存无限增长)
    _MAX_ERROR_HISTORY: int = field(default=5, repr=False)

    def increment_retry(self, error_type: ErrorType) -> int:
        """
        递增指定错误类型的重试计数

        同时更新 last_retry_at 时间戳。

        Args:
            error_type: 错误类型

        Returns:
            递增后的重试次数
        """
        if error_type in self.retry_counts:
            self.retry_counts[error_type] += 1
            self.last_retry_at = time()
        return self.retry_counts.get(error_type, 0)

    def get_retry_count(self, error_type: ErrorType) -> int:
        """
        获取指定错误类型的重试次数

        Args:
            error_type: 错误类型

        Returns:
            重试次数 (未重试过返回 0)
        """
        return self.retry_counts.get(error_type, 0)

    def add_error(self, error_type: ErrorType, message: str) -> None:
        """
        添加错误记录到历史

        自动维护历史记录数量，超过限制时移除最早的记录。

        Args:
            error_type: 错误类型
            message: 错误消息
        """
        self.error_history.append(
            ErrorRecord(
                timestamp=time(),
                error_type=error_type,
                message=message,
            )
        )
        # 保留最近 N 条记录
        if len(self.error_history) > self._MAX_ERROR_HISTORY:
            self.error_history = self.error_history[-self._MAX_ERROR_HISTORY :]

    @property
    def total_retries(self) -> int:
        """获取所有错误类型的总重试次数"""
        return sum(self.retry_counts.values())

    @property
    def has_errors(self) -> bool:
        """是否有错误记录"""
        return len(self.error_history) > 0

    @property
    def last_error(self) -> ErrorRecord | None:
        """获取最近一次错误记录，无错误时返回 None"""
        return self.error_history[-1] if self.error_history else None

    def reset_retry_count(self, error_type: ErrorType) -> None:
        """
        重置指定错误类型的重试计数

        Args:
            error_type: 错误类型
        """
        if error_type in self.retry_counts:
            self.retry_counts[error_type] = 0

    def reset_all(self) -> None:
        """重置所有状态 (重试计数、错误历史、最后重试时间)"""
        for error_type in self.retry_counts:
            self.retry_counts[error_type] = 0
        self.error_history.clear()
        self.last_retry_at = None
