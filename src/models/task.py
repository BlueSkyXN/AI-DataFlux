"""
任务元数据定义

TaskMetadata 类用于管理任务的内部状态，完全独立于业务数据。
采用 dataclass 实现，确保类型安全和代码简洁。
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
    
    Attributes:
        timestamp: 错误发生时间戳
        error_type: 错误类型
        message: 错误消息
    """
    timestamp: float
    error_type: ErrorType
    message: str


@dataclass
class TaskMetadata:
    """
    任务元数据
    
    管理任务的内部状态和重试信息，完全独立于业务数据。
    重试时通过 reload_task_data() 从数据源重新加载原始数据，避免内存泄漏。
    
    Attributes:
        record_id: 任务唯一标识
        created_at: 任务创建时间
        last_retry_at: 最后一次重试时间
        retry_counts: 按错误类型统计的重试次数
        error_history: 错误历史记录 (保留最近 N 条)
    
    Example:
        >>> metadata = TaskMetadata(record_id=123)
        >>> metadata.increment_retry(ErrorType.API)
        1
        >>> metadata.get_retry_count(ErrorType.API)
        1
    """
    
    record_id: Any
    created_at: float = field(default_factory=time)
    last_retry_at: float | None = None
    retry_counts: dict[ErrorType, int] = field(default_factory=lambda: {
        ErrorType.API: 0,
        ErrorType.CONTENT: 0,
        ErrorType.SYSTEM: 0,
    })
    error_history: list[ErrorRecord] = field(default_factory=list)
    
    # 最大保留的错误历史条数
    _MAX_ERROR_HISTORY: int = field(default=5, repr=False)
    
    def increment_retry(self, error_type: ErrorType) -> int:
        """
        递增指定错误类型的重试计数
        
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
            重试次数
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
        self.error_history.append(ErrorRecord(
            timestamp=time(),
            error_type=error_type,
            message=message,
        ))
        # 保留最近 N 条记录
        if len(self.error_history) > self._MAX_ERROR_HISTORY:
            self.error_history = self.error_history[-self._MAX_ERROR_HISTORY:]
    
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
        """获取最近一次错误记录"""
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
        """重置所有状态"""
        for error_type in self.retry_counts:
            self.retry_counts[error_type] = 0
        self.error_history.clear()
        self.last_retry_at = None
