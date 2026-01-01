"""
错误类型与异常定义

定义了系统中使用的错误类型枚举和异常类层次结构。
"""

from enum import Enum
from typing import Any


class ErrorType(str, Enum):
    """
    错误类型枚举
    
    用于分类处理过程中遇到的各种错误，每种错误类型有独立的重试策略。
    
    Attributes:
        API: API调用错误 (超时、HTTP错误、网络问题)
        CONTENT: 内容错误 (JSON解析失败、字段验证失败)
        SYSTEM: 系统错误 (内部异常、数据加载失败)
    """
    API = "api_error"
    CONTENT = "content_error"
    SYSTEM = "system_error"
    
    def __str__(self) -> str:
        return self.value


class AIDataFluxError(Exception):
    """
    AI-DataFlux 基础异常类
    
    所有自定义异常的基类，提供统一的错误信息格式。
    
    Attributes:
        message: 错误消息
        details: 附加的错误详情字典
    """
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | 详情: {self.details}"
        return self.message
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


class ConfigError(AIDataFluxError):
    """
    配置错误
    
    当配置文件无效、缺少必要字段或值不合法时抛出。
    """
    pass


class DataSourceError(AIDataFluxError):
    """
    数据源错误
    
    当数据源连接失败、读写操作失败时抛出。
    """
    pass


class APIError(AIDataFluxError):
    """
    API 调用错误
    
    当调用外部 API (如 AI 模型) 失败时抛出。
    
    Attributes:
        status_code: HTTP 状态码 (如适用)
        error_type: 对应的 ErrorType
    """
    error_type = ErrorType.API
    
    def __init__(
        self, 
        message: str, 
        status_code: int | None = None,
        details: dict[str, Any] | None = None
    ):
        self.status_code = status_code
        super().__init__(message, details)


class ContentError(AIDataFluxError):
    """
    内容处理错误
    
    当 AI 响应内容解析或验证失败时抛出。
    """
    error_type = ErrorType.CONTENT


class ValidationError(ContentError):
    """
    验证错误
    
    当字段值验证失败时抛出，是 ContentError 的子类。
    """
    pass
