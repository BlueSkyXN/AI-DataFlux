"""
错误类型与异常定义

本模块定义 AI-DataFlux 的错误分类系统和自定义异常类。
提供统一的错误处理框架，支持错误分类重试策略。

错误分类设计:
    ┌──────────────┬───────────────────────────────────────────────────┐
    │ 错误类型      │ 说明                                              │
    ├──────────────┼───────────────────────────────────────────────────┤
    │ API          │ API 调用错误: 超时、HTTP 错误、网络问题            │
    │ CONTENT      │ 内容错误: JSON 解析失败、字段验证失败              │
    │ SYSTEM       │ 系统错误: 内部异常、数据加载失败                   │
    └──────────────┴───────────────────────────────────────────────────┘

异常层次结构:
    Exception
    └── AIDataFluxError (基础异常)
        ├── ConfigError (配置错误)
        │   └─ 配置文件无效、缺少必要字段
        ├── DataSourceError (数据源错误)
        │   └─ 连接失败、读写操作失败
        ├── APIError (API 错误)
        │   └─ 外部 API 调用失败
        └── ContentError (内容错误)
            └─ AI 响应解析/验证失败
                └── ValidationError (验证错误)
                    └─ 字段值不在允许范围内

使用示例:
    from src.models.errors import ErrorType, APIError, ConfigError

    # 使用错误类型进行分类
    if error_type == ErrorType.API:
        # API 错误重试逻辑...
        pass

    # 抛出自定义异常
    raise APIError("连接超时", status_code=504, details={"url": "..."})

    # 捕获异常
    try:
        ...
    except AIDataFluxError as e:
        print(f"错误: {e.message}, 详情: {e.details}")
"""

from enum import Enum
from typing import Any


class ErrorType(str, Enum):
    """
    错误类型枚举

    用于分类处理过程中遇到的各种错误，每种错误类型有独立的重试策略。
    继承自 str 使得枚举值可以直接用于字符串操作。

    Attributes:
        API: API调用错误 (超时、HTTP错误、网络问题)
        CONTENT: 内容错误 (JSON解析失败、字段验证失败)
        SYSTEM: 系统错误 (内部异常、数据加载失败)

    重试策略映射:
        - API: 最多重试 3 次，触发 API 熔断暂停
        - CONTENT: 最多重试 1 次，不暂停
        - SYSTEM: 最多重试 2 次，重新加载数据
    """

    API = "api_error"
    CONTENT = "content_error"
    SYSTEM = "system_error"

    def __str__(self) -> str:
        return self.value


class AIDataFluxError(Exception):
    """
    AI-DataFlux 基础异常类

    所有自定义异常的基类，提供统一的错误信息格式和附加详情支持。

    Attributes:
        message: 错误消息文本
        details: 附加的错误详情字典 (可选)

    使用示例:
        raise AIDataFluxError("操作失败", details={"reason": "..."})
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

    常见场景:
        - 配置文件不存在
        - YAML 语法错误
        - 必需配置项缺失
        - 配置值类型不正确
    """

    pass


class DataSourceError(AIDataFluxError):
    """
    数据源错误

    当数据源连接失败、读写操作失败时抛出。

    常见场景:
        - 数据库连接失败
        - 文件读取/写入失败
        - SQL 执行错误
        - 数据格式不正确
    """

    pass


class APIError(AIDataFluxError):
    """
    API 调用错误

    当调用外部 API (如 AI 模型) 失败时抛出。

    Attributes:
        status_code: HTTP 状态码 (如适用)
        error_type: 对应的 ErrorType (固定为 API)

    常见场景:
        - HTTP 非 200 响应
        - 连接超时
        - SSL/TLS 错误
        - 请求体过大
    """

    error_type = ErrorType.API

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.status_code = status_code
        super().__init__(message, details)


class ContentError(AIDataFluxError):
    """
    内容处理错误

    当 AI 响应内容解析或验证失败时抛出。

    常见场景:
        - JSON 解析失败
        - 必需字段缺失
        - 响应格式不符合预期
    """

    error_type = ErrorType.CONTENT


class ValidationError(ContentError):
    """
    验证错误

    当字段值验证失败时抛出，是 ContentError 的子类。

    常见场景:
        - 字段值不在允许的枚举范围内
        - 数据类型不匹配
        - 值超出允许范围
    """

    pass
