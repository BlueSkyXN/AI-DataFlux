"""
数据模型与异常定义模块

本模块提供 AI-DataFlux 的核心数据模型和异常类定义。

模块内容:
    数据模型:
        - TaskMetadata: 任务元数据 (重试计数、错误历史)
        - ErrorRecord: 单次错误记录
    
    异常类:
        - AIDataFluxError: 基础异常类
        - ConfigError: 配置错误
        - DataSourceError: 数据源错误
        - APIError: API 调用错误
        - ContentError: 内容处理错误
        - ValidationError: 字段验证错误
    
    枚举:
        - ErrorType: 错误类型枚举 (API/CONTENT/SYSTEM)

异常层次结构:
    Exception
    └── AIDataFluxError (基础异常)
        ├── ConfigError (配置错误)
        ├── DataSourceError (数据源错误)
        ├── APIError (API 错误)
        └── ContentError (内容错误)
            └── ValidationError (验证错误)

使用示例:
    from src.models import ErrorType, TaskMetadata, APIError
    
    # 创建任务元数据
    metadata = TaskMetadata(record_id=123)
    metadata.increment_retry(ErrorType.API)
    
    # 抛出自定义异常
    raise APIError("连接超时", status_code=504)
"""

from .errors import ErrorType, AIDataFluxError, ConfigError, DataSourceError, APIError
from .task import TaskMetadata

__all__ = [
    "ErrorType",
    "AIDataFluxError",
    "ConfigError",
    "DataSourceError",
    "APIError",
    "TaskMetadata",
]
