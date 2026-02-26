"""
核心处理逻辑模块

本模块是 AI-DataFlux 的核心处理层，提供数据处理工作流的核心组件。
采用组件化架构，将复杂的处理逻辑拆分为独立、可测试的模块。

核心组件:
    - UniversalAIProcessor: 主处理器，协调所有组件完成数据处理
    - ShardedTaskManager: 分片调度器，管理大数据集的分片处理
    - JsonValidator: JSON 验证器，验证 AI 返回结果的格式和内容

架构设计:
    ┌─────────────────────────────────────────────────────────────┐
    │                 UniversalAIProcessor (协调者)                │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
    │  │TaskPool     │  │ShardedTask  │  │FluxAIClient         │ │
    │  │(数据源)     │  │Manager      │  │(API通信)            │ │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘ │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
    │  │Content      │  │TaskState    │  │RetryStrategy        │ │
    │  │Processor    │  │Manager      │  │(重试决策)           │ │
    │  │(内容处理)   │  │(状态管理)   │  │                     │ │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘ │
    └─────────────────────────────────────────────────────────────┘

处理流程:
    1. 加载配置 → 初始化组件
    2. 分片加载 → 获取任务批次
    3. 生成 Prompt → 调用 API → 解析响应
    4. 验证结果 → 写回数据源
    5. 错误处理 → 重试或跳过

模块级函数:
    __getattr__(name)
        - 功能: 延迟导入公开类，避免循环依赖和加载时间开销
        - 输入: name (str) — 请求的属性名
        - 输出: 对应的类对象 (JsonValidator / ShardedTaskManager / UniversalAIProcessor)
        - 异常: AttributeError — 请求了不存在的属性

    __dir__()
        - 功能: 返回模块的公开属性列表，支持 IDE 自动补全
        - 输出: list[str] — 排序后的 __all__ 列表

关键变量:
    __all__: 公开导出的类名列表

模块依赖:
    - .validator → JsonValidator
    - .scheduler → ShardedTaskManager
    - .processor → UniversalAIProcessor

使用示例:
    from src.core import UniversalAIProcessor

    processor = UniversalAIProcessor("config.yaml")
    await processor.process_shard_async_continuous()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .validator import JsonValidator
    from .scheduler import ShardedTaskManager
    from .processor import UniversalAIProcessor

__all__ = ["JsonValidator", "ShardedTaskManager", "UniversalAIProcessor"]


def __getattr__(name: str):
    """延迟导入公开类，按需加载避免循环依赖"""
    if name == "JsonValidator":
        from .validator import JsonValidator

        return JsonValidator
    if name == "ShardedTaskManager":
        from .scheduler import ShardedTaskManager

        return ShardedTaskManager
    if name == "UniversalAIProcessor":
        from .processor import UniversalAIProcessor

        return UniversalAIProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """返回模块公开属性列表，支持 dir() 和 IDE 自动补全"""
    return sorted(__all__)
