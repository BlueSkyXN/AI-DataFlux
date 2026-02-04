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
    return sorted(__all__)
