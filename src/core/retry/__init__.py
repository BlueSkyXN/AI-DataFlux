"""
重试策略模块

本模块提供错误重试决策的核心逻辑，实现分类重试和 API 熔断机制。
是错误处理的决策中枢。

模块导出:
    - RetryStrategy: 重试策略管理器
    - RetryAction: 重试动作枚举
    - RetryDecision: 重试决策结果数据类

错误分类重试策略:
    ┌──────────────┬──────────────────┬────────────┬─────────────────┐
    │ 错误类型      │ 默认最大重试次数  │ 是否重载   │ 是否触发熔断    │
    ├──────────────┼──────────────────┼────────────┼─────────────────┤
    │ API_ERROR    │ 3                │ ✓          │ ✓               │
    │ CONTENT_ERROR│ 1                │ ✗          │ ✗               │
    │ SYSTEM_ERROR │ 2                │ ✓          │ ✗               │
    └──────────────┴──────────────────┴────────────┴─────────────────┘

使用示例:
    from src.core.retry import RetryStrategy, RetryAction
    
    strategy = RetryStrategy(
        max_retries={ErrorType.API: 3, ErrorType.CONTENT: 1},
        api_pause_duration=2.0
    )
    
    decision = strategy.decide(ErrorType.API, task_metadata)
    if decision.action == RetryAction.PAUSE_THEN_RETRY:
        await asyncio.sleep(decision.pause_duration)
        # 重试...
"""

from .strategy import RetryStrategy, RetryAction, RetryDecision

__all__ = ["RetryStrategy", "RetryAction", "RetryDecision"]
