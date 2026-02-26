"""
重试策略模块

本模块提供错误重试决策的核心逻辑，实现分类重试和 API 熔断机制。
是错误处理的决策中枢。

类/函数清单:
    RetryAction (Enum):
        枚举值: RETRY (立即重试), FAIL (放弃), PAUSE_THEN_RETRY (暂停后重试)

    RetryDecision (dataclass):
        属性: action (RetryAction), pause_duration (float 秒),
              reload_data (bool 是否重载数据)

    RetryStrategy:
        - __init__(max_retries, api_pause_duration, api_error_trigger_window)
          初始化策略，配置各错误类型重试上限和熔断参数
          输入: Dict[ErrorType, int] 重试映射, float 暂停时长, float 触发窗口
        - decide(error_type, metadata) -> RetryDecision
          根据错误类型和任务元数据做出重试决策
          输入: ErrorType 错误类型, TaskMetadata 任务元数据
          输出: RetryDecision 决策结果
        - record_pause() -> None
          记录暂停结束时间，用于触发窗口计算

关键变量:
    - max_retries: Dict[ErrorType, int] 各错误类型最大重试次数
    - api_pause_duration: float API 熔断暂停时长 (秒)
    - api_error_trigger_window: float API 错误触发窗口 (秒)
    - last_pause_end_time: float 上次暂停结束时间戳

模块依赖:
    - src.models.errors.ErrorType: 错误类型枚举
    - src.models.task.TaskMetadata: 任务元数据 (含重试计数)

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
