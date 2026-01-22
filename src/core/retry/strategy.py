import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict

from ...models.errors import ErrorType
from ...models.task import TaskMetadata

class RetryAction(Enum):
    """重试决策动作"""
    RETRY = "retry"                     # 立即重试
    FAIL = "fail"                       # 标记失败，不再重试
    PAUSE_THEN_RETRY = "pause_then_retry"  # 暂停一段时间后重试（用于API熔断）

@dataclass
class RetryDecision:
    """重试决策结果"""
    action: RetryAction
    pause_duration: float = 0.0
    reload_data: bool = False  # 是否需要重新从数据源加载数据

class RetryStrategy:
    """
    重试策略管理器
    负责根据错误类型和当前状态决定下一步行动
    """

    def __init__(
        self,
        max_retries: Dict[ErrorType, int],
        api_pause_duration: float = 2.0,
        api_error_trigger_window: float = 2.0,
    ):
        """
        初始化重试策略

        Args:
            max_retries: 各错误类型的最大重试次数映射
            api_pause_duration: API 错误触发后的暂停时长（秒）
            api_error_trigger_window: API 错误触发的时间窗口（秒）
        """
        self.max_retries = max_retries
        self.api_pause_duration = api_pause_duration
        self.api_error_trigger_window = api_error_trigger_window
        self.last_pause_end_time = 0.0

    def decide(
        self,
        error_type: ErrorType,
        metadata: TaskMetadata
    ) -> RetryDecision:
        """
        根据错误类型和元数据做出决策

        Args:
            error_type: 错误类型
            metadata: 任务元数据

        Returns:
            RetryDecision: 决策结果
        """
        current_retries = metadata.get_retry_count(error_type)
        max_allowed = self.max_retries.get(error_type, 1)

        # 1. 检查是否超过最大重试次数
        if current_retries >= max_allowed:
            return RetryDecision(action=RetryAction.FAIL)

        # 2. API 错误特殊处理（熔断机制）
        if error_type == ErrorType.API:
            current_time = time.time()
            # 如果距离上次暂停结束时间超过了触发窗口，说明是新的一轮错误，触发暂停
            # 这里逻辑稍微有点绕：原逻辑是 api_error_in_batch 为 True 时检查时间
            # 我们简化为：每次 API 错误都检查是否需要暂停
            # 为了防止连续的 API 错误导致无限暂停，我们需要外部（Processor）配合更新 last_pause_end_time
            # 或者在这里简单的判断：如果最近没有暂停过，就建议暂停

            if (current_time - self.last_pause_end_time) > self.api_error_trigger_window:
                return RetryDecision(
                    action=RetryAction.PAUSE_THEN_RETRY,
                    pause_duration=self.api_pause_duration,
                    reload_data=True
                )

            # 如果刚暂停过，就普通重试
            return RetryDecision(
                action=RetryAction.RETRY,
                reload_data=True
            )

        # 3. 其他错误（内容错误、系统错误）
        # 内容错误（JSON解析失败）不需要重新加载数据，因为数据本身没变
        # 系统错误可能涉及数据处理异常，需要重新加载数据
        return RetryDecision(
            action=RetryAction.RETRY,
            reload_data=(error_type == ErrorType.SYSTEM)
        )

    def record_pause(self):
        """记录暂停发生，更新时间戳"""
        self.last_pause_end_time = time.time()
