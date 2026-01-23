"""
重试策略实现

本模块实现 AI-DataFlux 的核心重试决策逻辑，根据错误类型和
任务状态智能决定下一步行动。

设计理念:
    不同类型的错误有不同的恢复策略:
    - API 错误: 可能是临时的网络/服务问题，暂停后重试
    - 内容错误: AI 返回格式问题，直接重试可能有效
    - 系统错误: 可能是数据问题，重新加载数据后重试

API 熔断机制:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     API 熔断时序图                               │
    │                                                                  │
    │  API Error → 检查时间窗口 → 超过触发窗口 → PAUSE_THEN_RETRY     │
    │                    ↓                                            │
    │               窗口内 → RETRY (避免重复暂停)                      │
    │                                                                  │
    │  触发窗口 = api_error_trigger_window (默认 2 秒)                 │
    │  暂停时长 = api_pause_duration (默认 2 秒)                       │
    └─────────────────────────────────────────────────────────────────┘

决策流程:
    1. 检查是否超过最大重试次数 → FAIL
    2. API 错误且超过触发窗口 → PAUSE_THEN_RETRY
    3. API 错误但在窗口内 → RETRY (重载数据)
    4. 内容错误 → RETRY (不重载)
    5. 系统错误 → RETRY (重载数据)

配置说明:
    datasource:
      concurrency:
        retry_limits:
          api_error: 3        # API 错误最多重试 3 次
          content_error: 1    # 内容错误最多重试 1 次
          system_error: 2     # 系统错误最多重试 2 次
        api_pause_duration: 2.0     # 熔断暂停 2 秒
        api_error_trigger_window: 2.0  # 2 秒内的 API 错误不重复熔断
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict

from ...models.errors import ErrorType
from ...models.task import TaskMetadata


class RetryAction(Enum):
    """
    重试决策动作枚举
    
    定义重试策略可能返回的三种动作类型。
    """
    RETRY = "retry"                     # 立即重试
    FAIL = "fail"                       # 标记失败，不再重试
    PAUSE_THEN_RETRY = "pause_then_retry"  # 暂停一段时间后重试（用于API熔断）


@dataclass
class RetryDecision:
    """
    重试决策结果
    
    包含决策动作和相关参数，供调用方执行。
    
    Attributes:
        action: 决策动作 (RETRY, FAIL, PAUSE_THEN_RETRY)
        pause_duration: 暂停时长 (秒)，仅 PAUSE_THEN_RETRY 有效
        reload_data: 是否需要重新从数据源加载数据
    """
    action: RetryAction
    pause_duration: float = 0.0
    reload_data: bool = False


class RetryStrategy:
    """
    重试策略管理器
    
    根据错误类型和任务元数据决定下一步行动。
    实现分类重试和 API 熔断两个核心机制。
    
    分类重试:
        - 每种错误类型有独立的最大重试次数
        - 超过次数后标记为失败，不再重试
    
    API 熔断:
        - API 错误触发全局暂停
        - 时间窗口内的 API 错误不重复触发暂停
        - 暂停期间其他任务也会被阻塞
    
    Attributes:
        max_retries: 各错误类型的最大重试次数
        api_pause_duration: API 熔断暂停时长 (秒)
        api_error_trigger_window: API 错误触发窗口 (秒)
        last_pause_end_time: 上次暂停结束时间戳
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
                例: {ErrorType.API: 3, ErrorType.CONTENT: 1}
            api_pause_duration: API 错误触发后的暂停时长（秒）
            api_error_trigger_window: API 错误触发的时间窗口（秒）
                在此窗口内的多次 API 错误只会触发一次暂停
        """
        self.max_retries = max_retries
        self.api_pause_duration = api_pause_duration
        self.api_error_trigger_window = api_error_trigger_window
        self.last_pause_end_time = 0.0  # 上次暂停结束的时间戳

    def decide(
        self,
        error_type: ErrorType,
        metadata: TaskMetadata
    ) -> RetryDecision:
        """
        根据错误类型和元数据做出决策

        Args:
            error_type: 发生的错误类型
            metadata: 任务元数据，包含重试计数等信息

        Returns:
            RetryDecision: 决策结果，包含动作和参数
            
        决策逻辑:
            1. 检查重试次数是否超限 → FAIL
            2. API 错误 + 超过触发窗口 → PAUSE_THEN_RETRY
            3. API 错误 + 窗口内 → RETRY (重载数据)
            4. 内容错误 → RETRY (不重载)
            5. 系统错误 → RETRY (重载数据)
        """
        current_retries = metadata.get_retry_count(error_type)
        max_allowed = self.max_retries.get(error_type, 1)

        # 1. 检查是否超过最大重试次数
        if current_retries >= max_allowed:
            return RetryDecision(action=RetryAction.FAIL)

        # 2. API 错误特殊处理（熔断机制）
        if error_type == ErrorType.API:
            current_time = time.time()
            # 检查是否需要触发暂停:
            # 如果距离上次暂停结束超过了触发窗口，说明是新一轮错误
            if (current_time - self.last_pause_end_time) > self.api_error_trigger_window:
                return RetryDecision(
                    action=RetryAction.PAUSE_THEN_RETRY,
                    pause_duration=self.api_pause_duration,
                    reload_data=True  # API 错误需要重载数据
                )

            # 如果刚暂停过（在窗口内），就普通重试避免连续暂停
            return RetryDecision(
                action=RetryAction.RETRY,
                reload_data=True
            )

        # 3. 其他错误类型
        # 内容错误（JSON解析失败）: 不需要重新加载数据，因为数据本身没变
        # 系统错误: 可能涉及数据处理异常，需要重新加载数据
        return RetryDecision(
            action=RetryAction.RETRY,
            reload_data=(error_type == ErrorType.SYSTEM)
        )

    def record_pause(self):
        """
        记录暂停发生，更新时间戳
        
        在执行完暂停后调用，用于判断后续 API 错误
        是否在触发窗口内。
        """
        self.last_pause_end_time = time.time()
