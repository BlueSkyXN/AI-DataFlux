"""
重试策略单元测试

被测模块: src/core/retry/strategy.py (RetryStrategy, RetryAction)

测试 src/core/retry/strategy.py 的 RetryStrategy 类功能，包括：
- 重试决策逻辑 (RETRY/FAIL/PAUSE_THEN_RETRY)
- 错误类型分类处理
- API 熔断机制
- 最大总尝试次数限制

测试类/函数清单:
    TestRetryStrategy                          重试策略测试
        test_decide_retry_success              验证首次 API 错误触发 PAUSE_THEN_RETRY
        test_decide_retry_after_pause          验证暂停窗口内直接 RETRY 不再暂停
        test_decide_fail_max_attempts           验证总尝试次数耗尽返回 FAIL
        test_decide_content_error              验证内容错误直接 RETRY 且不重载数据
        test_decide_content_error_fail         验证内容错误超限返回 FAIL
        test_decide_system_error               验证系统错误 RETRY 并重载数据
        test_default_max_attempts               验证未配置类型默认仅首次执行
"""

import pytest
from src.core.retry.strategy import RetryStrategy, RetryAction
from src.models.errors import ErrorType
from src.models.task import TaskMetadata


class TestRetryStrategy:
    """重试策略测试"""

    @pytest.fixture
    def strategy(self):
        return RetryStrategy(
            max_attempts={
                ErrorType.API: 3,
                ErrorType.CONTENT: 2,
                ErrorType.SYSTEM: 2,
            },
            api_pause_duration=1.0,
            api_error_trigger_window=1.0,
        )

    @pytest.fixture
    def metadata(self):
        return TaskMetadata("task_1")

    def test_decide_retry_success(self, strategy, metadata):
        # 第一次 API 错误，应该暂停并重试
        decision = strategy.decide(ErrorType.API, metadata)
        assert decision.action == RetryAction.PAUSE_THEN_RETRY
        assert decision.pause_duration == 1.0
        assert decision.reload_data is True

    def test_decide_retry_after_pause(self, strategy, metadata):
        # 第一次错误，触发暂停
        strategy.decide(ErrorType.API, metadata)
        strategy.record_pause()  # 更新暂停时间

        # 马上又来一个错误（在窗口内），应该直接重试不暂停
        metadata.increment_retry(ErrorType.API)
        decision = strategy.decide(ErrorType.API, metadata)
        assert decision.action == RetryAction.RETRY
        assert decision.reload_data is True

    def test_decide_fail_max_attempts(self, strategy, metadata):
        # 首次执行加两次 retry 已用完 3 attempts
        metadata.increment_retry(ErrorType.API)
        metadata.increment_retry(ErrorType.API)

        decision = strategy.decide(ErrorType.API, metadata)
        assert decision.action == RetryAction.FAIL

    def test_decide_content_error(self, strategy, metadata):
        # 内容错误，直接重试但不重新加载数据
        decision = strategy.decide(ErrorType.CONTENT, metadata)
        assert decision.action == RetryAction.RETRY
        assert decision.reload_data is False

    def test_decide_content_error_fail(self, strategy, metadata):
        # 内容错误首次执行后已安排一次 retry，共 2 attempts
        metadata.increment_retry(ErrorType.CONTENT)
        decision = strategy.decide(ErrorType.CONTENT, metadata)
        assert decision.action == RetryAction.FAIL

    def test_decide_system_error(self, strategy, metadata):
        # 系统错误，重试并重新加载数据
        decision = strategy.decide(ErrorType.SYSTEM, metadata)
        assert decision.action == RetryAction.RETRY
        assert decision.reload_data is True

    def test_default_max_attempts(self, strategy, metadata):
        # 未配置类型默认只有首次 attempt，不再重试
        decision = strategy.decide(ErrorType.SOURCE, metadata)
        assert decision.action == RetryAction.FAIL

    def test_source_error_reloads_without_api_pause(self, metadata):
        strategy = RetryStrategy(
            max_attempts={ErrorType.SOURCE: 2},
            base_backoff_seconds=1,
        )

        decision = strategy.decide(ErrorType.SOURCE, metadata)

        assert decision.action == RetryAction.RETRY
        assert decision.reload_data is True
        assert decision.pause_duration == 0
        assert decision.retry_delay == 1
