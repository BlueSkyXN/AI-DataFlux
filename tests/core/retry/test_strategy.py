import pytest
from src.core.retry.strategy import RetryStrategy, RetryAction
from src.models.errors import ErrorType
from src.models.task import TaskMetadata

class TestRetryStrategy:

    @pytest.fixture
    def strategy(self):
        return RetryStrategy(
            max_retries={
                ErrorType.API: 3,
                ErrorType.CONTENT: 1
            },
            api_pause_duration=1.0,
            api_error_trigger_window=1.0
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
        strategy.record_pause() # 更新暂停时间

        # 马上又来一个错误（在窗口内），应该直接重试不暂停
        metadata.increment_retry(ErrorType.API)
        decision = strategy.decide(ErrorType.API, metadata)
        assert decision.action == RetryAction.RETRY
        assert decision.reload_data is True

    def test_decide_fail_max_retries(self, strategy, metadata):
        # API 错误超过最大次数
        metadata.increment_retry(ErrorType.API)
        metadata.increment_retry(ErrorType.API)
        metadata.increment_retry(ErrorType.API) # 3次

        decision = strategy.decide(ErrorType.API, metadata)
        assert decision.action == RetryAction.FAIL

    def test_decide_content_error(self, strategy, metadata):
        # 内容错误，直接重试但不重新加载数据
        decision = strategy.decide(ErrorType.CONTENT, metadata)
        assert decision.action == RetryAction.RETRY
        assert decision.reload_data is False

    def test_decide_content_error_fail(self, strategy, metadata):
        # 内容错误超过 1 次
        metadata.increment_retry(ErrorType.CONTENT)
        decision = strategy.decide(ErrorType.CONTENT, metadata)
        assert decision.action == RetryAction.FAIL

    def test_decide_system_error(self, strategy, metadata):
        # 系统错误，重试并重新加载数据
        decision = strategy.decide(ErrorType.SYSTEM, metadata)
        assert decision.action == RetryAction.RETRY
        assert decision.reload_data is True

    def test_default_max_retries(self, strategy, metadata):
        # 未配置的错误类型，默认最大重试 1 次
        decision = strategy.decide(ErrorType.SYSTEM, metadata)
        assert decision.action == RetryAction.RETRY

        metadata.increment_retry(ErrorType.SYSTEM)
        decision = strategy.decide(ErrorType.SYSTEM, metadata)
        assert decision.action == RetryAction.FAIL
