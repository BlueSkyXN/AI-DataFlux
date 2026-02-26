"""
模型数据结构测试

被测模块: src/models/task.py (TaskMetadata, ErrorRecord), src/models/errors.py (ErrorType)

测试 src/models/ 中的数据模型，包括：
- TaskMetadata: 任务元数据生命周期
- ErrorType: 错误类型枚举
- 重试计数管理
- 时间戳追踪

测试类/函数清单:
    TestTaskMetadata                           TaskMetadata 数据类测试
        test_create_task_metadata              验证创建元数据默认值
        test_increment_retry                   验证递增重试次数
        test_increment_retry_multiple_times    验证多次递增同一错误类型
        test_increment_retry_different_types   验证不同错误类型独立计数
        test_add_error_to_history              验证添加错误到历史记录
        test_error_history_limit               验证错误历史最多保留 5 条
        test_reset_retry_count                 验证重置指定类型重试次数
        test_reset_all                         验证重置所有状态
        test_total_retries_property            验证总重试次数属性
        test_has_errors_property               验证是否有错误属性
        test_last_error_property               验证最近错误属性
    TestErrorRecord                            ErrorRecord 数据类测试
        test_create_error_record               验证创建错误记录
    TestErrorType                              ErrorType 枚举测试
        test_error_types                       验证枚举值
        test_error_type_from_string            验证从字符串创建枚举
        test_error_type_comparison             验证枚举比较
        test_error_type_in_dict                验证枚举作为字典键
    TestTaskMetadataEdgeCases                  边界情况测试
        test_metadata_with_string_record_id    验证字符串 record_id
        test_metadata_with_none_record_id      验证 None record_id
        test_created_at_timestamp              验证创建时间戳范围
        test_last_retry_at_updates             验证最后重试时间递增
        test_get_retry_count_unknown_type      验证未递增类型返回 0
"""

import time


class TestTaskMetadata:
    """TaskMetadata 数据类测试"""

    def test_create_task_metadata(self):
        """测试创建任务元数据"""
        from src.models.task import TaskMetadata

        metadata = TaskMetadata(record_id=123)

        assert metadata.record_id == 123
        assert metadata.created_at > 0
        assert metadata.last_retry_at is None
        assert metadata.total_retries == 0

    def test_increment_retry(self):
        """测试递增重试次数"""
        from src.models.task import TaskMetadata
        from src.models.errors import ErrorType

        metadata = TaskMetadata(record_id=1)

        result = metadata.increment_retry(ErrorType.API)
        assert result == 1
        assert metadata.get_retry_count(ErrorType.API) == 1
        assert metadata.last_retry_at is not None

    def test_increment_retry_multiple_times(self):
        """测试多次递增"""
        from src.models.task import TaskMetadata
        from src.models.errors import ErrorType

        metadata = TaskMetadata(record_id=1)

        metadata.increment_retry(ErrorType.API)
        metadata.increment_retry(ErrorType.API)
        result = metadata.increment_retry(ErrorType.API)

        assert result == 3
        assert metadata.get_retry_count(ErrorType.API) == 3

    def test_increment_retry_different_types(self):
        """测试不同错误类型的重试"""
        from src.models.task import TaskMetadata
        from src.models.errors import ErrorType

        metadata = TaskMetadata(record_id=1)

        metadata.increment_retry(ErrorType.API)
        metadata.increment_retry(ErrorType.CONTENT)
        metadata.increment_retry(ErrorType.SYSTEM)

        assert metadata.get_retry_count(ErrorType.API) == 1
        assert metadata.get_retry_count(ErrorType.CONTENT) == 1
        assert metadata.get_retry_count(ErrorType.SYSTEM) == 1
        assert metadata.total_retries == 3

    def test_add_error_to_history(self):
        """测试添加错误历史"""
        from src.models.task import TaskMetadata
        from src.models.errors import ErrorType

        metadata = TaskMetadata(record_id=1)

        metadata.add_error(ErrorType.API, "Connection timeout")

        assert metadata.has_errors is True
        assert len(metadata.error_history) == 1
        assert metadata.last_error is not None
        assert metadata.last_error.message == "Connection timeout"
        assert metadata.last_error.error_type == ErrorType.API

    def test_error_history_limit(self):
        """测试错误历史限制"""
        from src.models.task import TaskMetadata
        from src.models.errors import ErrorType

        metadata = TaskMetadata(record_id=1)

        # 添加超过限制的错误记录
        for i in range(10):
            metadata.add_error(ErrorType.API, f"Error {i}")

        # 应该只保留最近的 5 条
        assert len(metadata.error_history) <= 5
        assert metadata.last_error.message == "Error 9"

    def test_reset_retry_count(self):
        """测试重置重试次数"""
        from src.models.task import TaskMetadata
        from src.models.errors import ErrorType

        metadata = TaskMetadata(record_id=1)
        metadata.increment_retry(ErrorType.API)
        metadata.increment_retry(ErrorType.API)

        metadata.reset_retry_count(ErrorType.API)

        assert metadata.get_retry_count(ErrorType.API) == 0

    def test_reset_all(self):
        """测试重置所有状态"""
        from src.models.task import TaskMetadata
        from src.models.errors import ErrorType

        metadata = TaskMetadata(record_id=1)
        metadata.increment_retry(ErrorType.API)
        metadata.increment_retry(ErrorType.CONTENT)
        metadata.add_error(ErrorType.API, "Test error")

        metadata.reset_all()

        assert metadata.total_retries == 0
        assert metadata.has_errors is False
        assert metadata.last_retry_at is None

    def test_total_retries_property(self):
        """测试总重试次数属性"""
        from src.models.task import TaskMetadata
        from src.models.errors import ErrorType

        metadata = TaskMetadata(record_id=1)

        metadata.increment_retry(ErrorType.API)
        metadata.increment_retry(ErrorType.API)
        metadata.increment_retry(ErrorType.CONTENT)

        assert metadata.total_retries == 3

    def test_has_errors_property(self):
        """测试是否有错误属性"""
        from src.models.task import TaskMetadata
        from src.models.errors import ErrorType

        metadata = TaskMetadata(record_id=1)

        assert metadata.has_errors is False

        metadata.add_error(ErrorType.API, "Test")

        assert metadata.has_errors is True

    def test_last_error_property(self):
        """测试最近错误属性"""
        from src.models.task import TaskMetadata
        from src.models.errors import ErrorType

        metadata = TaskMetadata(record_id=1)

        assert metadata.last_error is None

        metadata.add_error(ErrorType.API, "First error")
        metadata.add_error(ErrorType.CONTENT, "Second error")

        assert metadata.last_error.message == "Second error"
        assert metadata.last_error.error_type == ErrorType.CONTENT


class TestErrorRecord:
    """ErrorRecord 数据类测试"""

    def test_create_error_record(self):
        """测试创建错误记录"""
        from src.models.task import ErrorRecord
        from src.models.errors import ErrorType

        timestamp = time.time()
        record = ErrorRecord(
            timestamp=timestamp, error_type=ErrorType.API, message="Connection failed"
        )

        assert record.timestamp == timestamp
        assert record.error_type == ErrorType.API
        assert record.message == "Connection failed"


class TestErrorType:
    """ErrorType 枚举测试"""

    def test_error_types(self):
        """测试错误类型枚举"""
        from src.models.errors import ErrorType

        assert ErrorType.API.value == "api_error"
        assert ErrorType.CONTENT.value == "content_error"
        assert ErrorType.SYSTEM.value == "system_error"

    def test_error_type_from_string(self):
        """测试从字符串创建错误类型"""
        from src.models.errors import ErrorType

        error_type = ErrorType("api_error")
        assert error_type == ErrorType.API

    def test_error_type_comparison(self):
        """测试错误类型比较"""
        from src.models.errors import ErrorType

        assert ErrorType.API == ErrorType.API
        assert ErrorType.API != ErrorType.CONTENT

    def test_error_type_in_dict(self):
        """测试错误类型作为字典键"""
        from src.models.errors import ErrorType

        error_counts = {
            ErrorType.API: 3,
            ErrorType.CONTENT: 1,
            ErrorType.SYSTEM: 2,
        }

        assert error_counts[ErrorType.API] == 3
        assert error_counts[ErrorType.CONTENT] == 1
        assert error_counts[ErrorType.SYSTEM] == 2


class TestTaskMetadataEdgeCases:
    """TaskMetadata 边界情况测试"""

    def test_metadata_with_string_record_id(self):
        """测试字符串类型的 record_id"""
        from src.models.task import TaskMetadata

        metadata = TaskMetadata(record_id="task_abc_123")

        assert metadata.record_id == "task_abc_123"

    def test_metadata_with_none_record_id(self):
        """测试 None 类型的 record_id"""
        from src.models.task import TaskMetadata

        metadata = TaskMetadata(record_id=None)

        assert metadata.record_id is None

    def test_created_at_timestamp(self):
        """测试创建时间戳"""
        from src.models.task import TaskMetadata

        before = time.time()
        metadata = TaskMetadata(record_id=1)
        after = time.time()

        assert before <= metadata.created_at <= after

    def test_last_retry_at_updates(self):
        """测试最后重试时间更新"""
        from src.models.task import TaskMetadata
        from src.models.errors import ErrorType

        metadata = TaskMetadata(record_id=1)
        initial_last_retry = metadata.last_retry_at

        time.sleep(0.01)  # 短暂延迟

        metadata.increment_retry(ErrorType.API)
        first_retry_time = metadata.last_retry_at

        assert first_retry_time != initial_last_retry
        assert first_retry_time > 0

        time.sleep(0.01)

        metadata.increment_retry(ErrorType.API)
        second_retry_time = metadata.last_retry_at

        assert second_retry_time > first_retry_time

    def test_get_retry_count_unknown_type(self):
        """测试获取未知错误类型的重试次数"""
        from src.models.task import TaskMetadata
        from src.models.errors import ErrorType

        metadata = TaskMetadata(record_id=1)

        # 不递增任何类型
        count = metadata.get_retry_count(ErrorType.API)

        assert count == 0
