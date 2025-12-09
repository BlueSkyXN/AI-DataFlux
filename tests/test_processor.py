"""
处理器核心测试

测试 src/core/processor.py 的核心功能
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestPromptProcessing:
    """提示词处理测试"""

    def test_create_prompt_basic(self, sample_config_file):
        """测试基本提示词生成"""
        from src.core.processor import UniversalAIProcessor

        with patch.object(UniversalAIProcessor, '__init__', lambda x, y: None):
            processor = UniversalAIProcessor(None)
            processor.prompt_template = "处理数据: {record_json}"

            record_data = {"name": "测试", "value": 123}
            result = processor.create_prompt(record_data)

            assert "处理数据:" in result
            assert "测试" in result
            assert "123" in result

    def test_create_prompt_filters_internal_fields(self):
        """测试提示词过滤内部字段"""
        from src.core.processor import UniversalAIProcessor

        with patch.object(UniversalAIProcessor, '__init__', lambda x, y: None):
            processor = UniversalAIProcessor(None)
            processor.prompt_template = "{record_json}"

            record_data = {"name": "测试", "_internal": "忽略", "_error": "忽略"}
            result = processor.create_prompt(record_data)

            assert "_internal" not in result
            assert "_error" not in result
            assert "测试" in result

    def test_create_prompt_filters_none_values(self):
        """测试提示词过滤 None 值"""
        from src.core.processor import UniversalAIProcessor

        with patch.object(UniversalAIProcessor, '__init__', lambda x, y: None):
            processor = UniversalAIProcessor(None)
            processor.prompt_template = "{record_json}"

            record_data = {"name": "测试", "empty": None}
            result = processor.create_prompt(record_data)

            assert "empty" not in result
            assert "测试" in result

    def test_create_prompt_empty_template(self):
        """测试空模板返回空字符串"""
        from src.core.processor import UniversalAIProcessor

        with patch.object(UniversalAIProcessor, '__init__', lambda x, y: None):
            processor = UniversalAIProcessor(None)
            processor.prompt_template = ""

            result = processor.create_prompt({"data": "test"})
            assert result == ""


class TestJsonExtraction:
    """JSON 提取测试"""

    @pytest.fixture
    def processor(self):
        """创建带 JSON 提取配置的处理器"""
        from src.core.processor import UniversalAIProcessor
        from src.core.validator import JsonValidator

        with patch.object(UniversalAIProcessor, '__init__', lambda x, y: None):
            p = UniversalAIProcessor(None)
            p.required_fields = ["answer"]
            p.validator = JsonValidator()
            p.validator.configure({"enabled": False})
            return p

    def test_extract_simple_json(self, processor):
        """测试提取简单 JSON"""
        content = '{"answer": "这是答案"}'
        result = processor.extract_json_from_response(content)

        assert result.get("answer") == "这是答案"
        assert "_error" not in result

    def test_extract_json_from_markdown_block(self, processor):
        """测试从 Markdown 代码块提取 JSON"""
        content = '''这是一些说明文字
```json
{"answer": "代码块中的答案"}
```
后续文字'''
        result = processor.extract_json_from_response(content)

        assert result.get("answer") == "代码块中的答案"

    def test_extract_json_with_trailing_comma(self, processor):
        """测试处理尾随逗号"""
        content = '{"answer": "测试",}'
        result = processor.extract_json_from_response(content)

        assert result.get("answer") == "测试"

    def test_extract_empty_response(self, processor):
        """测试空响应返回错误"""
        result = processor.extract_json_from_response("")

        assert "_error" in result
        assert result["_error"] == "empty_ai_response"

    def test_extract_none_response(self, processor):
        """测试 None 响应返回错误"""
        result = processor.extract_json_from_response(None)

        assert "_error" in result
        assert result["_error"] == "empty_ai_response"

    def test_extract_invalid_json(self, processor):
        """测试无效 JSON 返回错误"""
        content = "这不是一个有效的 JSON 响应"
        result = processor.extract_json_from_response(content)

        assert "_error" in result

    def test_extract_missing_required_fields(self, processor):
        """测试缺少必需字段"""
        processor.required_fields = ["answer", "category"]
        content = '{"answer": "只有答案"}'
        result = processor.extract_json_from_response(content)

        # 缺少 category 字段
        assert "_error" in result or "category" not in result


class TestJsonSchema:
    """JSON Schema 构建测试"""

    def test_build_schema_basic(self):
        """测试基本 Schema 构建"""
        from src.core.processor import UniversalAIProcessor
        from src.core.validator import JsonValidator

        with patch.object(UniversalAIProcessor, '__init__', lambda x, y: None):
            processor = UniversalAIProcessor(None)
            processor.use_json_schema = True
            processor.required_fields = ["answer", "category"]
            processor.validator = JsonValidator()
            processor.validator.enabled = False
            processor.validator.field_rules = {}

            schema = processor.build_json_schema()

            assert schema is not None
            assert schema["type"] == "object"
            assert "answer" in schema["properties"]
            assert "category" in schema["properties"]
            assert "answer" in schema["required"]
            assert "category" in schema["required"]

    def test_build_schema_with_enum_rules(self):
        """测试带枚举规则的 Schema"""
        from src.core.processor import UniversalAIProcessor
        from src.core.validator import JsonValidator

        with patch.object(UniversalAIProcessor, '__init__', lambda x, y: None):
            processor = UniversalAIProcessor(None)
            processor.use_json_schema = True
            processor.required_fields = ["category"]
            processor.validator = JsonValidator()
            processor.validator.enabled = True
            processor.validator.field_rules = {"category": ["A", "B", "C"]}

            schema = processor.build_json_schema()

            assert schema is not None
            assert schema["properties"]["category"]["enum"] == ["A", "B", "C"]

    def test_build_schema_disabled(self):
        """测试禁用时返回 None"""
        from src.core.processor import UniversalAIProcessor

        with patch.object(UniversalAIProcessor, '__init__', lambda x, y: None):
            processor = UniversalAIProcessor(None)
            processor.use_json_schema = False

            schema = processor.build_json_schema()
            assert schema is None

    def test_build_schema_empty_required_fields(self):
        """测试空必需字段返回 None"""
        from src.core.processor import UniversalAIProcessor

        with patch.object(UniversalAIProcessor, '__init__', lambda x, y: None):
            processor = UniversalAIProcessor(None)
            processor.use_json_schema = True
            processor.required_fields = []

            schema = processor.build_json_schema()
            assert schema is None


class TestTaskStateManagement:
    """任务状态管理测试"""

    @pytest.fixture
    def processor(self):
        """创建带任务状态管理的处理器"""
        from src.core.processor import UniversalAIProcessor
        import threading

        with patch.object(UniversalAIProcessor, '__init__', lambda x, y: None):
            p = UniversalAIProcessor(None)
            p.tasks_in_progress = set()
            p.tasks_progress_lock = threading.Lock()
            p.task_metadata = {}
            p.metadata_lock = threading.Lock()
            return p

    def test_mark_task_in_progress(self, processor):
        """测试标记任务为处理中"""
        result = processor.mark_task_in_progress(1)
        assert result is True
        assert processor.is_task_in_progress(1)

    def test_mark_task_in_progress_duplicate(self, processor):
        """测试重复标记返回 False"""
        processor.mark_task_in_progress(1)
        result = processor.mark_task_in_progress(1)
        assert result is False

    def test_mark_task_completed(self, processor):
        """测试标记任务完成"""
        processor.mark_task_in_progress(1)
        processor.mark_task_completed(1)
        assert not processor.is_task_in_progress(1)

    def test_get_task_metadata_creates_new(self, processor):
        """测试获取元数据创建新实例"""
        from src.models.task import TaskMetadata

        metadata = processor.get_task_metadata(1)
        assert isinstance(metadata, TaskMetadata)
        assert metadata.record_id == 1

    def test_get_task_metadata_returns_existing(self, processor):
        """测试获取元数据返回已有实例"""
        meta1 = processor.get_task_metadata(1)
        meta2 = processor.get_task_metadata(1)
        assert meta1 is meta2

    def test_remove_task_metadata(self, processor):
        """测试移除任务元数据"""
        processor.get_task_metadata(1)
        processor.remove_task_metadata(1)

        # 重新获取应该是新实例
        meta = processor.get_task_metadata(1)
        assert meta.record_id == 1


class TestErrorHandling:
    """错误处理测试"""

    @pytest.fixture
    def processor(self):
        """创建带错误处理配置的处理器"""
        from src.core.processor import UniversalAIProcessor
        from src.models.errors import ErrorType
        import threading

        with patch.object(UniversalAIProcessor, '__init__', lambda x, y: None):
            p = UniversalAIProcessor(None)
            p.max_retry_counts = {
                ErrorType.API: 3,
                ErrorType.CONTENT: 1,
                ErrorType.SYSTEM: 2,
            }
            p.task_metadata = {}
            p.metadata_lock = threading.Lock()

            # Mock task_manager
            p.task_manager = MagicMock()
            p.task_manager.retried_tasks_count = {
                ErrorType.API: 0,
                ErrorType.CONTENT: 0,
                ErrorType.SYSTEM: 0,
            }
            p.task_manager.max_retries_exceeded_count = 0

            # Mock task_pool
            p.task_pool = MagicMock()
            p.task_pool.reload_task_data.return_value = {"question": "重新加载的数据"}

            return p

    def test_handle_error_result_retry(self, processor):
        """测试错误重试"""
        from src.models.errors import ErrorType
        from src.models.task import TaskMetadata

        metadata = TaskMetadata(1)
        result = {"_error": "api_error", "_error_type": ErrorType.API}

        retry_task = processor._handle_error_result(1, result, metadata, ErrorType.API)

        assert retry_task is not None
        assert retry_task[0] == 1
        assert metadata.get_retry_count(ErrorType.API) == 1

    def test_handle_error_result_max_retries(self, processor):
        """测试达到最大重试次数"""
        from src.models.errors import ErrorType
        from src.models.task import TaskMetadata

        metadata = TaskMetadata(1)
        # 已经重试了 3 次
        for _ in range(3):
            metadata.increment_retry(ErrorType.API)

        result = {"_error": "api_error", "_error_type": ErrorType.API}

        retry_task = processor._handle_error_result(1, result, metadata, ErrorType.API)

        assert retry_task is None
        assert processor.task_manager.max_retries_exceeded_count == 1
