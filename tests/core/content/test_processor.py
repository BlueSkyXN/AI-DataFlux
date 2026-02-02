"""
内容处理器单元测试

测试 src/core/content/processor.py 的 ContentProcessor 类功能，包括：
- Prompt 生成 (模板替换、JSON 序列化)
- AI 响应解析 (JSON 提取、Markdown 代码块处理)
- 字段验证 (必需字段检查、枚举值验证)
- JSON Schema 构建
"""

import pytest
from unittest.mock import MagicMock
from src.core.content.processor import ContentProcessor
from src.core.validator import JsonValidator
from src.models.errors import ErrorType

class TestContentProcessor:

    @pytest.fixture
    def mock_validator(self):
        validator = MagicMock(spec=JsonValidator)
        validator.enabled = True
        validator.field_rules = {"status": ["active", "inactive"]}
        validator.validate.return_value = (True, [])
        return validator

    @pytest.fixture
    def processor(self, mock_validator):
        return ContentProcessor(
            prompt_template="Analyze this: {record_json}",
            required_fields=["name", "status"],
            validator=mock_validator,
            use_json_schema=True
        )

    def test_create_prompt(self, processor):
        record = {"name": "Test", "status": "active", "_internal": 123}
        prompt = processor.create_prompt(record)
        assert "Analyze this:" in prompt
        assert '"name":"Test"' in prompt
        assert "_internal" not in prompt

    def test_create_prompt_serialization_error(self, processor):
        # 创建一个无法序列化的对象
        class Unserializable:
            pass

        record = {"obj": Unserializable()}
        prompt = processor.create_prompt(record)
        assert '{"error": "无法序列化数据"}' in prompt

    def test_parse_response_valid_json(self, processor):
        response = '{"name": "Alice", "status": "active"}'
        result = processor.parse_response(response)
        assert result["name"] == "Alice"
        assert result["status"] == "active"

    def test_parse_response_markdown_json(self, processor):
        response = """
        Here is the result:
        ```json
        {
            "name": "Bob",
            "status": "inactive"
        }
        ```
        """
        result = processor.parse_response(response)
        assert result["name"] == "Bob"

    def test_parse_response_with_trailing_comma(self, processor):
        response = '{"name": "Charlie", "status": "active",}'
        result = processor.parse_response(response)
        assert result["name"] == "Charlie"

    def test_parse_response_regex_extraction(self, processor):
        response = "Some text before { \"name\": \"Dave\", \"status\": \"active\" } some text after"
        result = processor.parse_response(response)
        assert result["name"] == "Dave"

    def test_parse_response_missing_fields(self, processor):
        response = '{"name": "Eve"}'  # 缺少 status
        result = processor.parse_response(response)
        assert result.get("_error_type") == ErrorType.CONTENT
        assert result.get("_error") == "invalid_or_missing_json"

    def test_parse_response_validation_failure(self, processor, mock_validator):
        # 模拟验证失败
        mock_validator.validate.return_value = (False, ["Invalid status"])

        response = '{"name": "Frank", "status": "invalid"}'
        result = processor.parse_response(response)

        assert result.get("_error_type") == ErrorType.CONTENT
        assert result.get("_error") == "invalid_field_values"
        assert "Invalid status" in result.get("_validation_errors", [])

    def test_build_schema(self, processor):
        schema = processor.build_schema()
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "status" in schema["properties"]
        assert schema["properties"]["status"]["enum"] == ["active", "inactive"]
        assert "name" in schema["required"]

    def test_exclude_fields_in_prompt(self, mock_validator):
        """测试 exclude_fields 功能：路由字段应当被提取但不出现在 Prompt JSON 中"""
        processor = ContentProcessor(
            prompt_template="Analyze: {record_json}",
            required_fields=["name"],
            validator=mock_validator,
            use_json_schema=False,
            exclude_fields=["BGBU", "category"],  # 模拟路由字段
        )

        record = {
            "name": "Alice",
            "BGBU": "type_a",  # 路由字段，应当被排除
            "category": "test",  # 另一个排除字段
            "description": "Some text"
        }

        prompt = processor.create_prompt(record)

        # 验证：BGBU 和 category 不应出现在生成的 Prompt 中
        assert "BGBU" not in prompt
        assert "type_a" not in prompt
        assert "category" not in prompt

        # 验证：其他字段正常包含
        assert '"name":"Alice"' in prompt
        assert "description" in prompt

    def test_exclude_fields_empty_list(self, mock_validator):
        """测试 exclude_fields 为空列表时的行为"""
        processor = ContentProcessor(
            prompt_template="Data: {record_json}",
            required_fields=[],
            validator=mock_validator,
            exclude_fields=[],
        )

        record = {"field1": "value1", "field2": "value2"}
        prompt = processor.create_prompt(record)

        assert "field1" in prompt
        assert "field2" in prompt

    def test_explicit_routing_field_included(self, mock_validator):
        """测试显式声明的路由字段应当包含在 Prompt 中"""
        # 模拟：用户在 columns_to_extract 中显式声明了路由字段
        # 此时 exclude_fields 为空（不排除）
        processor = ContentProcessor(
            prompt_template="Data: {record_json}",
            required_fields=["name"],
            validator=mock_validator,
            exclude_fields=[],  # 显式声明的字段不排除
        )

        record = {
            "name": "Alice",
            "BGBU": "type_a",  # 路由字段，显式声明，应当包含
            "content": "Some text"
        }

        prompt = processor.create_prompt(record)

        # 验证：BGBU 应当出现在 Prompt 中（作为业务字段）
        assert "BGBU" in prompt
        assert "type_a" in prompt
        assert "name" in prompt
        assert "content" in prompt

    def test_implicit_routing_field_excluded(self, mock_validator):
        """测试隐式路由字段（未显式声明）应当从 Prompt 中排除"""
        # 模拟：用户未在 columns_to_extract 中声明路由字段
        # 系统自动追加，但标记为排除
        processor = ContentProcessor(
            prompt_template="Data: {record_json}",
            required_fields=["name"],
            validator=mock_validator,
            exclude_fields=["BGBU"],  # 隐式字段，应当排除
        )

        record = {
            "name": "Bob",
            "BGBU": "type_b",  # 路由字段，隐式追加，应当排除
            "content": "Some text"
        }

        prompt = processor.create_prompt(record)

        # 验证：BGBU 不应出现在 Prompt 中
        assert "BGBU" not in prompt
        assert "type_b" not in prompt
        # 验证：其他字段正常包含
        assert "name" in prompt
        assert "content" in prompt
