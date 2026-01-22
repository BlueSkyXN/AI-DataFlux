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
