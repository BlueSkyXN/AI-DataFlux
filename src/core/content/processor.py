import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..validator import JsonValidator
from ...models.errors import ErrorType

class ContentProcessor:
    """
    内容处理器，负责：
    1. 将行数据渲染为 Prompt
    2. 解析 AI 响应并提取 JSON
    3. 构建 JSON Schema
    4. 验证内容合法性
    """

    def __init__(
        self,
        prompt_template: str,
        required_fields: List[str],
        validator: JsonValidator,
        use_json_schema: bool = False,
    ):
        """
        初始化内容处理器

        Args:
            prompt_template: Prompt 模板
            required_fields: 必需字段列表
            validator: JSON 验证器实例
            use_json_schema: 是否使用 JSON Schema
        """
        self.prompt_template = prompt_template
        self.required_fields = required_fields
        self.validator = validator
        self.use_json_schema = use_json_schema

    def create_prompt(self, record_data: Dict[str, Any]) -> str:
        """
        创建提示词

        Args:
            record_data: 记录数据字典

        Returns:
            渲染后的 Prompt 字符串
        """
        if not self.prompt_template:
            return ""

        try:
            # 过滤内部字段（以 _ 开头的字段）
            filtered_data = {
                k: v
                for k, v in record_data.items()
                if v is not None and not k.startswith("_")
            }
            # 使用 compact 格式序列化
            record_json_str = json.dumps(
                filtered_data, ensure_ascii=False, separators=(",", ":")
            )
        except (TypeError, ValueError) as e:
            logging.error(f"记录数据无法序列化为 JSON: {e}")
            return self.prompt_template.replace(
                "{record_json}", '{"error": "无法序列化数据"}'
            )

        return self.prompt_template.replace("{record_json}", record_json_str)

    def parse_response(self, content: Optional[str]) -> Dict[str, Any]:
        """
        从 AI 响应中提取 JSON

        Args:
            content: AI 返回的原始文本内容

        Returns:
            解析后的字典，包含数据或错误信息
        """
        if not content:
            return {"_error": "empty_ai_response", "_error_type": ErrorType.CONTENT}

        content = content.strip()
        parse_content = content

        # 1. 尝试提取 Markdown JSON 代码块
        # 支持 ```json ... ``` 和 ``` ... ```
        code_block_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE
        )
        if code_block_match:
            parse_content = code_block_match.group(1).strip()

        # 2. 尝试直接解析
        # 预处理：移除尾部多余的逗号
        try:
            parse_content_cleaned = re.sub(r",\s*([}\]])", r"\1", parse_content)
            data = json.loads(parse_content_cleaned)

            if isinstance(data, dict):
                missing_fields = self._check_missing_fields(data)
                if not missing_fields:
                    is_valid, errors = self.validator.validate(data)
                    if is_valid:
                        return data
                    else:
                        return {
                            "_error": "invalid_field_values",
                            "_error_type": ErrorType.CONTENT,
                            "_validation_errors": errors,
                        }
                # 如果缺少字段，继续尝试正则提取（也许有更完整的 JSON 在后面）
                # 但如果没有找到更好的，我们可能希望记录这个错误，不过目前的逻辑是继续
        except json.JSONDecodeError:
            pass

        # 3. 尝试正则提取所有可能的 JSON 对象
        pattern = r"(\{.*?\})"
        for match in re.finditer(pattern, content, re.DOTALL):
            match_str = match.group(1)
            try:
                match_str_cleaned = re.sub(r",\s*([}\]])", r"\1", match_str.strip())
                candidate = json.loads(match_str_cleaned)

                if isinstance(candidate, dict):
                    if not self._check_missing_fields(candidate):
                         is_valid, _ = self.validator.validate(candidate)
                         if is_valid:
                             return candidate
            except json.JSONDecodeError:
                continue

        logging.error(f"无法提取有效 JSON (必需字段: {self.required_fields})")
        return {"_error": "invalid_or_missing_json", "_error_type": ErrorType.CONTENT}

    def _check_missing_fields(self, data: Dict[str, Any]) -> List[str]:
        """检查缺失字段"""
        if not self.required_fields:
            return []
        return [k for k in self.required_fields if k not in data]

    def build_schema(self) -> Optional[Dict[str, Any]]:
        """
        构建 JSON Schema

        Returns:
            JSON Schema 字典或 None
        """
        if not self.use_json_schema:
            return None

        if not self.required_fields:
            logging.warning("JSON Schema 已启用但 required_fields 为空")
            return None

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": list(self.required_fields),
        }

        validation_rules = self.validator.field_rules if self.validator.enabled else {}

        for field in self.required_fields:
            prop_def: Dict[str, Any] = {}
            allowed_values = validation_rules.get(field)

            if allowed_values:
                first_val = allowed_values[0]
                if isinstance(first_val, bool):
                    prop_def["type"] = "boolean"
                elif isinstance(first_val, int):
                    prop_def["type"] = "integer"
                elif isinstance(first_val, float):
                    prop_def["type"] = "number"
                else:
                    prop_def["type"] = "string"
                prop_def["enum"] = allowed_values
            else:
                prop_def["type"] = "string"

            schema["properties"][field] = prop_def

        return schema
