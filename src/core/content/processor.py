"""
内容处理器

本模块实现 AI 交互的内容处理逻辑，是数据与 AI 模型之间的桥梁。
负责将原始数据转换为 Prompt，并从 AI 响应中提取结构化数据。

处理流程:
    ┌──────────────────────────────────────────────────────────────────┐
    │                        输入阶段                                   │
    │  原始数据 → 过滤内部字段 → JSON 序列化 → 模板替换 → Prompt        │
    └──────────────────────────────────────────────────────────────────┘
                                    ↓
                                 AI 模型
                                    ↓
    ┌──────────────────────────────────────────────────────────────────┐
    │                        输出阶段                                   │
    │  AI 响应 → Markdown 提取 → JSON 解析 → 字段验证 → 结构化数据      │
    └──────────────────────────────────────────────────────────────────┘

JSON 提取策略 (按优先级):
    1. Markdown 代码块: ```json {...} ``` 或 ``` {...} ```
    2. 直接 JSON 解析: 整个响应作为 JSON
    3. 正则提取: 搜索所有 {...} 模式，逐个尝试解析

JSON 预处理:
    - 移除尾部多余逗号 (常见的 AI 错误): {"a": 1,} → {"a": 1}
    - 过滤内部字段 (以 _ 开头)

验证层次:
    1. 必需字段检查: required_fields 中的字段必须存在
    2. 字段值验证: 通过 JsonValidator 验证枚举值合法性

使用示例:
    processor = ContentProcessor(
        prompt_template="请分析: {record_json}",
        required_fields=["category", "score"],
        validator=validator,
        use_json_schema=True
    )
    
    # 生成 Prompt
    prompt = processor.create_prompt({"title": "文章标题", "content": "..."})
    # => "请分析: {\"title\":\"文章标题\",\"content\":\"...\"}"
    
    # 解析响应
    result = processor.parse_response('```json\\n{"category": "tech", "score": 85}\\n```')
    # => {"category": "tech", "score": 85}
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..validator import JsonValidator
from ...models.errors import ErrorType


class ContentProcessor:
    """
    内容处理器
    
    负责 Prompt 生成、AI 响应解析、JSON 提取和字段验证。
    作为数据处理流程中的核心转换组件。
    
    职责:
        1. 将行数据渲染为 Prompt (create_prompt)
        2. 从 AI 响应中提取 JSON (parse_response)
        3. 构建 JSON Schema (build_schema)
        4. 验证内容合法性 (通过 validator)
    
    Attributes:
        prompt_template: Prompt 模板，使用 {record_json} 占位符
        required_fields: AI 响应中必须包含的字段列表
        validator: JsonValidator 实例，用于枚举值验证
        use_json_schema: 是否启用 JSON Schema 约束
    """

    def __init__(
        self,
        prompt_template: str,
        required_fields: List[str],
        validator: JsonValidator,
        use_json_schema: bool = False,
        exclude_fields: Optional[List[str]] = None,
    ):
        """
        初始化内容处理器

        Args:
            prompt_template: Prompt 模板，包含 {record_json} 占位符
            required_fields: 必需字段列表，解析时会检查这些字段是否存在
            validator: JSON 验证器实例，用于验证字段枚举值
            use_json_schema: 是否使用 JSON Schema 模式
            exclude_fields: 从 Prompt 中排除的字段列表（如路由字段），默认为 None
        """
        self.prompt_template = prompt_template
        self.required_fields = required_fields
        self.validator = validator
        self.use_json_schema = use_json_schema
        self.exclude_fields = exclude_fields or []

    def create_prompt(self, record_data: Dict[str, Any]) -> str:
        """
        创建提示词
        
        将原始记录数据转换为发送给 AI 的 Prompt 文本。
        数据会被序列化为紧凑格式的 JSON 字符串，
        然后替换模板中的 {record_json} 占位符。

        Args:
            record_data: 记录数据字典

        Returns:
            渲染后的 Prompt 字符串
            
        Note:
            - 以 _ 开头的内部字段会被过滤
            - None 值会被过滤
            - 使用紧凑 JSON 格式 (无空格)
        """
        if not self.prompt_template:
            return ""

        try:
            # 过滤内部字段（以 _ 开头的字段）、None 值和排除字段
            filtered_data = {
                k: v
                for k, v in record_data.items()
                if v is not None and not k.startswith("_") and k not in self.exclude_fields
            }
            # 使用 compact 格式序列化 (节省 token)
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
        
        使用多种策略尝试从 AI 返回的文本中提取有效的 JSON 数据。
        提取后会进行必需字段检查和枚举值验证。

        Args:
            content: AI 返回的原始文本内容

        Returns:
            解析后的字典:
            - 成功: 包含提取的数据字段
            - 失败: 包含 _error 和 _error_type 字段
            
        提取策略:
            1. 尝试提取 Markdown JSON 代码块 (```json...```)
            2. 尝试直接解析整个响应为 JSON
            3. 使用正则搜索所有 {...} 模式
            
        错误类型:
            - empty_ai_response: AI 返回空内容
            - invalid_field_values: 字段值不在允许范围内
            - invalid_or_missing_json: 无法提取有效 JSON
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
        # 预处理：移除尾部多余的逗号 (常见的 AI 格式错误)
        try:
            parse_content_cleaned = re.sub(r",\s*([}\]])", r"\1", parse_content)
            data = json.loads(parse_content_cleaned)

            if isinstance(data, dict):
                missing_fields = self._check_missing_fields(data)
                if not missing_fields:
                    # 字段完整，进行值验证
                    is_valid, errors = self.validator.validate(data)
                    if is_valid:
                        return data
                    else:
                        return {
                            "_error": "invalid_field_values",
                            "_error_type": ErrorType.CONTENT,
                            "_validation_errors": errors,
                        }
                # 如果缺少字段，继续尝试正则提取
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
        """
        检查缺失字段
        
        Args:
            data: 待检查的数据字典
            
        Returns:
            缺失字段名列表 (空列表表示字段完整)
        """
        if not self.required_fields:
            return []
        return [k for k in self.required_fields if k not in data]

    def build_schema(self) -> Optional[Dict[str, Any]]:
        """
        构建 JSON Schema
        
        根据 required_fields 和 validator.field_rules 构建
        符合 JSON Schema 规范的约束对象。用于 API 的
        response_format.json_schema 参数。

        Returns:
            JSON Schema 字典或 None (未启用时)
            
        Schema 结构示例:
            {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["A", "B", "C"]},
                    "score": {"type": "integer"}
                },
                "required": ["category", "score"]
            }
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

        # 获取验证规则 (如果有)
        validation_rules = self.validator.field_rules if self.validator.enabled else {}

        for field in self.required_fields:
            prop_def: Dict[str, Any] = {}
            allowed_values = validation_rules.get(field)

            if allowed_values:
                # 根据允许值的类型推断 Schema 类型
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
                # 默认字符串类型
                prop_def["type"] = "string"

            schema["properties"][field] = prop_def

        return schema
