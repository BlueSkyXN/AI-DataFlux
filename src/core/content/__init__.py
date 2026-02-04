"""
内容处理模块

本模块负责 AI 交互的内容处理，包括 Prompt 生成、响应解析、
JSON 提取和验证等功能。

模块职责:
    - ContentProcessor: 内容处理器，完成完整的输入输出转换

使用示例:
    from src.core.content import ContentProcessor
    from src.core.validator import JsonValidator

    validator = JsonValidator()
    processor = ContentProcessor(
        prompt_template="分析以下数据: {record_json}",
        required_fields=["category", "sentiment"],
        validator=validator
    )

    # 生成 Prompt
    prompt = processor.create_prompt({"title": "Test", "content": "..."})

    # 解析响应
    result = processor.parse_response(ai_response)
"""

from .processor import ContentProcessor

__all__ = ["ContentProcessor"]
