"""
内容处理模块

本模块负责 AI 交互的内容处理，包括 Prompt 生成、响应解析、
JSON 提取和验证等功能。

类/函数清单:
    ContentProcessor:
        - __init__(prompt_template, required_fields, validator, use_json_schema, exclude_fields)
          初始化处理器，配置模板、必需字段、验证器等
        - create_prompt(record_data) -> str
          将记录数据渲染为 Prompt 文本
          输入: Dict 记录数据 | 输出: str 渲染后的 Prompt
        - parse_response(content) -> Dict
          从 AI 响应中提取并验证 JSON 数据
          输入: str AI 原始响应 | 输出: Dict 解析结果 (含 _error 字段表示失败)
        - build_schema() -> Optional[Dict]
          构建 JSON Schema 约束对象
          输出: Dict JSON Schema 或 None
        - _check_missing_fields(data) -> List[str]
          检查缺失的必需字段
          输入: Dict 数据 | 输出: List 缺失字段名列表

关键变量:
    - prompt_template: Prompt 模板，{record_json} 占位符会被替换为记录 JSON
    - required_fields: AI 响应必须包含的字段列表
    - validator: JsonValidator 实例，验证字段枚举值合法性
    - use_json_schema: 是否启用 JSON Schema 输出约束
    - exclude_fields: 构建 Prompt 时排除的字段列表

模块依赖:
    - src.core.validator.JsonValidator: 字段值验证
    - src.models.errors.ErrorType: 错误类型枚举

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
