"""
JSON 字段验证器

本模块实现 AI 响应数据的字段值验证功能，确保 AI 返回的数据
符合预定义的枚举值约束。

设计目标:
    - 验证 AI 响应中的字段值是否在允许的范围内
    - 支持多字段同时验证
    - 提供详细的错误信息，便于重试决策

使用场景:
    当配置了 validation.field_rules 时，ContentProcessor 会在解析
    AI 响应后调用此验证器，检查返回的分类/枚举字段是否有效。

    例如: 情感分析任务要求返回 "positive", "neutral", "negative" 之一，
    但 AI 返回了 "good"，验证器会标记为无效。

类清单:
    JsonValidator
        - 功能: 根据预定义规则验证 JSON 字段值的合法性
        - 关键属性: enabled (bool), field_rules (dict[str, list[Any]])

        方法:
        ├── __init__()
        │     初始化验证器，设置默认禁用状态
        │
        ├── configure(validation_config) -> None
        │     输入: validation_config (dict | None) — 验证配置字典
        │     功能: 加载并解析 field_rules，启用/禁用验证
        │     副作用: 更新 self.enabled 和 self.field_rules
        │
        ├── validate(data) -> tuple[bool, list[str]]
        │     输入: data (dict[str, Any]) — 待验证的 JSON 数据
        │     输出: (是否通过, 错误消息列表)
        │     注意: 仅验证配置了规则且存在于 data 中的字段
        │
        └── get_rules_summary() -> dict[str, int]
              输出: {字段名: 允许值数量} 的摘要字典

模块依赖:
    - logging: 日志记录
    - typing.Any: 类型注解

配置示例:
    validation:
      enabled: true
      field_rules:
        category: ["technical", "business", "general"]
        sentiment: ["positive", "neutral", "negative"]
        priority: [1, 2, 3, 4, 5]

使用示例:
    validator = JsonValidator()
    validator.configure({
        "enabled": True,
        "field_rules": {
            "category": ["technical", "business"],
            "sentiment": ["positive", "negative"]
        }
    })
    is_valid, errors = validator.validate(ai_response)
"""

import logging
from typing import Any


class JsonValidator:
    """
    JSON 字段验证器

    根据配置的字段规则验证 JSON 数据中的字段值是否合法。

    验证逻辑:
        - 仅验证配置了规则的字段
        - 数据中不存在的字段不会触发验证失败
        - 值必须精确匹配（类型和值都要匹配）

    Attributes:
        enabled: 是否启用验证
        field_rules: 字段规则 {field: [allowed_values]}

    Example:
        >>> validator = JsonValidator()
        >>> validator.configure({
        ...     "enabled": True,
        ...     "field_rules": {
        ...         "category": ["technical", "business", "general"],
        ...         "sentiment": ["positive", "neutral", "negative"]
        ...     }
        ... })
        >>> is_valid, errors = validator.validate({"category": "technical", "sentiment": "positive"})
        >>> is_valid
        True
    """

    def __init__(self):
        """初始化验证器"""
        self.enabled = False
        self.field_rules: dict[str, list[Any]] = {}
        logging.info("JsonValidator 初始化")

    def configure(self, validation_config: dict[str, Any] | None) -> None:
        """
        从配置加载验证规则

        Args:
            validation_config: 验证配置字典，包含:
                - enabled: 是否启用验证
                - field_rules: 字段规则 {field: [allowed_values]}
        """
        if not validation_config:
            self.enabled = False
            logging.info("JSON 字段值验证配置未找到或为空，验证已禁用")
            return

        self.enabled = validation_config.get("enabled", False)
        if not self.enabled:
            logging.info("JSON 字段值验证功能已在配置中禁用")
            return

        rules = validation_config.get("field_rules", {})
        if not isinstance(rules, dict):
            logging.warning("validation.field_rules 配置格式错误，应为字典，验证已禁用")
            self.enabled = False
            return

        # 加载规则
        self.field_rules = {}
        loaded_count = 0

        for field, values in rules.items():
            if isinstance(values, list):
                self.field_rules[field] = values
                logging.debug(f"加载字段验证规则: '{field}' -> {len(values)} 个允许值")
                loaded_count += 1
            else:
                logging.warning(f"字段 '{field}' 的验证规则格式错误，应为列表，已忽略")

        if not self.field_rules:
            logging.warning("JSON 字段值验证已启用，但未加载任何有效规则")
            self.enabled = False
        else:
            logging.info(
                f"JSON 字段值验证功能已启用，共加载 {loaded_count} 个字段的规则"
            )

    def validate(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        验证数据

        Args:
            data: 要验证的 JSON 数据字典

        Returns:
            (是否通过验证, 错误消息列表)
        """
        if not self.enabled or not self.field_rules:
            return True, []

        errors: list[str] = []

        for field, allowed_values in self.field_rules.items():
            if field in data:
                value = data[field]
                if value not in allowed_values:
                    preview = str(allowed_values[:10])
                    errors.append(
                        f"字段 '{field}' 的值 '{value}' (类型: {type(value).__name__}) "
                        f"不在允许的范围内: {preview}..."
                    )

        is_valid = len(errors) == 0

        if not is_valid:
            logging.debug(f"JSON 字段值验证失败: {errors}")

        return is_valid, errors

    def get_rules_summary(self) -> dict[str, int]:
        """
        获取规则摘要

        Returns:
            {字段名: 允许值数量}
        """
        return {field: len(values) for field, values in self.field_rules.items()}
