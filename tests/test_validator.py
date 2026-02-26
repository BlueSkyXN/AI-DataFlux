"""
JSON 验证器测试

被测模块: src/core/validator.py (JsonValidator)

测试 src/core/validator.py 的字段验证功能，包括：
- 枚举值验证
- 多字段规则配置
- 验证启用/禁用
- 无效值检测与报告

测试类/函数清单:
    TestJsonValidator                  JSON 验证器测试
        test_valid_values              验证合法枚举值通过
        test_invalid_value             验证非法枚举值被检出
        test_missing_field             验证缺失字段不算错误（只验证已有字段）
        test_disabled_validator        验证禁用时总返回 True
        test_empty_data                验证空数据通过
        test_case_sensitivity          验证大小写敏感
        test_extra_fields              验证额外字段被忽略
    TestValidatorEdgeCases             边界情况测试
        test_numeric_values            验证数字类型不匹配字符串规则
        test_empty_rules               验证空规则时所有数据有效
        test_not_configured            验证未配置时默认禁用
"""

import pytest


class TestJsonValidator:
    """JSON 验证器测试"""

    @pytest.fixture
    def validator(self):
        """创建验证器实例"""
        from src.core.validator import JsonValidator

        v = JsonValidator()
        v.configure(
            {
                "enabled": True,
                "field_rules": {
                    "category": ["A", "B", "C"],
                    "status": ["active", "inactive"],
                },
            }
        )
        return v

    @pytest.fixture
    def validator_disabled(self):
        """创建禁用的验证器"""
        from src.core.validator import JsonValidator

        v = JsonValidator()
        v.configure({"enabled": False})
        return v

    def test_valid_values(self, validator):
        """测试有效值"""
        data = {"category": "A", "status": "active"}
        is_valid, errors = validator.validate(data)

        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_value(self, validator):
        """测试无效值"""
        data = {"category": "X", "status": "active"}
        is_valid, errors = validator.validate(data)

        assert is_valid is False
        assert "category" in str(errors)

    def test_missing_field(self, validator):
        """测试缺失字段 (应该通过，只验证存在的字段)"""
        data = {"category": "A"}  # 缺少 status
        is_valid, errors = validator.validate(data)

        # 缺失字段不算错误，只验证存在的字段
        assert is_valid is True

    def test_disabled_validator(self, validator_disabled):
        """测试禁用的验证器"""
        data = {"category": "INVALID", "status": "INVALID"}
        is_valid, errors = validator_disabled.validate(data)

        # 禁用时总是返回 True
        assert is_valid is True

    def test_empty_data(self, validator):
        """测试空数据"""
        is_valid, errors = validator.validate({})
        assert is_valid is True

    def test_case_sensitivity(self, validator):
        """测试大小写敏感"""
        data = {"category": "a"}  # 小写，规则是大写 "A"
        is_valid, errors = validator.validate(data)

        # 应该区分大小写
        assert is_valid is False

    def test_extra_fields(self, validator):
        """测试额外字段"""
        data = {
            "category": "A",
            "status": "active",
            "extra": "value",  # 规则中没有的字段
        }
        is_valid, errors = validator.validate(data)

        # 额外字段应该被忽略
        assert is_valid is True


class TestValidatorEdgeCases:
    """验证器边界情况测试"""

    def test_numeric_values(self):
        """测试数字值验证"""
        from src.core.validator import JsonValidator

        v = JsonValidator()
        v.configure({"enabled": True, "field_rules": {"score": ["1", "2", "3"]}})

        # 字符串 "1" 应该匹配
        assert v.validate({"score": "1"})[0] is True

        # 数字 1 不匹配字符串 "1"
        result, _ = v.validate({"score": 1})
        assert result is False

    def test_empty_rules(self):
        """测试空规则"""
        from src.core.validator import JsonValidator

        v = JsonValidator()
        v.configure({"enabled": True, "field_rules": {}})

        # 无规则时所有数据都有效
        assert v.validate({"any": "value"})[0] is True

    def test_not_configured(self):
        """测试未配置的验证器"""
        from src.core.validator import JsonValidator

        v = JsonValidator()
        # 不调用 configure

        # 未配置时默认禁用
        assert v.validate({"any": "value"})[0] is True
