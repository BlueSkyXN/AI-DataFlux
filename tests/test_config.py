"""
配置加载测试

被测模块: src/config/settings.py

测试 src/config/settings.py 的配置加载功能，包括：
- 有效配置文件加载
- 配置字段验证
- 错误配置处理 (文件不存在、格式错误)
- 默认配置深度合并

测试类/函数清单:
    TestConfigLoading              配置加载测试
        test_load_valid_config     验证有效 YAML 文件可正常加载并包含必要字段
        test_load_config_example   验证 config-example.yaml 可正常加载
        test_config_missing_file   验证不存在的文件抛出 FileNotFoundError
        test_config_invalid_yaml   验证格式错误的 YAML 抛出异常
    TestConfigValidation           配置验证测试
        test_datasource_type       验证数据源类型为合法值
        test_engine_options        验证引擎/读写器选项为合法值
        test_concurrency_config    验证并发配置为正整数
        test_columns_config        验证列配置为有效列表/字典
        test_prompt_config         验证提示词配置包含 template 或 system_prompt
"""

import pytest
from pathlib import Path
import yaml


class TestConfigLoading:
    """配置加载测试"""

    def test_load_valid_config(self, sample_config_file):
        """测试加载有效配置"""
        from src.config import load_config

        config = load_config(str(sample_config_file))

        assert "global" in config
        assert "datasource" in config
        assert "columns_to_extract" in config
        assert "columns_to_write" in config

    def test_load_config_example(self):
        """测试加载示例配置文件"""
        from src.config import load_config

        config_path = Path("config-example.yaml")
        if not config_path.exists():
            pytest.skip("config-example.yaml not found")

        config = load_config(str(config_path))

        # 验证必要字段
        assert "global" in config
        assert "datasource" in config
        assert "prompt" in config

    def test_config_missing_file(self, tmp_path):
        """测试加载不存在的配置文件"""
        from src.config import load_config

        with pytest.raises((FileNotFoundError, Exception)):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_config_invalid_yaml(self, tmp_path):
        """测试加载无效 YAML"""
        from src.config import load_config

        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: [unclosed")

        with pytest.raises(Exception):  # yaml.YAMLError
            load_config(str(invalid_file))


class TestConfigValidation:
    """配置验证测试"""

    def test_datasource_type(self, sample_config):
        """测试数据源类型验证"""
        ds = sample_config["datasource"]
        assert ds["type"] in ["mysql", "excel"]

    def test_engine_options(self, sample_config):
        """测试引擎选项"""
        ds = sample_config["datasource"]
        assert ds.get("engine", "auto") in ["auto", "pandas", "polars"]
        assert ds.get("excel_reader", "auto") in ["auto", "openpyxl", "calamine"]
        assert ds.get("excel_writer", "auto") in ["auto", "openpyxl", "xlsxwriter"]

    def test_concurrency_config(self, sample_config):
        """测试并发配置"""
        concurrency = sample_config["datasource"].get("concurrency", {})

        batch_size = concurrency.get("batch_size", 100)
        assert isinstance(batch_size, int)
        assert batch_size > 0

        save_interval = concurrency.get("save_interval", 300)
        assert isinstance(save_interval, int)
        assert save_interval > 0

    def test_columns_config(self, sample_config):
        """测试列配置"""
        extract = sample_config.get("columns_to_extract", [])
        write = sample_config.get("columns_to_write", {})

        assert isinstance(extract, list)
        assert isinstance(write, dict)
        assert len(extract) > 0 or len(write) > 0

    def test_prompt_config(self, sample_config):
        """测试提示词配置"""
        prompt = sample_config.get("prompt", {})

        assert "template" in prompt or "system_prompt" in prompt

        if "required_fields" in prompt:
            assert isinstance(prompt["required_fields"], list)


class TestConfigSemanticValidation:
    """配置语义校验测试"""

    def test_validate_config_rejects_startup_blocking_errors(self, sample_config):
        """测试 validate_config 能拦截启动阶段会失败的配置"""
        from src.config import validate_config

        sample_config["datasource"]["type"] = "mongodb"
        sample_config["datasource"]["concurrency"]["batch_size"] = 0
        sample_config["columns_to_extract"] = []

        result = validate_config(sample_config)

        assert any("datasource.type" in error for error in result["errors"])
        assert any("batch_size" in error for error in result["errors"])
        assert any("columns_to_extract" in error for error in result["errors"])

    def test_validate_config_accepts_datasource_type_case_runtime_support(
        self, sample_config
    ):
        """测试 datasource.type 大小写兼容 create_task_pool 的 lower 行为"""
        from src.config import validate_config

        sample_config["datasource"]["type"] = "Excel"

        result = validate_config(sample_config)

        assert not result["errors"]

    def test_validate_config_rejects_prompt_without_template(self, sample_config):
        """测试主配置缺少 prompt.template 时会被拦截"""
        from src.config import validate_config

        sample_config["prompt"] = {"system_prompt": "only system message"}

        result = validate_config(sample_config)

        assert any("prompt.template" in error for error in result["errors"])

    def test_validate_config_warns_ignored_legacy_concurrency_keys(self, sample_config):
        """测试已知会被忽略的旧并发配置键会给出 warning"""
        from src.config import validate_config

        sample_config["datasource"]["concurrency"]["max_workers"] = 8
        sample_config["datasource"]["concurrency"]["retry_times"] = 5

        result = validate_config(sample_config)

        assert not result["errors"]
        assert any("max_workers" in warning for warning in result["warnings"])
        assert any("retry_times" in warning for warning in result["warnings"])

    def test_validate_config_rejects_missing_routing_profile(
        self, sample_config, tmp_path
    ):
        """测试 routing profile 缺失时 validate_config 不再误报有效"""
        from src.config import validate_config

        config_path = tmp_path / "config.yaml"
        sample_config["routing"] = {
            "enabled": True,
            "field": "category",
            "subtasks": [{"match": "a", "profile": "missing.yaml"}],
        }

        result = validate_config(sample_config, config_path)

        assert any("profile 加载失败" in error for error in result["errors"])

    def test_validate_config_rejects_non_bool_routing_enabled(self, sample_config):
        """测试 routing.enabled 使用字符串时会被明确拦截"""
        from src.config import validate_config

        sample_config["routing"] = {"enabled": "false"}

        result = validate_config(sample_config)

        assert any("routing.enabled" in error for error in result["errors"])

    def test_validate_config_allows_partial_routing_profile_prompt(
        self, sample_config, tmp_path
    ):
        """测试 routing 子配置允许只覆盖 prompt 局部字段"""
        from src.config import validate_config

        profile_path = tmp_path / "rule.yaml"
        profile_path.write_text(
            yaml.safe_dump({"prompt": {"temperature": 0.1}}, allow_unicode=True),
            encoding="utf-8",
        )
        config_path = tmp_path / "config.yaml"
        sample_config["routing"] = {
            "enabled": True,
            "field": "category",
            "subtasks": [{"match": "a", "profile": "rule.yaml"}],
        }

        result = validate_config(sample_config, config_path)

        assert not result["errors"]

    def test_validate_config_rejects_forbidden_routing_profile_keys(
        self, sample_config, tmp_path
    ):
        """测试 routing 子配置包含非法顶层键时会失败"""
        from src.config import validate_config

        profile_path = tmp_path / "rule.yaml"
        profile_path.write_text(
            yaml.safe_dump(
                {
                    "prompt": {"template": "route: {record_json}"},
                    "datasource": {"type": "sqlite"},
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        config_path = tmp_path / "config.yaml"
        sample_config["routing"] = {
            "enabled": True,
            "field": "category",
            "subtasks": [{"match": "a", "profile": "rule.yaml"}],
        }

        result = validate_config(sample_config, config_path)

        assert any("仅允许 prompt/validation" in error for error in result["errors"])
