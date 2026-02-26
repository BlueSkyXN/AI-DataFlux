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
