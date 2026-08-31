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

import copy

import pytest
from pathlib import Path
import yaml


def _add_valid_gateway_config(config):
    config["channels"] = {
        "openai": {
            "base_url": "https://api.example.test",
            "endpoints": {"chat_completions": "/v1/chat/completions"},
            "timeout": 60,
        }
    }
    config["models"] = [
        {
            "id": "model-1",
            "name": "model-1",
            "model": "upstream-model-1",
            "channel_id": "openai",
            "api_key": "test-key",
            "timeout": 60,
            "weight": 1,
            "temperature": 0.5,
            "safe_rps": 10,
            "capabilities": ["chat_completions"],
        }
    ]


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

    def test_validate_config_rejects_removed_legacy_concurrency_keys(
        self, sample_config
    ):
        """v3.2 单一 schema 必须拒绝旧并发配置键。"""
        from src.config import validate_config

        sample_config["datasource"]["concurrency"]["max_workers"] = 8
        sample_config["datasource"]["concurrency"]["retry_times"] = 5

        result = validate_config(sample_config)

        assert any("max_workers" in error for error in result["errors"])
        assert any("retry_times" in error for error in result["errors"])

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

    def test_validate_config_accepts_canonical_gateway_contract(self, sample_config):
        from src.config import validate_config

        sample_config["channels"] = {
            "openai": {
                "name": "OpenAI",
                "base_url": "https://api.example.test",
                "endpoints": {
                    "chat_completions": "/v1/chat/completions",
                    "responses": "/v1/responses",
                },
            }
        }
        sample_config["models"] = [
            {
                "id": "model-a",
                "name": "Model A",
                "model": "upstream-a",
                "channel_id": "openai",
                "api_key": "local-test-key",
                "capabilities": [
                    "chat_completions",
                    "responses",
                    "stream",
                    "tools",
                    "previous_response_id",
                ],
            }
        ]

        assert validate_config(sample_config)["errors"] == []

    @pytest.mark.parametrize(
        ("section", "legacy_key"),
        [
            ("channel", "api_path"),
            ("model", "supports_json_schema"),
            ("token", "tiktoken_model"),
        ],
    )
    def test_validate_config_rejects_removed_gateway_and_token_keys(
        self, sample_config, section, legacy_key
    ):
        from src.config import validate_config

        sample_config["channels"] = {
            "openai": {
                "name": "OpenAI",
                "base_url": "https://api.example.test",
                "endpoints": {"chat_completions": "/v1/chat/completions"},
            }
        }
        sample_config["models"] = [
            {
                "id": "model-a",
                "name": "Model A",
                "model": "upstream-a",
                "channel_id": "openai",
                "api_key": "local-test-key",
                "capabilities": ["chat_completions"],
            }
        ]
        if section == "channel":
            sample_config["channels"]["openai"][legacy_key] = "/legacy"
        elif section == "model":
            sample_config["models"][0][legacy_key] = True
        else:
            sample_config.setdefault("token_estimation", {})[legacy_key] = "gpt-4"

        result = validate_config(sample_config)
        assert any(legacy_key in error for error in result["errors"])

    def test_model_endpoint_capability_requires_channel_endpoint(self, sample_config):
        from src.config import validate_config

        sample_config["channels"] = {
            "openai": {
                "name": "OpenAI",
                "base_url": "https://api.example.test",
                "endpoints": {"chat_completions": "/v1/chat/completions"},
            }
        }
        sample_config["models"] = [
            {
                "id": "model-a",
                "name": "Model A",
                "model": "upstream-a",
                "channel_id": "openai",
                "api_key": "local-test-key",
                "capabilities": ["responses"],
            }
        ]

        result = validate_config(sample_config)
        assert any("endpoint capability" in error for error in result["errors"])

    def test_workspace_resolution_rejects_symlink_escape(self, tmp_path):
        from src.config import resolve_workspace_path
        from src.models.errors import ConfigError

        root = tmp_path / "root"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        try:
            (root / "escape").symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are unavailable")
        config = {"workspace": {"roots": {"project": str(root)}}}

        with pytest.raises(ConfigError, match="超出允许根目录"):
            resolve_workspace_path(config, "project", "escape/file.yaml")

    def test_access_token_priority_and_non_loopback_requirement(self, monkeypatch):
        from src.config import resolve_access_token
        from src.models.errors import ConfigError

        config = {"server": {"token": "yaml-token"}}
        monkeypatch.setenv("DATAFLUX_TOKEN", "env-token")
        assert resolve_access_token(config, host="0.0.0.0").value == "env-token"
        monkeypatch.delenv("DATAFLUX_TOKEN")
        assert resolve_access_token(config, host="0.0.0.0").value == "yaml-token"
        with pytest.raises(ConfigError):
            resolve_access_token(
                {"server": {"token": ""}},
                host="0.0.0.0",
                allow_generate_loopback=True,
            )

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda c: c.update({"unknown_top_level": True}),
            lambda c: c["global"]["log"].update(
                {"level": "verbose", "format": "xml", "output": "remote"}
            ),
            lambda c: c["datasource"].update(
                {
                    "engine": "spark",
                    "excel_reader": "bad",
                    "excel_writer": "bad",
                    "require_all_input_fields": "yes",
                }
            ),
            lambda c: c["datasource"]["concurrency"].update(
                {
                    "batch_size": 0,
                    "max_in_flight": False,
                    "save_interval": -1,
                    "max_connections": 0,
                    "max_connections_per_host": -1,
                    "api_pause_duration": 0,
                }
            ),
            lambda c: c["datasource"]["concurrency"].update(
                {"min_shard_size": 10, "max_shard_size": 1}
            ),
            lambda c: c["datasource"]["concurrency"].update(
                {"retry_limits": {"api_error": -1}}
            ),
            lambda c: c.update(
                {"columns_to_extract": [""], "columns_to_write": {"": ""}}
            ),
            lambda c: c["prompt"].update(
                {
                    "required_fields": "answer",
                    "use_json_schema": "yes",
                    "temperature_override": "yes",
                    "temperature": 3,
                }
            ),
            lambda c: c.update({"validation": {"enabled": "yes", "field_rules": []}}),
            lambda c: c.update(
                {"validation": {"enabled": True, "field_rules": {"x": "bad"}}}
            ),
            lambda c: c.update(
                {
                    "token_estimation": {
                        "mode": "bad",
                        "sample_size": 0,
                        "encoding": "",
                    }
                }
            ),
            lambda c: c.update(
                {"gateway": {"max_connections": 0, "max_connections_per_host": 0}}
            ),
            lambda c: c.update(
                {
                    "scheduler": {
                        "max_active_jobs": 0,
                        "cpu_high_watermark": 101,
                        "memory_high_watermark": 0,
                        "min_free_memory_mb": -1,
                        "sample_interval_seconds": 0,
                    }
                }
            ),
            lambda c: c.update(
                {
                    "server": {
                        "host": "",
                        "control_port": 0,
                        "gateway_port": 70000,
                    }
                }
            ),
            lambda c: c.update({"workspace": {"roots": {}, "state_dir": "/outside"}}),
            lambda c: c.update(
                {
                    "workspace": {
                        "roots": {"bad/id": "", "project": "."},
                        "state_dir": "/outside",
                    }
                }
            ),
            lambda c: c.update(
                {
                    "datasource": {"type": "mysql", "concurrency": {}},
                    "mysql": {},
                }
            ),
            lambda c: c.update(
                {
                    "datasource": {"type": "feishu_bitable", "concurrency": {}},
                    "feishu": {},
                }
            ),
            lambda c: c.update({"channels": [], "models": {}}),
        ],
        ids=lambda mutate: str(id(mutate)),
    )
    def test_strict_schema_rejects_invalid_known_values(self, sample_config, mutate):
        from src.config import validate_config

        config = copy.deepcopy(sample_config)
        mutate(config)
        assert validate_config(config)["errors"]

    def test_strict_channel_and_model_shape_errors(self, sample_config):
        from src.config import validate_config

        sample_config["channels"] = {
            1: "not-a-map",
            "openai": {
                "base_url": "",
                "endpoints": {
                    "chat_completions": "",
                    "unknown": "/v1/unknown",
                },
                "unknown": True,
            },
        }
        sample_config["models"] = [
            {
                "id": "duplicate",
                "name": "",
                "model": "model",
                "channel_id": "missing",
                "capabilities": ["responses", "responses", "unknown"],
                "legacy": True,
            },
            {
                "id": "duplicate",
                "name": "second",
                "model": "model",
                "channel_id": "openai",
                "capabilities": ["previous_response_id"],
            },
        ]

        errors = validate_config(sample_config)["errors"]
        assert any("channel ID" in error for error in errors)
        assert any("重复" in error for error in errors)
        assert any("previous_response_id" in error for error in errors)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("timeout", 0),
            ("timeout", "60"),
            ("weight", -1),
            ("weight", 1.5),
            ("temperature", 3),
            ("temperature", "0.5"),
            ("safe_rps", 0),
            ("safe_rps", "10"),
        ],
    )
    def test_strict_model_numeric_fields(self, sample_config, field, value):
        from src.config import validate_config

        _add_valid_gateway_config(sample_config)
        sample_config["models"][0][field] = value

        errors = validate_config(sample_config)["errors"]
        assert any(f"models[0].{field}" in error for error in errors)

    @pytest.mark.parametrize("value", [0, -1, "60", 1.5])
    def test_strict_channel_timeout(self, sample_config, value):
        from src.config import validate_config

        _add_valid_gateway_config(sample_config)
        sample_config["channels"]["openai"]["timeout"] = value

        errors = validate_config(sample_config)["errors"]
        assert any("channels.openai.timeout" in error for error in errors)

    def test_strict_model_aliases_must_be_unambiguous(self, sample_config):
        from src.config import validate_config

        _add_valid_gateway_config(sample_config)
        duplicate = copy.deepcopy(sample_config["models"][0])
        duplicate.update({"id": "model-2", "channel_id": "openai"})
        sample_config["models"].append(duplicate)

        errors = validate_config(sample_config)["errors"]
        assert any("alias" in error for error in errors)

    def test_file_logging_without_path_warns_and_merge_get_nested_cover_defaults(
        self, sample_config
    ):
        from src.config import get_nested, merge_config, validate_config

        sample_config["global"]["log"] = {"output": "file"}
        result = validate_config(sample_config)
        assert any("file_path" in warning for warning in result["warnings"])
        assert get_nested({"a": {"b": 1}}, "a", "b") == 1
        assert get_nested({"a": 1}, "a", "b", default="missing") == "missing"
        assert merge_config({"a": {"b": 1}, "keep": True}, {"a": {"c": 2}}) == {
            "a": {"b": 1, "c": 2},
            "keep": True,
        }

    def test_loopback_token_generation_checker_and_redaction(self, monkeypatch):
        from src.config import (
            is_loopback_host,
            make_token_checker,
            redact_sensitive_text,
            resolve_access_token,
        )

        monkeypatch.delenv("DATAFLUX_TOKEN", raising=False)
        assert is_loopback_host("localhost") is True
        assert is_loopback_host("127.0.0.1") is True
        assert is_loopback_host("8.8.8.8") is False
        assert is_loopback_host("not-an-ip") is False
        generated = resolve_access_token({"server": {"token": ""}}, host="127.0.0.1")
        assert generated.generated is True
        checker = make_token_checker(generated)
        assert checker(generated.value) is True
        assert checker("wrong-token") is False
        assert checker("") is False
        redacted = redact_sensitive_text(
            "Bearer bearer-secret api_key=key-secret exact-secret",
            known_secrets=("", "exact-secret"),
        )
        assert "bearer-secret" not in redacted
        assert "key-secret" not in redacted
        assert "exact-secret" not in redacted
