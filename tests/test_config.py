"""Canonical v4 RootConfig contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _runtime(tmp_path: Path | None = None) -> dict:
    root = str(tmp_path) if tmp_path else "."
    return {
        "log": {"level": "error"},
        "auth": {"token": ""},
        "workspace": {
            "roots": {"project": root},
            "state_dir": "./.dataflux/jobs",
        },
    }


def _job(datasource: dict | None = None) -> dict:
    return {
        "gateway_url": "http://127.0.0.1:8787",
        "datasource": datasource
        or {
            "type": "csv",
            "input_path": "input.csv",
            "output_path": "output.csv",
            "engine": "pandas",
            "require_all_input_fields": True,
        },
        "columns": {"extract": ["input"], "write": {"answer": "result"}},
        "prompt": {"template": "{record_json}"},
    }


def _gateway() -> dict:
    return {
        "listen": {"host": "127.0.0.1", "port": 8787, "workers": 1},
        "channels": {
            "openai": {
                "base_url": "https://api.example.test",
                "endpoints": {
                    "chat_completions": "/v1/chat/completions",
                    "responses": "/v1/responses",
                },
                "timeout_seconds": 60,
            }
        },
        "routes": [
            {
                "id": "route-a",
                "display_name": "Route A",
                "aliases": ["alias-a"],
                "upstream_model": "upstream-a",
                "channel_id": "openai",
                "api_key": "test-key",
                "capabilities": ["chat_completions", "responses"],
                "weight": 1,
                "safe_rps": 10,
                "timeout_seconds": 60,
                "temperature": 0.3,
            }
        ],
        "fallback_groups": {"primary": ["route-a"]},
    }


def _root(*, job=None, gateway=None, control=None, tmp_path=None) -> dict:
    raw = {"schema_version": 4, "runtime": _runtime(tmp_path)}
    if job is not None:
        raw["job"] = job
    if gateway is not None:
        raw["gateway"] = gateway
    if control is not None:
        raw["control"] = control
    return raw


def test_load_config_returns_typed_defaults(sample_config_file):
    from src.config import RootConfig, load_config

    config = load_config(sample_config_file)

    assert isinstance(config, RootConfig)
    assert config.schema_version == 4
    assert config.job is not None
    assert config.job.retry.task_max_attempts.api_error == 4
    assert config.job.writeback.backoff_max_seconds == 30


def test_config_example_is_canonical_v4():
    from src.config import load_config

    config = load_config("config-example.yaml")

    assert config.schema_version == 4
    assert config.job is not None
    assert config.gateway is not None
    assert config.control is not None


def test_load_config_rejects_missing_and_invalid_yaml(tmp_path):
    from src.config import load_config
    from src.models.errors import ConfigError

    with pytest.raises(ConfigError, match="不存在"):
        load_config(tmp_path / "missing.yaml")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("invalid: yaml: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(invalid)


@pytest.mark.parametrize("version", [None, "4", 3, 5, 4.0, True])
def test_schema_version_must_be_integer_four(version):
    from src.config import validate_config

    raw = _root(control={"listen": {}})
    if version is None:
        raw.pop("schema_version")
    else:
        raw["schema_version"] = version

    errors = validate_config(raw)["errors"]

    assert errors
    assert any("schema_version" in error for error in errors)


@pytest.mark.parametrize(
    "legacy_key",
    [
        "global",
        "datasource",
        "mysql",
        "columns_to_extract",
        "columns_to_write",
        "models",
        "channels",
        "server",
    ],
)
def test_legacy_root_keys_are_unknown(legacy_key):
    from src.config import validate_config

    raw = _root(control={"listen": {}})
    raw[legacy_key] = {}

    errors = validate_config(raw)["errors"]

    assert any(legacy_key in error and "extra_forbidden" in error for error in errors)


def test_mixed_v3_v4_configuration_is_rejected():
    from src.config import validate_config

    raw = _root(job=_job())
    raw.update({"datasource": {"type": "csv"}, "models": []})

    errors = validate_config(raw)["errors"]

    assert any("datasource" in error for error in errors)
    assert any("models" in error for error in errors)


@pytest.mark.parametrize(
    "raw",
    [
        _root(job=_job()),
        _root(gateway=_gateway()),
        _root(control={"listen": {}}),
        _root(job=_job(), gateway=_gateway(), control={"listen": {}}),
    ],
    ids=["job-only", "gateway-only", "control-only", "combined"],
)
def test_component_combinations_are_valid(raw):
    from src.config import validate_config

    assert validate_config(raw)["errors"] == []


def test_at_least_one_component_is_required():
    from src.config import validate_config

    assert validate_config(_root())["errors"]


def test_component_require_helpers_fail_closed():
    from src.config import (
        RootConfig,
        require_control_config,
        require_gateway_config,
        require_job_config,
    )
    from src.models.errors import ConfigError

    config = RootConfig.model_validate(_root(control={"listen": {}}))
    assert require_control_config(config).listen.port == 8790
    with pytest.raises(ConfigError, match="job section"):
        require_job_config(config)
    with pytest.raises(ConfigError, match="gateway section"):
        require_gateway_config(config)


@pytest.mark.parametrize(
    "datasource",
    [
        {
            "type": "excel",
            "input_path": "in.xlsx",
            "output_path": "out.xlsx",
            "engine": "auto",
            "reader": "auto",
            "writer": "auto",
            "require_all_input_fields": True,
        },
        {
            "type": "csv",
            "input_path": "in.csv",
            "output_path": "out.csv",
            "engine": "pandas",
            "require_all_input_fields": True,
        },
        {
            "type": "sqlite",
            "db_path": "data.db",
            "table_name": "tasks",
            "require_all_input_fields": True,
        },
        {
            "type": "mysql",
            "host": "localhost",
            "port": 3306,
            "user": "user",
            "password": "secret",
            "database": "db",
            "table_name": "tasks",
            "pool_size": 2,
            "require_all_input_fields": True,
        },
        {
            "type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "user": "user",
            "password": "secret",
            "database": "db",
            "table_name": "tasks",
            "schema_name": "public",
            "pool_size": 2,
            "require_all_input_fields": True,
        },
        {
            "type": "feishu_bitable",
            "app_id": "app",
            "app_secret": "secret",
            "app_token": "token",
            "table_id": "table",
            "max_retries": 3,
            "qps_limit": 5.0,
            "require_all_input_fields": True,
        },
        {
            "type": "feishu_sheet",
            "app_id": "app",
            "app_secret": "secret",
            "spreadsheet_token": "sheet",
            "sheet_id": "0",
            "max_retries": 3,
            "qps_limit": 5.0,
            "require_all_input_fields": True,
        },
    ],
    ids=["excel", "csv", "sqlite", "mysql", "postgresql", "bitable", "sheet"],
)
def test_datasource_discriminated_union(datasource):
    from src.config import validate_config

    assert validate_config(_root(job=_job(datasource)))["errors"] == []


def test_datasource_union_rejects_wrong_fields_and_unknown_type():
    from src.config import validate_config

    excel = _job(
        {
            "type": "excel",
            "input_path": "in.xlsx",
            "output_path": "out.xlsx",
            "table_name": "not-allowed",
        }
    )
    unknown = _job({"type": "mongodb", "host": "localhost"})

    assert any(
        "table_name" in error for error in validate_config(_root(job=excel))["errors"]
    )
    assert any(
        "union_tag_invalid" in error
        for error in validate_config(_root(job=unknown))["errors"]
    )


def test_columns_and_model_selection_contracts():
    from src.config import validate_config

    empty_columns = _job()
    empty_columns["columns"] = {"extract": [], "write": {}}
    strict = _job()
    strict["model_selection"] = {"mode": "strict"}
    fallback = _job()
    fallback["model_selection"] = {"mode": "fallback_group"}

    assert validate_config(_root(job=empty_columns))["errors"]
    assert any(
        "route_id" in error for error in validate_config(_root(job=strict))["errors"]
    )
    assert any(
        "group" in error for error in validate_config(_root(job=fallback))["errors"]
    )


def test_prompt_rejects_model_and_unknown_nested_key():
    from src.config import validate_config

    job = _job()
    job["prompt"]["model"] = "legacy-model"
    job["retry"] = {"unknown": True}

    errors = validate_config(_root(job=job))["errors"]

    assert any("job.prompt.model" in error for error in errors)
    assert any("job.retry.unknown" in error for error in errors)


def test_routing_profile_allows_only_prompt_and_validation(tmp_path):
    from src.config import validate_config

    allowed = tmp_path / "allowed.yaml"
    allowed.write_text(
        yaml.safe_dump({"prompt": {"temperature": 0.1}}), encoding="utf-8"
    )
    job = _job()
    job["routing"] = {
        "enabled": True,
        "field": "kind",
        "subtasks": [{"match": "a", "profile": allowed.name}],
    }
    config_path = tmp_path / "config.yaml"
    assert (
        validate_config(_root(job=job, tmp_path=tmp_path), config_path)["errors"] == []
    )

    forbidden = tmp_path / "forbidden.yaml"
    forbidden.write_text(
        yaml.safe_dump({"prompt": {"temperature": 0.1}, "datasource": {}}),
        encoding="utf-8",
    )
    job["routing"]["subtasks"][0]["profile"] = forbidden.name
    errors = validate_config(_root(job=job, tmp_path=tmp_path), config_path)["errors"]
    assert any("datasource" in error and "extra_forbidden" in error for error in errors)


def test_missing_routing_profile_is_rejected(tmp_path):
    from src.config import validate_config

    job = _job()
    job["routing"] = {
        "enabled": True,
        "field": "kind",
        "subtasks": [{"match": "a", "profile": "missing.yaml"}],
    }

    errors = validate_config(
        _root(job=job, tmp_path=tmp_path), tmp_path / "config.yaml"
    )["errors"]

    assert any("profile_load_error" in error for error in errors)


def test_gateway_references_aliases_fallbacks_and_worker_count_are_strict():
    from src.config import validate_config

    unknown_channel = _gateway()
    unknown_channel["routes"][0]["channel_id"] = "missing"
    bad_group = _gateway()
    bad_group["fallback_groups"] = {"bad": ["missing"]}
    duplicate_group = _gateway()
    duplicate_group["fallback_groups"] = {"bad": ["route-a", "route-a"]}
    workers = _gateway()
    workers["listen"]["workers"] = 2

    for gateway in (unknown_channel, bad_group, duplicate_group, workers):
        assert validate_config(_root(gateway=gateway))["errors"]


def test_validation_errors_do_not_echo_secret_values():
    from src.config import validate_config

    raw = _root(gateway=_gateway())
    secret = "must-never-appear-in-validation"
    raw["gateway"]["routes"][0]["api_key"] = {"value": secret}

    rendered = "\n".join(validate_config(raw)["errors"])

    assert rendered
    assert secret not in rendered


def test_execution_hash_expands_defaults_and_resolved_profiles(tmp_path):
    from src.config import RootConfig, execution_config_hash

    profile = tmp_path / "profile.yaml"
    profile.write_text("prompt:\n  temperature: 0.1\n", encoding="utf-8")
    job = _job()
    job["routing"] = {
        "enabled": True,
        "field": "kind",
        "subtasks": [{"match": "a", "profile": profile.name}],
    }
    raw = _root(job=job, tmp_path=tmp_path)
    config = RootConfig.model_validate(raw)
    first = execution_config_hash(config, tmp_path / "config.yaml")
    profile.write_text("prompt:\n  temperature: 0.2\n", encoding="utf-8")
    second = execution_config_hash(config, tmp_path / "config.yaml")

    assert len(first) == 64
    assert first != second


def test_raw_yaml_etag_is_distinct_from_execution_hash(tmp_path):
    from src.config import execution_config_hash, load_config
    from src.control.config_api import config_revision

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(_root(control={"listen": {}}, tmp_path=tmp_path)),
        encoding="utf-8",
    )
    config = load_config(path)

    assert config_revision(path) != execution_config_hash(config, path)


def test_compile_job_config_uses_defaults_and_canonical_paths(tmp_path):
    from src.config import RootConfig, compile_job_config

    config = RootConfig.model_validate(_root(job=_job(), tmp_path=tmp_path))
    compiled = compile_job_config(config, tmp_path / "config.yaml")

    assert compiled["datasource"]["type"] == "csv"
    assert compiled["columns_to_extract"] == ["input"]
    assert compiled["retry"]["task_max_attempts"] == {
        "api_error": 4,
        "content_error": 2,
        "system_error": 3,
        "source_error": 3,
    }
    assert compiled["retry"]["model_max_attempts"] == 3


def test_workspace_resolution_rejects_symlink_escape(tmp_path):
    from src.config import RootConfig, resolve_workspace_path
    from src.models.errors import ConfigError

    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "escape").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    config = RootConfig.model_validate(_root(control={"listen": {}}, tmp_path=root))

    with pytest.raises(ConfigError, match="超出允许根目录"):
        resolve_workspace_path(config, "project", "escape/file.yaml")


def test_access_token_priority_and_non_loopback_requirement(monkeypatch):
    from src.config import RootConfig, resolve_access_token
    from src.models.errors import ConfigError

    raw = _root(control={"listen": {}})
    raw["runtime"]["auth"]["token"] = "yaml-token"
    config = RootConfig.model_validate(raw)
    monkeypatch.setenv("DATAFLUX_TOKEN", "env-token")
    assert resolve_access_token(config, host="0.0.0.0").value == "env-token"
    monkeypatch.delenv("DATAFLUX_TOKEN")
    assert resolve_access_token(config, host="0.0.0.0").value == "yaml-token"
    raw["runtime"]["auth"]["token"] = ""
    with pytest.raises(ConfigError, match="runtime.auth.token"):
        resolve_access_token(
            RootConfig.model_validate(raw),
            host="0.0.0.0",
            allow_generate_loopback=True,
        )


def test_redaction_and_token_checker(monkeypatch):
    from src.config import (
        make_token_checker,
        redact_sensitive_text,
        resolve_access_token,
    )

    monkeypatch.delenv("DATAFLUX_TOKEN", raising=False)
    token = resolve_access_token({"runtime": {"auth": {"token": ""}}}, host="127.0.0.1")
    checker = make_token_checker(token)
    assert checker(token.value)
    assert not checker("wrong")
    rendered = redact_sensitive_text(
        "Bearer bearer-secret api_key=key-secret exact-secret",
        known_secrets=("exact-secret",),
    )
    assert "secret" not in rendered
