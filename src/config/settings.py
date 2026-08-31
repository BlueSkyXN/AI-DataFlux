"""
配置管理模块

本模块提供 AI-DataFlux 的核心配置功能，包括：
- YAML 配置文件加载与解析
- 默认配置定义
- 日志系统初始化
- 配置工具函数

配置文件结构:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        config.yaml                               │
    ├─────────────────────────────────────────────────────────────────┤
    │ global:                                                          │
    │   flux_api_url: "http://..."     # API 网关地址                  │
    │   log:                                                           │
    │     level: info                  # 日志级别                      │
    │     format: text                 # 日志格式 (text/json)          │
    │     output: console              # 输出目标 (console/file)       │
    │                                                                  │
    │ datasource:                                                      │
    │   type: excel                    # 数据源类型                    │
    │   engine: pandas                 # 数据引擎                      │
    │   concurrency:                                                   │
    │     batch_size: 100              # 批处理大小                    │
    │     retry_limits: {...}          # 重试限制                      │
    │                                                                  │
    │ prompt:                                                          │
    │   template: "..."                # Prompt 模板                   │
    │   required_fields: [...]         # 必需字段                      │
    └─────────────────────────────────────────────────────────────────┘

配置合并策略:
    使用深度合并 (merge_config)，用户配置覆盖默认配置。
    对于嵌套字典，只覆盖指定的键，未指定的键保留默认值。

日志格式:
    - text: 传统文本格式，适合控制台阅读
      格式: "2024-01-01 12:00:00 [INFO] [logger] message"
    - json: JSON 格式，适合日志聚合系统
      格式: {"time": "...", "level": "...", "message": "..."}

函数清单:
    load_config(config_path: str | Path) -> dict[str, Any]
        加载并解析 YAML 配置文件
        输入: 配置文件路径 | 输出: 配置字典
        异常: ConfigError (文件不存在/格式错误/YAML 解析失败)

    init_logging(log_config: dict[str, Any] | None = None) -> None
        初始化 Python 标准日志系统
        输入: 日志配置字典 (level/format/output/file_path/date_format)
        副作用: 配置根日志器, 降低第三方库日志级别

    get_nested(config: dict, *keys: str, default: Any = None) -> Any
        安全获取嵌套字典值 (链式键路径)
        输入: 配置字典 + 键路径 | 输出: 目标值或默认值

    merge_config(base: dict, override: dict) -> dict[str, Any]
        深度递归合并两个配置字典
        输入: 基础配置 + 覆盖配置 | 输出: 合并后的新字典 (不修改原字典)

关键变量:
    DEFAULT_CONFIG: dict[str, Any]
        默认配置字典，包含以下顶层键:
        - global.log: 日志配置 (级别/格式/输出/路径)
        - global.flux_api_url: API 网关地址 (默认 http://127.0.0.1:8787)
        - datasource: 数据源配置 (类型/引擎/并发参数/重试限制)
        - token_estimation: Token 估算配置 (模式/采样/编码)

依赖模块:
    - yaml: YAML 文件解析
    - logging: Python 标准日志库
    - src.models.errors.ConfigError: 配置错误异常类

使用示例:
    # 加载配置
    config = load_config("config.yaml")

    # 初始化日志
    init_logging(config.get("global", {}).get("log"))

    # 获取嵌套配置
    batch_size = get_nested(config, "datasource", "concurrency", "batch_size", default=100)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from ..models.errors import ConfigError

# 默认配置值
# 用户配置会深度合并到此默认配置上
DEFAULT_CONFIG: dict[str, Any] = {
    "global": {
        "log": {
            "level": "info",
            "format": "text",
            "output": "console",
            "file_path": "./logs/ai_dataflux.log",
            "date_format": "%Y-%m-%d %H:%M:%S",
        },
        "flux_api_url": "http://127.0.0.1:8787",
    },
    "datasource": {
        "type": "excel",
        "engine": "pandas",  # 数据引擎: pandas | polars
        "require_all_input_fields": True,
        "concurrency": {
            "batch_size": 100,
            "max_in_flight": 100,
            "save_interval": 300,
            "shard_size": 10000,
            "min_shard_size": 1000,
            "max_shard_size": 50000,
            "api_pause_duration": 2.0,
            "api_error_trigger_window": 2.0,
            "max_connections": 1000,
            "max_connections_per_host": 0,
            "retry_limits": {
                "api_error": 3,
                "content_error": 1,
                "system_error": 2,
            },
        },
    },
    "token_estimation": {
        "mode": "io",
        "sample_size": -1,
        "encoding": "o200k_base",
    },
    "workspace": {
        "roots": {"project": "."},
        "state_dir": "./.dataflux/jobs",
    },
    "scheduler": {
        "max_active_jobs": "auto",
        "cpu_high_watermark": 85,
        "memory_high_watermark": 80,
        "min_free_memory_mb": 512,
        "sample_interval_seconds": 2,
    },
    "server": {
        "host": "127.0.0.1",
        "control_port": 8790,
        "gateway_port": 8787,
        "token": "",
    },
}

SUPPORTED_DATASOURCE_TYPES = {
    "excel",
    "csv",
    "mysql",
    "postgresql",
    "sqlite",
    "feishu_bitable",
    "feishu_sheet",
}
SUPPORTED_ENGINES = {"auto", "pandas", "polars"}
SUPPORTED_EXCEL_READERS = {"auto", "openpyxl", "calamine"}
SUPPORTED_EXCEL_WRITERS = {"auto", "openpyxl", "xlsxwriter"}
SUPPORTED_LOG_LEVELS = {"debug", "info", "warning", "error"}
SUPPORTED_LOG_FORMATS = {"text", "json"}
SUPPORTED_LOG_OUTPUTS = {"console", "file"}
ALLOWED_TOP_LEVEL_KEYS = {
    "global",
    "gateway",
    "datasource",
    "mysql",
    "excel",
    "postgresql",
    "sqlite",
    "csv",
    "feishu",
    "columns_to_extract",
    "columns_to_write",
    "validation",
    "models",
    "channels",
    "prompt",
    "token_estimation",
    "routing",
    "workspace",
    "scheduler",
    "server",
}

MODEL_CAPABILITIES = {
    "chat_completions",
    "responses",
    "stream",
    "multimodal",
    "tools",
    "n",
    "json_schema",
    "logprobs",
    "previous_response_id",
}

_SECTION_KEYS: dict[str, set[str]] = {
    "global": {"log", "flux_api_url"},
    "global.log": {"level", "format", "output", "file_path", "date_format"},
    "gateway": {"max_connections", "max_connections_per_host"},
    "datasource": {
        "type",
        "engine",
        "excel_reader",
        "excel_writer",
        "require_all_input_fields",
        "concurrency",
    },
    "datasource.concurrency": {
        "batch_size",
        "max_in_flight",
        "save_interval",
        "shard_size",
        "min_shard_size",
        "max_shard_size",
        "api_pause_duration",
        "api_error_trigger_window",
        "max_connections",
        "max_connections_per_host",
        "retry_limits",
    },
    "datasource.concurrency.retry_limits": {
        "api_error",
        "content_error",
        "system_error",
    },
    "workspace": {"roots", "state_dir"},
    "scheduler": {
        "max_active_jobs",
        "cpu_high_watermark",
        "memory_high_watermark",
        "min_free_memory_mb",
        "sample_interval_seconds",
    },
    "server": {"host", "control_port", "gateway_port", "token"},
    "token_estimation": {"mode", "sample_size", "encoding"},
    "prompt": {
        "required_fields",
        "use_json_schema",
        "temperature",
        "temperature_override",
        "system_prompt",
        "template",
    },
    "validation": {"enabled", "field_rules"},
    "excel": {"input_path", "output_path"},
    "csv": {"input_path", "output_path"},
    "sqlite": {"db_path", "table_name"},
    "mysql": {
        "host",
        "port",
        "user",
        "password",
        "database",
        "table_name",
        "pool_size",
    },
    "postgresql": {
        "host",
        "port",
        "user",
        "password",
        "database",
        "table_name",
        "schema_name",
        "pool_size",
    },
    "feishu": {
        "app_id",
        "app_secret",
        "app_token",
        "table_id",
        "spreadsheet_token",
        "sheet_id",
        "max_retries",
        "qps_limit",
    },
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    加载 YAML 配置文件

    从指定路径加载 YAML 格式的配置文件并解析为 Python 字典。

    Args:
        config_path: 配置文件路径 (相对或绝对路径)

    Returns:
        配置字典

    Raises:
        ConfigError: 配置文件不存在或格式错误

    Note:
        此函数只负责加载和解析，不进行与默认配置的合并。
        合并操作由调用方根据需要执行。
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"配置文件不存在: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ConfigError("配置文件格式错误: 根节点必须是字典")

        logging.info(f"配置文件 '{config_path}' 加载成功")
        return config

    except yaml.YAMLError as e:
        raise ConfigError(f"YAML 解析错误: {e}") from e
    except Exception as e:
        raise ConfigError(f"加载配置文件失败: {e}") from e


def validate_config(
    config: dict[str, Any], config_path: str | Path | None = None
) -> dict[str, list[str]]:
    """
    校验配置结构和本地可验证的运行前置条件。

    本函数不连接数据库、不访问远端 API，也不要求输入数据文件已经存在；
    它只拦截会在启动初始化阶段立即失败的配置错误，并提示已知会被忽略的旧键。
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(config, dict):
        return {"errors": ["配置根节点必须是字典"], "warnings": warnings}

    unknown_top_level = sorted(set(config) - ALLOWED_TOP_LEVEL_KEYS)
    if unknown_top_level:
        errors.append(f"发现未知顶层配置键: {unknown_top_level}")

    _validate_global_config(config.get("global", {}), errors, warnings)
    _validate_known_section(config.get("global", {}), "global", errors)
    global_map = config.get("global", {})
    if isinstance(global_map, dict):
        _validate_known_section(global_map.get("log", {}), "global.log", errors)
    datasource = _ensure_mapping(config.get("datasource", {}), "datasource", errors)
    datasource_type = (
        _normalize_nonempty_str(datasource.get("type")) or "excel"
    ).lower()
    if datasource_type not in SUPPORTED_DATASOURCE_TYPES:
        errors.append(
            f"datasource.type 不支持: {datasource_type!r}，"
            f"可选: {sorted(SUPPORTED_DATASOURCE_TYPES)}"
        )

    _validate_datasource_options(datasource, errors, warnings)
    _validate_known_section(datasource, "datasource", errors)
    if isinstance(datasource.get("concurrency"), dict):
        concurrency = datasource["concurrency"]
        _validate_known_section(concurrency, "datasource.concurrency", errors)
        _validate_known_section(
            concurrency.get("retry_limits", {}),
            "datasource.concurrency.retry_limits",
            errors,
        )
    _validate_columns_config(config, errors)
    _validate_prompt_config(config.get("prompt", {}), "prompt", errors)
    _validate_validation_config(config.get("validation", {}), "validation", errors)
    _validate_data_source_section(config, datasource_type, errors)
    _validate_routing_config(config, config_path, errors)
    _validate_runtime_sections(config, config_path, errors, warnings)

    for section in (
        "gateway",
        "workspace",
        "scheduler",
        "server",
        "token_estimation",
        "prompt",
        "validation",
        "excel",
        "csv",
        "sqlite",
        "mysql",
        "postgresql",
        "feishu",
    ):
        if section in config:
            _validate_known_section(config.get(section), section, errors)

    return {"errors": errors, "warnings": warnings}


def _normalize_nonempty_str(value: Any) -> str | None:
    """将配置值规范化为非空字符串。"""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text if text else None


def _ensure_mapping(value: Any, path: str, errors: list[str]) -> dict[str, Any]:
    """确认配置节点为 dict；否则记录错误并返回空 dict。"""
    if value is None:
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path} 必须是字典")
        return {}
    return value


def _validate_known_section(value: Any, path: str, errors: list[str]) -> None:
    """Reject unknown keys in canonical v3.2 configuration sections."""

    if value is None:
        return
    if not isinstance(value, dict):
        return
    allowed = _SECTION_KEYS.get(path)
    if allowed is None:
        return
    unknown = sorted(set(value) - allowed)
    if unknown:
        errors.append(f"{path} 包含未知或已移除的配置键: {unknown}")


def _validate_global_config(
    global_cfg: Any, errors: list[str], warnings: list[str]
) -> None:
    global_map = _ensure_mapping(global_cfg, "global", errors)
    log_cfg = _ensure_mapping(global_map.get("log", {}), "global.log", errors)

    level = _normalize_nonempty_str(log_cfg.get("level"))
    if level is not None and level.lower() not in SUPPORTED_LOG_LEVELS:
        errors.append(
            f"global.log.level 不支持: {level!r}，可选: {sorted(SUPPORTED_LOG_LEVELS)}"
        )

    fmt = _normalize_nonempty_str(log_cfg.get("format"))
    if fmt is not None and fmt.lower() not in SUPPORTED_LOG_FORMATS:
        errors.append(
            f"global.log.format 不支持: {fmt!r}，可选: {sorted(SUPPORTED_LOG_FORMATS)}"
        )

    output = _normalize_nonempty_str(log_cfg.get("output"))
    if output is not None and output.lower() not in SUPPORTED_LOG_OUTPUTS:
        errors.append(
            f"global.log.output 不支持: {output!r}，可选: {sorted(SUPPORTED_LOG_OUTPUTS)}"
        )

    if output == "file" and not _normalize_nonempty_str(log_cfg.get("file_path")):
        warnings.append("global.log.output=file 但未设置 file_path，将使用默认日志路径")


def _validate_datasource_options(
    datasource: dict[str, Any], errors: list[str], warnings: list[str]
) -> None:
    engine = _normalize_nonempty_str(datasource.get("engine")) or "auto"
    if engine not in SUPPORTED_ENGINES:
        errors.append(
            f"datasource.engine 不支持: {engine!r}，可选: {sorted(SUPPORTED_ENGINES)}"
        )

    reader = _normalize_nonempty_str(datasource.get("excel_reader")) or "auto"
    if reader not in SUPPORTED_EXCEL_READERS:
        errors.append(
            f"datasource.excel_reader 不支持: {reader!r}，"
            f"可选: {sorted(SUPPORTED_EXCEL_READERS)}"
        )

    writer = _normalize_nonempty_str(datasource.get("excel_writer")) or "auto"
    if writer not in SUPPORTED_EXCEL_WRITERS:
        errors.append(
            f"datasource.excel_writer 不支持: {writer!r}，"
            f"可选: {sorted(SUPPORTED_EXCEL_WRITERS)}"
        )

    if "require_all_input_fields" in datasource and not isinstance(
        datasource["require_all_input_fields"], bool
    ):
        errors.append("datasource.require_all_input_fields 必须是布尔值")

    concurrency = _ensure_mapping(
        datasource.get("concurrency", {}), "datasource.concurrency", errors
    )
    _validate_positive_int(concurrency, "batch_size", "datasource.concurrency", errors)
    _validate_positive_int(
        concurrency, "max_in_flight", "datasource.concurrency", errors
    )
    _validate_positive_int(
        concurrency, "save_interval", "datasource.concurrency", errors
    )
    _validate_positive_int(concurrency, "shard_size", "datasource.concurrency", errors)
    _validate_positive_int(
        concurrency, "min_shard_size", "datasource.concurrency", errors
    )
    _validate_positive_int(
        concurrency, "max_shard_size", "datasource.concurrency", errors
    )
    _validate_positive_int(
        concurrency, "max_connections", "datasource.concurrency", errors
    )
    _validate_nonnegative_int(
        concurrency, "max_connections_per_host", "datasource.concurrency", errors
    )
    _validate_positive_number(
        concurrency, "api_pause_duration", "datasource.concurrency", errors
    )
    _validate_positive_number(
        concurrency, "api_error_trigger_window", "datasource.concurrency", errors
    )

    min_shard = concurrency.get("min_shard_size")
    max_shard = concurrency.get("max_shard_size")
    if (
        isinstance(min_shard, int)
        and isinstance(max_shard, int)
        and min_shard > max_shard
    ):
        errors.append("datasource.concurrency.min_shard_size 不能大于 max_shard_size")

    retry_limits = _ensure_mapping(
        concurrency.get("retry_limits", {}),
        "datasource.concurrency.retry_limits",
        errors,
    )
    for key in ("api_error", "content_error", "system_error"):
        _validate_nonnegative_int(
            retry_limits, key, "datasource.concurrency.retry_limits", errors
        )

    ignored_keys = sorted(
        set(concurrency).intersection({"max_workers", "retry_times", "backoff_factor"})
    )
    for key in ignored_keys:
        errors.append(
            f"datasource.concurrency.{key} 已移除，请使用 batch_size/max_in_flight/retry_limits"
        )


def _validate_columns_config(config: dict[str, Any], errors: list[str]) -> None:
    columns_to_extract = config.get("columns_to_extract")
    if not isinstance(columns_to_extract, list) or not columns_to_extract:
        errors.append("columns_to_extract 必须是非空列表")
    elif not all(_normalize_nonempty_str(item) for item in columns_to_extract):
        errors.append("columns_to_extract 中的每一项都必须是非空字符串")

    columns_to_write = config.get("columns_to_write")
    if not isinstance(columns_to_write, dict) or not columns_to_write:
        errors.append("columns_to_write 必须是非空字典")
    else:
        invalid_items = [
            key
            for key, value in columns_to_write.items()
            if not _normalize_nonempty_str(key) or not _normalize_nonempty_str(value)
        ]
        if invalid_items:
            errors.append(
                f"columns_to_write 的键和值都必须是非空字符串，异常键: {invalid_items}"
            )


def _validate_prompt_config(
    value: Any, path: str, errors: list[str], *, require_template: bool = True
) -> None:
    prompt = _ensure_mapping(value, path, errors)

    template = _normalize_nonempty_str(prompt.get("template"))
    if require_template:
        if not template:
            errors.append(f"{path}.template 必须是非空字符串")
    elif "template" in prompt and not template:
        errors.append(f"{path}.template 必须是非空字符串")

    if "required_fields" in prompt:
        required_fields = prompt["required_fields"]
        if not isinstance(required_fields, list):
            errors.append(f"{path}.required_fields 必须是列表")
        elif not all(_normalize_nonempty_str(item) for item in required_fields):
            errors.append(f"{path}.required_fields 中的每一项都必须是非空字符串")

    if "use_json_schema" in prompt and not isinstance(prompt["use_json_schema"], bool):
        errors.append(f"{path}.use_json_schema 必须是布尔值")
    if "temperature_override" in prompt and not isinstance(
        prompt["temperature_override"], bool
    ):
        errors.append(f"{path}.temperature_override 必须是布尔值")
    if "temperature" in prompt:
        temperature = prompt["temperature"]
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not 0 <= temperature <= 2
        ):
            errors.append(f"{path}.temperature 必须在 0 到 2 之间")


def _validate_validation_config(value: Any, path: str, errors: list[str]) -> None:
    validation = _ensure_mapping(value, path, errors)
    if "enabled" in validation and not isinstance(validation["enabled"], bool):
        errors.append(f"{path}.enabled 必须是布尔值")

    if "field_rules" not in validation:
        return

    field_rules = validation["field_rules"]
    if not isinstance(field_rules, dict):
        errors.append(f"{path}.field_rules 必须是字典")
        return

    invalid_rules = [
        key for key, values in field_rules.items() if not isinstance(values, list)
    ]
    if invalid_rules:
        errors.append(
            f"{path}.field_rules 的规则值必须是列表，异常字段: {invalid_rules}"
        )


def _validate_data_source_section(
    config: dict[str, Any], datasource_type: str, errors: list[str]
) -> None:
    if datasource_type in {"excel", "csv"}:
        section = _ensure_mapping(
            config.get(datasource_type, {}), datasource_type, errors
        )
        if not _normalize_nonempty_str(section.get("input_path")):
            errors.append(f"{datasource_type}.input_path 是必填项")
        return

    if datasource_type == "sqlite":
        sqlite = _ensure_mapping(config.get("sqlite", {}), "sqlite", errors)
        for key in ("db_path", "table_name"):
            if not _normalize_nonempty_str(sqlite.get(key)):
                errors.append(f"sqlite.{key} 是必填项")
        return

    if datasource_type in {"mysql", "postgresql"}:
        section = _ensure_mapping(
            config.get(datasource_type, {}), datasource_type, errors
        )
        for key in ("host", "user", "password", "database", "table_name"):
            if not _normalize_nonempty_str(section.get(key)):
                errors.append(f"{datasource_type}.{key} 是必填项")
        return

    if datasource_type in {"feishu_bitable", "feishu_sheet"}:
        feishu = _ensure_mapping(config.get("feishu", {}), "feishu", errors)
        for key in ("app_id", "app_secret"):
            if not _normalize_nonempty_str(feishu.get(key)):
                errors.append(f"feishu.{key} 是必填项")

        if datasource_type == "feishu_bitable":
            app_token = _normalize_nonempty_str(feishu.get("app_token"))
            table_id = _normalize_nonempty_str(feishu.get("table_id"))
            if app_token is None:
                errors.append("feishu_bitable 需要 feishu.app_token")
            if table_id is None:
                errors.append("feishu_bitable 需要 feishu.table_id")
        else:
            spreadsheet_token = _normalize_nonempty_str(feishu.get("spreadsheet_token"))
            sheet_id = _normalize_nonempty_str(feishu.get("sheet_id"))
            if spreadsheet_token is None:
                errors.append("feishu_sheet 需要 feishu.spreadsheet_token")
            if sheet_id is None:
                errors.append("feishu_sheet 需要 feishu.sheet_id")


def _validate_runtime_sections(
    config: dict[str, Any],
    config_path: str | Path | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    gateway = _ensure_mapping(config.get("gateway", {}), "gateway", errors)
    _validate_positive_int(gateway, "max_connections", "gateway", errors)
    _validate_positive_int(gateway, "max_connections_per_host", "gateway", errors)

    token = _ensure_mapping(
        config.get("token_estimation", {}), "token_estimation", errors
    )
    if "tiktoken_model" in token:
        errors.append(
            "token_estimation.tiktoken_model 已移除，请使用 token_estimation.encoding"
        )
    mode = token.get("mode", "io")
    if mode not in {"in", "out", "io"}:
        errors.append("token_estimation.mode 必须是 in、out 或 io")
    sample_size = token.get("sample_size", -1)
    if (
        not isinstance(sample_size, int)
        or isinstance(sample_size, bool)
        or sample_size == 0
        or sample_size < -1
    ):
        errors.append("token_estimation.sample_size 必须为 -1 或正整数")
    if not _normalize_nonempty_str(token.get("encoding", "o200k_base")):
        errors.append("token_estimation.encoding 必须是非空字符串")

    scheduler = _ensure_mapping(config.get("scheduler", {}), "scheduler", errors)
    max_jobs = scheduler.get("max_active_jobs", "auto")
    if max_jobs != "auto" and (
        not isinstance(max_jobs, int) or isinstance(max_jobs, bool) or max_jobs <= 0
    ):
        errors.append("scheduler.max_active_jobs 必须是 auto 或正整数")
    for key in ("cpu_high_watermark", "memory_high_watermark"):
        value = scheduler.get(key)
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
            or value > 100
        ):
            errors.append(f"scheduler.{key} 必须在 0 到 100 之间")
    _validate_nonnegative_int(scheduler, "min_free_memory_mb", "scheduler", errors)
    _validate_positive_number(scheduler, "sample_interval_seconds", "scheduler", errors)

    server = _ensure_mapping(config.get("server", {}), "server", errors)
    if "host" in server and not _normalize_nonempty_str(server.get("host")):
        errors.append("server.host 必须是非空字符串")
    for key in ("control_port", "gateway_port"):
        value = server.get(key)
        if value is not None and (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= 65535
        ):
            errors.append(f"server.{key} 必须是 1-65535 的整数")

    _validate_workspace(config, config_path, errors, warnings)
    _validate_models_and_channels(config, errors)


def _validate_workspace(
    config: dict[str, Any],
    config_path: str | Path | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    workspace = _ensure_mapping(config.get("workspace", {}), "workspace", errors)
    roots = workspace.get("roots", {"project": "."})
    if not isinstance(roots, dict) or not roots:
        errors.append("workspace.roots 必须是非空的名称到路径映射")
        return
    for root_id, raw_path in roots.items():
        if not isinstance(root_id, str) or not re_full_identifier(root_id):
            errors.append(f"workspace.roots 名称非法: {root_id!r}")
        if not _normalize_nonempty_str(raw_path):
            errors.append(f"workspace.roots.{root_id} 必须是非空路径")

    try:
        resolved = resolve_workspace_roots(config, config_path)
    except ConfigError as exc:
        errors.append(str(exc))
        return
    for root_id, root_path in resolved.items():
        if not root_path.exists():
            warnings.append(f"workspace.roots.{root_id} 当前不存在: {root_path}")

    raw_state_dir = workspace.get("state_dir", "./.dataflux/jobs")
    state_path = _resolve_config_relative_path(raw_state_dir, config_path)
    if not any(_path_is_within(state_path, root) for root in resolved.values()):
        errors.append("workspace.state_dir 必须位于 workspace.roots 之一以内")


def _validate_models_and_channels(config: dict[str, Any], errors: list[str]) -> None:
    channels = config.get("channels", {})
    if not isinstance(channels, dict):
        errors.append("channels 必须是字典")
        channels = {}
    channel_ids: set[str] = set()
    channel_endpoints: dict[str, set[str]] = {}
    for raw_channel_id, raw_channel in channels.items():
        channel_id = str(raw_channel_id).strip()
        if not channel_id:
            errors.append("channel ID 必须是非空字符串")
            continue
        if not isinstance(raw_channel_id, str):
            errors.append(f"channel ID 必须使用字符串: {raw_channel_id!r}")
        channel_ids.add(channel_id)
        channel = _ensure_mapping(raw_channel, f"channels.{channel_id}", errors)
        allowed = {
            "name",
            "base_url",
            "endpoints",
            "timeout",
            "proxy",
            "ssl_verify",
            "ip_pool",
        }
        unknown = sorted(set(channel) - allowed)
        if unknown:
            errors.append(f"channels.{channel_id} 包含未知或已移除的键: {unknown}")
        if "api_path" in channel:
            errors.append(f"channels.{channel_id}.api_path 已移除，请使用 endpoints")
        if not _normalize_nonempty_str(channel.get("base_url")):
            errors.append(f"channels.{channel_id}.base_url 必须是非空字符串")
        _validate_positive_int(
            channel,
            "timeout",
            f"channels.{channel_id}",
            errors,
        )
        endpoints = _ensure_mapping(
            channel.get("endpoints", {}), f"channels.{channel_id}.endpoints", errors
        )
        endpoint_unknown = sorted(set(endpoints) - {"chat_completions", "responses"})
        if endpoint_unknown:
            errors.append(
                f"channels.{channel_id}.endpoints 包含未知键: {endpoint_unknown}"
            )
        for endpoint in ("chat_completions", "responses"):
            if endpoint in endpoints and not _normalize_nonempty_str(
                endpoints.get(endpoint)
            ):
                errors.append(
                    f"channels.{channel_id}.endpoints.{endpoint} 必须是非空字符串"
                )
        channel_endpoints[channel_id] = {
            endpoint
            for endpoint in ("chat_completions", "responses")
            if _normalize_nonempty_str(endpoints.get(endpoint))
        }

    models = config.get("models", [])
    if not isinstance(models, list):
        errors.append("models 必须是列表")
        return
    seen_ids: set[str] = set()
    alias_owners: dict[str, int] = {}
    allowed_model_keys = {
        "id",
        "name",
        "model",
        "channel_id",
        "api_key",
        "timeout",
        "weight",
        "temperature",
        "safe_rps",
        "capabilities",
    }
    for index, raw_model in enumerate(models):
        model = _ensure_mapping(raw_model, f"models[{index}]", errors)
        unknown = sorted(set(model) - allowed_model_keys)
        if unknown:
            errors.append(f"models[{index}] 包含未知或已移除的键: {unknown}")
        for key in ("id", "name", "model", "channel_id"):
            if not isinstance(model.get(key), str) or not model[key].strip():
                errors.append(f"models[{index}].{key} 必须是非空字符串")
        model_id = model.get("id")
        if isinstance(model_id, str):
            if model_id in seen_ids:
                errors.append(f"models[{index}].id 重复: {model_id}")
            seen_ids.add(model_id)
        for alias_key in ("id", "name", "model"):
            alias = model.get(alias_key)
            if not isinstance(alias, str) or not alias.strip():
                continue
            normalized_alias = alias.strip()
            previous_index = alias_owners.get(normalized_alias)
            if previous_index is not None and previous_index != index:
                errors.append(
                    f"models[{index}].{alias_key} alias 与 models[{previous_index}] 冲突: "
                    f"{normalized_alias}"
                )
            else:
                alias_owners[normalized_alias] = index
        model_channel_id = model.get("channel_id")
        if isinstance(model_channel_id, str) and model_channel_id not in channel_ids:
            errors.append(f"models[{index}].channel_id 不存在: {model_channel_id}")
        _validate_positive_int(model, "timeout", f"models[{index}]", errors)
        _validate_nonnegative_int(model, "weight", f"models[{index}]", errors)
        _validate_positive_number(model, "safe_rps", f"models[{index}]", errors)
        if "temperature" in model:
            temperature = model["temperature"]
            if (
                not isinstance(temperature, (int, float))
                or isinstance(temperature, bool)
                or not 0 <= temperature <= 2
            ):
                errors.append(f"models[{index}].temperature 必须在 0 到 2 之间")
        capabilities = model.get("capabilities")
        if not isinstance(capabilities, list) or not capabilities:
            errors.append(f"models[{index}].capabilities 必须是非空列表")
        else:
            if not all(
                isinstance(value, str) and value.strip() for value in capabilities
            ):
                errors.append(
                    f"models[{index}].capabilities 中的每一项都必须是非空字符串"
                )
            invalid = sorted(
                {str(value) for value in capabilities} - MODEL_CAPABILITIES
            )
            if invalid:
                errors.append(f"models[{index}].capabilities 不支持: {invalid}")
            normalized = {str(value) for value in capabilities}
            if len(normalized) != len(capabilities):
                errors.append(f"models[{index}].capabilities 不得包含重复项")
            endpoint_capabilities = normalized.intersection(
                {"chat_completions", "responses"}
            )
            if not endpoint_capabilities:
                errors.append(
                    f"models[{index}].capabilities 必须包含 chat_completions 或 responses"
                )
            available_endpoints = channel_endpoints.get(str(model_channel_id), set())
            missing_endpoints = sorted(endpoint_capabilities - available_endpoints)
            if missing_endpoints:
                errors.append(
                    f"models[{index}] 声明的 endpoint capability 未在 channel 中配置: "
                    f"{missing_endpoints}"
                )
            if "previous_response_id" in normalized and "responses" not in normalized:
                errors.append(
                    f"models[{index}].capabilities 使用 previous_response_id 时必须包含 responses"
                )


def re_full_identifier(value: str) -> bool:
    return bool(value) and value.replace("_", "a").replace("-", "a").isalnum()


def _resolve_config_relative_path(
    raw_path: str | Path, config_path: str | Path | None
) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        base = Path(config_path).resolve().parent if config_path else Path.cwd()
        path = base / path
    return path.resolve(strict=False)


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def resolve_workspace_roots(
    config: dict[str, Any], config_path: str | Path | None = None
) -> dict[str, Path]:
    workspace = config.get("workspace", {})
    roots = workspace.get("roots", {"project": "."})
    if not isinstance(roots, dict) or not roots:
        raise ConfigError("workspace.roots 必须是非空字典")
    result: dict[str, Path] = {}
    for root_id, raw_path in roots.items():
        if not isinstance(root_id, str) or not re_full_identifier(root_id):
            raise ConfigError(f"workspace root 名称非法: {root_id!r}")
        if not _normalize_nonempty_str(raw_path):
            raise ConfigError(f"workspace.roots.{root_id} 路径无效")
        result[root_id] = _resolve_config_relative_path(raw_path, config_path)
    return result


def resolve_workspace_path(
    config: dict[str, Any],
    root_id: str,
    relative_path: str = ".",
    config_path: str | Path | None = None,
) -> Path:
    roots = resolve_workspace_roots(config, config_path)
    if root_id not in roots:
        raise ConfigError(f"未知 workspace root: {root_id}")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ConfigError("workspace relative_path 不能是绝对路径")
    resolved = (roots[root_id] / relative).resolve(strict=False)
    if not _path_is_within(resolved, roots[root_id]):
        raise ConfigError("workspace 路径超出允许根目录")
    return resolved


def _validate_routing_config(
    config: dict[str, Any], config_path: str | Path | None, errors: list[str]
) -> None:
    routing = _ensure_mapping(config.get("routing", {}), "routing", errors)
    enabled = routing.get("enabled", False)
    if not isinstance(enabled, bool):
        errors.append("routing.enabled 必须是布尔值")
        return
    if not enabled:
        return

    field = _normalize_nonempty_str(routing.get("field"))
    if not field:
        errors.append("routing.enabled=true 时必须配置 routing.field")

    subtasks = routing.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        errors.append("routing.enabled=true 时 routing.subtasks 必须是非空列表")
        return

    base_dir = Path(config_path).parent if config_path is not None else Path.cwd()
    for idx, subtask in enumerate(subtasks):
        if not isinstance(subtask, dict):
            errors.append(f"routing.subtasks[{idx}] 必须是字典")
            continue

        if "match" not in subtask:
            errors.append(f"routing.subtasks[{idx}] 必须包含 match")

        profile = _normalize_nonempty_str(subtask.get("profile"))
        if profile is None:
            errors.append(f"routing.subtasks[{idx}] 必须包含非空 profile")
            continue

        profile_path = Path(profile)
        if not profile_path.is_absolute():
            profile_path = base_dir / profile_path

        try:
            profile_config = load_config(profile_path)
        except Exception as exc:
            errors.append(f"routing.subtasks[{idx}].profile 加载失败: {exc}")
            continue

        unknown_keys = sorted(set(profile_config) - {"prompt", "validation"})
        if unknown_keys:
            errors.append(
                f"routing.subtasks[{idx}].profile 仅允许 prompt/validation，"
                f"发现非法键: {unknown_keys}"
            )
            continue

        if "prompt" in profile_config:
            _validate_prompt_config(
                profile_config["prompt"],
                f"routing.subtasks[{idx}].profile.prompt",
                errors,
                require_template=False,
            )
        if "validation" in profile_config:
            _validate_validation_config(
                profile_config["validation"],
                f"routing.subtasks[{idx}].profile.validation",
                errors,
            )


def _validate_positive_int(
    mapping: dict[str, Any], key: str, path: str, errors: list[str]
) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{path}.{key} 必须是大于 0 的整数")


def _validate_nonnegative_int(
    mapping: dict[str, Any], key: str, path: str, errors: list[str]
) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(f"{path}.{key} 必须是大于等于 0 的整数")


def _validate_positive_number(
    mapping: dict[str, Any], key: str, path: str, errors: list[str]
) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        errors.append(f"{path}.{key} 必须是大于 0 的数字")


def init_logging(log_config: dict[str, Any] | None = None) -> None:
    """
    初始化日志系统

    配置 Python 标准日志库，支持控制台和文件输出，支持 text 和 json 两种格式。

    Args:
        log_config: 日志配置字典，包含以下可选键:
            - level: 日志级别 (debug/info/warning/error)
            - format: 日志格式 (text/json)
            - output: 输出目标 (console/file)
            - file_path: 日志文件路径 (当 output=file 时)
            - date_format: 日期格式

    日志级别映射:
        debug → DEBUG (10)
        info → INFO (20)
        warning → WARNING (30)
        error → ERROR (40)

    第三方库日志:
        自动将 aiohttp, asyncio, urllib3, mysql.connector 等
        库的日志级别设为 WARNING，减少干扰。
    """
    if log_config is None:
        log_config = {}

    # 解析配置
    level_str = log_config.get("level", "info").upper()
    level = getattr(logging, level_str, logging.INFO)

    # 选择日志格式
    log_format_type = log_config.get("format", "text")
    if log_format_type == "json":
        log_format = (
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"name": "%(name)s", "message": "%(message)s"}'
        )
    else:
        log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

    date_format = log_config.get("date_format", "%Y-%m-%d %H:%M:%S")
    output_type = log_config.get("output", "console")

    # 创建处理器
    handlers: list[logging.Handler] = []

    if output_type == "file":
        file_path = log_config.get("file_path", "./logs/ai_dataflux.log")
        try:
            log_dir = os.path.dirname(file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            handlers.append(file_handler)
            print(f"日志将输出到文件: {file_path}")
        except Exception as e:
            print(f"创建日志文件失败: {e}，回退到控制台", file=sys.stderr)
            output_type = "console"

    if output_type == "console" or not handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        handlers.append(console_handler)

    # 配置根日志器
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,  # 覆盖已有配置
    )

    # 降低第三方库的日志级别，减少干扰
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # 尝试降低 MySQL 日志级别 (如果可用)
    try:
        logging.getLogger("mysql.connector").setLevel(logging.WARNING)
    except Exception:
        pass

    logging.info(f"日志系统初始化完成 | 级别: {level_str}, 输出: {output_type}")


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    安全获取嵌套配置值

    遍历键路径获取嵌套字典中的值，任意一级不存在则返回默认值。

    Args:
        config: 配置字典
        *keys: 键路径 (可变参数)
        default: 键不存在时返回的默认值

    Returns:
        配置值或默认值

    Example:
        >>> config = {"a": {"b": {"c": 1}}}
        >>> get_nested(config, "a", "b", "c")
        1
        >>> get_nested(config, "a", "x", default=0)
        0
    """
    result: Any = config
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
        else:
            return default
        if result is None:
            return default
    return result


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    深度合并配置字典

    递归合并两个字典，override 中的值覆盖 base 中的同名键。
    对于嵌套字典，会递归合并而非直接替换。

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        合并后的配置 (新字典，不修改原始配置)

    Example:
        >>> base = {"a": {"b": 1, "c": 2}}
        >>> override = {"a": {"b": 10}}
        >>> merge_config(base, override)
        {"a": {"b": 10, "c": 2}}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value

    return result
