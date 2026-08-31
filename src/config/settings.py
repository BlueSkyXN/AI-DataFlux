"""Canonical v4 configuration loading, compilation, and runtime helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from ..models.errors import ConfigError
from .models import (
    ControlConfig,
    GatewayConfig,
    JobConfig,
    RootConfig,
    RoutingProfile,
)


def _read_yaml_mapping(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise ConfigError(f"配置文件不存在: {target}")
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"YAML 解析错误: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("配置文件格式错误: 根节点必须是字典")
    return raw


def _error_path(location: tuple[Any, ...]) -> str:
    result = ""
    for part in location:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += ("." if result else "") + str(part)
    return result or "$"


def _format_validation_errors(exc: ValidationError) -> list[str]:
    """Format validation failures without echoing input values or secrets."""

    errors: list[str] = []
    for item in exc.errors(include_input=False, include_url=False):
        path = _error_path(tuple(item.get("loc", ())))
        code = str(item.get("type", "validation_error"))
        message = str(item.get("msg", "invalid value"))
        errors.append(f"{path} [{code}]: {message}")
    return errors


def _validate_routing_profiles(
    config: RootConfig,
    config_path: str | Path | None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    errors: list[str] = []
    profiles: dict[str, dict[str, Any]] = {}
    if config.job is None or not config.job.routing.enabled:
        return errors, profiles

    base_dir = Path(config_path).resolve().parent if config_path else Path.cwd()
    for index, rule in enumerate(config.job.routing.subtasks):
        profile_path = Path(rule.profile).expanduser()
        if not profile_path.is_absolute():
            profile_path = base_dir / profile_path
        profile_path = profile_path.resolve(strict=False)
        try:
            raw = _read_yaml_mapping(profile_path)
            profile = RoutingProfile.model_validate(raw)
        except ValidationError as exc:
            for message in _format_validation_errors(exc):
                errors.append(f"job.routing.subtasks[{index}].profile.{message}")
            continue
        except ConfigError as exc:
            errors.append(
                f"job.routing.subtasks[{index}].profile [profile_load_error]: {exc.message}"
            )
            continue
        profiles[rule.profile] = profile.model_dump(exclude_none=True)
    return errors, profiles


def load_config(config_path: str | Path) -> RootConfig:
    """Load and strictly validate one canonical v4 RootConfig."""

    target = Path(config_path)
    raw = _read_yaml_mapping(target)
    try:
        config = RootConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError("; ".join(_format_validation_errors(exc))) from exc
    profile_errors, _ = _validate_routing_profiles(config, target)
    if profile_errors:
        raise ConfigError("; ".join(profile_errors))
    logging.info("配置文件 '%s' 加载成功", target)
    return config


def validate_config(
    raw: RootConfig | Mapping[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, list[str]]:
    """Validate RootConfig while preserving the Control/CLI response shape."""

    try:
        config = raw if isinstance(raw, RootConfig) else RootConfig.model_validate(raw)
    except ValidationError as exc:
        return {"errors": _format_validation_errors(exc), "warnings": []}
    except Exception as exc:
        return {
            "errors": [f"$ [config_type]: {type(exc).__name__}: {exc}"],
            "warnings": [],
        }

    profile_errors, _ = _validate_routing_profiles(config, config_path)
    warnings: list[str] = []
    try:
        roots = resolve_workspace_roots(config, config_path)
        for root_id, root_path in roots.items():
            if not root_path.exists():
                warnings.append(
                    f"runtime.workspace.roots.{root_id} 当前不存在: {root_path}"
                )
        state_dir = _resolve_config_relative_path(
            config.runtime.workspace.state_dir, config_path
        )
        if not any(_path_is_within(state_dir, root) for root in roots.values()):
            profile_errors.append(
                "runtime.workspace.state_dir [path_outside_workspace]: "
                "state_dir 必须位于 workspace roots 之一以内"
            )
    except ConfigError as exc:
        profile_errors.append(f"runtime.workspace [workspace_error]: {exc.message}")
    return {"errors": profile_errors, "warnings": warnings}


def require_job_config(config: RootConfig) -> JobConfig:
    if config.job is None:
        raise ConfigError("当前命令要求配置 job section")
    return config.job


def require_gateway_config(config: RootConfig) -> GatewayConfig:
    if config.gateway is None:
        raise ConfigError("当前命令要求配置 gateway section")
    return config.gateway


def require_control_config(config: RootConfig) -> ControlConfig:
    if config.control is None:
        raise ConfigError("当前命令要求配置 control section")
    return config.control


def load_routing_profile(path: str | Path) -> dict[str, Any]:
    """Load a partial routing profile; profiles are not RootConfig documents."""

    raw = _read_yaml_mapping(path)
    try:
        return RoutingProfile.model_validate(raw).model_dump(exclude_none=True)
    except ValidationError as exc:
        raise ConfigError("; ".join(_format_validation_errors(exc))) from exc


def canonical_config_payload(
    config: RootConfig,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return defaults-expanded config plus parsed routing profiles."""

    errors, profiles = _validate_routing_profiles(config, config_path)
    if errors:
        raise ConfigError("; ".join(errors))
    payload = config.model_dump(mode="json", exclude_none=False)
    payload["resolved_routing_profiles"] = profiles
    return payload


def execution_config_hash(
    config: RootConfig,
    config_path: str | Path | None = None,
) -> str:
    """Hash canonical execution semantics, distinct from raw-YAML ETags."""

    encoded = json.dumps(
        canonical_config_payload(config, config_path),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compile_job_config(
    config: RootConfig,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compile typed v4 JobConfig into the existing processing runtime shape."""

    job = require_job_config(config)
    if not job.prompt.template.strip():
        raise ConfigError("job.prompt.template 必须是非空字符串")

    datasource = job.datasource.model_dump(mode="python")
    datasource_type = str(datasource.pop("type"))
    require_all = bool(datasource.pop("require_all_input_fields", True))
    concurrency = job.concurrency.model_dump(mode="python")
    attempts = job.retry.task_max_attempts
    concurrency.update(
        {
            "api_pause_duration": job.retry.api_pause_duration_seconds,
            "api_error_trigger_window": job.retry.api_error_trigger_window_seconds,
            # The pre-H1.1 runtime consumes retry counts. H1.1 switches this
            # call site to total-attempt semantics without changing YAML.
            "retry_limits": {
                "api_error": attempts.api_error - 1,
                "content_error": attempts.content_error - 1,
                "system_error": attempts.system_error - 1,
                "source_error": attempts.source_error - 1,
            },
        }
    )
    datasource_runtime: dict[str, Any] = {
        "type": datasource_type,
        "require_all_input_fields": require_all,
        "concurrency": concurrency,
    }
    if datasource_type in {"excel", "csv"}:
        datasource_runtime["engine"] = datasource.pop("engine")
    if datasource_type == "excel":
        datasource_runtime["excel_reader"] = datasource.pop("reader")
        datasource_runtime["excel_writer"] = datasource.pop("writer")

    routing = job.routing.model_dump(mode="python")
    runtime: dict[str, Any] = {
        "global": {
            "log": config.runtime.log.model_dump(mode="python"),
            "flux_api_url": job.gateway_url,
        },
        "datasource": datasource_runtime,
        datasource_type: datasource,
        "columns_to_extract": list(job.columns.extract),
        "columns_to_write": dict(job.columns.write),
        "model_selection": job.model_selection.model_dump(
            mode="python", exclude_none=True
        ),
        "prompt": job.prompt.model_dump(mode="python"),
        "validation": job.validation.model_dump(mode="python"),
        "routing": routing,
        "token_estimation": config.runtime.token_estimation.model_dump(mode="python"),
        "workspace": config.runtime.workspace.model_dump(mode="python"),
        "scheduler": config.runtime.scheduler.model_dump(mode="python"),
        "server": {
            "token": config.runtime.auth.token,
            "host": (config.control.listen.host if config.control else "127.0.0.1"),
            "control_port": (config.control.listen.port if config.control else 8790),
            "gateway_port": (config.gateway.listen.port if config.gateway else 8787),
        },
        "writeback": job.writeback.model_dump(mode="python"),
        "execution_config_sha256": execution_config_hash(config, config_path),
    }
    return runtime


def compile_gateway_config(config: RootConfig) -> dict[str, Any]:
    """Compile typed v4 GatewayConfig for the current gateway engine."""

    gateway = require_gateway_config(config)
    channels: dict[str, dict[str, Any]] = {}
    for channel_id, channel in gateway.channels.items():
        channel_data = channel.model_dump(mode="python")
        channel_data["timeout"] = channel_data.pop("timeout_seconds")
        channels[channel_id] = channel_data

    models: list[dict[str, Any]] = []
    for route in gateway.routes:
        models.append(
            {
                "id": route.id,
                "name": route.display_name,
                "aliases": list(route.aliases),
                "model": route.upstream_model,
                "channel_id": route.channel_id,
                "api_key": route.api_key,
                "capabilities": list(route.capabilities),
                "weight": route.weight,
                "safe_rps": route.safe_rps,
                "timeout": route.timeout_seconds,
                "temperature": route.temperature,
            }
        )
    return {
        "global": {"log": config.runtime.log.model_dump(mode="python")},
        "server": {"token": config.runtime.auth.token},
        "gateway": gateway.connection_pool.model_dump(mode="python"),
        "gateway_listen": gateway.listen.model_dump(mode="python"),
        "gateway_retry": gateway.retry.model_dump(mode="python"),
        "gateway_affinity": gateway.affinity.model_dump(mode="python"),
        "fallback_groups": {
            name: list(route_ids) for name, route_ids in gateway.fallback_groups.items()
        },
        "channels": channels,
        "models": models,
    }


def _as_root_config(config: RootConfig | Mapping[str, Any]) -> RootConfig:
    if isinstance(config, RootConfig):
        return config
    try:
        return RootConfig.model_validate(config)
    except ValidationError as exc:
        raise ConfigError("; ".join(_format_validation_errors(exc))) from exc


def resolve_workspace_roots(
    config: RootConfig | Mapping[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    root = _as_root_config(config)
    result: dict[str, Path] = {}
    for root_id, raw_path in root.runtime.workspace.roots.items():
        result[root_id] = _resolve_config_relative_path(raw_path, config_path)
    return result


def resolve_workspace_path(
    config: RootConfig | Mapping[str, Any],
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


def _resolve_config_relative_path(
    raw_path: str | Path,
    config_path: str | Path | None,
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


def init_logging(log_config: Mapping[str, Any] | None = None) -> None:
    """Initialize the standard logging subsystem from runtime.log."""

    log_config = log_config or {}
    level_str = str(log_config.get("level", "info")).upper()
    level = getattr(logging, level_str, logging.INFO)
    if log_config.get("format", "text") == "json":
        log_format = (
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"name": "%(name)s", "message": "%(message)s"}'
        )
    else:
        log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

    handlers: list[logging.Handler] = []
    output_type = str(log_config.get("output", "console"))
    if output_type == "file":
        file_path = str(log_config.get("file_path", "./logs/ai_dataflux.log"))
        try:
            log_dir = os.path.dirname(file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            handlers.append(logging.FileHandler(file_path, encoding="utf-8"))
        except OSError as exc:
            print(f"创建日志文件失败: {exc}，回退到控制台", file=sys.stderr)
            output_type = "console"
    if output_type == "console" or not handlers:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for name in ("aiohttp", "asyncio", "urllib3", "mysql.connector"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_nested(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Read nested mappings. Kept as a general utility, not a config merger."""

    result: Any = config
    for key in keys:
        if not isinstance(result, Mapping):
            return default
        result = result.get(key)
        if result is None:
            return default
    return result
