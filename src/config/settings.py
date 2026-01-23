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
    result = config
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
