"""
配置管理模块

本模块提供 AI-DataFlux 的配置管理功能，包括配置文件加载、
默认配置定义和日志初始化。

导出清单 (均来自 settings.py):
    函数:
        load_config(config_path: str | Path) -> dict[str, Any]
            加载 YAML 配置文件并解析为字典
        init_logging(log_config: dict | None) -> None
            初始化日志系统 (支持 text/json 格式, console/file 输出)
        merge_config(base: dict, override: dict) -> dict
            深度合并两个配置字典 (override 覆盖 base)
        get_nested(config: dict, *keys: str, default=None) -> Any
            安全获取嵌套字典值 (任意层级不存在返回默认值)
    常量:
        DEFAULT_CONFIG: dict[str, Any]
            默认配置字典 (包含 global、datasource、token_estimation 等默认值)

配置层次:
    配置的优先级从高到低:
    1. 运行时参数 (命令行参数)
    2. 配置文件 (config.yaml)
    3. 默认配置 (DEFAULT_CONFIG)

使用示例:
    from src.config import load_config, init_logging

    config = load_config("config.yaml")
    init_logging(config.get("global", {}).get("log"))
"""

from .settings import (
    load_config,
    init_logging,
    DEFAULT_CONFIG,
    merge_config,
    get_nested,
)

__all__ = [
    "load_config",
    "init_logging",
    "DEFAULT_CONFIG",
    "merge_config",
    "get_nested",
]
