"""
配置管理模块

本模块提供 AI-DataFlux 的配置管理功能，包括配置文件加载、
默认配置定义和日志初始化。

核心功能:
    - load_config: 加载 YAML 配置文件
    - init_logging: 初始化日志系统
    - DEFAULT_CONFIG: 默认配置字典

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
