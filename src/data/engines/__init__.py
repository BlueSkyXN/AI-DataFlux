"""DataFrame 引擎抽象层

支持多种 DataFrame 引擎切换:
- pandas: 当前默认实现，使用 pandas + openpyxl
- polars: 高性能实现 (预留)，使用 polars + calamine + xlsxwriter
"""

from typing import Literal
from .base import BaseEngine
from .pandas_engine import PandasEngine

# 预留 Polars 引擎
# from .polars_engine import PolarsEngine

EngineType = Literal["pandas", "polars"]

_engines: dict[str, type[BaseEngine]] = {
    "pandas": PandasEngine,
    # "polars": PolarsEngine,  # 预留
}


def get_engine(engine_type: EngineType = "pandas") -> BaseEngine:
    """
    获取 DataFrame 引擎实例
    
    Args:
        engine_type: 引擎类型，可选 "pandas" 或 "polars"
        
    Returns:
        引擎实例
        
    Raises:
        ValueError: 不支持的引擎类型
    """
    if engine_type not in _engines:
        available = ", ".join(_engines.keys())
        raise ValueError(f"不支持的引擎类型: {engine_type}，可用引擎: {available}")
    return _engines[engine_type]()


def register_engine(name: str, engine_class: type[BaseEngine]) -> None:
    """
    注册新的引擎实现
    
    Args:
        name: 引擎名称
        engine_class: 引擎类
    """
    _engines[name] = engine_class


__all__ = [
    "BaseEngine",
    "PandasEngine",
    "EngineType",
    "get_engine",
    "register_engine",
]
