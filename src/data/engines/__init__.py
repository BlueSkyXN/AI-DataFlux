"""DataFrame 引擎抽象层

支持多种 DataFrame 引擎切换:
- pandas: 当前默认实现，使用 pandas + openpyxl (可选 calamine/xlsxwriter 加速)
- polars: 高性能实现，使用 polars + fastexcel + xlsxwriter

读写器选项:
- excel_reader: openpyxl (默认) | calamine (快速，需要 python-calamine/fastexcel)
- excel_writer: openpyxl (默认) | xlsxwriter (快速，需要 xlsxwriter)
"""

import logging
from typing import Literal, Any

from .base import BaseEngine
from .pandas_engine import PandasEngine

# 库可用性检测
POLARS_AVAILABLE = False
FASTEXCEL_AVAILABLE = False
XLSXWRITER_AVAILABLE = False
NUMPY_AVAILABLE = False

try:
    import polars  # noqa: F401
    POLARS_AVAILABLE = True
except ImportError:
    pass

try:
    import fastexcel  # noqa: F401
    FASTEXCEL_AVAILABLE = True
except ImportError:
    pass

try:
    import xlsxwriter  # noqa: F401
    XLSXWRITER_AVAILABLE = True
except ImportError:
    pass

try:
    import numpy  # noqa: F401
    NUMPY_AVAILABLE = True
except ImportError:
    pass


EngineType = Literal["pandas", "polars", "auto"]
ReaderType = Literal["openpyxl", "calamine", "auto"]
WriterType = Literal["openpyxl", "xlsxwriter", "auto"]


def _resolve_reader(reader_type: ReaderType) -> str:
    """解析实际使用的读取器"""
    if reader_type == "auto":
        if FASTEXCEL_AVAILABLE:
            return "calamine"
        return "openpyxl"
    elif reader_type == "calamine":
        if not FASTEXCEL_AVAILABLE:
            logging.warning("calamine/fastexcel 不可用，回退到 openpyxl")
            return "openpyxl"
        return "calamine"
    return "openpyxl"


def _resolve_writer(writer_type: WriterType) -> str:
    """解析实际使用的写入器"""
    if writer_type == "auto":
        if XLSXWRITER_AVAILABLE:
            return "xlsxwriter"
        return "openpyxl"
    elif writer_type == "xlsxwriter":
        if not XLSXWRITER_AVAILABLE:
            logging.warning("xlsxwriter 不可用，回退到 openpyxl")
            return "openpyxl"
        return "xlsxwriter"
    return "openpyxl"


def _resolve_engine(engine_type: EngineType) -> str:
    """解析实际使用的引擎"""
    if engine_type == "auto":
        if POLARS_AVAILABLE:
            return "polars"
        return "pandas"
    elif engine_type == "polars":
        if not POLARS_AVAILABLE:
            logging.warning("polars 不可用，回退到 pandas")
            return "pandas"
        return "polars"
    return "pandas"


def get_engine(
    engine_type: EngineType = "pandas",
    excel_reader: ReaderType = "auto",
    excel_writer: WriterType = "auto",
) -> BaseEngine:
    """
    获取 DataFrame 引擎实例
    
    Args:
        engine_type: 引擎类型 "pandas" | "polars" | "auto"
        excel_reader: Excel 读取器 "openpyxl" | "calamine" | "auto"
        excel_writer: Excel 写入器 "openpyxl" | "xlsxwriter" | "auto"
        
    Returns:
        引擎实例
        
    Raises:
        ValueError: 不支持的引擎类型
    """
    resolved_engine = _resolve_engine(engine_type)
    resolved_reader = _resolve_reader(excel_reader)
    resolved_writer = _resolve_writer(excel_writer)
    
    logging.info(
        f"引擎配置: engine={resolved_engine}, "
        f"reader={resolved_reader}, writer={resolved_writer}"
    )
    
    if resolved_engine == "polars":
        # 延迟导入 Polars 引擎
        try:
            from .polars_engine import PolarsEngine
            return PolarsEngine(
                excel_reader=resolved_reader,
                excel_writer=resolved_writer,
            )
        except ImportError as e:
            logging.warning(f"无法导入 PolarsEngine: {e}，回退到 PandasEngine")
            resolved_engine = "pandas"
    
    # 默认使用 PandasEngine
    return PandasEngine(
        excel_reader=resolved_reader,
        excel_writer=resolved_writer,
    )


def get_available_libraries() -> dict[str, bool]:
    """
    获取可用库的状态
    
    Returns:
        {库名: 是否可用}
    """
    return {
        "pandas": True,  # 核心依赖，始终可用
        "openpyxl": True,  # 核心依赖，始终可用
        "numpy": NUMPY_AVAILABLE,
        "polars": POLARS_AVAILABLE,
        "fastexcel": FASTEXCEL_AVAILABLE,
        "xlsxwriter": XLSXWRITER_AVAILABLE,
    }


def register_engine(name: str, engine_class: type[BaseEngine]) -> None:
    """
    注册新的引擎实现 (预留扩展接口)
    
    Args:
        name: 引擎名称
        engine_class: 引擎类
    """
    # 预留接口，目前引擎通过 get_engine 动态选择
    pass


__all__ = [
    "BaseEngine",
    "PandasEngine",
    "EngineType",
    "ReaderType",
    "WriterType",
    "get_engine",
    "get_available_libraries",
    "register_engine",
    "POLARS_AVAILABLE",
    "FASTEXCEL_AVAILABLE",
    "XLSXWRITER_AVAILABLE",
    "NUMPY_AVAILABLE",
]
