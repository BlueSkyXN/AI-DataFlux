"""
DataFrame 引擎抽象层模块

本模块提供 DataFrame 操作的引擎抽象，支持在不同的 DataFrame 框架之间切换。
通过策略模式实现引擎解耦，业务代码无需修改即可更换底层实现。

支持的引擎:
    - pandas: 默认引擎，生态丰富，兼容性最好
        - 读取器: openpyxl (默认) | calamine (10x 加速)
        - 写入器: openpyxl (默认) | xlsxwriter (3x 加速)
    
    - polars: 高性能引擎，多线程并行处理
        - 内置 Rust 读写器
        - 大文件处理性能优异

引擎选择策略:
    ┌─────────────────────────────────────────────────────┐
    │               engine_type="auto"                     │
    │  ┌─────────────┐        ┌─────────────────────────┐ │
    │  │ Polars 可用? │──Yes──▶│ 使用 PolarsEngine      │ │
    │  └──────┬──────┘        └─────────────────────────┘ │
    │         │ No                                         │
    │         ▼                                            │
    │  ┌─────────────────────────┐                        │
    │  │ 使用 PandasEngine       │                        │
    │  └─────────────────────────┘                        │
    └─────────────────────────────────────────────────────┘

安全性设计:
    某些库（如 polars, fastexcel）在特定平台上导入时可能导致进程崩溃
    （如 Windows ARM + Python 3.13 上的 "Fatal Python error: Illegal instruction"）。
    因此所有库可用性检测都在子进程中执行，避免影响主进程。

使用示例:
    from src.data.engines import get_engine, get_available_libraries
    
    # 获取引擎实例
    engine = get_engine(
        engine_type="auto",       # 自动选择最佳引擎
        excel_reader="calamine",  # 使用高性能读取器
        excel_writer="xlsxwriter" # 使用高性能写入器
    )
    
    # 使用引擎
    df = engine.read_excel("data.xlsx")
    engine.write_excel(df, "output.xlsx")
    
    # 检查可用库
    libs = get_available_libraries()
    print(libs)  # {'pandas': True, 'polars': True, ...}

性能对比:
    ┌─────────────────────┬────────────┬────────────┐
    │ 操作                │ 标准配置   │ 高性能配置 │
    ├─────────────────────┼────────────┼────────────┤
    │ 读取 100MB Excel    │ 45s        │ 4.5s       │
    │ 写入 100MB Excel    │ 30s        │ 10s        │
    │ DataFrame 处理      │ 1x         │ 2-5x       │
    └─────────────────────┴────────────┴────────────┘
    高性能配置: polars + calamine + xlsxwriter
"""

import logging
import subprocess
import sys
from typing import Literal

from .base import BaseEngine
from .pandas_engine import PandasEngine


def _safe_check_library(import_code: str, lib_name: str, timeout: int = 30) -> bool:
    """
    在子进程中安全检测库是否可用
    
    某些库（如 polars, fastexcel）在特定平台上导入时可能导致进程崩溃
    （如 Windows ARM + Python 3.13 上的 "Fatal Python error: Illegal instruction"）。
    因此必须在子进程中测试，避免崩溃主进程。
    
    实现原理:
        1. 启动一个新的 Python 子进程
        2. 在子进程中执行 import 语句
        3. 根据返回码判断是否成功
        4. 子进程崩溃不影响主进程

    Args:
        import_code: 要执行的 Python 导入代码
            例: "import polars; polars.DataFrame({'x': [1]})"
        lib_name: 库名称（仅用于日志输出）
        timeout: 超时时间（秒），防止子进程卡死

    Returns:
        bool: 库是否可用（导入成功）
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", import_code],
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            logging.debug(f"{lib_name} 库可用")
            return True
        else:
            # 导入失败，输出错误信息便于调试
            stderr = result.stderr.decode("utf-8", errors="replace")
            if stderr:
                logging.debug(f"{lib_name} 检测失败: {stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        logging.warning(f"{lib_name} 可用性检测超时")
        return False
    except Exception as e:
        logging.debug(f"{lib_name} 检测异常: {e}")
        return False


# ==================== 库可用性检测 ====================
# 全部使用子进程安全检测，防止导入崩溃

# Polars 检测：需要能创建 DataFrame（不只是导入）
POLARS_AVAILABLE = _safe_check_library(
    "import polars; polars.DataFrame({'x': [1]})", "Polars"
)

# fastexcel 检测：calamine 的 Python 绑定
FASTEXCEL_AVAILABLE = _safe_check_library("import fastexcel", "fastexcel")

# xlsxwriter 检测：高性能 Excel 写入器
XLSXWRITER_AVAILABLE = _safe_check_library("import xlsxwriter", "xlsxwriter")

# NumPy 检测：某些优化操作依赖
NUMPY_AVAILABLE = _safe_check_library("import numpy", "numpy")

# 记录引擎选择结果
if not POLARS_AVAILABLE:
    logging.debug("Polars 库不可用，将使用 Pandas 作为默认引擎")


# ==================== 类型定义 ====================
EngineType = Literal["pandas", "polars", "auto"]
ReaderType = Literal["openpyxl", "calamine", "auto"]
WriterType = Literal["openpyxl", "xlsxwriter", "auto"]


def _resolve_reader(reader_type: ReaderType) -> str:
    """
    解析实际使用的 Excel 读取器
    
    优先级: calamine > openpyxl（当 auto 时）
    
    Args:
        reader_type: 请求的读取器类型
        
    Returns:
        str: 实际使用的读取器名称
    """
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
    """
    解析实际使用的 Excel 写入器
    
    优先级: xlsxwriter > openpyxl（当 auto 时）
    
    Args:
        writer_type: 请求的写入器类型
        
    Returns:
        str: 实际使用的写入器名称
    """
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
    """
    解析实际使用的 DataFrame 引擎
    
    优先级: polars > pandas（当 auto 时）
    
    Args:
        engine_type: 请求的引擎类型
        
    Returns:
        str: 实际使用的引擎名称
    """
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
    
    工厂函数，根据配置创建合适的引擎实例。
    支持自动选择最佳引擎和回退机制。

    Args:
        engine_type: 引擎类型
            - "pandas": 使用 Pandas 引擎
            - "polars": 使用 Polars 引擎
            - "auto": 自动选择（优先 Polars）
        excel_reader: Excel 读取器
            - "openpyxl": 纯 Python 实现，功能完整
            - "calamine": Rust 实现，速度 10x
            - "auto": 自动选择（优先 calamine）
        excel_writer: Excel 写入器
            - "openpyxl": 纯 Python 实现，支持读写
            - "xlsxwriter": 仅写入，速度 3x
            - "auto": 自动选择（优先 xlsxwriter）

    Returns:
        BaseEngine: 引擎实例（PandasEngine 或 PolarsEngine）

    Raises:
        ValueError: 不支持的引擎类型
    
    示例:
        # 自动选择最佳配置
        engine = get_engine(engine_type="auto")
        
        # 指定使用 Pandas + 高性能读写器
        engine = get_engine(
            engine_type="pandas",
            excel_reader="calamine",
            excel_writer="xlsxwriter"
        )
    """
    # 解析实际使用的配置
    resolved_engine = _resolve_engine(engine_type)
    resolved_reader = _resolve_reader(excel_reader)
    resolved_writer = _resolve_writer(excel_writer)

    logging.info(
        f"引擎配置: engine={resolved_engine}, "
        f"reader={resolved_reader}, writer={resolved_writer}"
    )

    if resolved_engine == "polars":
        # 延迟导入 Polars 引擎，避免不需要时的导入开销
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
    获取所有可选库的可用状态
    
    用于诊断和显示系统配置信息。

    Returns:
        dict[str, bool]: {库名: 是否可用}
    
    示例:
        >>> get_available_libraries()
        {
            'pandas': True,     # 核心依赖，始终可用
            'openpyxl': True,   # 核心依赖，始终可用
            'numpy': True,
            'polars': False,    # 可选高性能引擎
            'fastexcel': True,  # 可选高性能读取器
            'xlsxwriter': True  # 可选高性能写入器
        }
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
    注册新的引擎实现（预留扩展接口）
    
    此接口为未来扩展预留，允许注册自定义引擎实现。
    目前引擎通过 get_engine() 动态选择。

    Args:
        name: 引擎名称
        engine_class: 继承自 BaseEngine 的引擎类
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
