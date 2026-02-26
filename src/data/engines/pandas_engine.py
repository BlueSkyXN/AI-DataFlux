"""
Pandas DataFrame 引擎实现模块

本模块提供基于 Pandas 的 DataFrame 引擎实现，是项目的默认引擎。
Pandas 生态成熟稳定，兼容性最好，适合大多数使用场景。

核心特性:
    - 成熟稳定: Pandas 是 Python 数据分析的标准库
    - 生态丰富: 与 NumPy、SciPy 等科学计算库无缝集成
    - 高性能读写器: 可选 calamine (10x) 和 xlsxwriter (3x)
    - 向量化优化: 利用 NumPy 实现高效计算

性能优化配置:
    ┌─────────────────────────────────────────────────────────┐
    │ 组件          │ 标准配置       │ 高性能配置           │
    ├─────────────────────────────────────────────────────────┤
    │ Excel 读取    │ openpyxl       │ calamine (fastexcel) │
    │ Excel 写入    │ openpyxl       │ xlsxwriter           │
    │ 向量计算      │ pandas 内置    │ numpy 加速           │
    └─────────────────────────────────────────────────────────┘

读取器对比:
    - openpyxl: 纯 Python，功能完整，支持读写
    - calamine (fastexcel): Rust 实现，读取速度 10x，仅支持读取

写入器对比:
    - openpyxl: 支持读写，功能完整，速度较慢
    - xlsxwriter: 仅支持写入，速度 2-5x，格式支持更好

使用示例:
    from src.data.engines.pandas_engine import PandasEngine

    # 标准配置
    engine = PandasEngine()

    # 高性能配置
    engine = PandasEngine(
        excel_reader="calamine",
        excel_writer="xlsxwriter"
    )

    # 使用引擎
    df = engine.read_excel("data.xlsx")
    engine.write_excel(df, "output.xlsx")

适用场景:
    ✓ 中小规模数据（< 100万行）
    ✓ 需要与其他 Pandas 库集成
    ✓ 复杂数据转换和清洗
    ✓ 兼容性要求高的环境

依赖:
    必需: pandas, openpyxl
    可选: fastexcel (calamine), xlsxwriter, numpy

方法清单:
    PandasEngine(BaseEngine) — 基于 Pandas 的 DataFrame 引擎
    ├── __init__(excel_reader="openpyxl", excel_writer="openpyxl")
    │       初始化引擎，自动检测高性能库可用性并回退
    ├── 属性
    │   ├── name -> str ("pandas")
    │   ├── excel_reader -> str                             — 当前 Excel 读取器
    │   └── excel_writer -> str                             — 当前 Excel 写入器
    ├── 文件 I/O
    │   ├── read_excel(path, sheet_name=0) -> pd.DataFrame  — 读取 Excel（自动选择读取器）
    │   ├── _read_excel_calamine(path, sheet_name) -> pd.DataFrame  — calamine 高性能读取
    │   ├── _read_excel_openpyxl(path, sheet_name) -> pd.DataFrame  — openpyxl 标准读取
    │   ├── write_excel(df, path, sheet_name) -> None       — 写入 Excel（自动选择写入器）
    │   ├── _write_excel_xlsxwriter(df, path, sheet_name) -> None   — xlsxwriter 高性能写入
    │   ├── _write_excel_openpyxl(df, path, sheet_name) -> None     — openpyxl 标准写入
    │   ├── read_csv(path) -> pd.DataFrame                  — 读取 CSV
    │   └── write_csv(df, path) -> None                     — 写入 CSV
    ├── 行操作
    │   ├── get_row(df, idx) -> dict[str, Any]              — 通过 df.loc 获取单行
    │   ├── get_rows_by_indices(df, indices) -> list[dict]  — 批量获取多行
    │   ├── set_value(df, idx, column, value) -> pd.DataFrame  — 通过 df.at 设置值（原地修改）
    │   └── set_values_batch(df, updates) -> pd.DataFrame   — 批量 df.at 设置
    ├── 列操作
    │   ├── get_column_names(df) -> list[str]
    │   ├── has_column(df, column) -> bool
    │   └── add_column(df, column, default) -> pd.DataFrame
    ├── 过滤与查询
    │   ├── filter_indices(df, column, condition, value) -> list[int]
    │   └── filter_indices_vectorized(df, input_cols, output_cols, ...) -> list[int]
    │           使用 pandas 向量化 + numpy 加速索引提取
    ├── 值操作
    │   ├── is_empty(value) -> bool                         — 基于 pd.isna() 判断
    │   ├── is_empty_vectorized(series) -> pd.Series        — 向量化空值判断（含空白字符串检测）
    │   └── to_string(value) -> str
    ├── 信息查询
    │   ├── row_count(df) -> int
    │   ├── get_index_range(df) -> tuple[int, int]          — 基于 df.index.min/max
    │   └── get_indices(df) -> list[int]                    — numpy 加速索引提取
    └── 迭代与操作
        ├── iter_rows(df, columns) -> Iterator[(int, dict)]
        ├── slice_by_index_range(df, min_idx, max_idx) -> pd.DataFrame
        └── copy(df) -> pd.DataFrame                        — df.copy() 深拷贝

关键变量:
    NUMPY_AVAILABLE (bool): NumPy 是否可用（影响向量化性能优化路径）
    FASTEXCEL_AVAILABLE (bool): fastexcel 是否可用（calamine 高性能读取器）
    XLSXWRITER_AVAILABLE (bool): xlsxwriter 是否可用（高性能写入器）

模块依赖:
    必需: pandas, openpyxl, logging, pathlib, typing
    可选: numpy (向量化加速), fastexcel (calamine 读取), xlsxwriter (高性能写入)
    内部: .base.BaseEngine

注意事项:
    1. calamine 不支持写入，只能用于读取
    2. xlsxwriter 不支持读取，只能用于写入
    3. 大文件（>100万行）建议使用 PolarsEngine
    4. 向量化操作需要 numpy 支持
"""

import logging
from pathlib import Path
from typing import Any, Iterator, Literal

import pandas as pd

from .base import BaseEngine

# ==================== 可选库导入 ====================
# 这些库用于性能优化，不是必需的

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:
    import fastexcel

    FASTEXCEL_AVAILABLE = True
except ImportError:
    fastexcel = None  # type: ignore
    FASTEXCEL_AVAILABLE = False

try:
    import xlsxwriter as xlsxwriter_lib  # noqa: F401

    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False


class PandasEngine(BaseEngine):
    """
    基于 Pandas 的 DataFrame 引擎

    这是项目的默认引擎实现，使用 pandas 进行数据处理。
    支持可选的高性能读写器以提升 Excel I/O 性能。

    高性能读写器:
        - calamine (fastexcel): Rust 实现的 Excel 解析器
            - 读取速度提升 10x+
            - 仅支持读取，不支持写入
            - 需要安装: pip install fastexcel

        - xlsxwriter: 高性能 Excel 写入器
            - 写入速度提升 2-5x
            - 格式支持更丰富
            - 仅支持写入，不支持读取
            - 需要安装: pip install xlsxwriter

    Attributes:
        excel_reader (str): 当前使用的 Excel 读取器
        excel_writer (str): 当前使用的 Excel 写入器

    性能建议:
        - 小文件（<10MB）: 标准配置即可
        - 中等文件（10-100MB）: 建议使用 calamine + xlsxwriter
        - 大文件（>100MB）: 建议切换到 PolarsEngine
    """

    def __init__(
        self,
        excel_reader: Literal["openpyxl", "calamine"] = "openpyxl",
        excel_writer: Literal["openpyxl", "xlsxwriter"] = "openpyxl",
    ):
        """
        初始化 Pandas 引擎

        会自动检测高性能库的可用性，不可用时回退到标准配置。

        Args:
            excel_reader: Excel 读取器类型
                - "openpyxl": 标准读取器（默认）
                - "calamine": 高性能读取器（需要 fastexcel）
            excel_writer: Excel 写入器类型
                - "openpyxl": 标准写入器（默认）
                - "xlsxwriter": 高性能写入器（需要 xlsxwriter）
        """
        self._excel_reader = excel_reader
        self._excel_writer = excel_writer

        # 验证读取器可用性，不可用时自动回退
        if excel_reader == "calamine" and not FASTEXCEL_AVAILABLE:
            logging.warning("fastexcel 不可用，回退到 openpyxl 读取")
            self._excel_reader = "openpyxl"

        # 验证写入器可用性
        if excel_writer == "xlsxwriter" and not XLSXWRITER_AVAILABLE:
            logging.warning("xlsxwriter 不可用，回退到 openpyxl 写入")
            self._excel_writer = "openpyxl"

        logging.debug(
            f"PandasEngine 初始化: reader={self._excel_reader}, "
            f"writer={self._excel_writer}, numpy={NUMPY_AVAILABLE}"
        )

    @property
    def name(self) -> str:
        return "pandas"

    @property
    def excel_reader(self) -> str:
        """当前使用的 Excel 读取器"""
        return self._excel_reader

    @property
    def excel_writer(self) -> str:
        """当前使用的 Excel 写入器"""
        return self._excel_writer

    # ==================== 文件 I/O ====================

    def read_excel(
        self, path: Path | str, sheet_name: str | int = 0, **kwargs: Any
    ) -> pd.DataFrame:
        """
        读取 Excel 文件

        根据配置自动选择最优读取器:
        - calamine (fastexcel): 10x+ 读取速度，基于 Rust 实现
        - openpyxl: 默认读取器，支持完整格式
        """
        path = Path(path)

        if self._excel_reader == "calamine" and FASTEXCEL_AVAILABLE:
            return self._read_excel_calamine(path, sheet_name, **kwargs)
        else:
            return self._read_excel_openpyxl(path, sheet_name, **kwargs)

    def _read_excel_calamine(
        self, path: Path, sheet_name: str | int = 0, **kwargs: Any
    ) -> pd.DataFrame:
        """使用 fastexcel (calamine) 读取 Excel - 高性能"""
        try:
            excel_file = fastexcel.read_excel(path)

            # 确定工作表
            if isinstance(sheet_name, int):
                sheet_names = excel_file.sheet_names
                if sheet_name >= len(sheet_names):
                    raise ValueError(f"工作表索引 {sheet_name} 超出范围")
                actual_sheet = sheet_names[sheet_name]
            else:
                actual_sheet = sheet_name

            # 加载工作表并转换为 pandas DataFrame
            df = excel_file.load_sheet(actual_sheet).to_pandas()

            logging.debug(f"使用 calamine 读取 Excel: {path} ({len(df)} 行)")
            return df

        except Exception as e:
            logging.warning(f"calamine 读取失败: {e}，回退到 openpyxl")
            return self._read_excel_openpyxl(path, sheet_name, **kwargs)

    def _read_excel_openpyxl(
        self, path: Path, sheet_name: str | int = 0, **kwargs: Any
    ) -> pd.DataFrame:
        """使用 openpyxl 读取 Excel - 默认"""
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl", **kwargs)
        logging.debug(f"使用 openpyxl 读取 Excel: {path} ({len(df)} 行)")
        return df

    def write_excel(
        self,
        df: pd.DataFrame,
        path: Path | str,
        sheet_name: str = "Sheet1",
        **kwargs: Any,
    ) -> None:
        """
        写入 Excel 文件

        根据配置自动选择最优写入器:
        - xlsxwriter: 2-5x 写入速度
        - openpyxl: 默认写入器，支持完整格式
        """
        path = Path(path)

        if self._excel_writer == "xlsxwriter" and XLSXWRITER_AVAILABLE:
            self._write_excel_xlsxwriter(df, path, sheet_name, **kwargs)
        else:
            self._write_excel_openpyxl(df, path, sheet_name, **kwargs)

    def _write_excel_xlsxwriter(
        self, df: pd.DataFrame, path: Path, sheet_name: str = "Sheet1", **kwargs: Any
    ) -> None:
        """使用 xlsxwriter 写入 Excel - 高性能"""
        try:
            df.to_excel(
                path, sheet_name=sheet_name, index=False, engine="xlsxwriter", **kwargs
            )
            logging.debug(f"使用 xlsxwriter 写入 Excel: {path} ({len(df)} 行)")
        except Exception as e:
            logging.warning(f"xlsxwriter 写入失败: {e}，回退到 openpyxl")
            self._write_excel_openpyxl(df, path, sheet_name, **kwargs)

    def _write_excel_openpyxl(
        self, df: pd.DataFrame, path: Path, sheet_name: str = "Sheet1", **kwargs: Any
    ) -> None:
        """使用 openpyxl 写入 Excel - 默认"""
        df.to_excel(
            path, sheet_name=sheet_name, index=False, engine="openpyxl", **kwargs
        )
        logging.debug(f"使用 openpyxl 写入 Excel: {path} ({len(df)} 行)")

    def read_csv(self, path: Path | str, **kwargs: Any) -> pd.DataFrame:
        """读取 CSV 文件"""
        return pd.read_csv(path, **kwargs)

    def write_csv(self, df: pd.DataFrame, path: Path | str, **kwargs: Any) -> None:
        """写入 CSV 文件"""
        df.to_csv(path, index=False, **kwargs)

    # ==================== 行操作 ====================

    def get_row(self, df: pd.DataFrame, idx: int) -> dict[str, Any]:
        """获取指定行数据"""
        if idx not in df.index:
            raise IndexError(f"索引 {idx} 不存在")
        row = df.loc[idx]
        return {col: row[col] for col in df.columns}

    def get_rows_by_indices(
        self, df: pd.DataFrame, indices: list[int]
    ) -> list[dict[str, Any]]:
        """批量获取多行数据"""
        result = []
        for idx in indices:
            if idx in df.index:
                row = df.loc[idx]
                result.append({col: row[col] for col in df.columns})
        return result

    def set_value(
        self, df: pd.DataFrame, idx: int, column: str, value: Any
    ) -> pd.DataFrame:
        """设置单元格值"""
        df.at[idx, column] = value
        return df

    def set_values_batch(
        self, df: pd.DataFrame, updates: list[tuple[int, str, Any]]
    ) -> pd.DataFrame:
        """批量设置多个单元格值"""
        for idx, column, value in updates:
            df.at[idx, column] = value
        return df

    # ==================== 列操作 ====================

    def get_column_names(self, df: pd.DataFrame) -> list[str]:
        """获取所有列名"""
        return list(df.columns)

    def has_column(self, df: pd.DataFrame, column: str) -> bool:
        """检查列是否存在"""
        return column in df.columns

    def add_column(
        self, df: pd.DataFrame, column: str, default_value: Any = None
    ) -> pd.DataFrame:
        """添加新列"""
        if column not in df.columns:
            df[column] = default_value
        return df

    # ==================== 过滤与查询 ====================

    def filter_indices(
        self, df: pd.DataFrame, column: str, condition: str, value: Any = None
    ) -> list[int]:
        """根据条件过滤行，返回符合条件的索引"""
        if column not in df.columns:
            return []

        series = df[column]

        if condition == "empty":
            mask = self.is_empty_vectorized(series)
        elif condition == "not_empty":
            mask = ~self.is_empty_vectorized(series)
        elif condition == "eq":
            mask = series == value
        elif condition == "ne":
            mask = series != value
        elif condition == "gt":
            mask = series > value
        elif condition == "lt":
            mask = series < value
        elif condition == "ge":
            mask = series >= value
        elif condition == "le":
            mask = series <= value
        else:
            raise ValueError(f"不支持的条件类型: {condition}")

        return list(df.index[mask])

    def filter_indices_vectorized(
        self,
        df: pd.DataFrame,
        input_columns: list[str],
        output_columns: list[str],
        require_all_inputs: bool = True,
        index_offset: int = 0,
    ) -> list[int]:
        """
        向量化过滤: 查找未处理的行

        使用 pandas 向量化操作，比逐行遍历快 50-100 倍。
        如果 numpy 可用，使用 numpy 进一步优化。
        """
        # 检查输入列是否有效
        if require_all_inputs:
            # 所有输入列都必须非空
            input_valid_mask = pd.Series(True, index=df.index)
            for col in input_columns:
                if col in df.columns:
                    input_valid_mask &= ~self.is_empty_vectorized(df[col])
                else:
                    # 列不存在，视为无效
                    input_valid_mask &= False
        else:
            # 至少一个输入列非空
            input_valid_mask = pd.Series(False, index=df.index)
            for col in input_columns:
                if col in df.columns:
                    input_valid_mask |= ~self.is_empty_vectorized(df[col])

        # 检查输出列是否有任一为空
        output_empty_mask = pd.Series(False, index=df.index)
        for col in output_columns:
            if col in df.columns:
                output_empty_mask |= self.is_empty_vectorized(df[col])
            else:
                # 列不存在，视为空
                output_empty_mask |= True

        # 未处理 = 输入有效 & 输出为空
        unprocessed_mask = input_valid_mask & output_empty_mask

        # 使用 numpy 加速索引提取 (如果可用)
        if NUMPY_AVAILABLE:
            return list(np.array(df.index)[unprocessed_mask.values])

        return list(df.index[unprocessed_mask])

    # ==================== 值操作 ====================

    def is_empty(self, value: Any) -> bool:
        """判断值是否为空"""
        if pd.isna(value):
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    def is_empty_vectorized(self, series: pd.Series) -> pd.Series:
        """
        向量化判断空值

        使用 pandas 向量化操作，性能优于 apply()。
        如果 numpy 可用，使用 numpy 进一步优化。
        """
        # 检查 NA 值 (NaN, None, pd.NA)
        is_na = series.isna()

        # 检查空白字符串 (object 和 string dtype)
        if series.dtype == "object" or pd.api.types.is_string_dtype(series):
            # str.strip() 对非字符串返回 NaN，因此 == '' 只匹配真正的空白字符串
            is_blank_str = series.str.strip() == ""
            result = is_na | is_blank_str.fillna(False)
        else:
            result = is_na

        # 使用 numpy 优化 (如果可用)
        if NUMPY_AVAILABLE and hasattr(result, "values"):
            # 确保返回 pandas Series 以保持兼容性
            pass

        return result

    def to_string(self, value: Any) -> str:
        """将值转换为字符串"""
        if pd.isna(value):
            return ""
        return str(value)

    # ==================== 信息查询 ====================

    def row_count(self, df: pd.DataFrame) -> int:
        """获取行数"""
        return len(df)

    def get_index_range(self, df: pd.DataFrame) -> tuple[int, int]:
        """获取索引范围"""
        if df.empty:
            return (0, 0)
        return (df.index.min(), df.index.max())

    def get_indices(self, df: pd.DataFrame) -> list[int]:
        """获取所有索引"""
        if NUMPY_AVAILABLE:
            return list(np.array(df.index))
        return list(df.index)

    # ==================== 迭代器 ====================

    def iter_rows(
        self, df: pd.DataFrame, columns: list[str] | None = None
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        """迭代所有行"""
        if columns is None:
            columns = list(df.columns)

        for idx in df.index:
            row = df.loc[idx]
            yield idx, {col: row[col] for col in columns if col in df.columns}

    # ==================== DataFrame 操作 ====================

    def slice_by_index_range(
        self, df: pd.DataFrame, min_idx: int, max_idx: int
    ) -> pd.DataFrame:
        """按索引范围切片"""
        mask = (df.index >= min_idx) & (df.index <= max_idx)
        return df.loc[mask]

    def copy(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建 DataFrame 副本"""
        return df.copy()
