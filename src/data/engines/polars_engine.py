"""
Polars DataFrame 引擎实现

基于 Polars + FastExcel (Calamine) + xlsxwriter 的高性能实现。

特点:
- 多线程并行处理
- 惰性求值 (LazyFrame)
- 内存效率高
- 适合大规模数据 (> 100万行)

依赖 (可选安装):
    pip install polars fastexcel xlsxwriter

使用方式:
    在 config.yaml 中设置:
    datasource:
      engine: polars

性能对比 (预估):
    | 操作 | pandas | polars | 提升倍数 |
    |------|--------|--------|---------|
    | Excel 读取 (100万行) | 60s | 6s | 10x |
    | DataFrame 过滤 | 10s | 0.5s | 20x |
    | Excel 写入 | 30s | 10s | 3x |
"""

import logging
from pathlib import Path
from typing import Any, Iterator, Literal

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore
    POLARS_AVAILABLE = False

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

from .base import BaseEngine


class PolarsEngine(BaseEngine):
    """
    基于 Polars 的高性能 DataFrame 引擎
    
    使用 Polars 进行 DataFrame 操作，FastExcel (Calamine) 读取 Excel，
    xlsxwriter 写入 Excel。
    
    特点:
    - 多线程并行处理
    - 惰性求值 (LazyFrame)
    - 内存效率高
    - 适合大规模数据 (> 100万行)
    
    Attributes:
        excel_reader: Excel 读取器类型 (polars 默认使用 fastexcel)
        excel_writer: Excel 写入器类型 (polars 默认使用 xlsxwriter)
    """
    
    def __init__(
        self,
        excel_reader: Literal["openpyxl", "calamine"] = "calamine",
        excel_writer: Literal["openpyxl", "xlsxwriter"] = "xlsxwriter",
    ):
        """
        初始化 Polars 引擎
        
        Args:
            excel_reader: Excel 读取器类型 (Polars 原生使用 calamine)
            excel_writer: Excel 写入器类型 (Polars 原生使用 xlsxwriter)
            
        Raises:
            ImportError: Polars 不可用
        """
        if not POLARS_AVAILABLE:
            raise ImportError(
                "Polars 不可用，请安装: pip install polars fastexcel xlsxwriter"
            )
        
        self._excel_reader = excel_reader
        self._excel_writer = excel_writer
        
        # Polars 原生支持 fastexcel，如果不可用则警告
        if excel_reader == "calamine" and not FASTEXCEL_AVAILABLE:
            logging.warning(
                "fastexcel 不可用，Polars 将使用内置方法读取 Excel"
            )
        
        logging.debug(
            f"PolarsEngine 初始化: reader={self._excel_reader}, "
            f"writer={self._excel_writer}"
        )
    
    @property
    def name(self) -> str:
        return "polars"
    
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
        self, 
        path: Path | str, 
        sheet_name: str | int = 0,
        **kwargs: Any
    ) -> pl.DataFrame:
        """
        读取 Excel 文件
        
        Polars 原生使用 fastexcel (Calamine) 读取，性能极高。
        """
        path = Path(path)
        
        try:
            if FASTEXCEL_AVAILABLE:
                # 使用 fastexcel 读取
                excel_file = fastexcel.read_excel(path)
                
                # 确定工作表
                if isinstance(sheet_name, int):
                    sheet_names = excel_file.sheet_names
                    if sheet_name >= len(sheet_names):
                        raise ValueError(f"工作表索引 {sheet_name} 超出范围")
                    actual_sheet = sheet_names[sheet_name]
                else:
                    actual_sheet = sheet_name
                
                # 加载并转换为 Polars DataFrame
                df = excel_file.load_sheet(actual_sheet).to_polars()
                logging.debug(f"使用 fastexcel 读取 Excel: {path} ({len(df)} 行)")
                return df
            else:
                # 回退到 Polars 内置方法
                df = pl.read_excel(
                    path,
                    sheet_name=sheet_name if isinstance(sheet_name, str) else None,
                    sheet_id=sheet_name if isinstance(sheet_name, int) else None,
                    **kwargs
                )
                logging.debug(f"使用 Polars 内置方法读取 Excel: {path} ({len(df)} 行)")
                return df
                
        except Exception as e:
            logging.error(f"Polars 读取 Excel 失败: {e}")
            raise
    
    def write_excel(
        self, 
        df: pl.DataFrame, 
        path: Path | str,
        sheet_name: str = "Sheet1",
        **kwargs: Any
    ) -> None:
        """
        写入 Excel 文件
        
        Polars 原生使用 xlsxwriter 写入。
        """
        path = Path(path)
        
        try:
            df.write_excel(
                path,
                worksheet=sheet_name,
                **kwargs
            )
            logging.debug(f"使用 Polars write_excel 写入: {path} ({len(df)} 行)")
        except Exception as e:
            logging.error(f"Polars 写入 Excel 失败: {e}")
            raise
    
    def read_csv(
        self,
        path: Path | str,
        **kwargs: Any
    ) -> pl.DataFrame:
        """读取 CSV 文件"""
        return pl.read_csv(path, **kwargs)
    
    def write_csv(
        self,
        df: pl.DataFrame,
        path: Path | str,
        **kwargs: Any
    ) -> None:
        """写入 CSV 文件"""
        df.write_csv(path, **kwargs)
    
    # ==================== 行操作 ====================
    
    def get_row(self, df: pl.DataFrame, idx: int) -> dict[str, Any]:
        """获取指定行数据"""
        # Polars 使用位置索引
        if idx < 0 or idx >= len(df):
            raise IndexError(f"索引 {idx} 超出范围")
        
        row = df.row(idx, named=True)
        return dict(row)
    
    def get_rows_by_indices(
        self, 
        df: pl.DataFrame, 
        indices: list[int]
    ) -> list[dict[str, Any]]:
        """批量获取多行数据"""
        result = []
        for idx in indices:
            if 0 <= idx < len(df):
                row = df.row(idx, named=True)
                result.append(dict(row))
        return result
    
    def set_value(
        self, 
        df: pl.DataFrame, 
        idx: int, 
        column: str, 
        value: Any
    ) -> pl.DataFrame:
        """
        设置单元格值
        
        注意: Polars DataFrame 是不可变的，此方法会返回新的 DataFrame。
        对于大量更新，建议使用批量操作。
        """
        df_with_idx = df.with_row_index("__row_idx__")
        updated = df_with_idx.with_columns(
            pl.when(pl.col("__row_idx__") == idx)
            .then(pl.lit(value))
            .otherwise(pl.col(column))
            .alias(column)
        )
        return updated.drop("__row_idx__")
    
    def set_values_batch(
        self,
        df: pl.DataFrame,
        updates: list[tuple[int, str, Any]]
    ) -> pl.DataFrame:
        """
        批量设置多个单元格值
        
        注意: Polars DataFrame 是不可变的，此方法会返回新的 DataFrame。
        """
        if not updates:
            return df
        
        df_with_idx = df.with_row_index("__row_idx__")
        row_idx_expr = pl.col("__row_idx__")
        
        updates_by_col: dict[str, list[tuple[int, Any]]] = {}
        for idx, column, value in updates:
            updates_by_col.setdefault(column, []).append((idx, value))
        
        expressions: list[pl.Expr] = []
        for column, col_updates in updates_by_col.items():
            expr = pl.col(column)
            for idx, value in col_updates:
                expr = pl.when(row_idx_expr == idx).then(pl.lit(value)).otherwise(expr)
            expressions.append(expr.alias(column))
        
        updated = df_with_idx.with_columns(expressions)
        return updated.drop("__row_idx__")
    
    # ==================== 列操作 ====================
    
    def get_column_names(self, df: pl.DataFrame) -> list[str]:
        """获取所有列名"""
        return df.columns
    
    def has_column(self, df: pl.DataFrame, column: str) -> bool:
        """检查列是否存在"""
        return column in df.columns
    
    def add_column(
        self, 
        df: pl.DataFrame, 
        column: str, 
        default_value: Any = None
    ) -> pl.DataFrame:
        """添加新列"""
        if column not in df.columns:
            return df.with_columns(pl.lit(default_value).alias(column))
        return df
    
    # ==================== 过滤与查询 ====================
    
    def filter_indices(
        self,
        df: pl.DataFrame,
        column: str,
        condition: str,
        value: Any = None
    ) -> list[int]:
        """根据条件过滤行，返回符合条件的索引"""
        if column not in df.columns:
            return []
        
        col = pl.col(column)
        
        if condition == "empty":
            mask = self._is_empty_expr(column)
        elif condition == "not_empty":
            mask = ~self._is_empty_expr(column)
        elif condition == "eq":
            mask = col == value
        elif condition == "ne":
            mask = col != value
        elif condition == "gt":
            mask = col > value
        elif condition == "lt":
            mask = col < value
        elif condition == "ge":
            mask = col >= value
        elif condition == "le":
            mask = col <= value
        else:
            raise ValueError(f"不支持的条件类型: {condition}")
        
        # 添加行号并过滤
        df_with_idx = df.with_row_index("__row_idx__")
        filtered = df_with_idx.filter(mask)
        
        return filtered["__row_idx__"].to_list()
    
    def _is_empty_expr(self, column: str) -> pl.Expr:
        """创建空值检查表达式"""
        col = pl.col(column)
        # 检查 null 或空白字符串
        return col.is_null() | (col.cast(pl.Utf8).str.strip_chars() == "")
    
    def filter_indices_vectorized(
        self,
        df: pl.DataFrame,
        input_columns: list[str],
        output_columns: list[str],
        require_all_inputs: bool = True,
        index_offset: int = 0
    ) -> list[int]:
        """
        向量化过滤: 查找未处理的行
        
        利用 Polars 的表达式 API，性能比 pandas 高 10-50 倍。
        """
        # 构建输入条件
        if require_all_inputs:
            # 所有输入列都必须非空
            input_conditions = []
            for col in input_columns:
                if col in df.columns:
                    input_conditions.append(~self._is_empty_expr(col))
                else:
                    # 列不存在，整体条件为 False
                    input_conditions.append(pl.lit(False))
            
            if input_conditions:
                input_valid = input_conditions[0]
                for cond in input_conditions[1:]:
                    input_valid = input_valid & cond
            else:
                input_valid = pl.lit(True)
        else:
            # 至少一个输入列非空
            input_conditions = []
            for col in input_columns:
                if col in df.columns:
                    input_conditions.append(~self._is_empty_expr(col))
            
            if input_conditions:
                input_valid = input_conditions[0]
                for cond in input_conditions[1:]:
                    input_valid = input_valid | cond
            else:
                input_valid = pl.lit(False)
        
        # 构建输出条件 (任一输出列为空)
        output_conditions = []
        for col in output_columns:
            if col in df.columns:
                output_conditions.append(self._is_empty_expr(col))
            else:
                # 列不存在，视为空
                output_conditions.append(pl.lit(True))
        
        if output_conditions:
            output_empty = output_conditions[0]
            for cond in output_conditions[1:]:
                output_empty = output_empty | cond
        else:
            output_empty = pl.lit(False)
        
        # 未处理 = 输入有效 & 输出为空
        unprocessed = input_valid & output_empty
        
        # 添加行号并过滤
        df_with_idx = df.with_row_index("__row_idx__")
        filtered = df_with_idx.filter(unprocessed)
        
        indices = filtered["__row_idx__"].to_list()
        if index_offset:
            return [idx + index_offset for idx in indices]
        return indices
    
    # ==================== 值操作 ====================
    
    def is_empty(self, value: Any) -> bool:
        """判断值是否为空"""
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False
    
    def is_empty_vectorized(self, series: pl.Series) -> pl.Series:
        """
        向量化判断空值
        
        Polars 的 null 检查比 pandas 更高效。
        """
        is_null = series.is_null()
        
        # 对字符串类型检查空白
        if series.dtype == pl.Utf8:
            is_blank = series.str.strip_chars() == ""
            return is_null | is_blank
        
        return is_null
    
    def to_string(self, value: Any) -> str:
        """将值转换为字符串"""
        if value is None:
            return ""
        return str(value)
    
    # ==================== 信息查询 ====================
    
    def row_count(self, df: pl.DataFrame) -> int:
        """获取行数"""
        return len(df)
    
    def get_index_range(self, df: pl.DataFrame) -> tuple[int, int]:
        """
        获取索引范围
        
        Polars 使用位置索引，范围为 [0, len-1]
        """
        if len(df) == 0:
            return (0, 0)
        return (0, len(df) - 1)
    
    def get_indices(self, df: pl.DataFrame) -> list[int]:
        """获取所有索引"""
        return list(range(len(df)))
    
    # ==================== 迭代器 ====================
    
    def iter_rows(
        self, 
        df: pl.DataFrame, 
        columns: list[str] | None = None
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        """迭代所有行"""
        if columns is None:
            columns = df.columns
        else:
            columns = [c for c in columns if c in df.columns]
        
        for idx in range(len(df)):
            row = df.row(idx, named=True)
            yield idx, {col: row[col] for col in columns}
    
    # ==================== DataFrame 操作 ====================
    
    def slice_by_index_range(
        self, 
        df: pl.DataFrame, 
        min_idx: int, 
        max_idx: int
    ) -> pl.DataFrame:
        """按索引范围切片"""
        # Polars 使用位置切片
        return df.slice(min_idx, max_idx - min_idx + 1)
    
    def copy(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        创建 DataFrame 副本
        
        Polars DataFrame 是不可变的，clone() 是轻量级操作。
        """
        return df.clone()
