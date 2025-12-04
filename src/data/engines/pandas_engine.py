"""
Pandas DataFrame 引擎实现

基于 pandas + openpyxl 的默认实现，提供完整的 DataFrame 操作功能。
"""

from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .base import BaseEngine


class PandasEngine(BaseEngine):
    """
    基于 pandas + openpyxl 的 DataFrame 引擎
    
    这是默认的引擎实现，使用 pandas 进行数据处理，
    openpyxl 进行 Excel 读写。
    
    特点:
    - 成熟稳定，生态丰富
    - 单线程处理
    - 内存占用较高
    - 适合中小规模数据 (< 100万行)
    """
    
    @property
    def name(self) -> str:
        return "pandas"
    
    # ==================== 文件 I/O ====================
    
    def read_excel(
        self, 
        path: Path | str, 
        sheet_name: str | int = 0,
        **kwargs: Any
    ) -> pd.DataFrame:
        """读取 Excel 文件"""
        return pd.read_excel(
            path, 
            sheet_name=sheet_name, 
            engine="openpyxl",
            **kwargs
        )
    
    def write_excel(
        self, 
        df: pd.DataFrame, 
        path: Path | str,
        sheet_name: str = "Sheet1",
        **kwargs: Any
    ) -> None:
        """写入 Excel 文件"""
        df.to_excel(
            path, 
            sheet_name=sheet_name,
            index=False, 
            engine="openpyxl",
            **kwargs
        )
    
    def read_csv(
        self,
        path: Path | str,
        **kwargs: Any
    ) -> pd.DataFrame:
        """读取 CSV 文件"""
        return pd.read_csv(path, **kwargs)
    
    def write_csv(
        self,
        df: pd.DataFrame,
        path: Path | str,
        **kwargs: Any
    ) -> None:
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
        self, 
        df: pd.DataFrame, 
        indices: list[int]
    ) -> list[dict[str, Any]]:
        """批量获取多行数据"""
        result = []
        for idx in indices:
            if idx in df.index:
                row = df.loc[idx]
                result.append({col: row[col] for col in df.columns})
        return result
    
    def set_value(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        column: str, 
        value: Any
    ) -> None:
        """设置单元格值"""
        df.at[idx, column] = value
    
    def set_values_batch(
        self,
        df: pd.DataFrame,
        updates: list[tuple[int, str, Any]]
    ) -> None:
        """批量设置多个单元格值"""
        for idx, column, value in updates:
            df.at[idx, column] = value
    
    # ==================== 列操作 ====================
    
    def get_column_names(self, df: pd.DataFrame) -> list[str]:
        """获取所有列名"""
        return list(df.columns)
    
    def has_column(self, df: pd.DataFrame, column: str) -> bool:
        """检查列是否存在"""
        return column in df.columns
    
    def add_column(
        self, 
        df: pd.DataFrame, 
        column: str, 
        default_value: Any = None
    ) -> pd.DataFrame:
        """添加新列"""
        if column not in df.columns:
            df[column] = default_value
        return df
    
    # ==================== 过滤与查询 ====================
    
    def filter_indices(
        self,
        df: pd.DataFrame,
        column: str,
        condition: str,
        value: Any = None
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
        require_all_inputs: bool = True
    ) -> list[int]:
        """
        向量化过滤: 查找未处理的行
        
        使用 pandas 向量化操作，比逐行遍历快 50-100 倍。
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
        """
        # 检查 NA 值 (NaN, None, pd.NA)
        is_na = series.isna()
        
        # 检查空白字符串 (仅对 object 类型)
        if series.dtype == "object":
            # str.strip() 对非字符串返回 NaN，因此 == '' 只匹配真正的空白字符串
            is_blank_str = series.str.strip() == ""
            return is_na | is_blank_str
        
        return is_na
    
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
        return list(df.index)
    
    # ==================== 迭代器 ====================
    
    def iter_rows(
        self, 
        df: pd.DataFrame, 
        columns: list[str] | None = None
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        """迭代所有行"""
        if columns is None:
            columns = list(df.columns)
        
        for idx in df.index:
            row = df.loc[idx]
            yield idx, {col: row[col] for col in columns if col in df.columns}
    
    # ==================== DataFrame 操作 ====================
    
    def slice_by_index_range(
        self, 
        df: pd.DataFrame, 
        min_idx: int, 
        max_idx: int
    ) -> pd.DataFrame:
        """按索引范围切片"""
        mask = (df.index >= min_idx) & (df.index <= max_idx)
        return df.loc[mask]
    
    def copy(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建 DataFrame 副本"""
        return df.copy()
