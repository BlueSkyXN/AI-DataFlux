"""
DataFrame 引擎抽象基类

定义统一的 DataFrame 操作接口，支持 pandas/polars 等多种实现。
通过 Strategy 模式实现引擎切换，业务代码无需修改即可更换底层引擎。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator


class BaseEngine(ABC):
    """
    DataFrame 引擎抽象基类

    定义了 DataFrame 操作的标准接口，包括:
    - Excel 文件读写
    - 行列操作
    - 值判断与转换
    - 迭代与过滤

    实现类需要处理各自引擎的特殊性，对外提供统一接口。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称标识"""
        pass

    # ==================== 文件 I/O ====================

    @abstractmethod
    def read_excel(
        self, path: Path | str, sheet_name: str | int = 0, **kwargs: Any
    ) -> Any:
        """
        读取 Excel 文件

        Args:
            path: 文件路径
            sheet_name: 工作表名称或索引
            **kwargs: 引擎特定参数

        Returns:
            DataFrame (类型取决于具体引擎)
        """
        pass

    @abstractmethod
    def write_excel(
        self, df: Any, path: Path | str, sheet_name: str = "Sheet1", **kwargs: Any
    ) -> None:
        """
        写入 Excel 文件

        Args:
            df: DataFrame
            path: 输出路径
            sheet_name: 工作表名称
            **kwargs: 引擎特定参数
        """
        pass

    @abstractmethod
    def read_csv(self, path: Path | str, **kwargs: Any) -> Any:
        """
        读取 CSV 文件

        Args:
            path: 文件路径
            **kwargs: 引擎特定参数

        Returns:
            DataFrame
        """
        pass

    @abstractmethod
    def write_csv(self, df: Any, path: Path | str, **kwargs: Any) -> None:
        """
        写入 CSV 文件

        Args:
            df: DataFrame
            path: 输出路径
            **kwargs: 引擎特定参数
        """
        pass

    # ==================== 行操作 ====================

    @abstractmethod
    def get_row(self, df: Any, idx: int) -> dict[str, Any]:
        """
        获取指定行数据

        Args:
            df: DataFrame
            idx: 行索引

        Returns:
            行数据字典 {列名: 值}
        """
        pass

    @abstractmethod
    def get_rows_by_indices(self, df: Any, indices: list[int]) -> list[dict[str, Any]]:
        """
        批量获取多行数据

        Args:
            df: DataFrame
            indices: 行索引列表

        Returns:
            行数据字典列表
        """
        pass

    @abstractmethod
    def set_value(self, df: Any, idx: int, column: str, value: Any) -> Any:
        """
        设置单元格值

        Args:
            df: DataFrame
            idx: 行索引
            column: 列名
            value: 值

        Returns:
            更新后的 DataFrame
        """
        pass

    @abstractmethod
    def set_values_batch(self, df: Any, updates: list[tuple[int, str, Any]]) -> Any:
        """
        批量设置多个单元格值

        Args:
            df: DataFrame
            updates: 更新列表 [(idx, column, value), ...]

        Returns:
            更新后的 DataFrame
        """
        pass

    # ==================== 列操作 ====================

    @abstractmethod
    def get_column_names(self, df: Any) -> list[str]:
        """
        获取所有列名

        Args:
            df: DataFrame

        Returns:
            列名列表
        """
        pass

    @abstractmethod
    def has_column(self, df: Any, column: str) -> bool:
        """
        检查列是否存在

        Args:
            df: DataFrame
            column: 列名

        Returns:
            是否存在
        """
        pass

    @abstractmethod
    def add_column(self, df: Any, column: str, default_value: Any = None) -> Any:
        """
        添加新列

        Args:
            df: DataFrame
            column: 列名
            default_value: 默认值

        Returns:
            修改后的 DataFrame
        """
        pass

    # ==================== 过滤与查询 ====================

    @abstractmethod
    def filter_indices(
        self, df: Any, column: str, condition: str, value: Any = None
    ) -> list[int]:
        """
        根据条件过滤行，返回符合条件的索引

        Args:
            df: DataFrame
            column: 列名
            condition: 条件类型 ("empty", "not_empty", "eq", "ne", "gt", "lt", "ge", "le")
            value: 比较值 (当 condition 为比较操作时需要)

        Returns:
            符合条件的行索引列表
        """
        pass

    @abstractmethod
    def filter_indices_vectorized(
        self,
        df: Any,
        input_columns: list[str],
        output_columns: list[str],
        require_all_inputs: bool = True,
        index_offset: int = 0,
    ) -> list[int]:
        """
        向量化过滤: 查找未处理的行

        未处理的定义:
        - 输入列: 根据 require_all_inputs 决定是否要求全部非空
        - 输出列: 任一输出列为空

        Args:
            df: DataFrame
            input_columns: 输入列名列表
            output_columns: 输出列名列表
            require_all_inputs: 是否要求所有输入列都非空
            index_offset: 索引偏移量 (用于切片后的局部索引场景)

        Returns:
            未处理的行索引列表
        """
        pass

    # ==================== 值操作 ====================

    @abstractmethod
    def is_empty(self, value: Any) -> bool:
        """
        判断值是否为空

        空值定义: None, NaN, 空字符串, 纯空白字符串

        Args:
            value: 要检查的值

        Returns:
            是否为空
        """
        pass

    @abstractmethod
    def is_empty_vectorized(self, series: Any) -> Any:
        """
        向量化判断空值

        Args:
            series: Series/列

        Returns:
            布尔 Series
        """
        pass

    @abstractmethod
    def to_string(self, value: Any) -> str:
        """
        将值转换为字符串

        空值转换为空字符串。

        Args:
            value: 值

        Returns:
            字符串
        """
        pass

    # ==================== 信息查询 ====================

    @abstractmethod
    def row_count(self, df: Any) -> int:
        """
        获取行数

        Args:
            df: DataFrame

        Returns:
            行数
        """
        pass

    @abstractmethod
    def get_index_range(self, df: Any) -> tuple[int, int]:
        """
        获取索引范围

        Args:
            df: DataFrame

        Returns:
            (最小索引, 最大索引)
        """
        pass

    @abstractmethod
    def get_indices(self, df: Any) -> list[int]:
        """
        获取所有索引

        Args:
            df: DataFrame

        Returns:
            索引列表
        """
        pass

    # ==================== 迭代器 ====================

    @abstractmethod
    def iter_rows(
        self, df: Any, columns: list[str] | None = None
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        """
        迭代所有行

        Args:
            df: DataFrame
            columns: 要提取的列名列表，None 表示所有列

        Yields:
            (索引, 行数据字典)
        """
        pass

    # ==================== DataFrame 操作 ====================

    @abstractmethod
    def slice_by_index_range(self, df: Any, min_idx: int, max_idx: int) -> Any:
        """
        按索引范围切片

        Args:
            df: DataFrame
            min_idx: 最小索引 (包含)
            max_idx: 最大索引 (包含)

        Returns:
            切片后的 DataFrame
        """
        pass

    @abstractmethod
    def copy(self, df: Any) -> Any:
        """
        创建 DataFrame 副本

        Args:
            df: DataFrame

        Returns:
            副本
        """
        pass
