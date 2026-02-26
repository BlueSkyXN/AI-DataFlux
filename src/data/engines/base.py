"""
DataFrame 引擎抽象基类模块

本模块定义了 DataFrame 操作的标准接口，是引擎抽象层的核心。
通过策略模式（Strategy Pattern）实现引擎切换，业务代码无需修改
即可在 Pandas、Polars 等不同框架之间切换。

设计原则:
    - 接口统一: 所有引擎提供相同的方法签名
    - 引擎无关: 业务代码不依赖具体引擎的 API
    - 可扩展性: 易于添加新的引擎实现
    - 性能透明: 引擎优化对业务代码透明

接口分类:
    ┌─────────────────────────────────────────────────────────┐
    │                    BaseEngine 接口                       │
    ├─────────────────────────────────────────────────────────┤
    │ 文件 I/O                                                │
    │   read_excel, write_excel, read_csv, write_csv         │
    ├─────────────────────────────────────────────────────────┤
    │ 行操作                                                   │
    │   get_row, get_rows_by_indices, set_value,             │
    │   set_values_batch                                      │
    ├─────────────────────────────────────────────────────────┤
    │ 列操作                                                   │
    │   get_column_names, has_column, add_column             │
    ├─────────────────────────────────────────────────────────┤
    │ 过滤与查询                                              │
    │   filter_indices, filter_indices_vectorized            │
    ├─────────────────────────────────────────────────────────┤
    │ 值操作                                                   │
    │   is_empty, is_empty_vectorized, to_string             │
    ├─────────────────────────────────────────────────────────┤
    │ 信息查询                                                 │
    │   row_count, get_index_range, get_indices              │
    ├─────────────────────────────────────────────────────────┤
    │ 迭代与操作                                              │
    │   iter_rows, slice_by_index_range, copy                │
    └─────────────────────────────────────────────────────────┘

实现类:
    - PandasEngine: 基于 pandas 的实现
    - PolarsEngine: 基于 polars 的实现

使用示例:
    from src.data.engines import get_engine, BaseEngine

    # 获取引擎（不关心具体实现）
    engine: BaseEngine = get_engine(engine_type="auto")

    # 使用统一接口
    df = engine.read_excel("data.xlsx")
    row = engine.get_row(df, 0)
    engine.set_value(df, 0, "column", "value")
    engine.write_excel(df, "output.xlsx")

扩展指南:
    添加新引擎时需要:
    1. 继承 BaseEngine
    2. 实现所有抽象方法
    3. 在 __init__.py 中注册引擎

方法签名索引:
    BaseEngine (抽象基类)
    ├── 属性
    │   └── name -> str                                     — 引擎名称标识
    ├── 文件 I/O
    │   ├── read_excel(path, sheet_name=0) -> DataFrame     — 读取 Excel 文件
    │   ├── write_excel(df, path, sheet_name) -> None       — 写入 Excel 文件
    │   ├── read_csv(path) -> DataFrame                     — 读取 CSV 文件
    │   └── write_csv(df, path) -> None                     — 写入 CSV 文件
    ├── 行操作
    │   ├── get_row(df, idx) -> dict[str, Any]              — 获取单行数据
    │   ├── get_rows_by_indices(df, indices) -> list[dict]  — 批量获取多行
    │   ├── set_value(df, idx, column, value) -> DataFrame  — 设置单元格值
    │   └── set_values_batch(df, updates) -> DataFrame      — 批量设置单元格
    ├── 列操作
    │   ├── get_column_names(df) -> list[str]               — 获取所有列名
    │   ├── has_column(df, column) -> bool                  — 检查列是否存在
    │   └── add_column(df, column, default) -> DataFrame    — 添加新列
    ├── 过滤与查询
    │   ├── filter_indices(df, column, condition, value) -> list[int]
    │   │       单列条件过滤，支持 empty/not_empty/eq/ne/gt/lt/ge/le
    │   └── filter_indices_vectorized(df, input_cols, output_cols, ...) -> list[int]
    │           向量化过滤未处理行，比逐行检查快 50-100x
    ├── 值操作
    │   ├── is_empty(value) -> bool                         — 判断值是否为空
    │   ├── is_empty_vectorized(series) -> BoolSeries       — 向量化空值判断
    │   └── to_string(value) -> str                         — 值转字符串
    ├── 信息查询
    │   ├── row_count(df) -> int                            — 获取行数
    │   ├── get_index_range(df) -> tuple[int, int]          — 获取索引范围 (min, max)
    │   └── get_indices(df) -> list[int]                    — 获取所有索引
    └── 迭代与操作
        ├── iter_rows(df, columns) -> Iterator[(idx, dict)] — 迭代所有行
        ├── slice_by_index_range(df, min, max) -> DataFrame — 按索引范围切片
        └── copy(df) -> DataFrame                           — 创建深拷贝

关键类型:
    df (Any): DataFrame 对象，具体类型取决于引擎 (pd.DataFrame / pl.DataFrame)
    series (Any): 列/Series 对象 (pd.Series / pl.Series)
    value (Any): 单元格值，可以是任意 Python 类型

模块依赖:
    标准库: abc (ABC, abstractmethod), pathlib (Path), typing (Any, Iterator)

注意事项:
    - DataFrame 类型是 Any，因为不同引擎有不同的 DataFrame 类型
    - 索引操作应保持一致性（整数索引）
    - 空值判断需处理各引擎的特殊类型（NaN, None, null 等）
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator


class BaseEngine(ABC):
    """
    DataFrame 引擎抽象基类

    定义了 DataFrame 操作的标准接口。所有具体引擎实现都必须
    继承此类并实现所有抽象方法，确保接口一致性。

    设计目标:
        - 统一接口: 业务代码使用统一的方法操作 DataFrame
        - 引擎透明: 切换引擎不需要修改业务代码
        - 性能优化: 各引擎可以使用自己的优化手段

    类型说明:
        - df: Any - DataFrame 类型，取决于具体引擎
            - PandasEngine: pandas.DataFrame
            - PolarsEngine: polars.DataFrame
        - series: Any - 列/Series 类型
        - value: Any - 单元格值

    索引约定:
        - 使用整数索引（0-based）
        - 索引应与行位置一致
        - 支持索引范围操作

    空值定义:
        - None: Python 空值
        - NaN: 浮点数空值
        - "": 空字符串
        - 纯空白字符串
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        引擎名称标识

        Returns:
            str: 引擎名称（如 "pandas", "polars"）
        """
        pass

    # ==================== 文件 I/O ====================

    @abstractmethod
    def read_excel(
        self, path: Path | str, sheet_name: str | int = 0, **kwargs: Any
    ) -> Any:
        """
        读取 Excel 文件

        将 Excel 文件加载为 DataFrame。支持 .xlsx 和 .xls 格式。

        Args:
            path: 文件路径
            sheet_name: 工作表名称或索引（0-based）
            **kwargs: 引擎特定参数
                - Pandas: header, usecols, dtype, na_values 等
                - Polars: infer_schema_length, read_csv_options 等

        Returns:
            DataFrame: 引擎特定的 DataFrame 类型

        Raises:
            FileNotFoundError: 文件不存在
            IOError: 文件读取失败
        """
        pass

    @abstractmethod
    def write_excel(
        self, df: Any, path: Path | str, sheet_name: str = "Sheet1", **kwargs: Any
    ) -> None:
        """
        写入 Excel 文件

        将 DataFrame 保存为 Excel 文件。

        Args:
            df: DataFrame
            path: 输出文件路径
            sheet_name: 工作表名称
            **kwargs: 引擎特定参数
                - Pandas: index, engine, freeze_panes 等
                - Polars: row_format, column_formats 等

        Raises:
            IOError: 文件写入失败
            UnicodeEncodeError: 编码问题
        """
        pass

    @abstractmethod
    def read_csv(self, path: Path | str, **kwargs: Any) -> Any:
        """
        读取 CSV 文件

        将 CSV 文件加载为 DataFrame。

        Args:
            path: 文件路径
            **kwargs: 引擎特定参数
                - encoding, delimiter, quotechar 等

        Returns:
            DataFrame
        """
        pass

    @abstractmethod
    def write_csv(self, df: Any, path: Path | str, **kwargs: Any) -> None:
        """
        写入 CSV 文件

        将 DataFrame 保存为 CSV 文件。

        Args:
            df: DataFrame
            path: 输出文件路径
            **kwargs: 引擎特定参数
        """
        pass

    # ==================== 行操作 ====================

    @abstractmethod
    def get_row(self, df: Any, idx: int) -> dict[str, Any]:
        """
        获取指定行数据

        通过索引获取单行数据，返回字典格式。

        Args:
            df: DataFrame
            idx: 行索引（整数）

        Returns:
            dict[str, Any]: 行数据 {列名: 值, ...}

        Raises:
            KeyError: 索引不存在
        """
        pass

    @abstractmethod
    def get_rows_by_indices(self, df: Any, indices: list[int]) -> list[dict[str, Any]]:
        """
        批量获取多行数据

        一次性获取多行，比循环调用 get_row() 更高效。

        Args:
            df: DataFrame
            indices: 行索引列表

        Returns:
            list[dict[str, Any]]: 行数据字典列表
        """
        pass

    @abstractmethod
    def set_value(self, df: Any, idx: int, column: str, value: Any) -> Any:
        """
        设置单元格值

        修改指定位置的单元格值。

        Args:
            df: DataFrame
            idx: 行索引
            column: 列名
            value: 要设置的值

        Returns:
            DataFrame: 更新后的 DataFrame
                注意: Polars 返回新 DataFrame，Pandas 可能原地修改
        """
        pass

    @abstractmethod
    def set_values_batch(self, df: Any, updates: list[tuple[int, str, Any]]) -> Any:
        """
        批量设置多个单元格值

        一次性设置多个单元格，性能优于多次调用 set_value()。

        Args:
            df: DataFrame
            updates: 更新列表 [(索引, 列名, 值), ...]

        Returns:
            DataFrame: 更新后的 DataFrame
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
            list[str]: 列名列表，保持原始顺序
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
            bool: 列是否存在
        """
        pass

    @abstractmethod
    def add_column(self, df: Any, column: str, default_value: Any = None) -> Any:
        """
        添加新列

        向 DataFrame 添加新列，所有行填充相同的默认值。

        Args:
            df: DataFrame
            column: 新列名
            default_value: 所有行的默认值

        Returns:
            DataFrame: 添加新列后的 DataFrame
        """
        pass

    # ==================== 过滤与查询 ====================

    @abstractmethod
    def filter_indices(
        self, df: Any, column: str, condition: str, value: Any = None
    ) -> list[int]:
        """
        根据条件过滤行，返回符合条件的索引

        单列条件过滤。

        Args:
            df: DataFrame
            column: 列名
            condition: 条件类型
                - "empty": 值为空
                - "not_empty": 值非空
                - "eq": 等于 value
                - "ne": 不等于 value
                - "gt": 大于 value
                - "lt": 小于 value
                - "ge": 大于等于 value
                - "le": 小于等于 value
            value: 比较值（当 condition 为比较操作时需要）

        Returns:
            list[int]: 符合条件的行索引列表
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

        高效的批量过滤操作，比逐行检查快 50-100 倍。

        未处理的定义:
            - 输入列: 根据 require_all_inputs 决定是否要求全部非空
            - 输出列: 任一输出列为空

        Args:
            df: DataFrame
            input_columns: 输入列名列表
            output_columns: 输出列名列表
            require_all_inputs: 输入列条件
                - True: 所有输入列都非空（AND 逻辑）
                - False: 任一输入列非空（OR 逻辑）
            index_offset: 索引偏移量
                用于切片后的 DataFrame，将局部索引转换为全局索引

        Returns:
            list[int]: 未处理的行索引列表（全局索引）
        """
        pass

    # ==================== 值操作 ====================

    @abstractmethod
    def is_empty(self, value: Any) -> bool:
        """
        判断值是否为空

        统一的空值判断逻辑，处理各引擎的特殊空值类型。

        空值定义:
            - None
            - NaN (float('nan'), numpy.nan, pandas.NA)
            - 空字符串 ""
            - 纯空白字符串（全是空格、制表符等）

        Args:
            value: 要检查的值

        Returns:
            bool: 是否为空
        """
        pass

    @abstractmethod
    def is_empty_vectorized(self, series: Any) -> Any:
        """
        向量化判断空值

        对整列进行空值判断，返回布尔 Series。
        性能优于逐个调用 is_empty()。

        Args:
            series: Series/列

        Returns:
            布尔 Series，True 表示对应位置为空
        """
        pass

    @abstractmethod
    def to_string(self, value: Any) -> str:
        """
        将值转换为字符串

        空值转换为空字符串，其他值转为字符串表示。

        Args:
            value: 值

        Returns:
            str: 字符串表示
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
            int: 行数
        """
        pass

    @abstractmethod
    def get_index_range(self, df: Any) -> tuple[int, int]:
        """
        获取索引范围

        返回 DataFrame 的最小和最大索引。

        Args:
            df: DataFrame

        Returns:
            tuple[int, int]: (最小索引, 最大索引)
        """
        pass

    @abstractmethod
    def get_indices(self, df: Any) -> list[int]:
        """
        获取所有索引

        Args:
            df: DataFrame

        Returns:
            list[int]: 索引列表
        """
        pass

    # ==================== 迭代器 ====================

    @abstractmethod
    def iter_rows(
        self, df: Any, columns: list[str] | None = None
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        """
        迭代所有行

        生成器函数，逐行返回数据。

        Args:
            df: DataFrame
            columns: 要提取的列名列表
                - None: 提取所有列
                - list[str]: 只提取指定列

        Yields:
            tuple[int, dict[str, Any]]: (索引, 行数据字典)
        """
        pass

    # ==================== DataFrame 操作 ====================

    @abstractmethod
    def slice_by_index_range(self, df: Any, min_idx: int, max_idx: int) -> Any:
        """
        按索引范围切片

        提取指定索引范围内的行。

        Args:
            df: DataFrame
            min_idx: 最小索引（包含）
            max_idx: 最大索引（包含）

        Returns:
            DataFrame: 切片后的 DataFrame
        """
        pass

    @abstractmethod
    def copy(self, df: Any) -> Any:
        """
        创建 DataFrame 副本

        深拷贝 DataFrame，修改副本不影响原始数据。

        Args:
            df: DataFrame

        Returns:
            DataFrame: 副本
        """
        pass
