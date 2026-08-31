"""
Excel 数据源任务池实现模块

本模块提供基于 DataFrame 的 Excel/CSV 文件任务池实现。
通过抽象引擎层，统一支持 Pandas 和 Polars 两种 DataFrame 框架，
并可选用高性能读写器提升 I/O 效率。

核心特性:
    - 多引擎支持: Pandas (兼容性好) 和 Polars (高性能)
    - 高性能读取: 可选 calamine (Rust) 引擎，比 openpyxl 快 10 倍
    - 高性能写入: 可选 xlsxwriter，比 openpyxl 快 3 倍
    - 向量化过滤: 使用 DataFrame 原生操作，避免逐行遍历
    - 自动保存: 定时持久化，防止数据丢失
    - 编码安全: Unicode 写入失败时不修改结果、不切换目标并返回失败
    - CSV 兼容: 自动检测文件类型，同一接口处理 Excel 和 CSV

架构设计:
    ┌─────────────────────────────────────────────────┐
    │              ExcelTaskPool                       │
    │  ┌─────────────┐   ┌─────────────────────────┐  │
    │  │ 任务队列    │   │ DataFrame (引擎特定)    │  │
    │  │ tasks[]     │   │ df                       │  │
    │  └─────────────┘   └─────────────────────────┘  │
    │         │                    │                   │
    │         ▼                    ▼                   │
    │  ┌─────────────────────────────────────────────┐│
    │  │            BaseEngine (抽象层)              ││
    │  │  ┌──────────────┐    ┌──────────────┐      ││
    │  │  │ PandasEngine │    │ PolarsEngine │      ││
    │  │  └──────────────┘    └──────────────┘      ││
    │  └─────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────┘

类清单:
    ExcelTaskPool(BaseTaskPool):
        Excel/CSV 数据源任务池，管理内存 DataFrame 与任务队列的同步。

        公开方法（BaseTaskPool 接口实现）:
            - __init__(input_path, output_path, columns_to_extract, columns_to_write,
                       save_interval, require_all_input_fields, engine_type,
                       excel_reader, excel_writer)
                初始化：读取文件 → 验证列 → 创建输出列
            - get_total_task_count() -> int
                统计未处理任务数（向量化过滤）
            - get_processed_task_count() -> int
                统计已处理任务数（遍历检查输出列）
            - get_id_boundaries() -> tuple[int, int]
                获取 DataFrame 索引边界 (min_idx, max_idx)
            - initialize_shard(shard_id, min_idx, max_idx) -> int
                加载分片到内存队列（向量化过滤 + 逐行提取）
            - get_task_batch(batch_size) -> list[tuple]
                从内存队列弹出一批任务
            - update_task_results(results) -> None
                写入 DataFrame + 检查自动保存间隔
            - reload_task_data(idx) -> dict | None
                从 DataFrame 重新读取指定索引的输入数据
            - close() -> None
                执行最终保存

        Token 估算方法:
            - sample_unprocessed_rows(sample_size) -> list[dict]
            - sample_processed_rows(sample_size) -> list[dict]
            - fetch_all_rows(columns) -> list[dict]
            - fetch_all_processed_rows(columns) -> list[dict]

        内部方法:
            - _validate_and_prepare_columns() -> None
                验证输入列存在、创建缺失的输出列
            - _filter_unprocessed_indices(min_idx, max_idx) -> list[int]
                向量化过滤未处理索引（核心性能优化）
            - _save_excel() -> None
                原子保存到配置的 Excel/CSV 目标

        关键属性:
            - input_path (Path): 输入文件路径
            - output_path (Path): 输出文件路径
            - engine (BaseEngine): DataFrame 引擎实例（Pandas/Polars）
            - df: 内存中的 DataFrame
            - save_interval (int): 自动保存间隔（秒）
            - last_save_time (float): 上次保存时间戳
            - _is_csv (bool): 是否为 CSV 文件

    模块依赖:
        - base.BaseTaskPool: 抽象基类
        - engines.get_engine: 引擎工厂函数
        - engines.BaseEngine: 引擎抽象基类

性能优化:
    1. 向量化过滤: filter_indices_vectorized() 比逐行快 50-100 倍
    2. 批量更新: set_values_batch() 减少内存分配
    3. 延迟保存: save_interval 机制减少磁盘 I/O
    4. 高性能库: calamine + xlsxwriter 组合最优

使用示例:
    from src.data.excel import ExcelTaskPool

    pool = ExcelTaskPool(
        input_path="data/input.xlsx",
        output_path="data/output.xlsx",
        columns_to_extract=["title", "content"],
        columns_to_write={"result": "ai_result", "score": "ai_score"},
        save_interval=300,  # 5分钟自动保存
        engine_type="auto",  # 自动选择引擎
        excel_reader="calamine",  # 使用高性能读取器
        excel_writer="xlsxwriter",  # 使用高性能写入器
    )

    # 获取任务批次
    batch = pool.get_task_batch(100)

    # 处理后更新结果
    results = {0: {"result": "分析结果", "score": "0.95"}}
    pool.update_task_results(results)

    # 关闭并保存
    pool.close()

配置选项:
    engine_type: "pandas" | "polars" | "auto"
        - pandas: 兼容性最好，内存占用较高
        - polars: 高性能，多线程，内存效率高
        - auto: 优先 polars，不可用时回退 pandas

    excel_reader: "openpyxl" | "calamine" | "auto"
        - openpyxl: 纯 Python，功能完整
        - calamine: Rust 实现，速度 10x，仅支持读取
        - auto: 优先 calamine

    excel_writer: "openpyxl" | "xlsxwriter" | "auto"
        - openpyxl: 支持读写，功能完整
        - xlsxwriter: 仅写入，速度 3x，格式支持更好
        - auto: 优先 xlsxwriter

注意事项:
    1. 大文件 (>100MB) 建议使用 polars + calamine
    2. 需要保留格式时使用 openpyxl
    3. CSV 文件不需要 excel_reader/excel_writer 配置
    4. 自动保存在锁外执行，避免阻塞
"""

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, cast

from .base import BaseTaskPool
from .contracts import (
    AdapterCapabilities,
    CommitDisposition,
    TaskBatch,
    TaskRecord,
    WritebackItem,
    WritebackReceipt,
)
from .engines import (
    BaseEngine,
    EngineType,
    ReaderType,
    WriterType,
    get_engine,
)


class ExcelTaskPool(BaseTaskPool):
    """
    Excel/CSV 数据源任务池

    从 Excel 或 CSV 文件读取任务数据，AI 处理后写回结果。
    核心职责是管理内存中的 DataFrame 和任务队列的同步。

    工作流程:
        1. 初始化: 读取文件 → 验证列 → 创建输出列
        2. 分片加载: 过滤未处理索引 → 提取数据 → 填充任务队列
        3. 任务获取: 从队列弹出批次
        4. 结果更新: 写入 DataFrame → 检查保存间隔
        5. 关闭: 执行最终保存

    Attributes:
        input_path (Path): 输入文件路径
        output_path (Path): 输出文件路径
        save_interval (int): 自动保存间隔（秒）
        last_save_time (float): 上次保存时间戳
        engine (BaseEngine): DataFrame 引擎实例
        df: 当前 DataFrame（Pandas 或 Polars）
        _is_csv (bool): 是否为 CSV 文件
        current_shard_id (int): 当前分片 ID
        current_min_idx (int): 当前分片最小索引
        current_max_idx (int): 当前分片最大索引

    线程安全:
        - 使用 self.lock 保护 DataFrame 和任务队列
        - 保存操作在锁外执行（避免长时间阻塞）
    """

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        columns_to_extract: list[str],
        columns_to_write: dict[str, str],
        save_interval: int = 300,
        require_all_input_fields: bool = True,
        engine_type: str = "pandas",
        excel_reader: str = "auto",
        excel_writer: str = "auto",
    ):
        """
        初始化 Excel/CSV 任务池

        创建流程:
            1. 验证输入文件存在
            2. 初始化基类（设置列配置）
            3. 获取并配置 DataFrame 引擎
            4. 检测文件类型（CSV 或 Excel）
            5. 读取文件到内存
            6. 验证和准备列

        Args:
            input_path: 输入文件路径（Excel 或 CSV）
            output_path: 输出文件路径（可与输入相同，原地修改）
            columns_to_extract: 需要提取的输入列名列表
                例: ["title", "content", "category"]
            columns_to_write: AI 输出字段映射 {别名: 实际列名}
                例: {"result": "ai_result", "confidence": "ai_confidence"}
            save_interval: 自动保存间隔（秒），默认 300（5分钟）
                设为 0 禁用自动保存（不推荐）
            require_all_input_fields: 是否要求所有输入字段都非空
                - True: 所有输入列都有值才视为有效任务
                - False: 任一输入列有值即为有效任务
            engine_type: DataFrame 引擎类型
                - "pandas": 使用 Pandas（兼容性好）
                - "polars": 使用 Polars（高性能）
                - "auto": 优先 Polars，不可用时回退 Pandas
            excel_reader: Excel 读取器（仅对 .xlsx 有效）
                - "openpyxl": 纯 Python 实现
                - "calamine": Rust 高性能实现（需安装 python-calamine）
                - "auto": 优先 calamine
            excel_writer: Excel 写入器（仅对 .xlsx 有效）
                - "openpyxl": 支持读写
                - "xlsxwriter": 仅写入，性能更好
                - "auto": 优先 xlsxwriter

        Raises:
            FileNotFoundError: 输入文件不存在
            IOError: 文件读取失败（格式错误、编码问题等）
            KeyError: 指定的列在文件中不存在

        示例:
            # 基本用法
            pool = ExcelTaskPool(
                input_path="data/tasks.xlsx",
                output_path="data/results.xlsx",
                columns_to_extract=["title"],
                columns_to_write={"result": "ai_result"},
            )

            # 高性能配置
            pool = ExcelTaskPool(
                input_path="big_data.xlsx",
                output_path="big_data_out.xlsx",
                columns_to_extract=["content"],
                columns_to_write={"summary": "ai_summary"},
                engine_type="polars",
                excel_reader="calamine",
                excel_writer="xlsxwriter",
                save_interval=60,  # 大文件建议更频繁保存
            )
        """
        # 验证输入文件
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Excel 输入文件不存在: {self.input_path}")

        # 初始化基类
        super().__init__(columns_to_extract, columns_to_write, require_all_input_fields)

        # 获取 DataFrame 引擎 (支持高性能读写器配置)
        self.engine: BaseEngine = get_engine(
            engine_type=cast(EngineType, engine_type),
            excel_reader=cast(ReaderType, excel_reader),
            excel_writer=cast(WriterType, excel_writer),
        )
        logging.info(f"使用 DataFrame 引擎: {self.engine.name}")

        # 显示读写器信息
        if hasattr(self.engine, "excel_reader"):
            logging.info(f"  - Excel 读取器: {self.engine.excel_reader}")
        if hasattr(self.engine, "excel_writer"):
            logging.info(f"  - Excel 写入器: {self.engine.excel_writer}")

        # 自动检测文件类型（CSV 或 Excel）
        self._is_csv = self.input_path.suffix.lower() == ".csv"

        # 读取文件（CSV 或 Excel）
        file_type = "CSV" if self._is_csv else "Excel"
        logging.info(f"正在读取 {file_type} 文件: {self.input_path}")
        try:
            if self._is_csv:
                self.df = self.engine.read_csv(self.input_path)
            else:
                self.df = self.engine.read_excel(self.input_path)
            row_count = self.engine.row_count(self.df)
            logging.info(f"{file_type} 文件读取成功，共 {row_count} 行")
        except Exception as e:
            raise IOError(f"无法读取 {file_type} 文件 {self.input_path}: {e}") from e

        # 保存相关
        self.save_interval = save_interval
        self.last_save_time = time.time()

        # 分片状态
        self.current_shard_id = -1
        self.current_min_idx = 0
        self.current_max_idx = 0

        # 列验证和准备
        self._validate_and_prepare_columns()

        logging.info(
            f"ExcelTaskPool 初始化完成 | 输入: {self.input_path}, 输出: {self.output_path}"
        )

    def _validate_and_prepare_columns(self) -> None:
        """
        验证和准备 DataFrame 列

        执行两项检查:
        1. 输入列验证: 检查 columns_to_extract 中的列是否存在
        2. 输出列准备: 如果 columns_to_write 中的列不存在，则创建

        注意:
            - 输入列不存在只发出警告，不阻止运行
            - 输出列不存在会自动创建（值为 None）
        """
        column_names = self.engine.get_column_names(self.df)

        # 检查输入列是否存在
        missing_extract = [c for c in self.columns_to_extract if c not in column_names]
        if missing_extract:
            logging.warning(f"输入列 {missing_extract} 在 Excel 中不存在")

        # 创建输出列（如果不存在）
        for alias, out_col in self.columns_to_write.items():
            if not self.engine.has_column(self.df, out_col):
                logging.warning(f"输出列 '{out_col}' 不存在，将创建新列")
                self.df = self.engine.add_column(self.df, out_col, None)

    # ==================== 核心接口实现 ====================

    def get_total_task_count(self) -> int:
        """
        获取未处理任务总数

        扫描整个 DataFrame，统计满足以下条件的行数:
        1. 输入列条件满足（根据 require_all_input_fields 配置）
        2. 任一输出列为空

        Returns:
            int: 未处理任务数量

        注意:
            此方法会扫描全量数据，大文件耗时较长。
            建议在启动时调用一次，而非循环中调用。
        """
        logging.info("正在计算 Excel 中未处理的任务总数...")

        min_idx, max_idx = self.engine.get_index_range(self.df)
        unprocessed = self._filter_unprocessed_indices(min_idx, max_idx)
        count = len(unprocessed)

        logging.info(f"Excel 中未处理的任务总数: {count}")
        return count

    def get_processed_task_count(self) -> int:
        """
        获取已处理任务总数

        统计所有输出列都有非空值的行数。
        用于进度统计和 Token 估算采样。

        Returns:
            int: 已处理任务数量

        算法:
            遍历所有行，检查每行的所有输出列是否都非空。
            使用引擎的 is_empty() 方法统一判断空值
            （包括 None、NaN、空字符串等）。
        """
        logging.info("正在计算 Excel 中已处理的任务总数...")

        output_columns = list(self.columns_to_write.values())
        if not output_columns:
            return 0

        processed_count = 0
        with self.lock:
            all_indices = self.engine.get_indices(self.df)

            for idx in all_indices:
                try:
                    row_data = self.engine.get_row(self.df, idx)
                    all_filled = True
                    for col in output_columns:
                        value = row_data.get(col)
                        # 使用引擎的 is_empty 方法统一处理空值判断
                        # 包括 None、NaN、空字符串等各种空值类型
                        if self.engine.is_empty(value):
                            all_filled = False
                            break

                    if all_filled:
                        processed_count += 1
                except Exception:
                    continue

        logging.info(f"Excel 中已处理的任务总数: {processed_count}")
        return processed_count

    def get_id_boundaries(self) -> tuple[int, int]:
        """
        获取 DataFrame 索引边界

        返回 DataFrame 的最小和最大索引值。
        用于分片调度器划分工作区间。

        Returns:
            tuple[int, int]: (最小索引, 最大索引)
            如果 DataFrame 为空，返回 (0, -1)

        注意:
            Excel/CSV 使用行号作为索引（从 0 开始）。
            如果 DataFrame 有自定义索引，使用引擎方法获取实际范围。
        """
        if self.engine.row_count(self.df) == 0:
            return (0, -1)

        min_idx, max_idx = self.engine.get_index_range(self.df)
        logging.info(f"Excel DataFrame 索引范围: {min_idx} - {max_idx}")
        return (min_idx, max_idx)

    def initialize_shard(self, shard_id: int, min_idx: int, max_idx: int) -> int:
        """
        初始化分片，加载指定范围的未处理任务

        分片调度的核心方法。将指定索引范围内的未处理任务
        加载到内存任务队列中，供后续 get_task_batch() 获取。

        工作流程:
            1. 过滤未处理索引（向量化操作）
            2. 提取每行的输入列数据
            3. 构建 (索引, 数据字典) 元组列表
            4. 更新内存任务队列

        Args:
            shard_id: 分片标识符（用于日志）
            min_idx: 分片起始索引（包含）
            max_idx: 分片结束索引（包含）

        Returns:
            int: 实际加载的任务数量

        注意:
            - 使用向量化过滤，性能比逐行遍历快 50-100 倍
            - 任务队列会被完全替换，而非追加
            - 分片状态（current_shard_id 等）会被更新
        """
        logging.info(f"开始初始化分片 {shard_id} (索引范围: {min_idx}-{max_idx})...")

        shard_tasks: list[tuple[Any, dict[str, Any]]] = []

        try:
            # 过滤未处理的索引
            unprocessed_indices = self._filter_unprocessed_indices(min_idx, max_idx)

            if unprocessed_indices:
                logging.debug(
                    f"分片 {shard_id}: 找到 {len(unprocessed_indices)} 个未处理索引，正在提取数据..."
                )

                for idx in unprocessed_indices:
                    try:
                        row_data = self.engine.get_row(self.df, idx)
                        record_dict = {
                            col: self.engine.to_string(row_data.get(col, ""))
                            for col in self.columns_to_extract
                        }
                        shard_tasks.append((idx, record_dict))
                    except Exception as e:
                        logging.error(
                            f"分片 {shard_id}: 提取索引 {idx} 数据时出错: {e}"
                        )
            else:
                logging.info(f"分片 {shard_id}: 在指定索引范围内未找到未处理的任务")

        except Exception as e:
            logging.error(
                f"初始化分片 {shard_id} (索引 {min_idx}-{max_idx}) 失败: {e}",
                exc_info=True,
            )
            shard_tasks = []

        # 更新任务队列
        with self.lock:
            self.tasks = shard_tasks

        # 更新分片状态
        self.current_shard_id = shard_id
        self.current_min_idx = min_idx
        self.current_max_idx = max_idx

        loaded_count = len(shard_tasks)
        logging.info(
            f"分片 {shard_id} (索引范围: {min_idx}-{max_idx}) 初始化完成，加载任务数: {loaded_count}"
        )

        return loaded_count

    def get_task_batch(self, batch_size: int) -> list[tuple[Any, dict[str, Any]]]:
        """
        从内存任务队列获取一批任务

        从队列头部弹出指定数量的任务，用于并发处理。

        Args:
            batch_size: 请求的任务数量

        Returns:
            list[tuple[Any, dict[str, Any]]]: 任务列表
                - 元组第一个元素是索引（用于结果写回）
                - 元组第二个元素是输入数据字典
            如果队列不足，返回剩余全部任务。

        线程安全:
            使用 self.lock 保护队列操作。
        """
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            atomic_batch=False,
            idempotent_write=True,
            resumable=True,
            full_scan=True,
        )

    async def scan(self, cursor: Any | None, limit: int) -> TaskBatch:
        """Read a stable file page using the DataFrame index as an opaque cursor."""

        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        if cursor is not None and (
            not isinstance(cursor, dict) or set(cursor) != {"last_index"}
        ):
            raise ValueError("invalid Excel/CSV cursor")

        def read_page() -> TaskBatch:
            min_idx, max_idx = self.get_id_boundaries()
            if max_idx < min_idx:
                return TaskBatch(records=(), next_cursor=None)

            indices = self._filter_unprocessed_indices(min_idx, max_idx)
            if cursor is not None:
                last_index = cursor["last_index"]
                indices = [idx for idx in indices if idx > last_index]
            page = indices[:limit]

            records: list[TaskRecord] = []
            for raw_index in page:
                record_id = (
                    raw_index.item() if hasattr(raw_index, "item") else raw_index
                )
                row_data = self.engine.get_row(self.df, raw_index)
                records.append(
                    TaskRecord(
                        record_id=record_id,
                        data={
                            col: self.engine.to_string(row_data.get(col, ""))
                            for col in self.columns_to_extract
                        },
                    )
                )

            next_cursor = (
                {"last_index": records[-1].record_id} if len(records) == limit else None
            )
            return TaskBatch(records=tuple(records), next_cursor=next_cursor)

        return await asyncio.to_thread(read_page)

    def update_task_results(
        self,
        batch_id: str,
        results: dict[int, dict[str, Any]],
    ) -> WritebackReceipt:
        """Update rows and acknowledge them only after an atomic file replace."""

        if not results:
            return WritebackReceipt.committed(batch_id, (), atomic=False)

        updated_indices: list[int] = []
        failures: dict[int, WritebackItem] = {}
        original_df = self.engine.copy(self.df)

        try:
            with self.lock:
                all_indices = set(self.engine.get_indices(self.df))
                for idx, row_result in results.items():
                    if "_error" in row_result:
                        failures[idx] = WritebackItem(
                            idx,
                            CommitDisposition.REJECTED,
                            "invalid_result",
                            "结果包含 _error",
                            False,
                        )
                        continue
                    if idx not in all_indices:
                        failures[idx] = WritebackItem(
                            record_id=idx,
                            disposition=CommitDisposition.REJECTED,
                            code="record_not_found",
                            message=f"索引 {idx} 不存在",
                            retryable=False,
                        )
                        continue

                    writable_fields = [
                        (alias, col_name)
                        for alias, col_name in self.columns_to_write.items()
                        if alias in row_result
                    ]
                    if not writable_fields:
                        failures[idx] = WritebackItem(
                            record_id=idx,
                            disposition=CommitDisposition.REJECTED,
                            code="no_writable_fields",
                            message="AI 结果不包含任何 columns_to_write alias",
                            retryable=False,
                        )
                        continue

                    record_failed = False
                    row_before = self.engine.get_row(self.df, idx)
                    applied_columns: list[str] = []
                    for alias, col_name in writable_fields:
                        if not self.engine.has_column(self.df, col_name):
                            record_failed = True
                            failures[idx] = WritebackItem(
                                record_id=idx,
                                disposition=CommitDisposition.REJECTED,
                                code="column_not_found",
                                message=f"输出列 {col_name} 不存在",
                                retryable=False,
                            )
                            break
                        try:
                            self.df = self.engine.set_value(
                                self.df, idx, col_name, row_result[alias]
                            )
                            applied_columns.append(col_name)
                        except Exception as exc:
                            record_failed = True
                            failures[idx] = WritebackItem(
                                record_id=idx,
                                disposition=CommitDisposition.REJECTED,
                                code="set_value_failed",
                                message=f"列 {col_name}: {exc}",
                                retryable=True,
                            )
                            break
                    if record_failed:
                        try:
                            for col_name in applied_columns:
                                self.df = self.engine.set_value(
                                    self.df,
                                    idx,
                                    col_name,
                                    row_before.get(col_name),
                                )
                        except Exception:
                            self.df = original_df
                            raise
                    if not record_failed:
                        updated_indices.append(idx)

            if updated_indices:
                self._save_excel()
                self.last_save_time = time.time()
        except Exception:
            self.df = original_df
            raise

        return WritebackReceipt(
            batch_id=batch_id,
            submitted_ids=tuple(results),
            items=tuple(
                (
                    WritebackItem(idx, CommitDisposition.COMMITTED)
                    if idx in updated_indices
                    else failures[idx]
                )
                for idx in results
            ),
            atomic=False,
        )

    def reload_task_data(self, idx: int) -> dict[str, Any] | None:
        """
        重新加载任务的原始输入数据

        从 DataFrame 中重新读取指定索引的输入列数据。
        用于 API 错误重试时重新获取原始数据，避免使用
        可能被污染的任务元数据。

        Args:
            idx: DataFrame 索引

        Returns:
            dict[str, Any] | None: 输入数据字典，如果索引不存在返回 None

        使用场景:
            当 API 调用失败需要重试时，RetryStrategy 会调用此方法
            重新获取干净的输入数据，确保重试使用正确的数据。
        """
        try:
            with self.lock:
                all_indices = self.engine.get_indices(self.df)
                if idx not in all_indices:
                    logging.warning(
                        f"尝试重载数据失败: 索引 {idx} 在 DataFrame 中不存在"
                    )
                    return None

                row_data = self.engine.get_row(self.df, idx)
                record_dict = {
                    col: self.engine.to_string(row_data.get(col, ""))
                    for col in self.columns_to_extract
                }
                return record_dict

        except Exception as e:
            logging.error(f"重载索引 {idx} 数据时发生错误: {e}", exc_info=True)
            return None

    def close(self) -> None:
        """
        关闭任务池并执行最终保存

        在处理结束时调用，确保所有内存中的数据都被持久化。

        注意:
            - 即使保存失败也不会抛出异常（已记录错误日志）
            - 调用后不应再使用此任务池实例
        """
        logging.info("正在执行 Excel 文件的最终保存操作...")
        self._save_excel()

    def close_readonly(self) -> None:
        """Excel token estimation owns no external handle and must not save."""

        self.clear_tasks()

    # ==================== 内部方法 ====================

    def _filter_unprocessed_indices(self, min_idx: int, max_idx: int) -> list[int]:
        """
        过滤指定范围内的未处理索引

        使用向量化操作，性能比逐行遍历快 50-100 倍。
        """
        # 使用引擎的向量化过滤方法
        output_columns = list(self.columns_to_write.values())

        # 获取范围内的子集
        sub_df = self.engine.slice_by_index_range(self.df, min_idx, max_idx)

        if self.engine.row_count(sub_df) == 0:
            return []

        # 向量化过滤
        try:
            unprocessed = self.engine.filter_indices_vectorized(
                sub_df,
                self.columns_to_extract,
                output_columns,
                self.require_all_input_fields,
                index_offset=min_idx,
            )

            logging.debug(
                f"过滤索引范围 {min_idx}-{max_idx} 完成，找到 {len(unprocessed)} 个未处理索引"
            )
            return unprocessed

        except Exception as e:
            logging.error(f"过滤未处理索引时出错: {e}", exc_info=True)
            raise

    def _save_excel(self) -> None:
        """
        保存文件（Excel 或 CSV）

        根据文件类型和配置的输出路径选择保存方式。
        只有配置目标的原子替换成功才会返回。
        """
        # 检查输出路径是否为 CSV（可能输入输出格式不同）
        output_is_csv = self.output_path.suffix.lower() == ".csv"
        logging.info(f"正在尝试保存 DataFrame 到: {self.output_path}")

        try:
            with self.lock:
                # 确保输出目录存在
                output_dir = self.output_path.parent
                if output_dir and not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)

                # CSV 文件直接保存
                if self._is_csv or output_is_csv:
                    self._atomic_write(self.df, self.output_path, csv=True)
                    logging.info(f"✅ DataFrame 已成功保存到: {self.output_path}")
                    return

                self._atomic_write(self.df, self.output_path, csv=False)
                logging.info(f"✅ DataFrame 已成功保存到: {self.output_path}")
                return

        except Exception as e:
            logging.error(f"❌ 保存文件失败: {e}", exc_info=True)
            raise IOError(f"保存文件失败: {e}") from e

    def _atomic_write(self, df: Any, destination: Path, *, csv: bool) -> None:
        """Write beside the destination, fsync it, then atomically replace."""

        suffix = destination.suffix or (".csv" if csv else ".xlsx")
        temp_path = destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}.tmp{suffix}"
        )
        try:
            if csv:
                self.engine.write_csv(df, temp_path)
            else:
                self.engine.write_excel(df, temp_path)
            with temp_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temp_path, destination)
            try:
                directory_fd = os.open(str(destination.parent), os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    # ==================== Token 估算采样 ====================

    def sample_unprocessed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """
        采样未处理的行 (用于输入 token 估算)

        Args:
            sample_size: 采样数量

        Returns:
            采样数据列表 [{column: value, ...}, ...]
        """
        min_idx, max_idx = self.engine.get_index_range(self.df)
        unprocessed_indices = self._filter_unprocessed_indices(min_idx, max_idx)

        if not unprocessed_indices:
            return []

        # 取前 sample_size 个
        sample_indices = unprocessed_indices[:sample_size]
        samples = []

        with self.lock:
            for idx in sample_indices:
                try:
                    row_data = self.engine.get_row(self.df, idx)
                    record_dict = {
                        col: self.engine.to_string(row_data.get(col, ""))
                        for col in self.columns_to_extract
                    }
                    samples.append(record_dict)
                except Exception as e:
                    logging.warning(f"采样索引 {idx} 失败: {e}")

        logging.info(f"采样 {len(samples)} 条未处理记录用于输入 token 估算")
        return samples

    def sample_processed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """
        采样已处理的行 (用于输出 token 估算)

        Args:
            sample_size: 采样数量

        Returns:
            采样数据列表 [{column: value, ...}, ...]，包含输出列
        """
        output_columns = list(self.columns_to_write.values())

        # 过滤已处理的行 (输出列都非空)
        processed_indices = []

        with self.lock:
            all_indices = self.engine.get_indices(self.df)

            for idx in all_indices:
                try:
                    row_data = self.engine.get_row(self.df, idx)

                    # 检查所有输出列是否都有值
                    all_filled = True
                    for col in output_columns:
                        value = row_data.get(col)
                        if value is None or (
                            isinstance(value, str) and not value.strip()
                        ):
                            all_filled = False
                            break

                    if all_filled:
                        processed_indices.append(idx)
                        if len(processed_indices) >= sample_size:
                            break
                except Exception:
                    continue

        if not processed_indices:
            return []

        # 提取数据
        samples = []
        with self.lock:
            for idx in processed_indices:
                try:
                    row_data = self.engine.get_row(self.df, idx)
                    # 只提取输出列
                    record_dict = {
                        col: self.engine.to_string(row_data.get(col, ""))
                        for col in output_columns
                    }
                    samples.append(record_dict)
                except Exception as e:
                    logging.warning(f"采样已处理索引 {idx} 失败: {e}")

        logging.info(f"采样 {len(samples)} 条已处理记录用于输出 token 估算")
        return samples

    def fetch_all_rows(self, columns: list[str]) -> list[dict[str, Any]]:
        """
        获取所有行 (忽略处理状态)

        Args:
            columns: 需要提取的列名列表

        Returns:
            所有行的数据列表 [{column: value, ...}, ...]
        """
        all_rows = []

        with self.lock:
            all_indices = self.engine.get_indices(self.df)

            for idx in all_indices:
                try:
                    row_data = self.engine.get_row(self.df, idx)
                    record_dict = {
                        col: self.engine.to_string(row_data.get(col, ""))
                        for col in columns
                    }
                    all_rows.append(record_dict)
                except Exception as e:
                    logging.warning(f"获取索引 {idx} 数据失败: {e}")

        logging.info(f"已获取 {len(all_rows)} 条记录 (忽略处理状态)")
        return all_rows

    def fetch_all_processed_rows(self, columns: list[str]) -> list[dict[str, Any]]:
        """
        获取所有已处理行 (仅输出已完成的记录)

        Args:
            columns: 需要提取的列名列表

        Returns:
            已处理行的数据列表 [{column: value, ...}, ...]
        """
        output_columns = list(self.columns_to_write.values())
        if not output_columns:
            return []

        processed_rows = []
        with self.lock:
            all_indices = self.engine.get_indices(self.df)

            for idx in all_indices:
                try:
                    row_data = self.engine.get_row(self.df, idx)
                    all_filled = True
                    for col in output_columns:
                        value = row_data.get(col)
                        if value is None or (
                            isinstance(value, str) and not value.strip()
                        ):
                            all_filled = False
                            break

                    if not all_filled:
                        continue

                    record_dict = {
                        col: self.engine.to_string(row_data.get(col, ""))
                        for col in columns
                    }
                    processed_rows.append(record_dict)
                except Exception as e:
                    logging.warning(f"获取已处理索引 {idx} 数据失败: {e}")

        logging.info(f"已获取 {len(processed_rows)} 条已处理记录")
        return processed_rows
