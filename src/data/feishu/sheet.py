"""
飞书电子表格（Sheet）数据源任务池

本模块实现基于飞书电子表格的 TaskPool，适用于将电子表格作为
AI-DataFlux 的数据源进行批量 AI 处理。

核心设计:
    1. 快照读取 —— 初始化时一次性拉取全部数据到内存二维数组
    2. 行号映射 —— 使用数据行索引（0-based）作为 task_id，实际行号 = task_id + 2 [已修正: 原注释误述为 1-based 行号]
    3. 串行写入 —— 飞书电子表格单文档须串行写入，写入时加锁
    4. 列名映射 —— 表头行建立 列名 → 列索引（0-based 整数）的映射 [已修正: 原注释误述为列字母]

电子表格范围约定:
    - 行号从 1 开始，第 1 行为表头
    - 数据行从第 2 行开始
    - 列号使用 A, B, C ... AA, AB 等字母表示
    - 范围格式: "{sheet_id}!A1:Z1000"

模块级函数:
    _col_index_to_letter(index) → str
        - 功能: 将 0-based 列索引转换为 Excel 风格列字母
        - 输入: index — 列索引（0=A, 25=Z, 26=AA）
        - 输出: 列字母字符串

类清单:
    FeishuSheetTaskPool(BaseTaskPool)
        飞书电子表格数据源任务池，继承自 BaseTaskPool

关键变量:
    _header_row         — list[str]: 表头行列名列表
    _data_rows          — list[list[Any]]: 数据行快照（二维数组，不含表头）
    _col_name_to_index  — dict[str, int]: 列名 → 0-based 列索引映射
    _snapshot_loaded    — bool: 快照是否已加载（双重检查锁保护）

方法清单:
    快照管理:
        _load_snapshot_sync()                   — 同步加载快照（线程安全，双重检查锁）
        _load_snapshot()                        — [async] 从飞书拉取全部工作表数据
        _get_cell(row, col_name) → str          — 获取行中指定列的值
        _is_unprocessed(row) → bool             — 判断行是否未处理（输入有效 & 输出缺失）
        _is_processed(row) → bool               — 判断行是否已处理（所有输出列非空）

    BaseTaskPool 接口实现:
        get_total_task_count() → int            — 统计未处理任务总数
        get_processed_task_count() → int        — 统计已处理任务总数
        get_id_boundaries() → (int, int)        — 返回 task_id 边界 (0, N-1)
        initialize_shard(shard_id, min_id, max_id) → int — 加载分片内未处理任务
        get_task_batch(batch_size) → list        — 从内存队列弹出一批任务
        update_task_results(results)             — 批量写回结果到飞书电子表格
        reload_task_data(task_id) → dict|None    — 从快照重新加载任务数据
        close()                                  — 关闭飞书客户端连接

    写入辅助:
        _write_results(col_data)                — [async] 按列分组写入，连续行合并为范围写入
        _group_consecutive(sorted_rows) → list  — [static] 将排序行按连续行号分组

    Token 估算采样:
        sample_unprocessed_rows(sample_size) → list — 采样未处理行（用于 Token 估算）
        sample_processed_rows(sample_size) → list   — 采样已处理行（用于 Token 估算）

模块依赖:
    logging, threading          — 日志与线程安全
    ..base.BaseTaskPool         — 任务池基类
    .run_async                  — 同步-异步桥接
    .client.FeishuClient        — 飞书 HTTP 客户端

使用示例:
    pool = FeishuSheetTaskPool(
        app_id="cli_xxx",
        app_secret="xxx",
        spreadsheet_token="shtcnXXX",
        sheet_id="0",
        columns_to_extract=["问题", "上下文"],
        columns_to_write={"answer": "AI回答", "category": "分类"},
    )
"""

import logging
import threading
from typing import Any

from ..base import BaseTaskPool
from ..contracts import (
    AdapterCapabilities,
    CommitDisposition,
    WritebackItem,
    WritebackReceipt,
)
from . import run_async
from .client import FeishuAPIError, FeishuClient, FeishuRateLimitError


def _sheet_error_item(record_id: int, error: BaseException) -> WritebackItem:
    """将 Sheet 写入异常分类为明确拒绝或结果不明。"""

    message = f"{type(error).__name__}: {error}"
    if (
        isinstance(error, FeishuAPIError)
        and error.code not in {-1}
        and not (500 <= error.code <= 599)
    ):
        return WritebackItem(
            record_id,
            CommitDisposition.REJECTED,
            "sheet_api_rejected",
            message,
            isinstance(error, FeishuRateLimitError) or error.code == 429,
        )
    return WritebackItem(
        record_id,
        CommitDisposition.INDETERMINATE,
        "sheet_write_indeterminate",
        message,
        False,
    )


def _col_index_to_letter(index: int) -> str:
    """
    将 0-based 列索引转换为 Excel 风格列字母

    0 → A, 1 → B, ..., 25 → Z, 26 → AA, 27 → AB
    """
    result = ""
    while True:
        result = chr(ord("A") + index % 26) + result
        index = index // 26 - 1
        if index < 0:
            break
    return result


class FeishuSheetTaskPool(BaseTaskPool):
    """
    飞书电子表格数据源任务池

    从飞书电子表格读取未处理的行，AI 处理后将结果写回对应单元格。

    Attributes:
        client: 飞书异步 HTTP 客户端
        spreadsheet_token: 电子表格 Token
        sheet_id: 工作表 ID（如 "0" 或 "SheetName"）
        _header_row: 表头行（列名列表）
        _data_rows: 数据行快照（二维数组，不含表头）
        _col_name_to_index: 列名 → 列索引 映射
    """

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        spreadsheet_token: str,
        sheet_id: str,
        columns_to_extract: list[str],
        columns_to_write: dict[str, str],
        require_all_input_fields: bool = True,
        max_retries: int = 3,
        qps_limit: float = 0,
    ):
        super().__init__(columns_to_extract, columns_to_write, require_all_input_fields)

        self.spreadsheet_token = spreadsheet_token
        self.sheet_id = sheet_id

        # 飞书客户端
        self.client = FeishuClient(
            app_id=app_id,
            app_secret=app_secret,
            max_retries=max_retries,
            qps_limit=qps_limit,
        )

        # 快照
        self._header_row: list[str] = []
        self._data_rows: list[list[Any]] = []
        self._col_name_to_index: dict[str, int] = {}
        self._snapshot_loaded = False

        # 写回列名
        self.write_aliases = list(self.columns_to_write.keys())
        self.write_colnames = list(self.columns_to_write.values())

        # 分片状态
        self.current_shard_id = -1
        self.current_min_id = 0
        self.current_max_id = 0

        self._snapshot_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._logger = logging.getLogger("feishu.sheet_pool")

    # ==================== 快照管理 ====================

    def _load_snapshot_sync(self) -> None:
        """同步加载快照（线程安全）"""
        if self._snapshot_loaded:
            return
        with self._snapshot_lock:
            if self._snapshot_loaded:
                return
            run_async(self._load_snapshot())

    async def _load_snapshot(self) -> None:
        """从飞书拉取全部工作表数据"""
        if self._snapshot_loaded:
            return

        self._logger.info(
            f"开始拉取电子表格快照: token={self.spreadsheet_token}, "
            f"sheet_id={self.sheet_id}"
        )

        # 获取电子表格元信息以确定数据范围
        meta = await self.client.sheet_get_meta(self.spreadsheet_token)
        sheets = meta.get("sheets", [])

        # 查找目标工作表
        target_sheet = None
        for s in sheets:
            sid = s.get("sheet_id", "")
            title = s.get("title", "")
            if sid == self.sheet_id or title == self.sheet_id:
                target_sheet = s
                break

        if not target_sheet:
            # 如果无法通过元数据找到工作表，使用默认范围
            self._logger.warning(
                f"未在元数据中找到工作表 {self.sheet_id}，使用直接范围读取"
            )
            row_count = 10000
            col_count = 100
        else:
            grid = target_sheet.get("grid_properties", {})
            row_count = grid.get("row_count", 10000)
            col_count = grid.get("column_count", 100)

        # 构建读取范围
        last_col = _col_index_to_letter(col_count - 1)
        range_str = f"{self.sheet_id}!A1:{last_col}{row_count}"

        self._logger.info(f"读取范围: {range_str}")
        all_rows = await self.client.sheet_read_range(self.spreadsheet_token, range_str)

        if not all_rows:
            self._logger.warning("电子表格数据为空")
            self._header_row = []
            self._data_rows = []
            self._snapshot_loaded = True
            return

        # 第一行为表头
        self._header_row = [
            str(cell) if cell is not None else "" for cell in all_rows[0]
        ]
        self._data_rows = all_rows[1:]

        # 建立列名映射
        self._col_name_to_index = {
            name: idx for idx, name in enumerate(self._header_row) if name
        }

        self._snapshot_loaded = True
        self._logger.info(
            f"快照加载完成: {len(self._header_row)} 列, "
            f"{len(self._data_rows)} 行数据"
        )

    def _get_cell(self, row: list[Any], col_name: str) -> str:
        """获取行中指定列的值"""
        idx = self._col_name_to_index.get(col_name)
        if idx is None or idx >= len(row):
            return ""
        val = row[idx]
        if val is None:
            return ""
        return str(val)

    def _is_unprocessed(self, row: list[Any]) -> bool:
        """判断行是否未处理"""
        # 输入有效性
        if self.require_all_input_fields:
            input_valid = all(
                self._get_cell(row, col).strip() != ""
                for col in self.columns_to_extract
            )
        else:
            input_valid = any(
                self._get_cell(row, col).strip() != ""
                for col in self.columns_to_extract
            )

        if not input_valid:
            return False

        # 输出列任一为空 → 未处理
        return any(
            self._get_cell(row, col).strip() == "" for col in self.write_colnames
        )

    def _is_processed(self, row: list[Any]) -> bool:
        """判断行是否已处理"""
        return all(
            self._get_cell(row, col).strip() != "" for col in self.write_colnames
        )

    # ==================== BaseTaskPool 抽象方法实现 ====================

    def get_total_task_count(self) -> int:
        """获取未处理任务总数"""
        self._load_snapshot_sync()
        count = sum(1 for row in self._data_rows if self._is_unprocessed(row))
        self._logger.info(f"电子表格未处理任务数: {count}")
        return count

    def get_processed_task_count(self) -> int:
        """获取已处理任务总数"""
        self._load_snapshot_sync()
        count = sum(1 for row in self._data_rows if self._is_processed(row))
        self._logger.info(f"电子表格已处理任务数: {count}")
        return count

    def get_id_boundaries(self) -> tuple[int, int]:
        """
        获取任务 ID 边界

        电子表格使用数据行索引（0-based）作为 task_id，
        对应实际行号为 task_id + 2（跳过表头行）。
        """
        self._load_snapshot_sync()
        if not self._data_rows:
            return (0, -1)
        return (0, len(self._data_rows) - 1)

    def initialize_shard(self, shard_id: int, min_id: int, max_id: int) -> int:
        """
        初始化分片

        Args:
            shard_id: 分片编号
            min_id: 数据行索引范围起始（包含）
            max_id: 数据行索引范围结束（包含）

        Returns:
            加载的任务数量
        """
        self._load_snapshot_sync()

        shard_tasks: list[tuple[int, dict[str, Any]]] = []

        for row_idx in range(min_id, min(max_id + 1, len(self._data_rows))):
            row = self._data_rows[row_idx]
            if self._is_unprocessed(row):
                record_dict = {
                    col: self._get_cell(row, col) for col in self.columns_to_extract
                }
                shard_tasks.append((row_idx, record_dict))

        with self.lock:
            self.tasks = shard_tasks

        self.current_shard_id = shard_id
        self.current_min_id = min_id
        self.current_max_id = max_id

        self._logger.info(
            f"分片 {shard_id} (行索引: {min_id}-{max_id}) 加载完成，"
            f"任务数: {len(shard_tasks)}"
        )
        return len(shard_tasks)

    def get_task_batch(self, batch_size: int) -> list[tuple[int, dict[str, Any]]]:
        """从内存队列获取一批任务"""
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

    def update_task_results(
        self,
        batch_id: str,
        results: dict[int, dict[str, Any]],
    ) -> WritebackReceipt:
        """
        批量写回任务结果到飞书电子表格

        将结果按行号写入对应单元格。Sheet API 串行写入。

        Args:
            results: {task_id(行索引): {alias: value, ...}, ...}
        """
        if not results:
            return WritebackReceipt.committed(batch_id, (), atomic=False)

        # 按输出列分组写入（每列一次 API 调用更高效）
        # 构建写入数据: {col_name: {row_idx: value}}
        col_data: dict[str, dict[int, Any]] = {}
        outcomes: dict[int, WritebackItem] = {}

        for task_id, row_result in results.items():
            if "_error" in row_result:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "invalid_result",
                    "结果包含 _error",
                    False,
                )
                continue

            if task_id < 0 or task_id >= len(self._data_rows):
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "record_not_found",
                    f"行索引 {task_id} 不存在",
                    False,
                )
                continue

            writable_fields = [
                (alias, col_name)
                for alias, col_name in self.columns_to_write.items()
                if alias in row_result
            ]
            if not writable_fields:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "no_writable_fields",
                    "结果不包含可写字段",
                    False,
                )
                continue

            missing_columns = [
                col_name
                for _, col_name in writable_fields
                if col_name not in self._col_name_to_index
            ]
            if missing_columns:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "column_not_found",
                    f"列不在表头中: {missing_columns}",
                    False,
                )
                continue

            for alias, col_name in writable_fields:
                col_data.setdefault(col_name, {})[task_id] = row_result[alias]

        if not col_data:
            return WritebackReceipt(
                batch_id=batch_id,
                submitted_ids=tuple(results),
                items=tuple(outcomes[task_id] for task_id in results),
                atomic=False,
            )

        with self._write_lock:
            write_outcomes = run_async(self._write_results(col_data))
        outcomes.update(write_outcomes)
        for task_id in results:
            outcomes.setdefault(
                task_id,
                WritebackItem(
                    task_id,
                    CommitDisposition.INDETERMINATE,
                    "sheet_result_missing",
                    "Sheet 写入结果未覆盖该记录",
                    False,
                ),
            )
        return WritebackReceipt(
            batch_id=batch_id,
            submitted_ids=tuple(results),
            items=tuple(outcomes[task_id] for task_id in results),
            atomic=False,
        )

    async def _write_results(
        self, col_data: dict[str, dict[int, Any]]
    ) -> dict[int, WritebackItem]:
        """
        异步写入结果到电子表格

        优化策略: 按列分组，将连续行号合并为一次范围写入。
        对于非连续行号，拆分为多个连续段分别写入。
        """
        success_count = 0
        error_count = 0
        expected_cells: dict[int, set[str]] = {}
        successful_cells: dict[int, set[str]] = {}
        failures: dict[int, list[WritebackItem]] = {}
        for col_name, rows in col_data.items():
            for task_id in rows:
                expected_cells.setdefault(task_id, set()).add(col_name)

        for col_name, rows in col_data.items():
            col_idx = self._col_name_to_index.get(col_name)
            if col_idx is None:
                self._logger.warning(f"列 '{col_name}' 不在表头中，跳过")
                error_count += len(rows)
                for task_id in rows:
                    failures.setdefault(task_id, []).append(
                        WritebackItem(
                            record_id=task_id,
                            disposition=CommitDisposition.REJECTED,
                            code="column_not_found",
                            message=f"列 {col_name} 不在表头中",
                            retryable=False,
                        )
                    )
                continue

            col_letter = _col_index_to_letter(col_idx)

            # 按行索引排序，找连续段合并写入
            sorted_rows = sorted(rows.items(), key=lambda x: x[0])
            segments = self._group_consecutive(sorted_rows)

            for segment in segments:
                first_row_idx = segment[0][0]
                last_row_idx = segment[-1][0]
                # 实际行号 = 数据行索引 + 2（跳过表头，1-based）
                start_row = first_row_idx + 2
                end_row = last_row_idx + 2
                range_str = (
                    f"{self.sheet_id}!{col_letter}{start_row}:{col_letter}{end_row}"
                )
                values = [[str(v) if v is not None else ""] for _, v in segment]

                try:
                    await self.client.sheet_write_range(
                        self.spreadsheet_token,
                        range_str,
                        values,
                    )
                    success_count += len(segment)

                    # 同步更新内存快照，防止多 shard 重复处理
                    for row_idx, value in segment:
                        successful_cells.setdefault(row_idx, set()).add(col_name)
                        # 飞书可能省略行尾空单元格，先补齐长度再写入
                        row = self._data_rows[row_idx]
                        if len(row) <= col_idx:
                            row.extend([""] * (col_idx + 1 - len(row)))
                        row[col_idx] = str(value) if value is not None else ""

                except Exception as e:
                    self._logger.error(f"写入 {range_str} 失败: {e}")
                    error_count += len(segment)
                    for row_idx, _ in segment:
                        failures.setdefault(row_idx, []).append(
                            _sheet_error_item(row_idx, e)
                        )

        self._logger.info(f"Sheet 写入完成，成功: {success_count}, 失败: {error_count}")
        outcomes: dict[int, WritebackItem] = {}
        for task_id, expected in expected_cells.items():
            succeeded = successful_cells.get(task_id, set())
            task_failures = failures.get(task_id, [])
            if succeeded == expected:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.COMMITTED,
                )
            elif succeeded:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.INDETERMINATE,
                    "partial_sheet_write",
                    "同一记录仅有部分期望单元格确认写入",
                    False,
                )
            elif any(
                item.disposition == CommitDisposition.INDETERMINATE
                for item in task_failures
            ):
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.INDETERMINATE,
                    "sheet_write_indeterminate",
                    "; ".join(item.message for item in task_failures),
                    False,
                )
            elif task_failures:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    task_failures[0].code,
                    "; ".join(item.message for item in task_failures),
                    all(item.retryable for item in task_failures),
                )
            else:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.INDETERMINATE,
                    "sheet_result_missing",
                    "Sheet 写入结果未覆盖该记录",
                    False,
                )
        return outcomes

    def reconcile_task_results(
        self,
        batch_id: str,
        results: dict[int, dict[str, Any]],
    ) -> WritebackReceipt:
        """按实际单元格范围重新读取并核对每个期望值。"""

        outcomes: dict[int, WritebackItem] = {}
        expected: dict[int, list[tuple[int, Any]]] = {}
        for task_id, row_result in results.items():
            if "_error" in row_result:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "invalid_result",
                    "结果包含 _error",
                    False,
                )
                continue
            if task_id < 0 or task_id >= len(self._data_rows):
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "record_not_found",
                    f"行索引 {task_id} 不存在",
                    False,
                )
                continue

            writable_fields = [
                (col_name, row_result[alias])
                for alias, col_name in self.columns_to_write.items()
                if alias in row_result
            ]
            if not writable_fields:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "no_writable_fields",
                    "结果不包含可写字段",
                    False,
                )
                continue
            missing_columns = [
                col_name
                for col_name, _ in writable_fields
                if col_name not in self._col_name_to_index
            ]
            if missing_columns:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "column_not_found",
                    f"列不在表头中: {missing_columns}",
                    False,
                )
                continue
            expected[task_id] = [
                (self._col_name_to_index[col_name], value)
                for col_name, value in writable_fields
            ]

        outcomes.update(run_async(self._reconcile_expected_cells(expected)))
        return self._receipt(batch_id, results, outcomes)

    async def _reconcile_expected_cells(
        self,
        expected: dict[int, list[tuple[int, Any]]],
    ) -> dict[int, WritebackItem]:
        outcomes: dict[int, WritebackItem] = {}
        for task_id, cells in expected.items():
            mismatched = False
            read_error: Exception | None = None
            actual_values: list[tuple[int, str]] = []
            for col_idx, expected_value in cells:
                cell = f"{_col_index_to_letter(col_idx)}{task_id + 2}"
                range_str = f"{self.sheet_id}!{cell}:{cell}"
                try:
                    rows = await self.client.sheet_read_range(
                        self.spreadsheet_token,
                        range_str,
                    )
                except Exception as exc:
                    read_error = exc
                    continue
                actual = ""
                if rows and rows[0]:
                    value = rows[0][0]
                    actual = "" if value is None else str(value)
                expected_text = "" if expected_value is None else str(expected_value)
                actual_values.append((col_idx, actual))
                if actual != expected_text:
                    mismatched = True

            if mismatched:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "sheet_reconciliation_mismatch",
                    "至少一个远端单元格与期望值不匹配",
                    True,
                )
            elif read_error is not None:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.INDETERMINATE,
                    "sheet_reconciliation_failed",
                    str(read_error),
                    False,
                )
            else:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.COMMITTED,
                )
                row = self._data_rows[task_id]
                for col_idx, actual in actual_values:
                    if len(row) <= col_idx:
                        row.extend([""] * (col_idx + 1 - len(row)))
                    row[col_idx] = actual
        return outcomes

    @staticmethod
    def _receipt(
        batch_id: str,
        results: dict[int, dict[str, Any]],
        outcomes: dict[int, WritebackItem],
    ) -> WritebackReceipt:
        return WritebackReceipt(
            batch_id=batch_id,
            submitted_ids=tuple(results),
            items=tuple(outcomes[task_id] for task_id in results),
            atomic=False,
        )

    @staticmethod
    def _group_consecutive(
        sorted_rows: list[tuple[int, Any]],
    ) -> list[list[tuple[int, Any]]]:
        """将排序后的 (row_idx, value) 列表按连续行号分组"""
        if not sorted_rows:
            return []
        segments: list[list[tuple[int, Any]]] = [[sorted_rows[0]]]
        for i in range(1, len(sorted_rows)):
            if sorted_rows[i][0] == sorted_rows[i - 1][0] + 1:
                segments[-1].append(sorted_rows[i])
            else:
                segments.append([sorted_rows[i]])
        return segments

    def reload_task_data(self, task_id: int) -> dict[str, Any] | None:
        """重新从快照加载任务数据"""
        if 0 <= task_id < len(self._data_rows):
            row = self._data_rows[task_id]
            return {col: self._get_cell(row, col) for col in self.columns_to_extract}
        self._logger.warning(f"task_id={task_id} 超出快照范围")
        return None

    def close(self) -> None:
        """关闭飞书客户端"""
        self._logger.info("关闭飞书电子表格任务池 ...")
        run_async(self.client.close())

    # ==================== Token 估算采样 ====================

    def sample_unprocessed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """采样未处理行"""
        self._load_snapshot_sync()
        samples: list[dict[str, Any]] = []
        for row in self._data_rows:
            if len(samples) >= sample_size:
                break
            if self._is_unprocessed(row):
                samples.append(
                    {col: self._get_cell(row, col) for col in self.columns_to_extract}
                )
        return samples

    def sample_processed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """采样已处理行"""
        self._load_snapshot_sync()
        samples: list[dict[str, Any]] = []
        for row in self._data_rows:
            if len(samples) >= sample_size:
                break
            if self._is_processed(row):
                samples.append(
                    {col: self._get_cell(row, col) for col in self.write_colnames}
                )
        return samples

    def fetch_all_rows(self, columns: list[str]) -> list[dict[str, Any]]:
        self._load_snapshot_sync()
        return [
            {col: self._get_cell(row, col) for col in columns}
            for row in self._data_rows
        ]

    def fetch_all_processed_rows(self, columns: list[str]) -> list[dict[str, Any]]:
        self._load_snapshot_sync()
        return [
            {col: self._get_cell(row, col) for col in columns}
            for row in self._data_rows
            if self._is_processed(row)
        ]
