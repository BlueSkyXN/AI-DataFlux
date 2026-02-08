"""
飞书电子表格（Sheet）数据源任务池

本模块实现基于飞书电子表格的 TaskPool，适用于将电子表格作为
AI-DataFlux 的数据源进行批量 AI 处理。

核心设计:
    1. 快照读取 —— 初始化时一次性拉取全部数据到内存二维数组
    2. 行号映射 —— 使用电子表格行号作为 task_id（1-based，第 1 行为表头）
    3. 串行写入 —— 飞书电子表格单文档须串行写入，写入时加锁
    4. 列名映射 —— 表头行建立 列名 → 列号（A, B, C ...）的映射

电子表格范围约定:
    - 行号从 1 开始，第 1 行为表头
    - 数据行从第 2 行开始
    - 列号使用 A, B, C ... AA, AB 等字母表示
    - 范围格式: "{sheet_id}!A1:Z1000"

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

import asyncio
import logging
from typing import Any

from ..base import BaseTaskPool
from .client import FeishuClient


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

        # 写入锁（Sheet 必须串行写入）
        self._write_lock = asyncio.Lock()

        self._logger = logging.getLogger("feishu.sheet_pool")

    # ==================== 快照管理 ====================

    def _load_snapshot_sync(self) -> None:
        """同步加载快照"""
        if self._snapshot_loaded:
            return

        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._load_snapshot())
                future.result()
        else:
            asyncio.run(self._load_snapshot())

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
        all_rows = await self.client.sheet_read_range(
            self.spreadsheet_token, range_str
        )

        if not all_rows:
            self._logger.warning("电子表格数据为空")
            self._header_row = []
            self._data_rows = []
            self._snapshot_loaded = True
            return

        # 第一行为表头
        self._header_row = [str(cell) if cell is not None else "" for cell in all_rows[0]]
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
            self._get_cell(row, col).strip() == ""
            for col in self.write_colnames
        )

    def _is_processed(self, row: list[Any]) -> bool:
        """判断行是否已处理"""
        return all(
            self._get_cell(row, col).strip() != ""
            for col in self.write_colnames
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
                    col: self._get_cell(row, col)
                    for col in self.columns_to_extract
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

    def update_task_results(self, results: dict[int, dict[str, Any]]) -> None:
        """
        批量写回任务结果到飞书电子表格

        将结果按行号写入对应单元格。Sheet API 串行写入。

        Args:
            results: {task_id(行索引): {alias: value, ...}, ...}
        """
        if not results:
            return

        # 按输出列分组写入（每列一次 API 调用更高效）
        # 构建写入数据: {col_name: {row_idx: value}}
        col_data: dict[str, dict[int, Any]] = {}

        for task_id, row_result in results.items():
            if "_error" in row_result:
                continue

            for alias, col_name in self.columns_to_write.items():
                if alias in row_result:
                    if col_name not in col_data:
                        col_data[col_name] = {}
                    col_data[col_name][task_id] = row_result[alias]

        if not col_data:
            return

        self._run_async(self._write_results(col_data))

    async def _write_results(self, col_data: dict[str, dict[int, Any]]) -> None:
        """异步写入结果到电子表格"""
        success_count = 0
        error_count = 0

        for col_name, rows in col_data.items():
            col_idx = self._col_name_to_index.get(col_name)
            if col_idx is None:
                self._logger.warning(f"列 '{col_name}' 不在表头中，跳过")
                error_count += len(rows)
                continue

            col_letter = _col_index_to_letter(col_idx)

            # 逐条写入（串行，避免并发冲突）
            for row_idx, value in rows.items():
                # 实际行号 = 数据行索引 + 2（跳过表头）
                actual_row = row_idx + 2
                range_str = f"{self.sheet_id}!{col_letter}{actual_row}"

                try:
                    await self.client.sheet_write_range(
                        self.spreadsheet_token,
                        range_str,
                        [[str(value) if value is not None else ""]],
                    )
                    success_count += 1
                except Exception as e:
                    self._logger.error(
                        f"写入 {range_str} 失败: {e}"
                    )
                    error_count += 1

        self._logger.info(
            f"Sheet 写入完成，成功: {success_count}, 失败: {error_count}"
        )

    def reload_task_data(self, task_id: int) -> dict[str, Any] | None:
        """重新从快照加载任务数据"""
        if 0 <= task_id < len(self._data_rows):
            row = self._data_rows[task_id]
            return {
                col: self._get_cell(row, col)
                for col in self.columns_to_extract
            }
        self._logger.warning(f"task_id={task_id} 超出快照范围")
        return None

    def close(self) -> None:
        """关闭飞书客户端"""
        self._logger.info("关闭飞书电子表格任务池 ...")
        self._run_async(self.client.close())

    # ==================== 工具方法 ====================

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """在同步上下文中运行异步协程"""
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    # ==================== Token 估算采样 ====================

    def sample_unprocessed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """采样未处理行"""
        self._load_snapshot_sync()
        samples = []
        for row in self._data_rows:
            if len(samples) >= sample_size:
                break
            if self._is_unprocessed(row):
                samples.append({
                    col: self._get_cell(row, col)
                    for col in self.columns_to_extract
                })
        return samples

    def sample_processed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """采样已处理行"""
        self._load_snapshot_sync()
        samples = []
        for row in self._data_rows:
            if len(samples) >= sample_size:
                break
            if self._is_processed(row):
                samples.append({
                    col: self._get_cell(row, col)
                    for col in self.write_colnames
                })
        return samples
