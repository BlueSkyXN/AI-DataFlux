"""
Excel 数据源任务池实现

基于 DataFrame 引擎抽象，支持 pandas 和 polars 等多种实现。
支持高性能读写器: calamine (fastexcel) 和 xlsxwriter。
"""

import logging
import time
from pathlib import Path
from typing import Any

from .base import BaseTaskPool
from .engines import get_engine, BaseEngine


class ExcelTaskPool(BaseTaskPool):
    """
    Excel 数据源任务池
    
    从 Excel 文件读取任务数据，处理后写回结果。
    支持定时保存、分片加载、向量化过滤等功能。
    
    Attributes:
        input_path: 输入 Excel 文件路径
        output_path: 输出 Excel 文件路径
        save_interval: 自动保存间隔 (秒)
        engine: DataFrame 引擎实例
        df: 当前 DataFrame (引擎特定类型)
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
        初始化 Excel 任务池
        
        Args:
            input_path: 输入 Excel 文件路径
            output_path: 输出 Excel 文件路径
            columns_to_extract: 需要提取的列名列表
            columns_to_write: 写回映射 {别名: 实际列名}
            save_interval: 自动保存间隔 (秒)
            require_all_input_fields: 是否要求所有输入字段都非空
            engine_type: DataFrame 引擎类型 ("pandas" | "polars" | "auto")
            excel_reader: Excel 读取器 ("openpyxl" | "calamine" | "auto")
            excel_writer: Excel 写入器 ("openpyxl" | "xlsxwriter" | "auto")
        
        Raises:
            FileNotFoundError: 输入文件不存在
            IOError: 文件读取失败
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
            engine_type=engine_type,
            excel_reader=excel_reader,
            excel_writer=excel_writer,
        )
        logging.info(f"使用 DataFrame 引擎: {self.engine.name}")
        
        # 显示读写器信息
        if hasattr(self.engine, 'excel_reader'):
            logging.info(f"  - Excel 读取器: {self.engine.excel_reader}")
        if hasattr(self.engine, 'excel_writer'):
            logging.info(f"  - Excel 写入器: {self.engine.excel_writer}")
        
        # 读取 Excel 文件
        logging.info(f"正在读取 Excel 文件: {self.input_path}")
        try:
            self.df = self.engine.read_excel(self.input_path)
            row_count = self.engine.row_count(self.df)
            logging.info(f"Excel 文件读取成功，共 {row_count} 行")
        except Exception as e:
            raise IOError(f"无法读取 Excel 文件 {self.input_path}: {e}") from e
        
        # 保存相关
        self.save_interval = save_interval
        self.last_save_time = time.time()
        
        # 分片状态
        self.current_shard_id = -1
        self.current_min_idx = 0
        self.current_max_idx = 0
        
        # 列验证和准备
        self._validate_and_prepare_columns()
        
        logging.info(f"ExcelTaskPool 初始化完成 | 输入: {self.input_path}, 输出: {self.output_path}")
    
    def _validate_and_prepare_columns(self) -> None:
        """验证和准备列"""
        column_names = self.engine.get_column_names(self.df)
        
        # 检查输入列
        missing_extract = [c for c in self.columns_to_extract if c not in column_names]
        if missing_extract:
            logging.warning(f"输入列 {missing_extract} 在 Excel 中不存在")
        
        # 创建输出列 (如果不存在)
        for alias, out_col in self.columns_to_write.items():
            if not self.engine.has_column(self.df, out_col):
                logging.warning(f"输出列 '{out_col}' 不存在，将创建新列")
                self.df = self.engine.add_column(self.df, out_col, None)
    
    # ==================== 核心接口实现 ====================
    
    def get_total_task_count(self) -> int:
        """获取未处理任务总数"""
        logging.info("正在计算 Excel 中未处理的任务总数...")
        
        min_idx, max_idx = self.engine.get_index_range(self.df)
        unprocessed = self._filter_unprocessed_indices(min_idx, max_idx)
        count = len(unprocessed)
        
        logging.info(f"Excel 中未处理的任务总数: {count}")
        return count
    
    def get_id_boundaries(self) -> tuple[int, int]:
        """获取索引边界"""
        if self.engine.row_count(self.df) == 0:
            return (0, -1)
        
        min_idx, max_idx = self.engine.get_index_range(self.df)
        logging.info(f"Excel DataFrame 索引范围: {min_idx} - {max_idx}")
        return (min_idx, max_idx)
    
    def initialize_shard(self, shard_id: int, min_idx: int, max_idx: int) -> int:
        """初始化分片，加载指定范围的未处理任务"""
        logging.info(f"开始初始化分片 {shard_id} (索引范围: {min_idx}-{max_idx})...")
        
        shard_tasks: list[tuple[Any, dict[str, Any]]] = []
        
        try:
            # 过滤未处理的索引
            unprocessed_indices = self._filter_unprocessed_indices(min_idx, max_idx)
            
            if unprocessed_indices:
                logging.debug(f"分片 {shard_id}: 找到 {len(unprocessed_indices)} 个未处理索引，正在提取数据...")
                
                for idx in unprocessed_indices:
                    try:
                        row_data = self.engine.get_row(self.df, idx)
                        record_dict = {
                            col: self.engine.to_string(row_data.get(col, ""))
                            for col in self.columns_to_extract
                        }
                        shard_tasks.append((idx, record_dict))
                    except Exception as e:
                        logging.error(f"分片 {shard_id}: 提取索引 {idx} 数据时出错: {e}")
            else:
                logging.info(f"分片 {shard_id}: 在指定索引范围内未找到未处理的任务")
                
        except Exception as e:
            logging.error(f"初始化分片 {shard_id} (索引 {min_idx}-{max_idx}) 失败: {e}", exc_info=True)
            shard_tasks = []
        
        # 更新任务队列
        with self.lock:
            self.tasks = shard_tasks
        
        # 更新分片状态
        self.current_shard_id = shard_id
        self.current_min_idx = min_idx
        self.current_max_idx = max_idx
        
        loaded_count = len(shard_tasks)
        logging.info(f"分片 {shard_id} (索引范围: {min_idx}-{max_idx}) 初始化完成，加载任务数: {loaded_count}")
        
        return loaded_count
    
    def get_task_batch(self, batch_size: int) -> list[tuple[Any, dict[str, Any]]]:
        """从内存队列获取一批任务"""
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch
    
    def update_task_results(self, results: dict[int, dict[str, Any]]) -> None:
        """批量写回任务结果"""
        if not results:
            return
        
        updated_indices: list[int] = []
        needs_save = False
        
        try:
            with self.lock:
                for idx, row_result in results.items():
                    # 跳过错误结果
                    if "_error" in row_result:
                        continue
                    
                    # 检查索引是否存在
                    all_indices = self.engine.get_indices(self.df)
                    if idx not in all_indices:
                        logging.warning(f"尝试更新 Excel 中不存在的索引 {idx}，跳过")
                        continue
                    
                    # 写入结果
                    for alias, col_name in self.columns_to_write.items():
                        if self.engine.has_column(self.df, col_name):
                            value = row_result.get(alias, "")
                            try:
                                self.df = self.engine.set_value(self.df, idx, col_name, value)
                            except Exception as e:
                                logging.warning(f"设置索引 {idx} 列 '{col_name}' 值失败: {e}")
                    
                    updated_indices.append(idx)
                
                if updated_indices:
                    logging.info(f"已在内存中更新 {len(updated_indices)} 条 Excel 记录")
                    
                    # 检查是否需要自动保存
                    current_time = time.time()
                    if current_time - self.last_save_time >= self.save_interval:
                        needs_save = True
                        self.last_save_time = current_time
                        
        except Exception as e:
            logging.error(f"更新 Excel DataFrame 时发生错误: {e}", exc_info=True)
            needs_save = False
        
        # 在锁外执行保存
        if needs_save:
            logging.info(f"达到保存间隔 ({self.save_interval}s)，准备保存 Excel 文件...")
            try:
                self._save_excel()
            except Exception as e:
                logging.error(f"自动保存 Excel 文件失败: {e}")
    
    def reload_task_data(self, idx: int) -> dict[str, Any] | None:
        """重新加载任务的原始输入数据"""
        try:
            with self.lock:
                all_indices = self.engine.get_indices(self.df)
                if idx not in all_indices:
                    logging.warning(f"尝试重载数据失败: 索引 {idx} 在 DataFrame 中不存在")
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
        """关闭并保存文件"""
        logging.info("正在执行 Excel 文件的最终保存操作...")
        try:
            self._save_excel()
        except Exception as e:
            logging.error(f"最终保存 Excel 文件失败: {e}")
    
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
                index_offset=min_idx
            )
            logging.debug(f"过滤索引范围 {min_idx}-{max_idx} 完成，找到 {len(unprocessed)} 个未处理索引")
            return unprocessed
            
        except Exception as e:
            logging.error(f"过滤未处理索引时出错: {e}", exc_info=True)
            return []
    
    def _save_excel(self) -> None:
        """
        保存 Excel 文件
        
        处理 Unicode 编码问题，必要时清空问题单元格或回退到 CSV。
        """
        logging.info(f"正在尝试保存 DataFrame 到: {self.output_path}")
        
        try:
            with self.lock:
                # 确保输出目录存在
                output_dir = self.output_path.parent
                if output_dir and not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)
                
                # 策略1: 直接保存
                try:
                    self.engine.write_excel(self.df, self.output_path)
                    logging.info(f"✅ DataFrame 已成功保存到: {self.output_path}")
                    return
                    
                except UnicodeEncodeError as e:
                    logging.error(f"❌ Unicode 编码问题: {e}")
                    logging.info("🧹 开始清空 AI 输出列中的问题单元格...")
                    
                    # 策略2: 清空问题单元格
                    fixed_df = self.engine.copy(self.df)
                    fixed_df, cleared_count = self._clear_problematic_cells(fixed_df)
                    
                    if cleared_count > 0:
                        logging.info(f"🧹 已清空 {cleared_count} 个问题单元格，重新尝试保存...")
                        
                        try:
                            self.engine.write_excel(fixed_df, self.output_path)
                            logging.info(f"✅ DataFrame 已成功保存 (已清空 {cleared_count} 个问题单元格)")
                            self.df = fixed_df
                            return
                        except UnicodeEncodeError:
                            logging.warning("⚠️ 清空 AI 输出列后仍有问题，可能来自原始数据")
                    
                    # 策略3: CSV 备选方案
                    csv_path = self.output_path.with_suffix(".csv")
                    logging.warning(f"⚠️ Excel 保存失败，尝试保存为 CSV: {csv_path}")
                    
                    df_to_save = fixed_df if cleared_count > 0 else self.df
                    self.engine.write_csv(df_to_save, csv_path)
                    logging.warning(f"✅ 已保存为 CSV: {csv_path}")
                    
        except Exception as e:
            logging.error(f"❌ 保存文件失败: {e}", exc_info=True)
            raise IOError(f"保存文件失败: {e}") from e
    
    def _clear_problematic_cells(self, df: Any) -> tuple[Any, int]:
        """
        清空 DataFrame 中有编码问题的单元格
        
        只检查 AI 输出列，返回更新后的 DataFrame 和清空的单元格数量。
        """
        cleared_count = 0
        ai_columns = list(self.columns_to_write.values())
        updated_df = df
        
        for col_name in ai_columns:
            if not self.engine.has_column(df, col_name):
                continue
            
            for idx, row_data in self.engine.iter_rows(df, [col_name]):
                value = row_data.get(col_name)
                
                if isinstance(value, str) and value:
                    try:
                        value.encode("utf-8")
                    except UnicodeEncodeError:
                        logging.warning(f"❌ 清空问题单元格: 第 {idx} 行, '{col_name}' 列")
                        updated_df = self.engine.set_value(updated_df, idx, col_name, "")
                        cleared_count += 1
        
        return updated_df, cleared_count
    
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
                        if value is None or (isinstance(value, str) and not value.strip()):
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
