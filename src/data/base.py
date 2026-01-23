"""
数据源任务池抽象基类模块

本模块定义数据源任务池的标准接口，是所有数据源实现的基础。
通过抽象基类确保各数据源实现提供一致的接口，便于处理器统一调用。

核心概念:
    - 任务池 (TaskPool): 管理待处理任务的数据结构
    - 分片 (Shard): 将大数据集分割为小块处理
    - 任务 (Task): 单条待处理记录，由 ID 和数据字典组成

接口分类:
    1. 统计接口:
       - get_total_task_count(): 未处理任务数
       - get_processed_task_count(): 已处理任务数
       - get_id_boundaries(): ID 范围（用于分片）
    
    2. 分片加载:
       - initialize_shard(): 将分片数据加载到内存队列
    
    3. 任务获取:
       - get_task_batch(): 获取一批任务
       - reload_task_data(): 重新加载单个任务数据
    
    4. 结果写回:
       - update_task_results(): 批量写回处理结果
    
    5. 队列操作:
       - add_task_to_front(): 重试时放回队头
       - add_task_to_back(): 延迟处理时放回队尾
       - has_tasks(): 检查队列是否为空

实现指南:
    创建新数据源需要：
    1. 继承 BaseTaskPool
    2. 实现所有抽象方法
    3. 在 factory.py 中添加创建逻辑
    4. 添加相应的测试用例

使用示例:
    class MyTaskPool(BaseTaskPool):
        def get_total_task_count(self) -> int:
            # 实现具体逻辑
            pass
        
        # 实现其他抽象方法...
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any


class BaseTaskPool(ABC):
    """
    数据源任务池抽象基类
    
    定义了数据源操作的标准接口，所有具体数据源实现必须继承此类
    并实现所有抽象方法。
    
    Attributes:
        columns_to_extract (list[str]): 需要从数据源提取的列名列表
        columns_to_write (dict[str, str]): 结果写回映射 {别名: 实际列名}
        require_all_input_fields (bool): 是否要求所有输入字段都非空
        tasks (list): 当前分片的内存任务队列 [(task_id, data_dict), ...]
        lock (threading.Lock): 线程锁，保护 tasks 队列的并发访问
    
    线程安全:
        tasks 队列的所有操作都通过 self.lock 保护，支持多线程并发访问。
    """

    def __init__(
        self,
        columns_to_extract: list[str],
        columns_to_write: dict[str, str],
        require_all_input_fields: bool = True,
    ):
        """
        初始化任务池基类
        
        Args:
            columns_to_extract: 需要提取的列名列表，用于构建任务数据
            columns_to_write: 写回映射 {别名: 实际列名}，将处理结果写入对应列
            require_all_input_fields: 是否要求所有输入字段都非空才处理
        """
        self.columns_to_extract = columns_to_extract
        self.columns_to_write = columns_to_write
        self.require_all_input_fields = require_all_input_fields
        self.tasks: list[tuple[Any, dict[str, Any]]] = []
        self.lock = threading.Lock()

    # ==================== 抽象方法（子类必须实现） ====================

    @abstractmethod
    def get_total_task_count(self) -> int:
        """
        获取未处理任务总数
        
        扫描数据源，统计所有输出列为空的记录数量。
        
        Returns:
            int: 未处理任务数量
        """
        pass

    @abstractmethod
    def get_processed_task_count(self) -> int:
        """
        获取已处理任务总数
        
        扫描数据源，统计所有输出列非空的记录数量。
        
        Returns:
            int: 已处理任务数量
        """
        pass

    @abstractmethod
    def get_id_boundaries(self) -> tuple[Any, Any]:
        """
        获取任务 ID 边界
        
        返回数据源中未处理任务的 ID 范围，用于分片处理。
        对于 Excel 类数据源，ID 通常是行索引；
        对于数据库，ID 通常是主键。

        Returns:
            tuple: (最小ID, 最大ID)，如果无任务返回 (0, -1)
        """
        pass

    @abstractmethod
    def initialize_shard(self, shard_id: int, min_id: Any, max_id: Any) -> int:
        """
        初始化分片并加载到内存队列
        
        将指定 ID 范围内的未处理任务加载到 self.tasks 队列。
        这是分片处理的核心方法。

        Args:
            shard_id: 分片编号（从 0 开始）
            min_id: 最小 ID（包含）
            max_id: 最大 ID（包含）

        Returns:
            int: 实际加载的任务数量
        """
        pass

    @abstractmethod
    def get_task_batch(self, batch_size: int) -> list[tuple[Any, dict[str, Any]]]:
        """
        从内存队列获取一批任务
        
        获取的任务会从队列中移除，确保不会重复处理。
        如果队列中任务不足，返回所有剩余任务。

        Args:
            batch_size: 期望获取的任务数量

        Returns:
            list: 任务列表 [(task_id, data_dict), ...]
        """
        pass

    @abstractmethod
    def update_task_results(self, results: dict[Any, dict[str, Any]]) -> None:
        """
        批量写回任务结果
        
        将处理结果写入数据源对应位置。
        对于 Excel 类数据源，可能需要定时保存文件；
        对于数据库，通常立即提交事务。

        Args:
            results: 结果字典 {task_id: {field: value, ...}, ...}
                    field 是 columns_to_write 中的别名
        """
        pass

    @abstractmethod
    def reload_task_data(self, task_id: Any) -> dict[str, Any] | None:
        """
        重新加载任务的原始输入数据
        
        从数据源重新读取任务数据，用于重试场景。
        这样可以避免使用可能被污染的内存缓存数据。

        Args:
            task_id: 任务 ID

        Returns:
            dict: 原始数据字典 {column: value, ...}
            None: 如果加载失败或任务不存在
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        关闭资源并清理
        
        释放数据库连接、保存文件、清空缓存等清理操作。
        调用后任务池不应再被使用。
        """
        pass

    # ==================== 队列操作（具体实现） ====================

    def add_task_to_front(
        self, task_id: Any, record_dict: dict[str, Any] | None
    ) -> None:
        """
        将任务放回队列头部（用于优先重试）
        
        当任务需要立即重试时使用，例如遇到可恢复错误后。

        Args:
            task_id: 任务 ID
            record_dict: 任务数据，如果为 None 则忽略
        """
        if record_dict is None:
            logging.warning(f"尝试将任务 {task_id} 放回队列失败: 数据为 None")
            return

        with self.lock:
            self.tasks.insert(0, (task_id, record_dict))
            logging.debug(f"任务 {task_id} 已放回队列头部")

    def add_task_to_back(
        self, task_id: Any, record_dict: dict[str, Any] | None
    ) -> None:
        """
        将任务放到队列尾部（用于延迟处理）
        
        当任务需要稍后重试时使用，例如遇到限流后。

        Args:
            task_id: 任务 ID
            record_dict: 任务数据，如果为 None 则忽略
        """
        if record_dict is None:
            logging.warning(f"尝试将任务 {task_id} 放回队列失败: 数据为 None")
            return

        with self.lock:
            self.tasks.append((task_id, record_dict))
            logging.debug(f"任务 {task_id} 已放回队列尾部")

    def has_tasks(self) -> bool:
        """
        检查内存队列是否还有任务
        
        Returns:
            bool: True 表示还有任务待处理
        """
        with self.lock:
            return len(self.tasks) > 0

    def get_remaining_count(self) -> int:
        """
        获取内存队列中剩余的任务数量
        
        Returns:
            int: 剩余任务数
        """
        with self.lock:
            return len(self.tasks)

    def clear_tasks(self) -> None:
        """
        清空内存队列
        
        用于切换分片或重置状态时调用。
        """
        with self.lock:
            self.tasks.clear()

    # ==================== Token 估算采样（子类可覆盖） ====================

    def sample_unprocessed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """
        采样未处理的行（用于输入 Token 估算）
        
        默认实现：初始化第一个分片并返回采样数据。
        子类可以覆盖以提供更高效的实现。

        Args:
            sample_size: 采样数量

        Returns:
            list: 采样数据列表 [{column: value, ...}, ...]
        """
        # 默认实现：使用分片加载
        min_id, max_id = self.get_id_boundaries()
        if min_id > max_id:
            return []

        # 初始化分片加载部分数据
        self.initialize_shard(0, min_id, max_id)

        with self.lock:
            samples = [data for _, data in self.tasks[:sample_size]]

        return samples

    def sample_processed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """
        采样已处理的行（用于输出 Token 估算）
        
        默认实现返回空列表，子类需要覆盖以提供实际实现。

        Args:
            sample_size: 采样数量

        Returns:
            list: 采样数据列表 [{column: value, ...}, ...]，包含输出列
        """
        logging.warning(
            f"{self.__class__.__name__} 未实现 sample_processed_rows，返回空列表"
        )
        return []

    def fetch_all_rows(self, columns: list[str]) -> list[dict[str, Any]]:
        """
        获取所有行（忽略处理状态）
        
        用于导出或统计等场景。

        Args:
            columns: 需要提取的列名列表

        Returns:
            list: 所有行的数据列表 [{column: value, ...}, ...]
        """
        logging.warning(f"{self.__class__.__name__} 未实现 fetch_all_rows，返回空列表")
        return []

    def fetch_all_processed_rows(self, columns: list[str]) -> list[dict[str, Any]]:
        """
        获取所有已处理行
        
        仅返回输出已完成的记录。

        Args:
            columns: 需要提取的列名列表

        Returns:
            list: 已处理行的数据列表 [{column: value, ...}, ...]
        """
        logging.warning(
            f"{self.__class__.__name__} 未实现 fetch_all_processed_rows，返回空列表"
        )
        return []
