"""
数据源任务池抽象基类

定义了数据源任务池的标准接口，支持 MySQL 和 Excel 两种实现。
所有数据源实现必须继承此基类并实现抽象方法。
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any


class BaseTaskPool(ABC):
    """
    数据源任务池抽象基类
    
    定义了数据源操作的标准接口，包括:
    - 任务统计
    - 分片加载
    - 批量获取
    - 结果写回
    - 数据重载
    
    Attributes:
        columns_to_extract: 需要从数据源提取的列名列表
        columns_to_write: 结果写回映射 {别名: 实际列名}
        require_all_input_fields: 是否要求所有输入字段都非空
        tasks: 当前分片的内存任务队列 [(task_id, data_dict), ...]
        lock: 线程锁，保护 tasks 队列的并发访问
    """
    
    def __init__(
        self, 
        columns_to_extract: list[str], 
        columns_to_write: dict[str, str],
        require_all_input_fields: bool = True
    ):
        """
        初始化任务池
        
        Args:
            columns_to_extract: 需要提取的列名列表
            columns_to_write: 写回映射 {别名: 实际列名}
            require_all_input_fields: 是否要求所有输入字段都非空才处理
        """
        self.columns_to_extract = columns_to_extract
        self.columns_to_write = columns_to_write
        self.require_all_input_fields = require_all_input_fields
        self.tasks: list[tuple[Any, dict[str, Any]]] = []
        self.lock = threading.Lock()
    
    # ==================== 抽象方法 ====================
    
    @abstractmethod
    def get_total_task_count(self) -> int:
        """
        获取未处理任务总数
        
        Returns:
            未处理任务数量
        """
        pass
    
    @abstractmethod
    def get_id_boundaries(self) -> tuple[Any, Any]:
        """
        获取任务 ID 边界
        
        用于分片处理时确定范围。
        
        Returns:
            (最小ID, 最大ID)
        """
        pass
    
    @abstractmethod
    def initialize_shard(self, shard_id: int, min_id: Any, max_id: Any) -> int:
        """
        初始化分片
        
        将指定范围内的未处理任务加载到内存队列 (self.tasks)。
        
        Args:
            shard_id: 分片编号
            min_id: 最小 ID (包含)
            max_id: 最大 ID (包含)
            
        Returns:
            加载的任务数量
        """
        pass
    
    @abstractmethod
    def get_task_batch(self, batch_size: int) -> list[tuple[Any, dict[str, Any]]]:
        """
        从内存队列获取一批任务
        
        获取的任务会从队列中移除。
        
        Args:
            batch_size: 批量大小
            
        Returns:
            任务列表 [(task_id, data_dict), ...]
        """
        pass
    
    @abstractmethod
    def update_task_results(self, results: dict[Any, dict[str, Any]]) -> None:
        """
        批量写回任务结果
        
        Args:
            results: 结果字典 {task_id: {field: value, ...}, ...}
        """
        pass
    
    @abstractmethod
    def reload_task_data(self, task_id: Any) -> dict[str, Any] | None:
        """
        重新加载任务的原始输入数据
        
        用于重试时从数据源获取干净的原始数据，而非使用可能被污染的内存缓存。
        
        Args:
            task_id: 任务 ID
            
        Returns:
            原始数据字典，加载失败返回 None
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        关闭资源
        
        释放连接、保存文件等清理操作。
        """
        pass
    
    # ==================== 队列操作 (具体实现) ====================
    
    def add_task_to_front(self, task_id: Any, record_dict: dict[str, Any] | None) -> None:
        """
        将任务放回队列头部 (用于重试)
        
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
    
    def add_task_to_back(self, task_id: Any, record_dict: dict[str, Any] | None) -> None:
        """
        将任务放到队列尾部 (用于延迟处理)
        
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
        """检查内存队列是否还有任务"""
        with self.lock:
            return len(self.tasks) > 0
    
    def get_remaining_count(self) -> int:
        """获取内存队列中剩余的任务数量"""
        with self.lock:
            return len(self.tasks)
    
    def clear_tasks(self) -> None:
        """清空内存队列"""
        with self.lock:
            self.tasks.clear()
    
    # ==================== Token 估算采样 (子类可覆盖) ====================
    
    def sample_unprocessed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """
        采样未处理的行 (用于输入 token 估算)
        
        默认实现: 初始化第一个分片并返回采样数据。
        子类可以覆盖以提供更高效的实现。
        
        Args:
            sample_size: 采样数量
            
        Returns:
            采样数据列表 [{column: value, ...}, ...]
        """
        # 默认实现: 使用分片加载
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
        采样已处理的行 (用于输出 token 估算)
        
        默认实现返回空列表，子类需要覆盖以提供实际实现。
        
        Args:
            sample_size: 采样数量
            
        Returns:
            采样数据列表 [{column: value, ...}, ...]，包含输出列
        """
        logging.warning(
            f"{self.__class__.__name__} 未实现 sample_processed_rows，返回空列表"
        )
        return []

    def fetch_all_rows(self, columns: list[str]) -> list[dict[str, Any]]:
        """
        获取所有行 (忽略处理状态)

        Args:
            columns: 需要提取的列名列表

        Returns:
            所有行的数据列表 [{column: value, ...}, ...]
        """
        logging.warning(
            f"{self.__class__.__name__} 未实现 fetch_all_rows，返回空列表"
        )
        return []
