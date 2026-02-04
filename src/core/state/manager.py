"""
任务状态管理器

本模块实现任务处理状态的线程安全管理。在连续任务流模式下，
多个协程可能同时处理任务，状态管理器确保一致性和防重复。

设计目标:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     TaskStateManager                             │
    │  ┌───────────────────────┐  ┌─────────────────────────────────┐│
    │  │ _tasks_in_progress    │  │ _task_metadata                  ││
    │  │ Set[task_id]          │  │ Dict[task_id, TaskMetadata]     ││
    │  │ (正在处理的任务集合)  │  │ (任务元数据: 重试次数、错误历史)││
    │  └───────────────────────┘  └─────────────────────────────────┘│
    │           │                              │                       │
    │     _progress_lock               _metadata_lock                  │
    │     (threading.Lock)             (threading.Lock)                │
    └─────────────────────────────────────────────────────────────────┘

元数据分离设计:
    传统方式: 将重试次数等信息存储在任务数据中
        问题: API 错误重试时需要重新加载数据，内部状态会丢失

    分离方式: 元数据存储在 TaskStateManager 中
        优势:
        - 重试时可以重新加载原始数据而不丢失状态
        - 任务完成后自动清理元数据
        - 定期清理过期元数据防止内存泄漏

线程安全:
    使用 threading.Lock 保护共享数据结构。
    虽然主处理是 asyncio 协程，但锁仍然必要，因为:
    - 回调函数可能在不同线程执行
    - 未来可能扩展多进程/多线程模式

使用示例:
    manager = TaskStateManager()

    # 开始处理
    if manager.try_start_task(record_id):
        metadata = manager.get_metadata(record_id)
        # ... 处理任务 ...
        if error:
            metadata.increment_retry(error_type)
        manager.complete_task(record_id)

    # 定期清理
    manager.cleanup_expired(max_age_hours=24)
"""

import logging
import threading
import time
from typing import Any, Dict, Set

from ...models.task import TaskMetadata


class TaskStateManager:
    """
    任务状态管理器

    负责线程安全地管理任务的处理状态和元数据。

    核心职责:
        1. 正在进行的任务集合 (_tasks_in_progress)
        2. 任务元数据 (_task_metadata): 重试次数、错误历史等

    生命周期:
        try_start_task → [处理中] → complete_task → remove_metadata
             ↓                            ↓
        get_metadata                 cleanup_expired
        (创建/获取元数据)            (定期清理过期数据)

    Attributes:
        _tasks_in_progress: 正在处理的任务 ID 集合
        _task_metadata: 任务 ID 到元数据的映射
    """

    def __init__(self):
        """初始化状态管理器"""
        # 任务状态追踪 (正在处理的任务集合)
        self._tasks_in_progress: Set[Any] = set()
        self._progress_lock = threading.Lock()

        # 任务元数据管理 (重试次数、错误历史等)
        self._task_metadata: Dict[Any, TaskMetadata] = {}
        self._metadata_lock = threading.Lock()

    def try_start_task(self, task_id: Any) -> bool:
        """
        尝试标记任务为处理中

        原子操作: 检查是否已在处理中，如果不在则标记。

        Args:
            task_id: 任务唯一标识符

        Returns:
            如果成功标记为处理中返回 True，如果已经在处理中返回 False

        Note:
            这是防止重复处理的关键方法。在获取任务后、开始处理前调用。
        """
        with self._progress_lock:
            if task_id in self._tasks_in_progress:
                return False
            self._tasks_in_progress.add(task_id)
            return True

    def complete_task(self, task_id: Any) -> None:
        """
        标记任务处理完成

        无论任务成功还是失败，都需要调用此方法释放处理槽位。

        Args:
            task_id: 任务唯一标识符
        """
        with self._progress_lock:
            self._tasks_in_progress.discard(task_id)

    def is_task_in_progress(self, task_id: Any) -> bool:
        """
        检查任务是否处于处理中

        Args:
            task_id: 任务唯一标识符

        Returns:
            是否正在处理中
        """
        with self._progress_lock:
            return task_id in self._tasks_in_progress

    def get_active_count(self) -> int:
        """
        获取当前活动任务数

        用于监控并发度和进度报告。

        Returns:
            当前正在处理的任务数量
        """
        with self._progress_lock:
            return len(self._tasks_in_progress)

    def get_metadata(self, task_id: Any) -> TaskMetadata:
        """
        获取或创建任务元数据

        如果元数据不存在，会自动创建新的 TaskMetadata 实例。

        Args:
            task_id: 任务唯一标识符

        Returns:
            任务元数据对象
        """
        with self._metadata_lock:
            if task_id not in self._task_metadata:
                self._task_metadata[task_id] = TaskMetadata(task_id)
            return self._task_metadata[task_id]

    def remove_metadata(self, task_id: Any) -> None:
        """
        移除任务元数据

        任务最终完成 (成功或达到重试上限) 后调用，释放内存。

        Args:
            task_id: 任务唯一标识符
        """
        with self._metadata_lock:
            self._task_metadata.pop(task_id, None)

    def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """
        清理过期的任务元数据

        定期调用以清理长时间未完成的任务元数据，防止内存泄漏。
        默认清理超过 24 小时的元数据。

        Args:
            max_age_hours: 最大保留小时数

        Returns:
            清理的元数据数量
        """
        cutoff_time = time.time() - (max_age_hours * 3600)

        with self._metadata_lock:
            to_remove = [
                task_id
                for task_id, meta in self._task_metadata.items()
                if meta.created_at < cutoff_time
            ]

            for task_id in to_remove:
                del self._task_metadata[task_id]

            if to_remove:
                logging.info(f"[元数据清理] 清理了 {len(to_remove)} 个过期的任务元数据")

            return len(to_remove)
