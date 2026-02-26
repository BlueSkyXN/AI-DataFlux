"""
任务状态管理模块

本模块提供任务处理状态的集中管理，确保并发环境下的
状态一致性，是连续任务流模式的关键组件。

类/函数清单:
    TaskStateManager:
        - __init__() -> None
          初始化管理器，创建任务集合和元数据字典及对应的锁
        - try_start_task(task_id) -> bool
          原子操作: 尝试标记任务为处理中 (防重复处理)
          输入: Any 任务 ID | 输出: bool 是否成功标记
        - complete_task(task_id) -> None
          标记任务处理完成，释放处理槽位
          输入: Any 任务 ID
        - is_task_in_progress(task_id) -> bool
          检查任务是否正在处理中
          输入: Any 任务 ID | 输出: bool
        - get_active_count() -> int
          获取当前活动任务数量
          输出: int 正在处理的任务数
        - get_metadata(task_id) -> TaskMetadata
          获取或自动创建任务元数据
          输入: Any 任务 ID | 输出: TaskMetadata 元数据对象
        - remove_metadata(task_id) -> None
          移除任务元数据，释放内存
          输入: Any 任务 ID
        - cleanup_expired(max_age_hours=24) -> int
          清理过期元数据，防止内存泄漏
          输入: int 最大保留小时数 | 输出: int 清理数量

关键变量:
    - _tasks_in_progress: Set 正在处理的任务 ID 集合
    - _progress_lock: threading.Lock 任务集合的线程锁
    - _task_metadata: Dict[Any, TaskMetadata] 任务元数据映射
    - _metadata_lock: threading.Lock 元数据字典的线程锁

模块依赖:
    - src.models.task.TaskMetadata: 任务元数据数据类

核心功能:
    1. 防止重复处理: 同一任务不会被同时处理
    2. 元数据分离: 将任务元数据与业务数据解耦
    3. 生命周期管理: 自动清理过期元数据，防止内存泄漏

使用示例:
    from src.core.state import TaskStateManager

    manager = TaskStateManager()

    # 尝试开始处理任务
    if manager.try_start_task(task_id):
        # 处理任务...
        manager.complete_task(task_id)
    else:
        # 任务已在处理中，跳过
        pass
"""

from .manager import TaskStateManager

__all__ = ["TaskStateManager"]
