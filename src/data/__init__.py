"""
数据源抽象层模块 (src.data)

本模块提供数据源的统一抽象，支持多种数据源类型的任务池实现。
通过工厂模式创建具体的任务池实例，统一接口便于处理器调用。

支持的数据源:
    - MySQL: 企业级关系型数据库，适合大规模数据
    - PostgreSQL: 高级特性丰富，适合复杂查询场景
    - SQLite: 本地轻量级数据库，适合开发测试
    - Excel: Excel 电子表格文件（.xlsx、.xls）
    - CSV: 逗号分隔值文件

模块架构:
    ┌──────────────────────────────────────────────────┐
    │              create_task_pool()                   │
    │                 (factory.py)                      │
    │     根据 datasource.type 创建对应实例              │
    └─────────────────┬────────────────────────────────┘
                      │
    ┌─────────────────▼────────────────────────────────┐
    │              BaseTaskPool                         │
    │                (base.py)                          │
    │        抽象基类，定义统一接口                        │
    └─────────────────┬────────────────────────────────┘
                      │
    ┌─────────┬───────┴───────┬─────────┬─────────────┐
    │ MySQL   │ PostgreSQL    │ SQLite  │ Excel/CSV   │
    │TaskPool │   TaskPool    │TaskPool │ TaskPool    │
    └─────────┴───────────────┴─────────┴─────────────┘

核心接口:
    - get_total_task_count(): 获取未处理任务数
    - initialize_shard(): 加载分片数据到内存
    - get_task_batch(): 获取一批任务
    - update_task_results(): 写回处理结果
    - reload_task_data(): 重新加载任务数据（用于重试）

使用示例:
    from src.data import create_task_pool
    
    pool = create_task_pool(config, columns_to_extract, columns_to_write)
    pool.initialize_shard(0, min_id, max_id)
    
    while pool.has_tasks():
        batch = pool.get_task_batch(100)
        # 处理任务...
        pool.update_task_results(results)
    
    pool.close()

依赖模块:
    - engines: DataFrame 引擎抽象（Pandas/Polars）
"""

from .base import BaseTaskPool, validate_sql_identifier
from .factory import create_task_pool, MYSQL_AVAILABLE, EXCEL_ENABLED

__all__ = [
    "BaseTaskPool",
    "create_task_pool",
    "validate_sql_identifier",
    "MYSQL_AVAILABLE",
    "EXCEL_ENABLED",
]
