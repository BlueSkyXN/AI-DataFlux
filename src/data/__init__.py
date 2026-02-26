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
    - Feishu Bitable: 飞书多维表格
    - Feishu Sheet: 飞书电子表格

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

导出清单:
    - BaseTaskPool: 数据源任务池抽象基类（来自 base.py）
    - create_task_pool(config, columns_to_extract, columns_to_write) -> BaseTaskPool:
        工厂函数，根据配置创建具体任务池实例（来自 factory.py）
    - MYSQL_AVAILABLE (bool): MySQL 连接器是否可用
    - EXCEL_ENABLED (bool): Excel 依赖（pandas + openpyxl）是否可用
    - FEISHU_AVAILABLE (bool): 飞书依赖（aiohttp）是否可用

依赖模块:
    - engines: DataFrame 引擎抽象（Pandas/Polars）
"""

from .base import BaseTaskPool
from .factory import create_task_pool, EXCEL_ENABLED, FEISHU_AVAILABLE, MYSQL_AVAILABLE

__all__ = [
    "BaseTaskPool",
    "create_task_pool",
    "MYSQL_AVAILABLE",
    "EXCEL_ENABLED",
    "FEISHU_AVAILABLE",
]
