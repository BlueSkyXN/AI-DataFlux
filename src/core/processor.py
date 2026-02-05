"""
通用 AI 数据处理器

本模块实现 AI-DataFlux 的核心处理引擎，采用组件化架构设计。
作为系统的协调者 (Coordinator)，负责编排各个组件完成数据处理工作流。

架构设计:
    ┌─────────────────────────────────────────────────────────────────┐
    │              UniversalAIProcessor (协调者/Coordinator)           │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
    │  │ TaskPool    │  │ Scheduler   │  │ FluxAIClient            │ │
    │  │ 数据源读写  │  │ 分片调度    │  │ API 通信/超时管理       │ │
    │  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘ │
    │         │                │                      │              │
    │         ▼                ▼                      ▼              │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
    │  │ Content     │  │ TaskState   │  │ RetryStrategy           │ │
    │  │ Processor   │  │ Manager     │  │ 错误分类/重试决策       │ │
    │  │ Prompt/解析 │  │ 状态追踪    │  │ API熔断机制             │ │
    │  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────┘

核心处理模式 - 连续任务流 (Continuous Task Flow):
    传统批处理: 加载批次 → 并发处理 → 等待全部完成 → 写入结果 → 下一批次
    连续任务流: 动态填充任务池 → 任一完成即处理 → 实时写入 → 持续补充新任务

    优势:
    - 无需等待整批完成，响应更快
    - 更灵活的错误处理和重试机制
    - 资源利用率更高

错误处理策略:
    ┌──────────────┬──────────────────┬────────────┬─────────────────┐
    │ 错误类型      │ 处理动作          │ 数据重载   │ 是否暂停        │
    ├──────────────┼──────────────────┼────────────┼─────────────────┤
    │ API_ERROR    │ PAUSE_THEN_RETRY │ ✓          │ ✓ (api_pause)   │
    │ CONTENT_ERROR│ RETRY            │ ✗          │ ✗               │
    │ SYSTEM_ERROR │ RETRY            │ ✓          │ ✗               │
    └──────────────┴──────────────────┴────────────┴─────────────────┘

使用示例:
    # 基本使用
    processor = UniversalAIProcessor("config.yaml")
    processor.run()  # 同步运行

    # 异步使用
    await processor.process_shard_async_continuous()

配置要点:
    - global.flux_api_url: API 网关地址 (必需)
    - datasource.concurrency.batch_size: 最大并发任务数
    - datasource.concurrency.api_pause_duration: API 熔断暂停时长
    - datasource.concurrency.retry_limits: 各类错误最大重试次数

重构历史:
    2026-01-22: 采用组件化架构重构，拆分为独立的处理组件
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp

from ..config.settings import load_config, init_logging, merge_config, DEFAULT_CONFIG
from ..models.errors import ErrorType
from ..data import create_task_pool
from .scheduler import ShardedTaskManager
from .validator import JsonValidator

# 新组件
from .content import ContentProcessor
from .clients import FluxAIClient
from .state import TaskStateManager
from .retry import RetryStrategy, RetryAction


class UniversalAIProcessor:
    """
    通用 AI 数据处理器

    作为协调者 (Coordinator)，负责编排各组件完成数据处理工作流：
    1. TaskPool: 数据源读写
    2. ShardedTaskManager: 分片管理
    3. ContentProcessor: Prompt生成与结果解析
    4. FluxAIClient: API 通信
    5. TaskStateManager: 状态管理
    6. RetryStrategy: 错误重试决策

    组件职责分离:
        - 本类只负责编排协调，不直接实现具体逻辑
        - 各组件独立可测试，便于单元测试和功能扩展
        - 组件之间通过明确接口通信，降低耦合度

    生命周期:
        __init__ → run() → process_shard_async_continuous() → finalize()
              ↓           ↓
         配置加载     异步处理循环
         组件初始化   分片轮转
                     任务处理
                     结果写入

    Attributes:
        config: 配置字典
        flux_api_url: API 网关 URL
        batch_size: 最大并发任务数
        client: FluxAIClient 实例
        content_processor: ContentProcessor 实例
        state_manager: TaskStateManager 实例
        retry_strategy: RetryStrategy 实例
        task_pool: BaseTaskPool 实例
        task_manager: ShardedTaskManager 实例
    """

    def __init__(self, config_path: str, progress_file: str = None):
        """
        初始化处理器

        Args:
            config_path: 配置文件路径
            progress_file: 进度文件路径 (可选，用于 GUI 控制面板)

        Raises:
            ValueError: 配置加载失败
        """
        # 保存进度文件路径
        self.progress_file = progress_file

        # 1. 加载配置
        try:
            config_path_obj = Path(config_path)
            self.config_base_dir = config_path_obj.parent
            user_config = load_config(config_path)
            self.config = merge_config(DEFAULT_CONFIG, user_config)
        except Exception as e:
            raise ValueError(f"无法加载配置文件: {e}") from e

        # 初始化日志
        global_cfg = self.config.get("global", {})
        init_logging(global_cfg.get("log"))

        # 2. 读取关键配置
        # API 配置
        self.flux_api_url = global_cfg.get("flux_api_url")
        if not self.flux_api_url:
            raise ValueError("配置文件 [global] 部分缺少 'flux_api_url'")

        # 数据源配置
        datasource_cfg = self.config.get("datasource", {})

        # 并发配置
        concurrency_cfg = datasource_cfg.get("concurrency", {})
        self.batch_size = concurrency_cfg.get("batch_size", 100)
        self.max_connections = concurrency_cfg.get("max_connections", 1000)
        self.max_connections_per_host = concurrency_cfg.get(
            "max_connections_per_host", 0
        )

        api_pause_duration = float(concurrency_cfg.get("api_pause_duration", 2.0))
        api_error_trigger_window = float(
            concurrency_cfg.get("api_error_trigger_window", 2.0)
        )

        # 重试限制
        retry_limits_cfg = concurrency_cfg.get("retry_limits", {})
        max_retry_counts = {
            ErrorType.API: retry_limits_cfg.get("api_error", 3),
            ErrorType.CONTENT: retry_limits_cfg.get("content_error", 1),
            ErrorType.SYSTEM: retry_limits_cfg.get("system_error", 2),
        }

        # 3. 初始化各组件

        # API 客户端
        logging.info(f"API 端点: {self.flux_api_url}")
        self.client = FluxAIClient(self.flux_api_url)

        # 默认验证器
        self.validator = JsonValidator()
        self.validator.configure(self.config.get("validation"))

        # 状态管理器
        self.state_manager = TaskStateManager()

        # 重试策略
        self.retry_strategy = RetryStrategy(
            max_retries=max_retry_counts,
            api_pause_duration=api_pause_duration,
            api_error_trigger_window=api_error_trigger_window,
        )

        # 4. 初始化数据源和分片管理器
        # 注意：需要在路由初始化之前处理 columns_to_extract
        self.columns_to_extract = self.config.get("columns_to_extract", [])
        self.columns_to_write = self.config.get("columns_to_write", {})

        if not self.columns_to_extract or not self.columns_to_write:
            raise ValueError("缺少 columns_to_extract 或 columns_to_write 配置")

        # 处理路由字段
        # - 如果用户显式声明 → 作为业务字段，提供给 AI
        # - 如果用户未声明 → 自动追加但排除出 Prompt（仅用于路由决策）
        self.routing_enabled = False
        self.routing_field = None
        self.routing_field_is_implicit = False  # 是否为隐式路由字段
        self.routing_contexts: dict[str, dict[str, Any]] = {}

        routing_cfg = self.config.get("routing", {})
        if routing_cfg.get("enabled", False):
            routing_field = routing_cfg.get("field")
            if routing_field:
                self.routing_field = routing_field
                self.routing_enabled = True

                if routing_field not in self.columns_to_extract:
                    # 隐式路由字段：自动追加，标记为排除
                    logging.info(
                        f"路由字段 '{routing_field}' 自动追加到 columns_to_extract（不发给 AI）"
                    )
                    self.columns_to_extract.append(routing_field)
                    self.routing_field_is_implicit = True
                else:
                    # 显式路由字段：用户明确声明，作为业务字段
                    logging.info(
                        f"路由字段 '{routing_field}' 为显式声明的业务字段（会发给 AI）"
                    )
                    self.routing_field_is_implicit = False

        # 提前读取 prompt 配置（_init_routing_contexts 需要这些属性）
        prompt_cfg = self.config.get("prompt", {})
        self.ai_model = prompt_cfg.get("model", "auto")
        self.ai_temperature = prompt_cfg.get("temperature", 0.7)
        self.ai_temperature_override = prompt_cfg.get("temperature_override", True)
        self.ai_system_prompt = prompt_cfg.get("system_prompt")
        self.ai_use_json_schema = prompt_cfg.get("use_json_schema", False)

        # 初始化规则路由上下文（需要在 routing_field_is_implicit 和 ai_* 属性设置后调用）
        self._init_routing_contexts()

        # 创建默认内容处理器（需要在路由初始化后创建，以确定 exclude_fields）
        # 只有隐式路由字段才排除（用户未显式声明的字段）
        exclude_fields = []
        if (
            self.routing_enabled
            and self.routing_field
            and self.routing_field_is_implicit
        ):
            exclude_fields = [self.routing_field]

        self.content_processor = ContentProcessor(
            prompt_template=prompt_cfg.get("template", ""),
            required_fields=prompt_cfg.get("required_fields", []),
            validator=self.validator,
            use_json_schema=self.ai_use_json_schema,
            exclude_fields=exclude_fields,
        )

        try:
            self.task_pool = create_task_pool(
                self.config,
                self.columns_to_extract,
                self.columns_to_write,
            )
        except Exception as e:
            raise RuntimeError(f"无法初始化数据源任务池: {e}") from e

        shard_size = concurrency_cfg.get("shard_size", 10000)
        min_shard_size = concurrency_cfg.get("min_shard_size", 1000)
        max_shard_size = concurrency_cfg.get("max_shard_size", 50000)

        try:
            self.task_manager = ShardedTaskManager(
                self.task_pool,
                shard_size,
                min_shard_size,
                max_shard_size,
                max_retry_counts,
            )
        except Exception as e:
            raise RuntimeError(f"无法初始化分片任务管理器: {e}") from e

        logging.info("UniversalAIProcessor 初始化完成 (组件化版本)")

    def _write_progress(self) -> None:
        """
        写入进度文件 (GUI 控制面板读取)

        仅当 progress_file 参数被设置时才写入。
        使用原子写入 (写临时文件后 rename) 保证数据一致性。
        """
        if not self.progress_file:
            return

        try:
            # 计算当前分片
            current_shard = min(
                self.task_manager.current_shard_index, self.task_manager.total_shards
            )

            data = {
                "total": self.task_manager.total_estimated,
                "processed": self.task_manager.total_processed_successfully,
                "active": self.state_manager.get_active_count(),
                "shard": f"{current_shard}/{self.task_manager.total_shards}",
                "errors": self.task_manager.max_retries_exceeded_count,
                "ts": time.time(),
            }

            tmp_path = self.progress_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, self.progress_file)
        except Exception as e:
            logging.debug(f"写入进度文件失败: {e}")

    def _cleanup_progress(self) -> None:
        """
        清理进度文件

        在处理正常结束时调用，删除进度文件。
        """
        if not self.progress_file:
            return

        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
        except OSError:
            pass

    def _init_routing_contexts(self) -> None:
        """
        初始化规则路由上下文

        加载所有子配置文件，为每个路由规则创建独立的 ContentProcessor 和 Validator。

        前提条件:
            - self.routing_enabled 已设置
            - self.routing_field 已设置
            - self.routing_field_is_implicit 已设置
        """
        if not self.routing_enabled:
            return

        routing_cfg = self.config.get("routing", {})
        subtasks = routing_cfg.get("subtasks")
        if not isinstance(subtasks, list) or not subtasks:
            raise ValueError("routing 配置缺少 subtasks")

        for idx, subtask in enumerate(subtasks):
            if "match" not in subtask or "profile" not in subtask:
                raise ValueError(f"routing.subtasks[{idx}] 必须包含 match 和 profile")

            match_value = str(subtask["match"])
            profile_path = subtask["profile"]
            profile_config = self._load_routing_profile(profile_path)

            allowed_keys = {"prompt", "validation"}
            unknown_keys = set(profile_config.keys()) - allowed_keys
            if unknown_keys:
                raise ValueError(
                    f"routing 子配置仅允许 prompt/validation，发现非法键: {sorted(unknown_keys)}"
                )

            merged_config = merge_config(self.config, profile_config)
            prompt_cfg = merged_config.get("prompt", {})
            validator = JsonValidator()
            validator.configure(merged_config.get("validation"))

            # 路由上下文的 ContentProcessor 也遵循隐式/显式规则
            exclude_fields = []
            if self.routing_field and self.routing_field_is_implicit:
                exclude_fields = [self.routing_field]

            processor = ContentProcessor(
                prompt_template=prompt_cfg.get("template", ""),
                required_fields=prompt_cfg.get("required_fields", []),
                validator=validator,
                use_json_schema=prompt_cfg.get("use_json_schema", False),
                exclude_fields=exclude_fields,
            )

            self.routing_contexts[match_value] = {
                "content_processor": processor,
                "validator": validator,
                "model": prompt_cfg.get("model", self.ai_model),
                "temperature": prompt_cfg.get("temperature", self.ai_temperature),
                "temperature_override": prompt_cfg.get(
                    "temperature_override", self.ai_temperature_override
                ),
                "system_prompt": prompt_cfg.get("system_prompt"),
                "use_json_schema": prompt_cfg.get(
                    "use_json_schema", self.ai_use_json_schema
                ),
            }

    def _load_routing_profile(self, profile_path: str) -> dict[str, Any]:
        """
        加载规则路由的子配置文件

        支持绝对路径和相对路径（相对于主配置文件目录）

        Args:
            profile_path: 子配置文件路径

        Returns:
            子配置字典
        """
        profile_path_obj = Path(profile_path)
        if not profile_path_obj.is_absolute():
            profile_path_obj = self.config_base_dir / profile_path_obj

        return load_config(profile_path_obj)

    def _get_routing_context(self, row_data: Dict[str, Any]) -> dict[str, Any] | None:
        """
        根据记录数据获取路由上下文

        根据 routing.field 指定的字段值，查找匹配的路由上下文。

        行为说明:
            - 路由未启用时返回 None（使用默认配置）
            - 路由字段不存在于记录中时返回 None（使用默认配置）
            - 路由字段值没有匹配规则时返回 None（使用默认配置）
            - 匹配成功时返回对应的路由上下文

        Args:
            row_data: 记录数据字典

        Returns:
            匹配的路由上下文或 None
        """
        if not self.routing_enabled or not self.routing_field:
            return None

        # 字段不存在时使用默认配置
        if self.routing_field not in row_data:
            logging.debug(
                f"routing.field '{self.routing_field}' 不存在于记录中，使用默认配置"
            )
            return None

        match_value = str(row_data.get(self.routing_field))
        context = self.routing_contexts.get(match_value)

        if context is None:
            logging.debug(
                f"routing 未找到匹配规则: {self.routing_field}='{match_value}'，使用默认配置"
            )

        return context

    async def process_shard_async_continuous(self) -> None:
        """
        连续任务流模式的异步处理

        核心处理入口，实现连续任务流 (Continuous Task Flow) 模式:
        1. 初始化分片管理器
        2. 创建 HTTP 连接池
        3. 进入主处理循环
        4. 完成后关闭资源

        连续任务流的关键特性:
            - 动态任务填充: 任务完成后立即补充新任务，保持并发度
            - 实时结果处理: 无需等待整批完成，单个任务完成即处理结果
            - 灵活错误处理: 可针对单个任务进行重试，不影响其他任务

        HTTP 连接池配置:
            - limit: 总连接数上限 (max_connections)
            - limit_per_host: 单主机连接数 (max_connections_per_host)

        Raises:
            Exception: 处理过程中的异常会被记录，但不会中断整体流程
        """
        if not self.task_manager.initialize():
            logging.info("无任务或初始化失败")
            return

        try:
            # 创建连接池
            connector = aiohttp.TCPConnector(
                limit=self.max_connections, limit_per_host=self.max_connections_per_host
            )
            async with aiohttp.ClientSession(connector=connector) as session:
                await self._process_loop(session)
        finally:
            self.task_manager.finalize()

    async def _process_loop(self, session: aiohttp.ClientSession) -> None:
        """
        主处理循环

        实现连续任务流的核心循环逻辑，包含以下 8 个阶段:

        处理流程:
            ┌─────────────────────────────────────────────────────────┐
            │                      主循环开始                          │
            └─────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   │
            ┌───────────────┐                          │
            │ 1. 分片轮转检查│ ─────── 无更多分片 ─────→ 结束
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ 2. 填充任务池  │ ← Backpressure 控制
            └───────┬───────┘   (保持 batch_size 并发度)
                    ▼
            ┌───────────────┐
            │ 3. 等待任务完成│ ← asyncio.wait(FIRST_COMPLETED)
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ 4. 处理完成任务│ → 成功: 存入结果缓冲
            └───────┬───────┘   失败: 重试决策
                    ▼
            ┌───────────────┐
            │ 5. API 熔断检查│ → 触发: 暂停 api_pause_duration 秒
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ 6. 重试任务入队│ → 加到队首优先处理
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ 7. 批量写回结果│
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ 8. 监控与日志  │ ← 每 5 秒输出进度
            └───────┬───────┘   清理过期元数据
                    │
                    └─────────────────→ 返回循环开始

        Args:
            session: aiohttp 客户端会话
        """
        current_shard_num = 0
        active_tasks: Set[asyncio.Task] = set()
        task_id_map: Dict[asyncio.Task, Tuple[Any, Dict[str, Any]]] = {}
        results_buffer: Dict[Any, Dict[str, Any]] = {}

        last_progress_time = time.time()

        while True:
            # 1. 分片轮转检查
            if not self.task_pool.has_tasks() and len(active_tasks) == 0:
                if not self.task_manager.load_next_shard():
                    logging.info("所有分片加载完毕")
                    break
                current_shard_num += 1
                logging.info(
                    f"--- 开始处理分片 {current_shard_num}/{self.task_manager.total_shards} ---"
                )

            # 2. 填充任务池 (Backpressure 控制)
            space_available = self.batch_size - len(active_tasks)
            if space_available > 0 and self.task_pool.has_tasks():
                fetch_count = min(space_available, self.task_pool.get_remaining_count())
                tasks_batch = self.task_pool.get_task_batch(fetch_count)

                for record_id, data in tasks_batch:
                    if self.state_manager.try_start_task(record_id):
                        # 确保元数据已创建
                        self.state_manager.get_metadata(record_id)

                        task = asyncio.create_task(
                            self._process_one_record(session, record_id, data)
                        )
                        task_id_map[task] = (record_id, data)
                        active_tasks.add(task)
                    else:
                        # 如果已经在处理中（理论上不应发生，除非 task pool 返回重复任务），放回队列
                        self.task_pool.add_task_to_front(record_id, data)

            if not active_tasks:
                await asyncio.sleep(0.1)
                continue

            # 3. 等待任一任务完成
            done, pending = await asyncio.wait(
                active_tasks, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
            )
            active_tasks = pending

            # 4. 处理完成的任务
            tasks_to_retry: List[Tuple[Any, Dict[str, Any]]] = []
            should_pause_api = False
            pause_duration = 0.0

            for completed_task in done:
                record_id, _ = task_id_map.pop(completed_task)
                self.state_manager.complete_task(record_id)

                try:
                    result = completed_task.result()

                    error_type = result.get("_error_type")
                    if error_type:
                        # 失败处理
                        metadata = self.state_manager.get_metadata(record_id)
                        # 更新重试计数
                        metadata.increment_retry(error_type)
                        metadata.add_error(error_type, result.get("_error", ""))

                        # 决策
                        decision = self.retry_strategy.decide(error_type, metadata)

                        if decision.action in [
                            RetryAction.RETRY,
                            RetryAction.PAUSE_THEN_RETRY,
                        ]:
                            self.task_manager.retried_tasks_count[error_type] += 1
                            logging.warning(
                                f"记录[{record_id}] {error_type.value}: {result.get('_error')} -> 重试"
                            )

                            # 是否需要重新加载数据
                            retry_data = None
                            if decision.reload_data:
                                retry_data = self.task_pool.reload_task_data(record_id)

                            if retry_data:
                                tasks_to_retry.append((record_id, retry_data))
                            else:
                                # 重新加载失败，算作系统错误或重试失败
                                logging.error(
                                    f"记录[{record_id}] 重新加载数据失败，放弃任务"
                                )
                                self.task_manager.max_retries_exceeded_count += 1
                                self.state_manager.remove_metadata(record_id)

                            # 处理 API 暂停
                            if decision.action == RetryAction.PAUSE_THEN_RETRY:
                                should_pause_api = True
                                pause_duration = decision.pause_duration
                                self.retry_strategy.record_pause()

                        else:  # FAIL
                            logging.error(
                                f"记录[{record_id}] {error_type.value} 超过最大重试次数"
                            )
                            self.task_manager.max_retries_exceeded_count += 1

                            # 将错误信息写回数据源
                            error_result = {}
                            error_msg = f"ERROR: {error_type.value} - {result.get('_error', 'unknown')}"

                            # 对所有输出列填充错误信息
                            for alias in self.columns_to_write.keys():
                                error_result[alias] = error_msg

                            results_buffer[record_id] = error_result
                            self.state_manager.remove_metadata(record_id)
                    else:
                        # 成功
                        results_buffer[record_id] = result
                        self.task_manager.total_processed_successfully += 1
                        self.state_manager.remove_metadata(record_id)

                except Exception as e:
                    logging.error(f"处理结果时发生未捕获异常: {e}")
                    self.state_manager.remove_metadata(record_id)

            # 5. 执行 API 暂停
            if should_pause_api:
                logging.warning(f"触发 API 熔断，暂停 {pause_duration}s...")
                await asyncio.sleep(pause_duration)

            # 6. 重新入队重试任务
            for r_id, r_data in tasks_to_retry:
                self.task_pool.add_task_to_front(r_id, r_data)

            # 7. 批量回写结果
            if results_buffer:
                self.task_pool.update_task_results(results_buffer)
                results_buffer.clear()

            # 8. 监控与日志
            current_time = time.time()
            if current_time - last_progress_time >= 5.0:
                self.task_manager.monitor_memory_usage()
                logging.info(
                    f"进度: {self.task_manager.progress_percent:.1f}% | "
                    f"成功: {self.task_manager.total_processed_successfully} | "
                    f"活动: {self.state_manager.get_active_count()}"
                )
                last_progress_time = current_time
                # 清理过期元数据
                self.state_manager.cleanup_expired()
                # 写入进度文件 (GUI 控制面板读取)
                self._write_progress()

    async def _process_one_record(
        self, session: aiohttp.ClientSession, record_id: Any, row_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理单条记录

        完成单条数据的完整处理流程:
        1. 生成 Prompt - 使用 ContentProcessor 将原始数据转换为 AI 输入
        2. 调用 API - 通过 FluxAIClient 发送请求并获取响应
        3. 解析结果 - 提取 AI 返回的结构化数据

        异常处理策略:
            - aiohttp 异常 (连接失败、超时): 返回 API_ERROR
            - Prompt 生成失败: 返回 SYSTEM_ERROR
            - 其他未知异常: 返回 SYSTEM_ERROR 并记录详情

        Args:
            session: aiohttp 客户端会话
            record_id: 记录唯一标识符
            row_data: 原始记录数据字典

        Returns:
            处理结果字典:
            - 成功: 包含解析后的 AI 响应字段
            - 失败: 包含 _error, _error_type, _details 字段

        Example:
            # 成功返回
            {"field1": "value1", "field2": "value2"}

            # 失败返回
            {"_error": "api_call_failed: TimeoutError",
             "_error_type": ErrorType.API,
             "_details": "Connection timeout after 30s"}
        """
        try:
            routing_context = self._get_routing_context(row_data)
            if routing_context:
                content_processor = routing_context["content_processor"]
                model = routing_context["model"]
                temperature = routing_context["temperature"]
                temperature_override = routing_context["temperature_override"]
                system_prompt = routing_context["system_prompt"]
                use_schema = routing_context["use_json_schema"]
            else:
                content_processor = self.content_processor
                model = self.ai_model
                temperature = self.ai_temperature
                temperature_override = self.ai_temperature_override
                system_prompt = self.ai_system_prompt
                use_schema = self.ai_use_json_schema

            # 1. 生成 Prompt
            prompt = content_processor.create_prompt(row_data)
            if not prompt:
                return {
                    "_error": "prompt_generation_failed",
                    "_error_type": ErrorType.SYSTEM,
                }

            # 2. 调用 API
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response_content = await self.client.call(
                session,
                messages,
                model=model,
                temperature=temperature if temperature_override else None,
                use_json_schema=use_schema,
            )

            # 3. 解析结果
            result = content_processor.parse_response(response_content)
            return result

        except (aiohttp.ClientResponseError, TimeoutError, aiohttp.ClientError) as e:
            return {
                "_error": f"api_call_failed: {type(e).__name__}",
                "_error_type": ErrorType.API,
                "_details": str(e)[:200],
            }
        except Exception as e:
            logging.exception(f"记录[{record_id}] 处理异常: {e}")
            return {
                "_error": f"unexpected_error: {str(e)}",
                "_error_type": ErrorType.SYSTEM,
            }

    def run(self) -> None:
        """
        运行处理器（同步入口）

        为异步处理提供同步包装，适用于命令行直接调用。
        内部使用 asyncio.run() 启动事件循环。

        使用示例:
            processor = UniversalAIProcessor("config.yaml")
            processor.run()  # 阻塞直到处理完成
        """
        logging.info("启动 AI 数据处理引擎...")
        try:
            asyncio.run(self.process_shard_async_continuous())
        except Exception:
            # 异常退出时保留进度文件，便于 GUI 侧基于 ts 判断超时/残留
            raise
        else:
            # 正常结束才清理进度文件
            self._cleanup_progress()
        logging.info("AI 数据处理引擎已停止")
