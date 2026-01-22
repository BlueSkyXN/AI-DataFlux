"""
通用 AI 数据处理器
重构版本 (2026-01-22)：采用组件化架构
"""

import asyncio
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp

from ..config.settings import load_config, init_logging
from ..models.errors import ErrorType
from ..models.task import TaskMetadata
from ..data import create_task_pool, BaseTaskPool
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
    """

    def __init__(self, config_path: str):
        """
        初始化处理器

        Args:
            config_path: 配置文件路径
        """
        # 1. 加载配置
        try:
            self.config = load_config(config_path)
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
        self.max_connections_per_host = concurrency_cfg.get("max_connections_per_host", 0)

        api_pause_duration = float(concurrency_cfg.get("api_pause_duration", 2.0))
        api_error_trigger_window = float(concurrency_cfg.get("api_error_trigger_window", 2.0))

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

        # 验证器
        self.validator = JsonValidator()
        self.validator.configure(self.config.get("validation"))

        # 内容处理器
        prompt_cfg = self.config.get("prompt", {})
        self.content_processor = ContentProcessor(
            prompt_template=prompt_cfg.get("template", ""),
            required_fields=prompt_cfg.get("required_fields", []),
            validator=self.validator,
            use_json_schema=prompt_cfg.get("use_json_schema", False)
        )
        # 保存一些 content processor 需要但在 init 中没用到的参数（为了 API 调用）
        self.ai_model = prompt_cfg.get("model", "auto")
        self.ai_temperature = prompt_cfg.get("temperature", 0.7)

        # 状态管理器
        self.state_manager = TaskStateManager()

        # 重试策略
        self.retry_strategy = RetryStrategy(
            max_retries=max_retry_counts,
            api_pause_duration=api_pause_duration,
            api_error_trigger_window=api_error_trigger_window
        )

        # 4. 初始化数据源和分片管理器
        self.columns_to_extract = self.config.get("columns_to_extract", [])
        self.columns_to_write = self.config.get("columns_to_write", {})

        if not self.columns_to_extract or not self.columns_to_write:
            raise ValueError("缺少 columns_to_extract 或 columns_to_write 配置")

        try:
            self.task_pool = create_task_pool(
                self.config, self.columns_to_extract, self.columns_to_write
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

    async def process_shard_async_continuous(self) -> None:
        """连续任务流模式的异步处理"""
        if not self.task_manager.initialize():
            logging.info("无任务或初始化失败")
            return

        try:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host
            )
            async with aiohttp.ClientSession(connector=connector) as session:
                await self._process_loop(session)
        finally:
            self.task_manager.finalize()

    async def _process_loop(self, session: aiohttp.ClientSession) -> None:
        """主处理循环"""
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
                logging.info(f"--- 开始处理分片 {current_shard_num}/{self.task_manager.total_shards} ---")

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

                        if decision.action in [RetryAction.RETRY, RetryAction.PAUSE_THEN_RETRY]:
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
                                logging.error(f"记录[{record_id}] 重新加载数据失败，放弃任务")
                                self.task_manager.max_retries_exceeded_count += 1
                                self.state_manager.remove_metadata(record_id)

                            # 处理 API 暂停
                            if decision.action == RetryAction.PAUSE_THEN_RETRY:
                                should_pause_api = True
                                pause_duration = decision.pause_duration
                                self.retry_strategy.record_pause()

                        else: # FAIL
                            logging.error(f"记录[{record_id}] {error_type.value} 超过最大重试次数")
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

    async def _process_one_record(
        self, session: aiohttp.ClientSession, record_id: Any, row_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理单条记录"""
        try:
            # 1. 生成 Prompt
            prompt = self.content_processor.create_prompt(row_data)
            if not prompt:
                return {
                    "_error": "prompt_generation_failed",
                    "_error_type": ErrorType.SYSTEM,
                }

            # 2. 调用 API
            messages = []
            if self.config.get("prompt", {}).get("system_prompt"):
                messages.append({"role": "system", "content": self.config["prompt"]["system_prompt"]})
            messages.append({"role": "user", "content": prompt})

            use_schema = self.config.get("prompt", {}).get("use_json_schema", False)

            response_content = await self.client.call(
                session,
                messages,
                model=self.ai_model,
                temperature=self.ai_temperature,
                use_json_schema=use_schema
            )

            # 3. 解析结果
            result = self.content_processor.parse_response(response_content)
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
        """运行处理器（同步入口）"""
        logging.info("启动 AI 数据处理引擎...")
        asyncio.run(self.process_shard_async_continuous())
        logging.info("AI 数据处理引擎已停止")
