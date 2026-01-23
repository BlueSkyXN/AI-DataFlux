"""
模型调度器模块

本模块实现模型配置管理和调度逻辑，包括：
- ModelConfig: 单个模型的配置封装
- ModelDispatcher: 模型调度器，实现加权随机选择和故障退避

核心功能:
    - 模型配置解析：从配置字典创建 ModelConfig 对象
    - 加权随机选择：根据权重随机选择可用模型
    - 故障退避：API 错误时使用指数退避算法暂停模型
    - 可用性缓存：减少频繁的可用性检查开销
    - 性能指标追踪：成功率、响应时间等

退避算法:
    失败次数 <= 3: 退避时间 = 失败次数 * 2 秒
    失败次数 > 3:  退避时间 = min(6 + 2^(失败次数-3), 60) 秒
    
    示例：
    - 第 1 次失败: 2 秒
    - 第 2 次失败: 4 秒
    - 第 3 次失败: 6 秒
    - 第 4 次失败: 8 秒
    - 第 5 次失败: 10 秒
    ...
    - 最大: 60 秒

可用性缓存:
    为避免频繁获取锁检查可用性，使用带 TTL 的缓存。
    默认 TTL 为 0.5 秒，在此期间直接返回缓存结果。

使用示例:
    # 创建调度器
    dispatcher = ModelDispatcher(models)
    
    # 选择模型
    model = dispatcher.select_model(exclude_model_ids={"model-1"})
    
    # 标记结果
    if success:
        dispatcher.mark_model_success(model.id)
    else:
        dispatcher.mark_model_failed(model.id, "api_error")
"""

import logging
import random
import time
from typing import Any

from .limiter import RWLock


class ModelConfig:
    """
    模型配置类
    
    封装单个模型的所有配置信息，包括 API 端点、认证、超时、权重等。
    从配置字典和通道配置中提取并验证必要字段。

    Attributes:
        id (str): 模型唯一标识符
        name (str): 模型显示名称
        model (str): 上游 API 的模型标识符
        channel_id (str): 所属通道 ID
        api_key (str): API 认证密钥
        timeout (int): 模型级超时时间（秒）
        weight (int): 加权选择的权重
        temperature (float): 默认温度参数
        safe_rps (float): 安全请求速率（用于限流）
        base_url (str): API 基础 URL
        api_url (str): 完整 API URL
        proxy (str): 代理地址（可选）
        ssl_verify (bool): 是否验证 SSL 证书
    """

    def __init__(self, model_dict: dict[str, Any], channels: dict[str, Any]):
        """
        从配置字典初始化模型配置
        
        解析模型配置和关联的通道配置，验证必填字段，
        计算最终超时时间和 API URL。

        Args:
            model_dict: 模型配置字典，需包含 id、model、channel_id
            channels: 通道配置字典，键为 channel_id
        
        Raises:
            ValueError: 缺少必填字段或通道不存在
        """
        # 基本配置
        self.id = str(model_dict.get("id", ""))
        self.name = model_dict.get("name", self.id)
        self.model = model_dict.get("model", "")
        self.channel_id = str(model_dict.get("channel_id", ""))
        self.api_key = model_dict.get("api_key", "")
        self.timeout = int(model_dict.get("timeout", 300))
        self.weight = int(model_dict.get("weight", 1))
        self.temperature = float(model_dict.get("temperature", 0.7))
        
        # safe_rps: 用于限流，默认基于权重计算
        self.safe_rps = float(
            model_dict.get("safe_rps", max(0.5, min(self.weight / 10, 10)))
        )
        
        # 功能支持标志
        self.supports_json_schema = bool(model_dict.get("supports_json_schema", False))
        self.supports_advanced_params = bool(
            model_dict.get("supports_advanced_params", False)
        )

        # 从通道配置获取 API 端点信息
        channel = channels.get(self.channel_id, {})
        self.base_url = channel.get("base_url", "")
        self.api_path = channel.get("api_path", "/v1/chat/completions")
        self.proxy = channel.get("proxy", "")
        self.ssl_verify = channel.get("ssl_verify", True)
        self.channel_name = channel.get("name", self.channel_id)
        self.channel_timeout = int(channel.get("timeout", 600))

        # 验证必填字段
        if not self.id or not self.model or not self.channel_id:
            raise ValueError(
                f"模型配置缺少必填字段: id={self.id}, model={self.model}, channel_id={self.channel_id}"
            )
        if self.channel_id not in channels:
            raise ValueError(
                f"模型 {self.id}: channel_id='{self.channel_id}' 在 channels 中不存在"
            )
        if not self.base_url:
            raise ValueError(f"通道 '{self.channel_id}' 缺少 base_url 配置")

        # 计算最终超时时间（取模型和通道中的较小值）
        self.final_timeout = min(self.timeout, self.channel_timeout)
        self.connect_timeout = 30  # 固定的连接超时时间
        self.read_timeout = self.final_timeout  # 总读取/处理超时时间

        # 构建完整 API URL
        self.api_url = self.base_url.rstrip("/") + self.api_path

    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            dict: 模型配置的字典表示
        """
        return {
            "id": self.id,
            "name": self.name,
            "model": self.model,
            "channel_id": self.channel_id,
            "weight": self.weight,
            "safe_rps": self.safe_rps,
            "supports_json_schema": self.supports_json_schema,
            "supports_advanced_params": self.supports_advanced_params,
            "channel_name": self.channel_name,
        }


class ModelDispatcher:
    """
    模型调度器
    
    负责模型的选择和状态管理，实现：
    - 加权随机选择：根据模型权重进行随机选择
    - 故障退避：API 错误时暂时禁用模型
    - 可用性缓存：减少锁竞争
    - 性能指标：追踪成功率和响应时间

    Attributes:
        backoff_factor (int): 指数退避算法的基数（默认 2）
        models (dict): 模型 ID 到 ModelConfig 的映射
        _model_state (dict): 模型运行时状态（失败次数、可用时间等）
        _availability_cache (dict): 可用性缓存
    
    线程安全:
        使用读写锁 (RWLock) 保护共享状态，允许并发读取但独占写入。
    """

    def __init__(self, models: list[ModelConfig], backoff_factor: int = 2):
        """
        初始化调度器

        Args:
            models: ModelConfig 对象列表
            backoff_factor: 指数退避算法的基数（默认 2）
        """
        self.backoff_factor = backoff_factor
        self.models = {m.id: m for m in models}

        # 初始化每个模型的运行时状态
        self._model_state: dict[str, dict[str, Any]] = {}
        for m in models:
            self._model_state[m.id] = {
                "fail_count": 0,           # 连续失败次数
                "next_available_ts": 0,    # 下次可用的时间戳
                "success_count": 0,        # 总成功次数
                "error_count": 0,          # 总错误次数
                "avg_response_time": 0.0,  # 平均响应时间（指数移动平均）
            }

        # 读写锁和可用性缓存
        self._rwlock = RWLock()
        self._availability_cache: dict[str, bool] = {}
        self._cache_last_update: float = 0
        self._cache_ttl: float = 0.5  # 缓存 TTL（秒）

        # 初始化缓存
        self._update_availability_cache()

    def _update_availability_cache(self) -> None:
        """更新可用性缓存"""
        with self._rwlock.write_lock():
            current_time = time.time()
            new_cache = {}
            for model_id, state in self._model_state.items():
                new_cache[model_id] = current_time >= state["next_available_ts"]
            self._availability_cache = new_cache
            self._cache_last_update = current_time

    def update_model_metrics(
        self, model_id: str, response_time: float, success: bool
    ) -> None:
        """更新模型性能指标"""
        with self._rwlock.write_lock():
            state = self._model_state.get(model_id)
            if not state:
                return

            if success:
                state["success_count"] += 1
            else:
                state["error_count"] += 1

            # 指数移动平均更新响应时间
            total_calls = state["success_count"] + state["error_count"]
            if total_calls == 1:
                state["avg_response_time"] = response_time
            else:
                weight = 0.1
                current_avg = state.get("avg_response_time", response_time)
                state["avg_response_time"] = (
                    current_avg * (1 - weight) + response_time * weight
                )

    def get_model_success_rate(self, model_id: str) -> float:
        """获取模型成功率"""
        with self._rwlock.read_lock():
            state = self._model_state.get(model_id)
            if not state:
                return 0.0
            total = state["success_count"] + state["error_count"]
            return state["success_count"] / total if total > 0 else 1.0

    def get_model_avg_response_time(self, model_id: str) -> float:
        """获取模型平均响应时间"""
        with self._rwlock.read_lock():
            state = self._model_state.get(model_id)
            return (state.get("avg_response_time", 1.0) or 1.0) if state else 1.0

    def is_model_available(self, model_id: str) -> bool:
        """判断模型是否可用"""
        current_time = time.time()

        with self._rwlock.read_lock():
            cache_expired = current_time - self._cache_last_update >= self._cache_ttl
            if not cache_expired and model_id in self._availability_cache:
                return self._availability_cache[model_id]

        if cache_expired:
            self._update_availability_cache()
            with self._rwlock.read_lock():
                return self._availability_cache.get(model_id, False)

        with self._rwlock.read_lock():
            state = self._model_state.get(model_id)
            return (current_time >= state["next_available_ts"]) if state else False

    def mark_model_success(self, model_id: str) -> None:
        """标记模型调用成功"""
        with self._rwlock.write_lock():
            state = self._model_state.get(model_id)
            if state:
                was_unavailable = time.time() < state["next_available_ts"]
                state["fail_count"] = 0
                state["next_available_ts"] = 0
                self._availability_cache[model_id] = True

                if was_unavailable:
                    logging.info(f"模型[{model_id}] 调用成功，恢复可用")

    def mark_model_failed(self, model_id: str, error_type: str = "api_error") -> None:
        """标记模型调用失败"""
        if error_type != "api_error":
            logging.warning(f"模型[{model_id}] 遇到 {error_type}，不执行退避")
            return

        with self._rwlock.write_lock():
            state = self._model_state.get(model_id)
            if not state:
                return

            state["fail_count"] += 1
            fail_count = state["fail_count"]

            # 计算退避时间
            if fail_count <= 3:
                backoff_seconds = fail_count * 2
            else:
                backoff_seconds = min(6 + (self.backoff_factor ** (fail_count - 3)), 60)

            state["next_available_ts"] = time.time() + backoff_seconds
            self._availability_cache[model_id] = False

            logging.warning(
                f"模型[{model_id}] API 调用失败，第 {fail_count} 次，"
                f"进入退避 {backoff_seconds:.2f} 秒"
            )

    def get_available_models(
        self, exclude_model_ids: set[str] | None = None
    ) -> list[str]:
        """获取所有可用模型 ID"""
        exclude_ids = exclude_model_ids or set()

        with self._rwlock.read_lock():
            all_model_ids = list(self._model_state.keys())

        return [
            model_id
            for model_id in all_model_ids
            if model_id not in exclude_ids and self.is_model_available(model_id)
        ]

    def select_model(
        self, exclude_model_ids: set[str] | None = None
    ) -> ModelConfig | None:
        """
        使用加权随机算法选择一个可用模型

        Args:
            exclude_model_ids: 要排除的模型 ID 集合

        Returns:
            选中的模型配置，如果没有可用模型返回 None
        """
        available_ids = self.get_available_models(exclude_model_ids)

        if not available_ids:
            return None

        # 获取可用模型及其权重
        available_models = [
            self.models[model_id]
            for model_id in available_ids
            if model_id in self.models
        ]

        if not available_models:
            return None

        # 加权随机选择
        weights = [m.weight for m in available_models]
        selected = random.choices(available_models, weights=weights, k=1)

        return selected[0] if selected else None

    def get_model_config(self, model_id: str) -> ModelConfig | None:
        """获取模型配置"""
        return self.models.get(model_id)

    def get_all_model_stats(self) -> list[dict[str, Any]]:
        """获取所有模型的统计信息"""
        stats = []

        with self._rwlock.read_lock():
            for model_id, state in self._model_state.items():
                model = self.models.get(model_id)
                if not model:
                    continue

                total = state["success_count"] + state["error_count"]
                success_rate = state["success_count"] / total if total > 0 else 1.0

                stats.append(
                    {
                        "id": model_id,
                        "name": model.name,
                        "model": model.model,
                        "weight": model.weight,
                        "success_rate": success_rate,
                        "avg_response_time": state.get("avg_response_time", 0),
                        "available": self._availability_cache.get(model_id, False),
                        "channel": model.channel_name,
                    }
                )

        return stats
