"""
Token 估算器

本模块用于估算 AI 数据处理任务的 token 用量，帮助用户在大规模
处理前预测 API 成本。支持输入和输出两个维度的估算。

估算模式:
    ┌──────────┬─────────────────────────────────────────────────┐
    │ 模式      │ 说明                                            │
    ├──────────┼─────────────────────────────────────────────────┤
    │ in       │ 仅估算输入 token (系统提示 + 用户提示)           │
    │ out      │ 仅估算输出 token (基于已处理数据采样)            │
    │ io       │ 同时估算输入和输出 token (默认)                  │
    └──────────┴─────────────────────────────────────────────────┘

采样策略:
    - sample_size = -1: 全量计算，遍历所有记录 (精确但较慢)
    - sample_size > 0: 随机采样，按平均值推算总量 (快速估算)

编码器选择:
    - 默认使用 o200k_base (GPT-4o, GPT-4 Turbo 等新模型)
    - 可通过配置指定其他编码器 (如 cl100k_base)

输出统计:
    - total_estimated: 预估总 token 数
    - avg: 平均每条记录的 token 数
    - min/max: 最小/最大值
    - p50/p90/p99: 百分位数统计

使用示例:
    from src.core.token_estimator import run_token_estimation
    
    # 估算输入+输出 token
    result = run_token_estimation("config.yaml", mode="io")
    print(f"预估输入 token: {result['input_tokens']['total_estimated']}")
    print(f"预估输出 token: {result['output_tokens']['total_estimated']}")

配置示例:
    token_estimation:
      mode: io           # in, out, 或 io
      sample_size: 100   # -1 表示全量
      encoding: o200k_base
"""

import json
import logging
from typing import Any

# 延迟导入 tiktoken，避免在未安装时导入失败
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None  # type: ignore


def normalize_mode(mode: str) -> str:
    """
    规范化模式值
    
    将用户输入的各种模式表示统一转换为内部使用的标准形式。

    Args:
        mode: 原始模式值 (如 "input", "in", "io" 等)

    Returns:
        规范化后的模式: "in", "out", 或 "io"
        
    Example:
        >>> normalize_mode("input")
        "in"
        >>> normalize_mode("input_output")
        "io"
    """
    mode_mapping = {
        "input": "in",
        "input_output": "io",
        "in": "in",
        "out": "out",
        "io": "io",
    }
    normalized = mode_mapping.get(mode.lower(), "in")
    if mode.lower() not in mode_mapping:
        logging.warning(f"未知的估算模式 '{mode}'，使用默认值 'in'")
    return normalized


class TokenEstimator:
    """
    Token 估算器

    计算 AI 数据处理任务的预估 token 用量:
    - 输入 token: 系统提示词 + 用户提示词 (含 {record_json} 替换)
    - 输出 token: 基于已处理数据采样，假设纯 JSON 响应
    
    计算流程:
        1. 从数据源采样/获取记录
        2. 构建完整的 API 请求消息 (system + user)
        3. 使用 tiktoken 编码计算 token 数
        4. 统计平均值、百分位数等
        5. 按平均值推算总量

    Attributes:
        config: 完整配置字典
        mode: 估算模式 ("in", "out", 或 "io")
        encoding: tiktoken 编码器
        sample_size: 采样大小 (-1 表示全量)
        system_prompt: 系统提示词
        prompt_template: 用户提示词模板
    """

    def __init__(self, config: dict[str, Any]):
        """
        初始化 Token 估算器

        Args:
            config: 完整配置字典

        Raises:
            ImportError: tiktoken 未安装
            ValueError: 配置无效
        """
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken 未安装，请运行: pip install tiktoken")

        self.config = config

        # Token 估算配置
        token_cfg = config.get("token_estimation", {})
        raw_mode = token_cfg.get("mode", "io")
        self.mode = normalize_mode(raw_mode)
        self.sample_size = token_cfg.get("sample_size", -1)

        # 获取 tiktoken 编码器
        encoding_name = token_cfg.get("encoding", None)

        # 固定使用 o200k_base，除非用户显式指定其他编码
        if not encoding_name:
            encoding_name = "o200k_base"
            logging.info("默认使用 o200k_base 编码器")

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
            logging.info(f"使用 tiktoken 编码: {encoding_name}")
        except Exception as e:
            logging.warning(f"无法获取指定编码 {encoding_name}，尝试回退: {e}")
            if encoding_name != "o200k_base":
                try:
                    self.encoding = tiktoken.get_encoding("o200k_base")
                    logging.info("回退到 o200k_base 编码器")
                except Exception:
                    logging.warning("o200k_base 不可用，回退到 cl100k_base")
                    self.encoding = tiktoken.get_encoding("cl100k_base")
            else:
                logging.warning("o200k_base 不可用，回退到 cl100k_base")
                self.encoding = tiktoken.get_encoding("cl100k_base")

        # 提示词配置
        prompt_cfg = config.get("prompt", {})
        self.system_prompt = prompt_cfg.get("system_prompt", "")
        self.prompt_template = prompt_cfg.get("template", "")
        self.required_fields = prompt_cfg.get("required_fields", [])

        # 列配置
        self.columns_to_extract = config.get("columns_to_extract", [])
        self.columns_to_write = config.get("columns_to_write", {})

        if not self.columns_to_extract:
            raise ValueError("缺少 columns_to_extract 配置")

        logging.info(f"TokenEstimator 初始化完成 | 模式: {self.mode}")

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_message_tokens(self, messages: list[dict[str, str]]) -> int:
        """
        计算消息列表的 token 数量

        直接对拼接后的完整输入文本进行 tiktoken 计数，
        不包含 chat 消息格式的固定开销。
        """
        parts: list[str] = []
        for message in messages:
            content = message.get("content")
            if content:
                parts.append(str(content))

        combined_text = "\n".join(parts)
        return self.count_tokens(combined_text)

    def create_prompt(self, record_data: dict[str, Any]) -> str:
        """创建提示词 (与 processor 保持一致)"""
        if not self.prompt_template:
            return ""

        try:
            filtered_data = {
                k: v
                for k, v in record_data.items()
                if v is not None and not k.startswith("_")
            }
            record_json_str = json.dumps(
                filtered_data, ensure_ascii=False, separators=(",", ":")
            )
        except (TypeError, ValueError) as e:
            logging.error(f"记录数据无法序列化为 JSON: {e}")
            return self.prompt_template.replace(
                "{record_json}", '{"error": "无法序列化数据"}'
            )

        return self.prompt_template.replace("{record_json}", record_json_str)

    def build_messages(self, prompt: str) -> list[dict[str, str]]:
        """构建消息列表 (与 processor 保持一致)"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def estimate_input_tokens_for_record(self, record_data: dict[str, Any]) -> int:
        """估算单条记录的输入 token 数量"""
        prompt = self.create_prompt(record_data)
        messages = self.build_messages(prompt)
        return self.count_message_tokens(messages)

    def estimate_output_tokens_for_record(self, output_data: dict[str, Any]) -> int:
        """
        估算单条记录的输出 token 数量

        假设输出为纯 JSON 格式，使用 columns_to_write 的别名作为字段名。
        """
        # 构建输出 JSON (使用别名作为键)
        output_json = {}
        for alias, col_name in self.columns_to_write.items():
            value = output_data.get(col_name, "")
            if value is not None:
                output_json[alias] = value

        # 序列化为 JSON 字符串
        try:
            json_str = json.dumps(
                output_json, ensure_ascii=False, separators=(",", ":")
            )
        except (TypeError, ValueError):
            json_str = "{}"

        return self.count_tokens(json_str)

    def estimate(
        self, input_pool: Any, output_pool: Any | None = None
    ) -> dict[str, Any]:
        """
        执行 token 估算

        Args:
            input_pool: 输入数据源任务池 (用于输入 token 估算)
            output_pool: 输出数据源任务池 (用于输出 token 估算，可选)
                         如果为 None，则使用 input_pool

        Returns:
            估算结果字典，包含:
            - total_rows: 采样行数
            - input_tokens: 输入 token 统计 (仅 in/io 模式)
            - output_tokens: 输出 token 统计 (仅 out/io 模式)
            - request_count: 预估请求次数
        """
        # 如果未提供 output_pool，使用 input_pool (MySQL 场景)
        if output_pool is None:
            output_pool = input_pool

        result: dict[str, Any] = {
            "mode": self.mode,
            "sample_size": self.sample_size,
            "total_rows": 0,
            "sampled_rows": 0,
            "request_count": 0,
            "processed_total_rows": 0,
        }

        # 获取总任务数 (未处理)
        total_task_count = input_pool.get_total_task_count()
        result["total_rows"] = total_task_count
        result["request_count"] = total_task_count

        # 获取已处理总数 (用于输出估算)
        processed_total_count = output_pool.get_processed_task_count()
        result["processed_total_rows"] = processed_total_count

        # 输入 token 估算 (in 或 io 模式)
        if self.mode in ("in", "io"):
            if self.sample_size == -1:
                # 全量模式: 获取所有行 (忽略处理状态)
                logging.info("正在获取所有输入记录进行全量 Token 计算...")
                unprocessed_samples = input_pool.fetch_all_rows(self.columns_to_extract)
            else:
                # 采样模式: 仅获取未处理行
                unprocessed_samples = input_pool.sample_unprocessed_rows(
                    self.sample_size
                )

            unprocessed_count = len(unprocessed_samples)

            if unprocessed_count == 0:
                logging.warning("未找到记录，无法估算输入 token")
                result["input_tokens"] = {"error": "no_rows_found"}
            else:
                if self.sample_size == -1:
                    result["total_rows"] = unprocessed_count
                    result["request_count"] = unprocessed_count

                result["sampled_rows"] = unprocessed_count
                input_tokens_list = []
                log_step = 1000
                for idx, record_data in enumerate(unprocessed_samples, start=1):
                    tokens = self.estimate_input_tokens_for_record(record_data)
                    input_tokens_list.append(tokens)
                    if self.sample_size == -1 and idx % log_step == 0:
                        logging.info(f"输入 Token 计算进度: {idx}/{unprocessed_count}")

                if self.sample_size == -1:
                    logging.info("输入 Token 计算完成")

                # 如果是全量模式，total_rows 应该等于 sample_count，避免放大
                calc_total_rows = (
                    unprocessed_count if self.sample_size == -1 else total_task_count
                )
                result["input_tokens"] = self._compute_stats(
                    input_tokens_list, calc_total_rows
                )

        # 输出 token 估算 (out 或 io 模式)
        if self.mode in ("out", "io"):
            if self.sample_size == -1:
                # 全量模式: 获取所有已处理行
                logging.info("正在获取所有输出记录进行全量 Token 计算...")
                write_cols = list(self.columns_to_write.values())
                processed_samples = output_pool.fetch_all_processed_rows(write_cols)
            else:
                processed_samples = output_pool.sample_processed_rows(self.sample_size)

            processed_count = len(processed_samples)

            if processed_count == 0:
                logging.warning("未找到输出记录，无法估算输出 token")
                result["output_tokens"] = {"error": "no_rows_found"}
            else:
                if self.mode == "out":
                    result["total_rows"] = processed_total_count
                    result["request_count"] = processed_total_count

                output_tokens_list = []
                log_step = 1000
                for idx, output_data in enumerate(processed_samples, start=1):
                    tokens = self.estimate_output_tokens_for_record(output_data)
                    output_tokens_list.append(tokens)
                    if self.sample_size == -1 and idx % log_step == 0:
                        logging.info(f"输出 Token 计算进度: {idx}/{processed_count}")

                if self.sample_size == -1:
                    logging.info("输出 Token 计算完成")

                # 输出估算基于已处理总数
                calc_total_rows = (
                    processed_count if self.sample_size == -1 else processed_total_count
                )

                result["output_tokens"] = self._compute_stats(
                    output_tokens_list, calc_total_rows
                )
                result["output_sampled_rows"] = processed_count

        return result

    def _compute_stats(self, tokens_list: list[int], total_rows: int) -> dict[str, Any]:
        """计算 token 统计信息"""
        if not tokens_list:
            return {"total": 0, "avg": 0, "min": 0, "max": 0}

        avg = sum(tokens_list) / len(tokens_list)

        # 按百分位排序
        sorted_tokens = sorted(tokens_list)

        def percentile(p: float) -> int:
            idx = int(len(sorted_tokens) * p / 100)
            idx = min(idx, len(sorted_tokens) - 1)
            return sorted_tokens[idx]

        return {
            "total_estimated": int(avg * total_rows),
            "avg": round(avg, 2),
            "min": min(tokens_list),
            "max": max(tokens_list),
            "p50": percentile(50),
            "p90": percentile(90),
            "p99": percentile(99),
            "sample_count": len(tokens_list),
        }


def run_token_estimation(config_path: str, mode: str | None = None) -> dict[str, Any]:
    """
    运行 token 估算

    Args:
        config_path: 配置文件路径
        mode: 可选覆盖模式 ("in", "out", 或 "io")

    Returns:
        估算结果字典
    """
    import copy
    from pathlib import Path
    from ..config.settings import load_config, init_logging
    from ..data import create_task_pool

    # 加载配置
    config = load_config(config_path)

    # 初始化日志 (token 命令也输出进度日志)
    global_cfg = config.get("global", {})
    init_logging(global_cfg.get("log", {}))

    # 覆盖模式
    if mode:
        if "token_estimation" not in config:
            config["token_estimation"] = {}
        config["token_estimation"]["mode"] = mode
        logging.info(f"使用命令行模式覆盖: {mode}")

    # 创建估算器
    estimator = TokenEstimator(config)

    # 创建任务池 (只读模式)
    columns_to_extract = config.get("columns_to_extract", [])
    columns_to_write = config.get("columns_to_write", {})

    # 创建输入池 (Excel: 若输入文件缺失且输出文件存在，则回退使用输出文件)
    try:
        input_pool = create_task_pool(config, columns_to_extract, columns_to_write)
    except FileNotFoundError:
        datasource_type = config.get("datasource", {}).get("type", "excel")
        if datasource_type == "excel":
            excel_cfg = config.get("excel", {})
            input_path = excel_cfg.get("input_path")
            output_path = excel_cfg.get("output_path")
            if output_path and output_path != input_path and Path(output_path).exists():
                logging.warning(
                    f"输入文件不存在，回退使用输出文件作为输入: {output_path}"
                )
                input_config = copy.deepcopy(config)
                input_config["excel"]["input_path"] = output_path
                input_pool = create_task_pool(
                    input_config, columns_to_extract, columns_to_write
                )
            else:
                raise
        else:
            raise

    # 创建输出池 (仅 Excel 数据源，且 out/io 模式需要)
    output_pool = None
    datasource_type = config.get("datasource", {}).get("type", "excel")

    if datasource_type == "excel" and estimator.mode in ("out", "io"):
        excel_cfg = config.get("excel", {})
        output_path = excel_cfg.get("output_path")

        if output_path and output_path != excel_cfg.get("input_path"):
            # 创建输出池配置副本，将 input_path 指向 output_path
            output_config = copy.deepcopy(config)
            output_config["excel"]["input_path"] = output_path

            try:
                output_pool = create_task_pool(
                    output_config, columns_to_extract, columns_to_write
                )
            except FileNotFoundError:
                logging.error(f"输出文件不存在: {output_path}")
                return {
                    "mode": estimator.mode,
                    "error": "output_file_not_found",
                    "message": f"输出文件不存在: {output_path}",
                }

    # 如果没有单独的输出池，使用输入池 (MySQL 或相同文件场景)
    if output_pool is None:
        output_pool = input_pool

    try:
        # 执行估算
        result = estimator.estimate(input_pool, output_pool)
        return result
    finally:
        # 关闭任务池 (不保存)
        pass  # Excel 任务池的 close() 会保存，这里跳过
