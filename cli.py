#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-DataFlux 统一命令行接口 (CLI) 入口模块

本模块提供 AI-DataFlux 的完整命令行界面，集成了数据处理、API 网关、
Token 估算、版本信息和库状态检查等所有功能的统一入口。

核心子命令:
    process  - 运行数据处理引擎，从数据源读取任务并调用 AI API
    gateway  - 启动 OpenAI 兼容的 API 网关服务
    token    - 估算处理任务的 Token 用量（用于成本预估）
    version  - 显示版本信息
    check    - 检查高性能库（Polars、calamine 等）的安装状态

使用示例:
    python cli.py process --config config.yaml     # 运行数据处理
    python cli.py process -c config.yaml --validate  # 仅验证配置
    python cli.py gateway --port 8787              # 启动 API 网关
    python cli.py gateway -p 8787 -w 4             # 4 worker 进程
    python cli.py token --config config.yaml       # 估算输入+输出 Token
    python cli.py token -c config.yaml --mode in   # 仅估算输入 Token
    python cli.py version                          # 显示版本号
    python cli.py check                            # 检查库安装状态

架构说明:
    CLI 是用户与系统交互的主要入口，内部调用各模块的核心功能：
    - process: 调用 src.core.UniversalAIProcessor
    - gateway: 调用 src.gateway.app.run_server
    - token: 调用 src.core.token_estimator.run_token_estimation
    - check: 调用 src.data.engines.get_available_libraries

退出码:
    0 - 执行成功
    1 - 用户中断或发生错误

依赖模块:
    - src.core: 核心处理引擎
    - src.gateway: API 网关
    - src.config: 配置管理
    - src.data.engines: 数据引擎
    - src.utils.console: 控制台输出工具

函数清单:
    辅助函数:
        _validate_port(value: str) -> int
            验证端口号范围 (1024-65535)
            输入: 端口号字符串 | 输出: 验证通过的端口号整数
        _validate_config_path(value: str) -> str
            验证配置文件路径格式 (.yaml/.yml 扩展名)
            输入: 文件路径字符串 | 输出: 验证通过的路径字符串
        _check_rlimit() -> None
            检查 Unix 文件描述符限制，低于 256 时输出警告

    子命令处理函数 (均接收 argparse.Namespace 参数，返回 int 退出码):
        cmd_process(args) -> int
            执行数据处理：支持 --validate 仅验证模式和完整处理模式
        cmd_gateway(args) -> int
            启动 API 网关服务：配置监听地址、端口、工作进程数
        cmd_version(args) -> int
            显示 AI-DataFlux 版本号
        cmd_check(args) -> int
            检查可选高性能库 (polars/calamine/xlsxwriter) 安装状态
        cmd_token(args) -> int
            估算 Token 用量：支持 in/out/io 三种模式
        cmd_gui(args) -> int
            启动 Web GUI 控制面板 (仅完整版可用)

    主入口:
        main() -> int
            CLI 主入口：创建参数解析器、注册子命令、分发执行

关键变量:
    resource: Unix 资源限制模块 (Windows 下为 None)

作者: AI-DataFlux Team
版本: 参见 src/__init__.py
"""

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

try:
    import resource  # Unix-only
except Exception:
    resource = None


EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_JOB_FAILED = 3
EXIT_CONNECTION_ERROR = 4
EXIT_NOT_FOUND_OR_CONFLICT = 5


def _write_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _error_payload(code: str, message: str, details=None) -> dict:
    return {"error": {"code": code, "message": message, "details": details}}


def _validate_port(value: str) -> int:
    """
    验证端口号范围

    Args:
        value: 用户输入的端口号字符串

    Returns:
        int: 验证通过的端口号

    Raises:
        argparse.ArgumentTypeError: 端口号无效
    """
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid port: {value!r} is not a number")

    if not (1024 <= port <= 65535):
        raise argparse.ArgumentTypeError(
            f"Port must be between 1024 and 65535, got {port}"
        )
    return port


def _validate_config_path(value: str) -> str:
    """
    验证配置文件路径（仅检查基本格式，不检查存在性）

    验证模式时会检查文件存在性。启动服务时可能配置文件稍后创建。

    Args:
        value: 用户输入的配置文件路径

    Returns:
        str: 配置文件路径

    Raises:
        argparse.ArgumentTypeError: 路径无效
    """
    if not value:
        raise argparse.ArgumentTypeError("Config path cannot be empty")

    # 检查文件扩展名是否合理
    if not value.endswith((".yaml", ".yml")):
        raise argparse.ArgumentTypeError(
            f"Config file should have .yaml or .yml extension, got: {value}"
        )

    return value


def _check_rlimit(*, stream=None):
    """检查文件描述符限制"""
    target = stream or sys.stdout
    if resource is None or sys.platform not in ("darwin", "linux"):
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Current limits: ({soft}, {hard})", file=target)
        if soft <= 256:
            from src.utils.console import console

            print(
                f"{console.warn} File descriptor limit is too low ({soft}). This triggers crashes on macOS.",
                file=target,
            )
            print(
                f"{console.tip} Run 'ulimit -n 10240' or higher before running this program.",
                file=target,
            )
    except Exception:
        pass


def cmd_process(args):
    """
    执行数据处理子命令

    Args:
        args: argparse 解析后的命令行参数对象
            - config (str): 配置文件路径
            - validate (bool): 是否仅验证配置
            - progress_file (str): 进度文件路径 (可选，用于 GUI 控制面板)

    Returns:
        int: 退出码，0 表示成功

    工作流程:
        1. 验证模式：加载配置并显示关键信息（数据源、引擎、列配置）
        2. 处理模式：创建处理器并执行完整的数据处理流程
    """
    json_output = bool(getattr(args, "json", False))
    from src.utils.console import console
    from src.core import UniversalAIProcessor

    if args.validate:
        _check_rlimit(stream=sys.stderr if json_output else sys.stdout)
        from src.config import compile_job_config, load_config, validate_config

        try:
            config = load_config(args.config)
            validation = validate_config(config, args.config)
            compiled = compile_job_config(config, args.config)
        except Exception as exc:
            if json_output:
                _write_json(_error_payload("config_invalid", str(exc)))
            else:
                print(f"{console.error} Config invalid: {exc}")
            return EXIT_CONFIG_ERROR

        if json_output:
            payload = {
                "valid": not validation["errors"],
                "config": str(Path(args.config).expanduser()),
                "errors": validation["errors"],
                "warnings": validation["warnings"],
            }
            _write_json(payload)
            return EXIT_CONFIG_ERROR if validation["errors"] else EXIT_OK

        for warning in validation["warnings"]:
            print(f"{console.warn} {warning}")

        if validation["errors"]:
            print(f"{console.error} Config invalid: {args.config}")
            for error in validation["errors"]:
                print(f"  - {error}")
            return EXIT_CONFIG_ERROR

        print(f"{console.ok} Config valid: {args.config}")
        print(f"  - Datasource: {config.job.datasource.type}")
        print(f"  - Engine: {getattr(config.job.datasource, 'engine', 'n/a')}")
        print(f"  - Input columns: {config.job.columns.extract}")
        print(f"  - Output columns: {list(config.job.columns.write.values())}")
        # 显示路由配置信息（如果启用）
        routing = compiled.get("routing", {})
        if routing.get("enabled"):
            subtasks = routing.get("subtasks", [])
            print(
                f"  - Routing: enabled on '{routing.get('field', 'N/A')}' ({len(subtasks)} rules)"
            )
        return EXIT_OK

    # 获取进度文件路径 (可选)
    progress_file = getattr(args, "progress_file", None)

    # 创建处理器并执行（使用配置文件路径）
    try:
        processor = UniversalAIProcessor(args.config, progress_file=progress_file)
        completed = processor.run()
    except Exception as exc:
        if json_output:
            _write_json(_error_payload("process_failed", str(exc)))
        else:
            print(f"{console.error} Processing failed: {exc}", file=sys.stderr)
        return EXIT_JOB_FAILED

    manager = processor.task_manager
    failed = manager.max_retries_exceeded_count
    payload = {
        "status": (
            "cancelled"
            if not completed
            else "completed_with_errors" if failed else "completed"
        ),
        "persisted": manager.total_processed_successfully,
        "failed": failed,
        "discovered": manager.total_estimated,
        "retries": sum(manager.retried_tasks_count.values()),
    }
    if json_output:
        _write_json(payload)
    return EXIT_OK if completed and not failed else EXIT_JOB_FAILED


def cmd_gateway(args):
    """
    启动 API 网关子命令

    启动 OpenAI 兼容的 API 网关服务，提供多模型负载均衡、
    自动故障切换、令牌桶限流等功能。

    Args:
        args: argparse 解析后的命令行参数对象
            - config (str): 配置文件路径
            - host (str): 监听地址，默认 0.0.0.0
            - port (int): 监听端口，默认 8787
            - workers (int): 工作进程数，默认 1
            - reload (bool): 是否启用热重载（开发模式）

    Returns:
        int: 退出码，0 表示成功

    网关功能:
        - OpenAI 兼容的 /v1/chat/completions 端点
        - 加权随机模型选择
        - 模型故障自动切换
        - 令牌桶限流保护
        - IP 池 DNS 轮询
    """
    _check_rlimit()
    from src.config import load_config, require_gateway_config
    from src.gateway.app import run_server

    gateway_config = require_gateway_config(load_config(args.config))
    if args.workers != 1:
        raise ValueError("AI-DataFlux 4.0 H1-H2 仅支持 gateway workers=1")

    run_server(
        config_path=args.config,
        host=args.host or gateway_config.listen.host,
        port=args.port or gateway_config.listen.port,
        workers=args.workers,
        reload=args.reload,
    )
    return 0


def cmd_version(args):
    """
    显示版本信息子命令

    从 src 包的 __version__ 变量读取并显示当前版本号。

    Args:
        args: argparse 解析后的命令行参数对象（本命令不使用）

    Returns:
        int: 退出码，0 表示成功
    """
    from src import __version__

    print(f"AI-DataFlux v{__version__}")
    return 0


def cmd_check(args):
    """
    检查库安装状态子命令

    检测高性能可选库（Polars、calamine、xlsxwriter 等）的安装状态，
    并提供缺失库的安装建议。

    Args:
        args: argparse 解析后的命令行参数对象（本命令不使用）

    Returns:
        int: 退出码，0 表示成功

    检查的库:
        - polars: 高性能 DataFrame 库（多线程，比 Pandas 快）
        - calamine: 高性能 Excel 读取器（比 openpyxl 快 10 倍）
        - xlsxwriter: 高性能 Excel 写入器（比 openpyxl 快 3 倍）

    输出格式:
        显示每个库的安装状态（✓ 或 ✗），并在最后给出缺失库的安装命令
    """
    from src.data.engines import get_available_libraries
    from src.utils.console import console, print_status

    print("AI-DataFlux Library Status\n")
    print("=" * 40)

    # 获取并显示所有库的可用性状态
    libs = get_available_libraries()
    for name, available in libs.items():
        print_status(available, name)

    print("=" * 40)

    # 提供缺失库的安装建议
    missing = [name for name, avail in libs.items() if not avail]
    if missing:
        print(f"\n{console.tip} Install high-performance libraries:")
        print(f"   pip install {' '.join(missing)}")
    else:
        print(f"\n{console.ok} All high-performance libraries installed!")

    return 0


def cmd_token(args):
    """
    Token 用量估算子命令

    基于配置文件和数据源，估算处理任务所需的 Token 数量，
    用于 API 成本预估和预算规划。

    Args:
        args: argparse 解析后的命令行参数对象
            - config (str): 配置文件路径
            - mode (str): 估算模式
                - 'in': 仅估算输入 Token（从未处理数据）
                - 'out': 仅估算输出 Token（从已处理数据）
                - 'io' 或 None: 同时估算输入和输出 Token

    Returns:
        int: 退出码，0 表示成功，1 表示失败

    依赖:
        需要安装 tiktoken 库用于 Token 计数
        pip install tiktoken

    输出内容:
        - 总行数和采样行数
        - 预估请求数
        - 输入 Token 统计（总量、平均、最小、最大、百分位数）
        - 输出 Token 统计（总量、平均、最小、最大、百分位数）
    """
    from src.utils.console import console
    from src.core.token_estimator import run_token_estimation

    # 获取估算模式参数
    mode = args.mode if hasattr(args, "mode") and args.mode else None

    try:
        # 执行 Token 估算
        result = run_token_estimation(args.config, mode)
    except ImportError as e:
        # tiktoken 库未安装
        print(f"{console.error} {e}")
        print(f"{console.tip} Install tiktoken: pip install tiktoken")
        return 1
    except Exception as e:
        # 其他估算错误
        print(f"{console.error} Token estimation failed: {e}")
        return 1

    # 检查估算结果是否有错误
    if result.get("error"):
        print(f"{console.error} {result.get('message', result.get('error'))}")
        return 1

    # 打印估算结果标题
    print("\n" + "=" * 50)
    print("  Token Estimation Results")
    print("=" * 50)

    # 显示基本信息
    mode_display = result.get("mode", "in")
    mode_desc = {"in": "input only", "out": "output only", "io": "input + output"}
    print(f"\n{console.info} Mode: {mode_display} ({mode_desc.get(mode_display, '')})")
    print(f"{console.info} Total rows: {result.get('total_rows', 0)}")
    print(f"{console.info} Sampled rows: {result.get('sampled_rows', 0)}")
    if result.get("processed_total_rows", 0):
        print(f"{console.info} Processed rows: {result.get('processed_total_rows', 0)}")
    if result.get("output_sampled_rows", 0):
        print(
            f"{console.info} Output sampled rows: {result.get('output_sampled_rows', 0)}"
        )
    print(f"{console.info} Estimated requests: {result.get('request_count', 0)}")

    # 显示输入 Token 统计（适用于 in/io 模式）
    input_stats = result.get("input_tokens", {})
    if input_stats and "error" not in input_stats:
        print(f"\n{console.ok} Input Token Estimation:")
        print(f"   Total (estimated): {input_stats.get('total_estimated', 0):,}")
        print(f"   Per-request avg:   {input_stats.get('avg', 0):.1f}")
        print(f"   Per-request min:   {input_stats.get('min', 0)}")
        print(f"   Per-request max:   {input_stats.get('max', 0)}")
        print(
            f"   P50: {input_stats.get('p50', 0)} | P90: {input_stats.get('p90', 0)} | P99: {input_stats.get('p99', 0)}"
        )
    elif mode_display in ("in", "io") and (not input_stats or "error" in input_stats):
        print(
            f"\n{console.warn} Input estimation unavailable (no unprocessed rows found)"
        )

    # 显示输出 Token 统计（适用于 out/io 模式）
    output_stats = result.get("output_tokens", {})
    if output_stats and "error" not in output_stats:
        print(f"\n{console.ok} Output Token Estimation:")
        print(f"   Total (estimated): {output_stats.get('total_estimated', 0):,}")
        print(f"   Per-response avg:  {output_stats.get('avg', 0):.1f}")
        print(f"   Per-response min:  {output_stats.get('min', 0)}")
        print(f"   Per-response max:  {output_stats.get('max', 0)}")
        print(
            f"   P50: {output_stats.get('p50', 0)} | P90: {output_stats.get('p90', 0)} | P99: {output_stats.get('p99', 0)}"
        )
    elif mode_display in ("out", "io") and (
        not output_stats or "error" in output_stats
    ):
        print(
            f"\n{console.warn} Output estimation unavailable (no processed rows found)"
        )

    print("\n" + "=" * 50)

    return 0


def cmd_config_validate(args):
    """Validate the canonical v4 schema with stable JSON output."""

    from src.config import load_config, validate_config

    try:
        config = load_config(args.config)
        result = validate_config(config, args.config)
    except Exception as exc:
        payload = _error_payload("config_invalid", str(exc))
        if args.json:
            _write_json(payload)
        else:
            print(f"Config invalid: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    payload = {
        "valid": not result["errors"],
        "config": str(Path(args.config).expanduser()),
        "errors": result["errors"],
        "warnings": result["warnings"],
    }
    if args.json:
        _write_json(payload)
    else:
        for warning in result["warnings"]:
            print(f"WARNING: {warning}", file=sys.stderr)
        if result["errors"]:
            for error in result["errors"]:
                print(f"ERROR: {error}", file=sys.stderr)
        else:
            print(f"Config valid: {args.config}")
    return EXIT_CONFIG_ERROR if result["errors"] else EXIT_OK


async def _run_worker(config_path: str) -> None:
    from src.control.job_service import JobService

    service = JobService(config_path)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass
    loop_task = asyncio.create_task(service.run_loop())
    try:
        await stop_event.wait()
    finally:
        await service.stop()
        await asyncio.gather(loop_task, return_exceptions=True)


def cmd_worker(args):
    """Run the durable background Worker without the GUI."""

    try:
        if args.json:
            _write_json({"status": "starting", "mode": "worker"})
        asyncio.run(_run_worker(args.config))
        return EXIT_OK
    except KeyboardInterrupt:
        return EXIT_OK
    except Exception as exc:
        if args.json:
            _write_json(_error_payload("worker_failed", str(exc)))
        else:
            print(f"Worker failed: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR


def _local_job_service(config_path: str):
    from src.control.job_service import JobService

    return JobService(config_path)


def _workspace_reference(service, config_path: str) -> tuple[str, str]:
    target = Path(config_path).expanduser().resolve()
    for root_id, root in service.roots.items():
        try:
            relative = target.relative_to(root)
        except ValueError:
            continue
        return root_id, str(relative)
    raise ValueError("config path is outside workspace.roots")


def _control_client(args):
    from src.control.client import ControlClient

    token = os.getenv("DATAFLUX_TOKEN", "").strip()
    if not token:
        try:
            from src.config import load_config

            token = str(
                load_config(getattr(args, "config", "config.yaml")).runtime.auth.token
            ).strip()
        except Exception:
            token = ""
    if not token:
        raise RuntimeError("DATAFLUX_TOKEN or runtime.auth.token is required")
    return ControlClient(args.server, token)


def _print_job_payload(payload, *, json_output: bool) -> None:
    if json_output:
        _write_json(payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def _job_error_exit(exc, *, json_output: bool) -> int:
    from src.control.client import ControlClientError
    from src.jobs import JobNotFoundError

    if isinstance(exc, ControlClientError):
        payload = _error_payload(exc.code or "control_error", str(exc), exc.details)
        if json_output:
            _write_json(payload)
        else:
            print(str(exc), file=sys.stderr)
        if exc.status_code in {404, 409}:
            return EXIT_NOT_FOUND_OR_CONFLICT
        return EXIT_CONNECTION_ERROR
    if isinstance(exc, JobNotFoundError):
        payload = _error_payload("job_not_found", "Job not found")
        if json_output:
            _write_json(payload)
        else:
            print("Job not found", file=sys.stderr)
        return EXIT_NOT_FOUND_OR_CONFLICT
    if isinstance(exc, ValueError):
        if json_output:
            _write_json(_error_payload("job_conflict", str(exc)))
        else:
            print(str(exc), file=sys.stderr)
        return EXIT_NOT_FOUND_OR_CONFLICT
    if json_output:
        _write_json(_error_payload("job_error", str(exc)))
    else:
        print(str(exc), file=sys.stderr)
    return EXIT_RUNTIME_ERROR


def cmd_job_submit(args):
    try:
        if args.server:
            relative_path = args.config
            if Path(relative_path).is_absolute():
                raise ValueError(
                    "remote submit requires a workspace-relative --config path"
                )
            payload = _control_client(args).request_json(
                "POST",
                "/api/v1/jobs",
                body={
                    "root_id": args.root_id,
                    "relative_path": relative_path,
                    "options": {},
                },
            )
        else:
            service = _local_job_service(args.config)
            root_id, relative_path = _workspace_reference(service, args.config)
            payload = service.submit(
                root_id=root_id,
                relative_path=relative_path,
            ).to_dict()
        _print_job_payload(payload, json_output=args.json)
        return EXIT_OK
    except Exception as exc:
        return _job_error_exit(exc, json_output=args.json)


def cmd_job_list(args):
    try:
        if args.server:
            payload = _control_client(args).request_json("GET", "/api/v1/jobs")
        else:
            service = _local_job_service(args.config)
            payload = {
                "jobs": [state.to_dict() for state in service.list_states()],
                "resource": service.resource_status(),
            }
        _print_job_payload(payload, json_output=args.json)
        return EXIT_OK
    except Exception as exc:
        return _job_error_exit(exc, json_output=args.json)


def cmd_job_status(args):
    try:
        if args.server:
            payload = _control_client(args).request_json(
                "GET", f"/api/v1/jobs/{args.job_id}"
            )
        else:
            payload = (
                _local_job_service(args.config)
                .repository.get_state(args.job_id)
                .to_dict()
            )
        _print_job_payload(payload, json_output=args.json)
        return EXIT_OK
    except Exception as exc:
        return _job_error_exit(exc, json_output=args.json)


def _cmd_job_action(args, action: str) -> int:
    try:
        if args.server:
            payload = _control_client(args).request_json(
                "POST", f"/api/v1/jobs/{args.job_id}/{action}", body={}
            )
        else:
            service = _local_job_service(args.config)
            payload = getattr(service, action)(args.job_id).to_dict()
        _print_job_payload(payload, json_output=args.json)
        return EXIT_OK
    except Exception as exc:
        return _job_error_exit(exc, json_output=args.json)


def cmd_job_cancel(args):
    return _cmd_job_action(args, "cancel")


def cmd_job_resume(args):
    return _cmd_job_action(args, "resume")


def _emit_event(event: dict, *, json_output: bool) -> None:
    if json_output:
        _write_json(event)
    else:
        print(json.dumps(event, ensure_ascii=False))


def cmd_job_events(args):
    try:
        if args.server and args.follow:
            for event in _control_client(args).stream_events(
                args.job_id, after_seq=args.after_seq
            ):
                _emit_event(event, json_output=args.json)
            return EXIT_OK
        if args.server:
            payload = _control_client(args).request_json(
                "GET",
                f"/api/v1/jobs/{args.job_id}/events",
                query={"after_seq": args.after_seq, "limit": args.limit},
            )
            for event in payload.get("events", []):
                _emit_event(event, json_output=args.json)
            return EXIT_OK

        service = _local_job_service(args.config)
        cursor = args.after_seq
        while True:
            events, cursor = service.list_events(
                args.job_id,
                after_seq=cursor,
                limit=args.limit,
            )
            for event in events:
                _emit_event(event, json_output=args.json)
            if not args.follow:
                break
            state = service.repository.get_state(args.job_id)
            if (state.is_terminal or state.status.value == "blocked") and not events:
                break
            time.sleep(0.5)
        return EXIT_OK
    except Exception as exc:
        return _job_error_exit(exc, json_output=args.json)


def _parse_age(value: str) -> float:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    text = value.strip().lower()
    multiplier = units.get(text[-1:], 1)
    number = text[:-1] if text[-1:] in units else text
    seconds = float(number) * multiplier
    if seconds <= 0:
        raise ValueError("older-than must be positive")
    return seconds


def cmd_job_prune(args):
    try:
        if args.server:
            raise ValueError("remote prune is not exposed by the Control API")
        service = _local_job_service(args.config)
        preview = service.repository.preview_prune(
            older_than=time.time() - _parse_age(args.older_than)
        )
        result = service.repository.prune(preview, confirm=args.confirm)
        payload = {
            "confirmed": result.confirmed,
            "selected_job_ids": list(result.selected_job_ids),
            "deleted_job_ids": list(result.deleted_job_ids),
            "reclaimed_bytes": result.reclaimed_bytes,
        }
        _print_job_payload(payload, json_output=args.json)
        return EXIT_OK
    except Exception as exc:
        return _job_error_exit(exc, json_output=args.json)


def cmd_gui(args):
    """
    启动 GUI 控制面板子命令

    启动 Web GUI 控制面板，提供配置编辑、进程管理和日志查看功能。

    Args:
        args: argparse 解析后的命令行参数对象
            - port (int): 控制服务器端口，默认 8790
            - no_browser (bool): 是否禁止自动打开浏览器

    Returns:
        int: 退出码，0 表示成功

    使用示例:
        python cli.py gui              # 启动并打开浏览器
        python cli.py gui -p 8080      # 使用自定义端口
        python cli.py gui --no-browser # 不打开浏览器
    """
    try:
        from src.control.server import run_control_server
    except ImportError:
        print("❌ 此版本不包含 GUI 功能，请下载完整版")
        return 1

    from src.config import load_config, require_control_config

    control_config = require_control_config(load_config(args.config))
    port = args.port or control_config.listen.port
    open_browser = not getattr(args, "no_browser", False)

    # 启动 Control Server
    run_control_server(
        host=args.host or control_config.listen.host,
        port=port,
        open_browser=open_browser,
        config_path=args.config,
        supervise_worker=True,
    )
    return 0


def main():
    """
    CLI 主入口函数

    创建命令行参数解析器，注册所有子命令，解析参数并分发到对应的命令处理函数。

    Returns:
        int: 退出码，0 表示成功，1 表示失败

    子命令:
        process     - 运行数据处理
        gateway     - 启动 API 网关
        version     - 显示版本信息
        check       - 检查库安装状态
        token       - 估算 Token 用量

    异常处理:
        - KeyboardInterrupt: 用户中断，返回 1
        - Exception: 打印错误信息和堆栈，返回 1
    """
    # 创建主解析器
    parser = argparse.ArgumentParser(
        prog="ai-dataflux",
        description="AI-DataFlux: High-performance batch AI data processing engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ===== process 子命令：数据处理 =====
    p_process = subparsers.add_parser(
        "process",
        help="Run data processing",
        description="Run data processing with config file (supports rule routing)",
    )
    p_process.add_argument(
        "-c", "--config", default="config.yaml", help="Config file path"
    )
    p_process.add_argument(
        "--validate", action="store_true", help="Only validate config"
    )
    p_process.add_argument(
        "--progress-file",
        help="Progress file path (used by GUI control panel)",
    )
    p_process.add_argument(
        "--json", action="store_true", help="Write one stable JSON result to stdout"
    )
    p_process.set_defaults(func=cmd_process)

    # ===== gateway 子命令：API 网关 =====
    p_gateway = subparsers.add_parser("gateway", help="Start API gateway")
    p_gateway.add_argument(
        "-c", "--config", default="config.yaml", help="Config file path"
    )
    p_gateway.add_argument(
        "--host", default=None, help="Listen address (default: gateway.listen.host)"
    )
    p_gateway.add_argument(
        "-p",
        "--port",
        type=_validate_port,
        default=None,
        help="Listen port (default: gateway.listen.port)",
    )
    p_gateway.add_argument(
        "-w", "--workers", type=int, default=1, help="Worker processes"
    )
    p_gateway.add_argument("--reload", action="store_true", help="Auto reload")
    p_gateway.set_defaults(func=cmd_gateway)

    # ===== worker 子命令：后台 Job Worker =====
    p_worker = subparsers.add_parser("worker", help="Run background Job worker")
    p_worker.add_argument(
        "-c", "--config", default="config.yaml", help="Config file path"
    )
    p_worker.add_argument(
        "--json", action="store_true", help="Write stable JSON status to stdout"
    )
    p_worker.set_defaults(func=cmd_worker)

    # ===== job 子命令组：durable Job lifecycle =====
    p_job = subparsers.add_parser("job", help="Manage durable background Jobs")
    job_subparsers = p_job.add_subparsers(dest="job_command", required=True)

    def add_job_common(parser, *, job_id: bool = False):
        if job_id:
            parser.add_argument("job_id", help="Job UUID")
        parser.add_argument(
            "-c", "--config", default="config.yaml", help="Local config path"
        )
        parser.add_argument(
            "--server",
            help="Remote Control API base URL (for example http://host:8790)",
        )
        parser.add_argument(
            "--json", action="store_true", help="Write stable JSON/JSONL to stdout"
        )

    p_job_submit = job_subparsers.add_parser("submit", help="Submit a Job")
    p_job_submit.add_argument("-c", "--config", required=True, help="Job config path")
    p_job_submit.add_argument("--server", help="Remote Control API base URL")
    p_job_submit.add_argument(
        "--root-id", default="project", help="Remote workspace root id"
    )
    p_job_submit.add_argument("--json", action="store_true")
    p_job_submit.set_defaults(func=cmd_job_submit)

    p_job_list = job_subparsers.add_parser("list", help="List Jobs")
    add_job_common(p_job_list)
    p_job_list.set_defaults(func=cmd_job_list)

    p_job_status = job_subparsers.add_parser("status", help="Show Job state")
    add_job_common(p_job_status, job_id=True)
    p_job_status.set_defaults(func=cmd_job_status)

    p_job_cancel = job_subparsers.add_parser("cancel", help="Cancel a Job")
    add_job_common(p_job_cancel, job_id=True)
    p_job_cancel.set_defaults(func=cmd_job_cancel)

    p_job_resume = job_subparsers.add_parser("resume", help="Resume a Job")
    add_job_common(p_job_resume, job_id=True)
    p_job_resume.set_defaults(func=cmd_job_resume)

    p_job_events = job_subparsers.add_parser("events", help="Read Job events")
    add_job_common(p_job_events, job_id=True)
    p_job_events.add_argument("--after-seq", type=int, default=0)
    p_job_events.add_argument("--limit", type=int, default=100)
    p_job_events.add_argument("--follow", action="store_true")
    p_job_events.set_defaults(func=cmd_job_events)

    p_job_prune = job_subparsers.add_parser(
        "prune", help="Preview or confirm deletion of old terminal Jobs"
    )
    add_job_common(p_job_prune)
    p_job_prune.add_argument(
        "--older-than", required=True, help="Age such as 7d or 12h"
    )
    p_job_prune.add_argument(
        "--confirm", action="store_true", help="Actually delete previewed Jobs"
    )
    p_job_prune.set_defaults(func=cmd_job_prune)

    # ===== config 子命令组 =====
    p_config = subparsers.add_parser("config", help="Validate configuration")
    config_subparsers = p_config.add_subparsers(dest="config_command", required=True)
    p_config_validate = config_subparsers.add_parser(
        "validate", help="Validate canonical v4 config"
    )
    p_config_validate.add_argument(
        "-c", "--config", default="config.yaml", help="Config file path"
    )
    p_config_validate.add_argument("--json", action="store_true")
    p_config_validate.set_defaults(func=cmd_config_validate)

    # ===== version 子命令：版本信息 =====
    p_version = subparsers.add_parser("version", help="Show version info")
    p_version.set_defaults(func=cmd_version)

    # ===== check 子命令：库状态检查 =====
    p_check = subparsers.add_parser("check", help="Check library status")
    p_check.set_defaults(func=cmd_check)

    # ===== token 子命令：Token 估算 =====
    p_token = subparsers.add_parser("token", help="Estimate token usage")
    p_token.add_argument(
        "-c", "--config", default="config.yaml", help="Config file path"
    )
    p_token.add_argument(
        "--mode",
        choices=["in", "out", "io"],
        help="Estimation mode: in (input from input file), out (output from output file), io (both)",
    )
    p_token.set_defaults(func=cmd_token)

    # ===== gui 子命令：Web GUI 控制面板（可选特性，仅完整版包含）=====
    try:
        gui_available = importlib.util.find_spec("src.control.server") is not None
    except (ModuleNotFoundError, ImportError):
        gui_available = False

    if gui_available:
        p_gui = subparsers.add_parser("gui", help="Start GUI control panel")
        p_gui.add_argument(
            "-p",
            "--port",
            type=_validate_port,
            default=None,
            help="Control server port (default: control.listen.port)",
        )
        p_gui.add_argument(
            "-c", "--config", default="config.yaml", help="Config file path"
        )
        p_gui.add_argument(
            "--host",
            default=None,
            help="Control server listen address (default: control.listen.host)",
        )
        p_gui.add_argument(
            "--no-browser", action="store_true", help="Don't open browser automatically"
        )
        p_gui.set_defaults(func=cmd_gui)

    # 解析命令行参数
    args = parser.parse_args()

    # 未指定子命令时显示帮助
    if not args.command:
        parser.print_help()
        return 0

    try:
        # 调用对应的命令处理函数
        return args.func(args)
    except KeyboardInterrupt:
        if not getattr(args, "json", False):
            print("\nInterrupted by user", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        json_output = bool(getattr(args, "json", False))
        try:
            from src.control.client import ControlClientError
        except ImportError:
            ControlClientError = ()  # type: ignore[assignment]
        if isinstance(e, ControlClientError):
            code = EXIT_CONNECTION_ERROR
        elif isinstance(e, (ValueError, FileNotFoundError)):
            code = EXIT_CONFIG_ERROR
        else:
            code = EXIT_RUNTIME_ERROR
        if json_output:
            _write_json(_error_payload("command_failed", str(e)))
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        return code


if __name__ == "__main__":
    sys.exit(main())
