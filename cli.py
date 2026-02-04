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

作者: AI-DataFlux Team
版本: 参见 src/__init__.py
"""

import argparse
import sys

try:
    import resource  # Unix-only
except Exception:
    resource = None


def _check_rlimit():
    """检查文件描述符限制"""
    if resource is None or sys.platform not in ("darwin", "linux"):
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Current limits: ({soft}, {hard})")
        if soft <= 256:
            from src.utils.console import console

            print(
                f"{console.warn} File descriptor limit is too low ({soft}). This triggers crashes on macOS."
            )
            print(
                f"{console.tip} Run 'ulimit -n 10240' or higher before running this program."
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
    from src.utils.console import console
    from src.core import UniversalAIProcessor

    if args.validate:
        _check_rlimit()
        from src.config import load_config

        config = load_config(args.config)
        print(f"{console.ok} Config valid: {args.config}")
        print(f"  - Datasource: {config.get('datasource', {}).get('type', 'excel')}")
        print(f"  - Engine: {config.get('datasource', {}).get('engine', 'auto')}")
        print(f"  - Input columns: {config.get('columns_to_extract', [])}")
        print(
            f"  - Output columns: {list(config.get('columns_to_write', {}).values())}"
        )
        # 显示路由配置信息（如果启用）
        routing = config.get("routing", {})
        if routing.get("enabled"):
            subtasks = routing.get("subtasks", [])
            print(
                f"  - Routing: enabled on '{routing.get('field', 'N/A')}' ({len(subtasks)} rules)"
            )
        return 0

    # 获取进度文件路径 (可选)
    progress_file = getattr(args, "progress_file", None)

    # 创建处理器并执行（使用配置文件路径）
    processor = UniversalAIProcessor(args.config, progress_file=progress_file)
    processor.run()
    return 0


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
    from src.gateway.app import run_server

    run_server(
        config_path=args.config,
        host=args.host,
        port=args.port,
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
    from src.control.server import run_control_server

    port = args.port
    open_browser = not getattr(args, "no_browser", False)

    # 启动 Control Server
    run_control_server(host="127.0.0.1", port=port, open_browser=open_browser)
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
    p_process.set_defaults(func=cmd_process)

    # ===== gateway 子命令：API 网关 =====
    p_gateway = subparsers.add_parser("gateway", help="Start API gateway")
    p_gateway.add_argument(
        "-c", "--config", default="config.yaml", help="Config file path"
    )
    p_gateway.add_argument("--host", default="0.0.0.0", help="Listen address")
    p_gateway.add_argument("-p", "--port", type=int, default=8787, help="Listen port")
    p_gateway.add_argument(
        "-w", "--workers", type=int, default=1, help="Worker processes"
    )
    p_gateway.add_argument("--reload", action="store_true", help="Auto reload")
    p_gateway.set_defaults(func=cmd_gateway)

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

    # ===== gui 子命令：Web GUI 控制面板 =====
    p_gui = subparsers.add_parser("gui", help="Start GUI control panel")
    p_gui.add_argument(
        "-p", "--port", type=int, default=8790, help="Control server port"
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
        # 用户按 Ctrl+C 中断
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        # 未处理的异常
        from src.utils.console import console

        print(f"\n{console.error} {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
