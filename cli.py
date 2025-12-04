#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-DataFlux 统一命令行入口

用法:
    python cli.py process --config config.yaml     # 运行数据处理
    python cli.py gateway --port 8787              # 启动 API 网关
    python cli.py version                          # 显示版本信息
    python cli.py check                            # 检查依赖库状态
"""

import argparse
import sys


def cmd_process(args):
    """运行数据处理"""
    from src.core import UniversalAIProcessor
    
    if args.validate:
        from src.config import load_config
        config = load_config(args.config)
        print(f"✓ 配置文件有效: {args.config}")
        print(f"  - 数据源类型: {config.get('datasource', {}).get('type', 'excel')}")
        print(f"  - 引擎: {config.get('datasource', {}).get('engine', 'auto')}")
        print(f"  - 输入列: {config.get('columns_to_extract', [])}")
        print(f"  - 输出列: {list(config.get('columns_to_write', {}).values())}")
        return 0
    
    processor = UniversalAIProcessor(args.config)
    processor.run()
    return 0


def cmd_gateway(args):
    """启动 API 网关"""
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
    """显示版本信息"""
    from src import __version__
    print(f"AI-DataFlux v{__version__}")
    return 0


def cmd_check(args):
    """检查依赖库状态"""
    from src.data.engines import get_available_libraries
    
    print("AI-DataFlux 依赖库状态\n")
    print("=" * 40)
    
    libs = get_available_libraries()
    for name, available in libs.items():
        status = "✅" if available else "❌"
        print(f"{status} {name}: {'可用' if available else '未安装'}")
    
    print("=" * 40)
    
    # 推荐安装
    missing = [name for name, avail in libs.items() if not avail]
    if missing:
        print(f"\n💡 推荐安装高性能库:")
        print(f"   pip install {' '.join(missing)}")
    else:
        print(f"\n✅ 所有高性能库已安装，性能最优！")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="ai-dataflux",
        description="AI-DataFlux: 高性能批量 AI 数据处理引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # process 子命令
    p_process = subparsers.add_parser("process", help="运行数据处理")
    p_process.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    p_process.add_argument("--validate", action="store_true", help="仅验证配置")
    p_process.set_defaults(func=cmd_process)
    
    # gateway 子命令
    p_gateway = subparsers.add_parser("gateway", help="启动 API 网关")
    p_gateway.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    p_gateway.add_argument("--host", default="0.0.0.0", help="监听地址")
    p_gateway.add_argument("-p", "--port", type=int, default=8787, help="监听端口")
    p_gateway.add_argument("-w", "--workers", type=int, default=1, help="工作进程数")
    p_gateway.add_argument("--reload", action="store_true", help="自动重载")
    p_gateway.set_defaults(func=cmd_gateway)
    
    # version 子命令
    p_version = subparsers.add_parser("version", help="显示版本信息")
    p_version.set_defaults(func=cmd_version)
    
    # check 子命令
    p_check = subparsers.add_parser("check", help="检查依赖库状态")
    p_check.set_defaults(func=cmd_check)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
