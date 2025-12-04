#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-DataFlux Unified CLI Entry Point

Usage:
    python cli.py process --config config.yaml     # Run data processing
    python cli.py gateway --port 8787              # Start API gateway
    python cli.py version                          # Show version info
    python cli.py check                            # Check library status
"""

import argparse
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def cmd_process(args):
    """Run data processing"""
    from src.core import UniversalAIProcessor
    
    if args.validate:
        from src.config import load_config
        config = load_config(args.config)
        print(f"[OK] Config valid: {args.config}")
        print(f"  - Datasource: {config.get('datasource', {}).get('type', 'excel')}")
        print(f"  - Engine: {config.get('datasource', {}).get('engine', 'auto')}")
        print(f"  - Input columns: {config.get('columns_to_extract', [])}")
        print(f"  - Output columns: {list(config.get('columns_to_write', {}).values())}")
        return 0
    
    processor = UniversalAIProcessor(args.config)
    processor.run()
    return 0


def cmd_gateway(args):
    """Start API gateway"""
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
    """Show version info"""
    from src import __version__
    print(f"AI-DataFlux v{__version__}")
    return 0


def cmd_check(args):
    """Check library status"""
    from src.data.engines import get_available_libraries
    
    print("AI-DataFlux Library Status\n")
    print("=" * 40)
    
    libs = get_available_libraries()
    for name, available in libs.items():
        status = "[OK]" if available else "[--]"
        state = "installed" if available else "not installed"
        print(f"{status} {name}: {state}")
    
    print("=" * 40)
    
    # Recommend installation
    missing = [name for name, avail in libs.items() if not avail]
    if missing:
        print(f"\nTip: Install high-performance libraries:")
        print(f"   pip install {' '.join(missing)}")
    else:
        print(f"\n[OK] All high-performance libraries installed!")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="ai-dataflux",
        description="AI-DataFlux: High-performance batch AI data processing engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # process subcommand
    p_process = subparsers.add_parser("process", help="Run data processing")
    p_process.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    p_process.add_argument("--validate", action="store_true", help="Only validate config")
    p_process.set_defaults(func=cmd_process)
    
    # gateway subcommand
    p_gateway = subparsers.add_parser("gateway", help="Start API gateway")
    p_gateway.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    p_gateway.add_argument("--host", default="0.0.0.0", help="Listen address")
    p_gateway.add_argument("-p", "--port", type=int, default=8787, help="Listen port")
    p_gateway.add_argument("-w", "--workers", type=int, default=1, help="Worker processes")
    p_gateway.add_argument("--reload", action="store_true", help="Auto reload")
    p_gateway.set_defaults(func=cmd_gateway)
    
    # version subcommand
    p_version = subparsers.add_parser("version", help="Show version info")
    p_version.set_defaults(func=cmd_version)
    
    # check subcommand
    p_check = subparsers.add_parser("check", help="Check library status")
    p_check.set_defaults(func=cmd_check)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
