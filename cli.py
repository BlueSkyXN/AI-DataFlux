#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-DataFlux Unified CLI Entry Point

Usage:
    python cli.py process --config config.yaml     # Run data processing
    python cli.py gateway --port 8787              # Start API gateway
    python cli.py token --config config.yaml       # Estimate token usage
    python cli.py version                          # Show version info
    python cli.py check                            # Check library status
"""

import argparse
import sys


def cmd_process(args):
    """Run data processing"""
    from src.utils.console import console
    from src.core import UniversalAIProcessor
    
    if args.validate:
        from src.config import load_config
        config = load_config(args.config)
        print(f"{console.ok} Config valid: {args.config}")
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
    from src.utils.console import console, print_status
    
    print("AI-DataFlux Library Status\n")
    print("=" * 40)
    
    libs = get_available_libraries()
    for name, available in libs.items():
        print_status(available, name)
    
    print("=" * 40)
    
    # Recommend installation
    missing = [name for name, avail in libs.items() if not avail]
    if missing:
        print(f"\n{console.tip} Install high-performance libraries:")
        print(f"   pip install {' '.join(missing)}")
    else:
        print(f"\n{console.ok} All high-performance libraries installed!")
    
    return 0


def cmd_token(args):
    """Estimate token usage"""
    from src.utils.console import console
    from src.core.token_estimator import run_token_estimation
    
    mode = args.mode if hasattr(args, 'mode') and args.mode else None
    
    try:
        result = run_token_estimation(args.config, mode)
    except ImportError as e:
        print(f"{console.error} {e}")
        print(f"{console.tip} Install tiktoken: pip install tiktoken")
        return 1
    except Exception as e:
        print(f"{console.error} Token estimation failed: {e}")
        return 1
    
    # Check for errors
    if result.get("error"):
        print(f"{console.error} {result.get('message', result.get('error'))}")
        return 1
    
    # Print results
    print("\n" + "=" * 50)
    print("  Token Estimation Results")
    print("=" * 50)
    
    mode_display = result.get('mode', 'in')
    mode_desc = {"in": "input only", "out": "output only", "io": "input + output"}
    print(f"\n{console.info} Mode: {mode_display} ({mode_desc.get(mode_display, '')})")
    print(f"{console.info} Total rows: {result.get('total_rows', 0)}")
    print(f"{console.info} Sampled rows: {result.get('sampled_rows', 0)}")
    print(f"{console.info} Estimated requests: {result.get('request_count', 0)}")
    
    # Input tokens (in/io modes)
    input_stats = result.get("input_tokens", {})
    if input_stats and "error" not in input_stats:
        print(f"\n{console.ok} Input Token Estimation:")
        print(f"   Total (estimated): {input_stats.get('total_estimated', 0):,}")
        print(f"   Per-request avg:   {input_stats.get('avg', 0):.1f}")
        print(f"   Per-request min:   {input_stats.get('min', 0)}")
        print(f"   Per-request max:   {input_stats.get('max', 0)}")
        print(f"   P50: {input_stats.get('p50', 0)} | P90: {input_stats.get('p90', 0)} | P99: {input_stats.get('p99', 0)}")
    elif mode_display in ('in', 'io') and (not input_stats or "error" in input_stats):
        print(f"\n{console.warn} Input estimation unavailable (no unprocessed rows found)")
    
    # Output tokens (out/io modes)
    output_stats = result.get("output_tokens", {})
    if output_stats and "error" not in output_stats:
        print(f"\n{console.ok} Output Token Estimation:")
        print(f"   Total (estimated): {output_stats.get('total_estimated', 0):,}")
        print(f"   Per-response avg:  {output_stats.get('avg', 0):.1f}")
        print(f"   Per-response min:  {output_stats.get('min', 0)}")
        print(f"   Per-response max:  {output_stats.get('max', 0)}")
        print(f"   P50: {output_stats.get('p50', 0)} | P90: {output_stats.get('p90', 0)} | P99: {output_stats.get('p99', 0)}")
    elif mode_display in ('out', 'io') and (not output_stats or "error" in output_stats):
        print(f"\n{console.warn} Output estimation unavailable (no processed rows found)")
    
    print("\n" + "=" * 50)
    
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
    
    # token subcommand
    p_token = subparsers.add_parser("token", help="Estimate token usage")
    p_token.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    p_token.add_argument(
        "--mode", 
        choices=["in", "out", "io"], 
        help="Estimation mode: in (input from input file), out (output from output file), io (both)"
    )
    p_token.set_defaults(func=cmd_token)
    
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
        from src.utils.console import console
        print(f"\n{console.error} {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
