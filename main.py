#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-DataFlux 数据处理入口

运行方式:
    python main.py --config config.yaml
"""

import argparse
import sys

from src.core import UniversalAIProcessor


def main() -> int:
    """主入口"""
    parser = argparse.ArgumentParser(
        description="AI-DataFlux 高性能批量 AI 数据处理引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py --config config.yaml
    python main.py -c my_config.yaml --validate
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="仅验证配置文件，不执行处理",
    )

    args = parser.parse_args()

    try:
        if args.validate:
            # 仅验证配置
            from src.config import load_config

            config = load_config(args.config)
            print(f"✓ 配置文件有效: {args.config}")
            print(
                f"  - 数据源类型: {config.get('datasource', {}).get('type', 'excel')}"
            )
            print(f"  - 输入列: {config.get('columns_to_extract', [])}")
            print(f"  - 输出列: {list(config.get('columns_to_write', {}).values())}")
            return 0

        # 运行处理器
        processor = UniversalAIProcessor(args.config)
        processor.run()
        return 0

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        return 1

    except Exception as e:
        print(f"\n\n❌ 程序执行出错: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
