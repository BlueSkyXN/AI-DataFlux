#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-DataFlux 数据处理引擎主入口模块

本模块是 AI-DataFlux 数据处理引擎的直接入口点，提供简洁的命令行接口来启动
批量 AI 数据处理任务。它是 cli.py 中 process 子命令的精简替代方案。

核心功能:
    - 加载并验证配置文件
    - 初始化并运行 UniversalAIProcessor 处理器
    - 支持仅验证模式（--validate）用于配置检查

架构位置:
    本模块位于应用程序入口层，直接调用核心处理器 (src.core.UniversalAIProcessor)。
    处理器负责协调数据源读取、AI API 调用、结果写回的完整流程。

运行方式:
    # 使用默认配置文件 config.yaml
    python main.py

    # 指定自定义配置文件
    python main.py --config my_config.yaml
    python main.py -c my_config.yaml

    # 仅验证配置文件有效性
    python main.py --config config.yaml --validate

退出码:
    0 - 执行成功
    1 - 用户中断或发生错误

依赖模块:
    - src.core.UniversalAIProcessor: 核心处理器
    - src.config.load_config: 配置加载器（仅验证模式）

作者: AI-DataFlux Team
版本: 参见 src/__init__.py
"""

import argparse
import sys

from src.core import UniversalAIProcessor


def main() -> int:
    """
    数据处理引擎主入口函数

    解析命令行参数，根据用户选择执行配置验证或完整的数据处理流程。

    工作流程:
        1. 解析命令行参数（配置文件路径、验证模式）
        2. 如果是验证模式，仅加载并显示配置信息
        3. 否则创建处理器实例并运行完整处理流程
        4. 捕获并处理用户中断和异常

    Returns:
        int: 退出码，0 表示成功，1 表示失败

    Raises:
        不直接抛出异常，所有异常在内部捕获并转换为退出码
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="AI-DataFlux 高性能批量 AI 数据处理引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py --config config.yaml
    python main.py -c my_config.yaml --validate
        """,
    )

    # 配置文件路径参数
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )

    # 验证模式参数：仅检查配置有效性
    parser.add_argument(
        "--validate",
        action="store_true",
        help="仅验证配置文件，不执行处理",
    )

    args = parser.parse_args()

    try:
        if args.validate:
            # 验证模式：仅加载配置并显示关键信息
            from src.config import load_config

            config = load_config(args.config)
            print(f"✓ 配置文件有效: {args.config}")
            print(
                f"  - 数据源类型: {config.get('datasource', {}).get('type', 'excel')}"
            )
            print(f"  - 输入列: {config.get('columns_to_extract', [])}")
            print(f"  - 输出列: {list(config.get('columns_to_write', {}).values())}")
            return 0

        # 生产模式：创建处理器并执行完整处理流程
        processor = UniversalAIProcessor(args.config)
        processor.run()
        return 0

    except KeyboardInterrupt:
        # 用户按 Ctrl+C 中断程序
        print("\n\n程序被用户中断")
        return 1

    except Exception as e:
        # 捕获所有未处理异常，打印详细堆栈信息
        print(f"\n\n❌ 程序执行出错: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
