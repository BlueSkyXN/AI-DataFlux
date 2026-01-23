"""
工具模块

本模块提供 AI-DataFlux 的通用工具函数和类。

模块内容:
    - console: 跨平台控制台输出实例
    - print_status: 状态行打印 (带图标)
    - print_error: 错误消息打印
    - print_tip: 提示消息打印
    - supports_unicode: Unicode 支持检测

使用示例:
    from src.utils import console, print_status
    
    # 打印带图标的状态
    console.print_ok("操作成功")     # ✓ 操作成功
    console.print_error("操作失败")  # ✗ 操作失败
    
    # 打印依赖检查状态
    print_status(True, "pandas")     # ✅ pandas: installed
    print_status(False, "polars")    # ❌ polars: not installed
"""

from .console import (
    console as console,
    print_status as print_status,
    print_error as print_error,
    print_tip as print_tip,
    supports_unicode as supports_unicode,
)
