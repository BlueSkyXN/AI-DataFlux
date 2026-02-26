#!/usr/bin/env python3
"""
检查可选依赖可用性

CI 脚本：在 GitHub Actions 中运行，验证可选依赖库是否正确安装。
用于 test.yml 工作流的 "Verify dependencies availability" 步骤。

检查流程:
    1. 逐一尝试导入 LIBS 列表中的库
    2. 成功则打印版本号，失败则记录到 missing 列表
    3. 若关键依赖 (CRITICAL) 缺失，输出警告到 stderr
"""
import sys

# 需要检查的可选依赖库列表
LIBS = ["aiohttp", "psutil", "polars", "fastexcel", "xlsxwriter"]
# 关键依赖集合：缺失时会影响核心功能（异步HTTP客户端和系统监控）
CRITICAL = {"aiohttp", "psutil"}

missing = []
for lib in LIBS:
    try:
        # 动态导入库并获取版本号
        mod = __import__(lib)
        version = getattr(mod, "__version__", "N/A")
        print(f"✓ {lib} {version}")
    except ModuleNotFoundError:
        print(f"✗ {lib} (missing)")
        missing.append(lib)

# 检查关键依赖是否缺失（取交集）
if CRITICAL & set(missing):
    print("⚠️  WARNING: Core functionality may be limited", file=sys.stderr)
