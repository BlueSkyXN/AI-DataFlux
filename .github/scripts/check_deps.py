#!/usr/bin/env python3
"""检查可选依赖可用性"""
import sys

LIBS = ["aiohttp", "psutil", "polars", "fastexcel", "xlsxwriter"]
CRITICAL = {"aiohttp", "psutil"}

missing = []
for lib in LIBS:
    try:
        mod = __import__(lib)
        version = getattr(mod, "__version__", "N/A")
        print(f"✓ {lib} {version}")
    except ModuleNotFoundError:
        print(f"✗ {lib} (missing)")
        missing.append(lib)

if CRITICAL & set(missing):
    print("⚠️  WARNING: Core functionality may be limited", file=sys.stderr)
