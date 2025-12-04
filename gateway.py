#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flux API 网关入口

运行方式:
    python gateway.py --config config.yaml --port 8787
"""

import sys

from src.gateway.app import main


if __name__ == "__main__":
    sys.exit(main() or 0)
