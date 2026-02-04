"""
跨平台控制台工具

本模块提供跨平台的控制台输出工具，自动检测终端对 Unicode 的
支持能力，并提供合适的输出格式。

设计目标:
    - 在支持 Unicode 的终端显示美观的图标 (✓ ✗ ℹ 💡 ⚠)
    - 在不支持 Unicode 的终端 (如 Windows cmd) 显示 ASCII 替代 ([OK] [ERROR])
    - 处理编码错误，防止程序崩溃

Unicode 检测逻辑:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Unicode 支持检测流程                         │
    ├─────────────────────────────────────────────────────────────────┤
    │ 1. 检查环境变量 FORCE_ASCII/FORCE_UNICODE (用户覆盖)            │
    │ 2. Windows 平台:                                                 │
    │    - CI 环境 (GitHub Actions): 不支持                           │
    │    - Windows Terminal (WT_SESSION): 支持                        │
    │    - VS Code 终端: 支持                                         │
    │    - ConEmu/Cmder: 支持                                         │
    │    - 代码页 65001 (UTF-8): 支持                                 │
    │    - 默认 cmd.exe: 不支持                                       │
    │ 3. Unix 平台:                                                    │
    │    - 检查 locale 编码                                           │
    │    - 检查 stdout 编码                                           │
    │    - 默认支持                                                    │
    └─────────────────────────────────────────────────────────────────┘

符号映射:
    ┌──────────────┬────────────┬────────────┐
    │ 语义          │ Unicode    │ ASCII      │
    ├──────────────┼────────────┼────────────┤
    │ 成功          │ ✓          │ [OK]       │
    │ 错误          │ ✗          │ [ERROR]    │
    │ 信息          │ ℹ          │ [INFO]     │
    │ 提示          │ 💡         │ [TIP]      │
    │ 警告          │ ⚠          │ [WARN]     │
    │ 复选 (选中)   │ ✅         │ [OK]       │
    │ 复选 (未选)   │ ❌         │ [--]       │
    └──────────────┴────────────┴────────────┘

使用示例:
    from src.utils.console import console, print_status

    # 使用全局 console 实例
    console.print_ok("操作成功")
    console.print_error("操作失败")
    console.print_info("正在处理...")

    # 获取符号
    print(f"{console.check} 测试通过")

    # 打印状态行
    print_status(True, "pandas", "已安装", "未安装")
"""

import sys
import os
import locale


def _safe_print(text: str):
    """
    安全打印，优雅处理编码错误

    在 Windows cp1252 环境下，Unicode 字符会被替换为 ASCII 替代符。

    Args:
        text: 要打印的文本
    """
    try:
        print(text)
    except UnicodeEncodeError:
        # 回退: 将无法编码的字符替换为 ?
        ascii_text = text.encode("ascii", errors="replace").decode("ascii")
        print(ascii_text)


def _configure_windows_console():
    """
    配置 Windows 控制台以获得更好的 Unicode 支持

    尝试:
    1. 将 stdout/stderr 重新配置为 UTF-8
    2. 设置控制台代码页为 65001 (UTF-8)
    """
    if sys.platform != "win32":
        return

    try:
        # 尝试设置 UTF-8 模式
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    try:
        # 尝试设置控制台代码页为 UTF-8
        import subprocess

        subprocess.run(["chcp", "65001"], shell=True, capture_output=True)
    except Exception:
        pass


def supports_unicode() -> bool:
    """
    检测控制台是否支持 Unicode 输出

    Returns:
        True 如果支持 Unicode，否则 False

    环境变量覆盖:
        - FORCE_ASCII=1: 强制使用 ASCII
        - FORCE_UNICODE=1: 强制使用 Unicode
    """
    # 检查环境变量覆盖
    if os.environ.get("FORCE_ASCII", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("FORCE_UNICODE", "").lower() in ("1", "true", "yes"):
        return True

    # Windows 平台检测
    if sys.platform == "win32":
        # CI 环境 - GitHub Actions Windows 使用 cp1252
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            return False

        # Windows Terminal (设置 WT_SESSION 环境变量)
        if os.environ.get("WT_SESSION"):
            return True
        # VS Code 终端
        if os.environ.get("TERM_PROGRAM") == "vscode":
            return True
        # ConEmu/Cmder
        if os.environ.get("ConEmuANSI") == "ON":
            return True
        # 检查控制台代码页
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            # 65001 是 UTF-8 代码页
            if kernel32.GetConsoleOutputCP() == 65001:
                return True
        except Exception:
            pass
        # 默认: Windows cmd.exe 不能很好地支持 Unicode
        return False

    # Unix 平台: 检查是否在 CI 环境
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return True

    # Unix 平台: 检查编码
    try:
        encoding = locale.getpreferredencoding(False).lower()
        if "utf" in encoding:
            return True
    except Exception:
        pass

    # 检查 stdout 编码
    try:
        if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
            if "utf" in sys.stdout.encoding.lower():
                return True
    except Exception:
        pass

    # Unix 平台默认支持
    return sys.platform != "win32"


class Console:
    """
    跨平台控制台输出类

    自动检测 Unicode 支持并提供合适的输出方法。
    使用惰性求值缓存检测结果。

    Attributes:
        ok: 成功符号 (✓ 或 [OK])
        error: 错误符号 (✗ 或 [ERROR])
        info: 信息符号 (ℹ 或 [INFO])
        tip: 提示符号 (💡 或 [TIP])
        warn: 警告符号 (⚠ 或 [WARN])
        check: 复选选中符号 (✅ 或 [OK])
        cross: 复选未选符号 (❌ 或 [--])

    Usage:
        from src.utils.console import console

        console.print_ok("Operation successful")    # ✓ or [OK]
        console.print_error("Something failed")     # ✗ or [ERROR]
    """

    # Unicode 符号
    UNICODE_OK = "✓"
    UNICODE_ERROR = "✗"
    UNICODE_INFO = "ℹ"
    UNICODE_TIP = "💡"
    UNICODE_WARN = "⚠"
    UNICODE_CHECK = "✅"
    UNICODE_CROSS = "❌"

    # ASCII 回退符号
    ASCII_OK = "[OK]"
    ASCII_ERROR = "[ERROR]"
    ASCII_INFO = "[INFO]"
    ASCII_TIP = "[TIP]"
    ASCII_WARN = "[WARN]"
    ASCII_CHECK = "[OK]"
    ASCII_CROSS = "[--]"

    def __init__(self):
        self._unicode = None  # 惰性求值

    @property
    def unicode(self) -> bool:
        """检查是否支持 Unicode (带缓存)"""
        if self._unicode is None:
            self._unicode = supports_unicode()
        return self._unicode

    def reset(self):
        """重置 Unicode 检测缓存"""
        self._unicode = None

    @property
    def ok(self) -> str:
        return self.UNICODE_OK if self.unicode else self.ASCII_OK

    @property
    def error(self) -> str:
        return self.UNICODE_ERROR if self.unicode else self.ASCII_ERROR

    @property
    def info(self) -> str:
        return self.UNICODE_INFO if self.unicode else self.ASCII_INFO

    @property
    def tip(self) -> str:
        return self.UNICODE_TIP if self.unicode else self.ASCII_TIP

    @property
    def warn(self) -> str:
        return self.UNICODE_WARN if self.unicode else self.ASCII_WARN

    @property
    def check(self) -> str:
        return self.UNICODE_CHECK if self.unicode else self.ASCII_CHECK

    @property
    def cross(self) -> str:
        return self.UNICODE_CROSS if self.unicode else self.ASCII_CROSS

    def print_ok(self, message: str):
        """打印成功消息"""
        _safe_print(f"{self.ok} {message}")

    def print_error(self, message: str):
        """打印错误消息"""
        _safe_print(f"{self.error} {message}")

    def print_info(self, message: str):
        """打印信息消息"""
        _safe_print(f"{self.info} {message}")

    def print_tip(self, message: str):
        """打印提示消息"""
        _safe_print(f"{self.tip} {message}")

    def print_warn(self, message: str):
        """打印警告消息"""
        _safe_print(f"{self.warn} {message}")


# 全局 console 实例
console = Console()


def print_status(
    available: bool,
    name: str,
    state_true: str = "installed",
    state_false: str = "not installed",
):
    """
    打印状态行，带复选图标

    Args:
        available: 状态是否为真
        name: 项目名称
        state_true: 状态为真时的文本
        state_false: 状态为假时的文本

    Example:
        print_status(True, "pandas")    # ✅ pandas: installed
        print_status(False, "polars")   # ❌ polars: not installed
    """
    status = console.check if available else console.cross
    state = state_true if available else state_false
    _safe_print(f"{status} {name}: {state}")


def print_error(message: str):
    """打印错误消息 (快捷函数)"""
    console.print_error(message)


def print_tip(message: str):
    """打印提示消息 (快捷函数)"""
    console.print_tip(message)
