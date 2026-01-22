"""
Console utilities for cross-platform Unicode support

Automatically detects console capabilities and provides appropriate
output functions that work on Windows, Linux, and macOS.
"""

import sys
import os
import locale


def _safe_print(text: str):
    """
    Safe print that handles encoding errors gracefully.

    On Windows with cp1252, Unicode characters will be replaced with ASCII alternatives.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: replace problematic characters
        ascii_text = text.encode("ascii", errors="replace").decode("ascii")
        print(ascii_text)


def _configure_windows_console():
    """Configure Windows console for better Unicode support."""
    if sys.platform != "win32":
        return

    try:
        # Try to set UTF-8 mode
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    try:
        # Try to set console code page to UTF-8
        import subprocess

        subprocess.run(["chcp", "65001"], shell=True, capture_output=True)
    except Exception:
        pass


def supports_unicode() -> bool:
    """
    Check if the console supports Unicode output.

    Returns:
        True if Unicode is supported, False otherwise
    """
    # Check environment variable override
    if os.environ.get("FORCE_ASCII", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("FORCE_UNICODE", "").lower() in ("1", "true", "yes"):
        return True

    # Check platform
    if sys.platform == "win32":
        # Check if running in CI environment - GitHub Actions Windows uses cp1252
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            # GitHub Actions Windows runner doesn't support Unicode well
            return False

        # Windows: check if using Windows Terminal or compatible console
        # Windows Terminal sets WT_SESSION
        if os.environ.get("WT_SESSION"):
            return True
        # VS Code terminal
        if os.environ.get("TERM_PROGRAM") == "vscode":
            return True
        # ConEmu/Cmder
        if os.environ.get("ConEmuANSI") == "ON":
            return True
        # Check console code page
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            # 65001 is UTF-8 code page
            if kernel32.GetConsoleOutputCP() == 65001:
                return True
        except Exception:
            pass
        # Default: Windows cmd.exe doesn't support Unicode well
        return False

    # Unix-like systems: check if running in CI
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return True

    # Unix-like systems: check encoding
    try:
        encoding = locale.getpreferredencoding(False).lower()
        if "utf" in encoding:
            return True
    except Exception:
        pass

    # Check stdout encoding
    try:
        if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
            if "utf" in sys.stdout.encoding.lower():
                return True
    except Exception:
        pass

    # Default to True for Unix-like systems
    return sys.platform != "win32"


class Console:
    """
    Cross-platform console output with automatic Unicode detection.

    Usage:
        from src.utils.console import console

        console.print_ok("Operation successful")    # ✓ or [OK]
        console.print_error("Something failed")     # ✗ or [ERROR]
    """

    # Unicode symbols
    UNICODE_OK = "✓"
    UNICODE_ERROR = "✗"
    UNICODE_INFO = "ℹ"
    UNICODE_TIP = "💡"
    UNICODE_WARN = "⚠"
    UNICODE_CHECK = "✅"
    UNICODE_CROSS = "❌"

    # ASCII fallbacks
    ASCII_OK = "[OK]"
    ASCII_ERROR = "[ERROR]"
    ASCII_INFO = "[INFO]"
    ASCII_TIP = "[TIP]"
    ASCII_WARN = "[WARN]"
    ASCII_CHECK = "[OK]"
    ASCII_CROSS = "[--]"

    def __init__(self):
        self._unicode = None  # Lazy evaluation

    @property
    def unicode(self) -> bool:
        """Check if Unicode is supported (cached)."""
        if self._unicode is None:
            self._unicode = supports_unicode()
        return self._unicode

    def reset(self):
        """Reset the Unicode detection cache."""
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
        """Print success message."""
        _safe_print(f"{self.ok} {message}")

    def print_error(self, message: str):
        """Print error message."""
        _safe_print(f"{self.error} {message}")

    def print_info(self, message: str):
        """Print info message."""
        _safe_print(f"{self.info} {message}")

    def print_tip(self, message: str):
        """Print tip message."""
        _safe_print(f"{self.tip} {message}")

    def print_warn(self, message: str):
        """Print warning message."""
        _safe_print(f"{self.warn} {message}")


# Global console instance
console = Console()


def print_status(
    available: bool,
    name: str,
    state_true: str = "installed",
    state_false: str = "not installed",
):
    """Print status line with check mark or X."""
    status = console.check if available else console.cross
    state = state_true if available else state_false
    _safe_print(f"{status} {name}: {state}")


def print_error(message: str):
    """Print error message."""
    console.print_error(message)


def print_tip(message: str):
    """Print tip message."""
    console.print_tip(message)
