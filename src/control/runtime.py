"""
运行时环境工具

用于区分源码运行 vs. 打包运行（PyInstaller / Nuitka），并提供：
  - project root 解析
  - 静态资源路径查找
  - 子进程启动命令构造

函数清单:
    _looks_like_project_root(path) -> bool
        判断给定目录是否"看起来像项目根目录"
        输入: path (Path) - 待检测目录
        输出: bool - 目录中是否存在配置文件标记 (config.yaml 等)

    is_packaged() -> bool
        检测当前是否运行在打包环境 (PyInstaller / Nuitka)
        输出: bool - sys.frozen 或 sys._MEIPASS 是否存在

    get_repo_root() -> Path
        获取源码模式下的仓库根目录（基于本文件位置向上两级）
        输出: Path - 仓库根目录绝对路径

    get_project_root() -> Path
        获取项目根目录（语义化，区分源码/打包环境）
        输出: Path - 项目根目录
        环境变量覆盖: DATAFLUX_PROJECT_ROOT, AI_DATAFLUX_PROJECT_ROOT

    get_embedded_root() -> Optional[Path]
        获取打包环境下的资源根目录
        输出: Optional[Path] - PyInstaller _MEIPASS 或可执行文件目录

    find_web_dist_dir(project_root) -> Optional[Path]
        查找 web/dist 前端构建产物目录
        输入: project_root (Optional[Path]) - 项目根目录（可选）
        输出: Optional[Path] - 第一个存在的 web/dist 目录，或 None

模块依赖:
    - os, sys, pathlib: 路径与环境检测
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _looks_like_project_root(path: Path) -> bool:
    """
    判断给定目录是否"看起来像项目根目录"

    通过检测目录下是否存在配置文件标记来判断。
    用于打包环境下优先使用 cwd 还是可执行文件目录的决策。

    Args:
        path: 待检测目录

    Returns:
        bool: 目录中存在任一配置文件标记则返回 True
    """
    markers = (
        "config.yaml",
        "config.yml",
        "config-example.yaml",
        "config-example.yml",
    )
    return any((path / name).is_file() for name in markers)


def is_packaged() -> bool:
    """
    是否处于打包后的可执行文件环境。

    - PyInstaller: sys.frozen == True 且通常存在 sys._MEIPASS
    - Nuitka: 通常也会设置 sys.frozen（兼容 PyInstaller 行为）
    """
    return bool(getattr(sys, "frozen", False)) or bool(getattr(sys, "_MEIPASS", None))


def get_repo_root() -> Path:
    """源码模式下的仓库根目录 (src/control/* -> repo root)"""
    return Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """
    获取“项目根目录”的语义：

    - 源码运行：仓库根目录
    - 打包运行：优先使用当前工作目录（当其“看起来像项目根”时），否则使用可执行文件所在目录
      允许通过环境变量覆盖：
        - DATAFLUX_PROJECT_ROOT
        - AI_DATAFLUX_PROJECT_ROOT
    """
    override = os.environ.get("DATAFLUX_PROJECT_ROOT") or os.environ.get(
        "AI_DATAFLUX_PROJECT_ROOT"
    )
    if override:
        return Path(override).expanduser().resolve()

    if is_packaged():
        cwd_root = Path(os.getcwd()).resolve()
        try:
            exe_root = Path(sys.executable).resolve().parent
        except Exception:
            exe_root = cwd_root

        # 终端启动通常在项目目录下运行：若 cwd 看起来像项目根则优先使用 cwd
        if _looks_like_project_root(cwd_root):
            return cwd_root

        # 双击/非预期 cwd：优先使用可执行文件目录（默认白名单更小、更可控）
        return exe_root.resolve()
    return get_repo_root()


def get_embedded_root() -> Optional[Path]:
    """
    获取打包环境下的资源根目录。

    - PyInstaller onefile: sys._MEIPASS
    - 其他场景：优先尝试可执行文件所在目录
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass).resolve()

    try:
        return Path(sys.executable).resolve().parent
    except Exception:
        return None


def find_web_dist_dir(project_root: Optional[Path] = None) -> Optional[Path]:
    """
    查找 web/dist 目录（优先顺序）：
      1) 源码仓库根目录下的 web/dist
      2) project_root 下的 web/dist（允许用户手动构建后运行）
      3) 打包资源目录（PyInstaller/Nuitka）中的 web/dist
      4) 可执行文件目录下的 web/dist（standalone）
    """
    repo_root = get_repo_root()
    project_root = project_root or get_project_root()
    embedded_root = get_embedded_root()

    candidates: list[Path] = [
        repo_root / "web" / "dist",
        project_root / "web" / "dist",
    ]
    if embedded_root:
        candidates.append(embedded_root / "web" / "dist")
    try:
        candidates.append(Path(sys.executable).resolve().parent / "web" / "dist")
    except Exception:
        pass

    for path in candidates:
        if path.is_dir():
            return path
    return None
