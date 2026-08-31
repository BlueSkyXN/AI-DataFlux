"""
配置文件 API 模块

提供配置文件的读写接口，保留原始 YAML 格式和注释。

功能:
    - 读取配置文件 (raw text)
    - 写入配置文件 (原子写入 + 单份备份)
    - 路径校验 (防止路径穿越)

安全:
    - 使用 os.path.realpath 解析路径
    - 校验路径必须在项目目录内
    - 原子写入避免部分写入

函数清单:
    _validate_path(path) -> str
        路径安全校验，防止路径穿越攻击
        输入: path (str) - 相对于项目根目录的路径
        输出: str - 解析后的绝对路径
        异常: HTTPException(403) - 路径在项目目录外

    _validate_read_target(path, real_path) -> None
        读取目标类型校验，仅允许 YAML 文件
        输入: path (str) - 原始路径, real_path (str) - 绝对路径
        异常: HTTPException(403) - 非 YAML 文件

    _validate_write_target(path, real_path) -> None
        写入目标类型校验，仅允许 YAML 文件
        输入: path (str) - 原始路径, real_path (str) - 绝对路径
        异常: HTTPException(403) - 非 YAML 文件

    read_config(path) -> str
        读取配置文件内容 (raw text)
        输入: path (str) - 相对路径
        输出: str - 文件内容
        异常: HTTPException(403/404)

    write_config(path, content) -> dict
        写入配置文件（自动备份 + 原子写入）
        输入: path (str) - 相对路径, content (str) - 文件内容
        输出: dict - {"success", "path", "backed_up"}
        异常: HTTPException(403/500)

    get_project_root() -> str
        获取项目根目录路径

关键变量:
    PROJECT_ROOT: str - 项目根目录绝对路径
    ALLOWED_CONFIG_EXTENSIONS: set - 允许操作的文件扩展名 {".yaml", ".yml"}

模块依赖:
    - os, shutil: 文件系统操作
    - fastapi.HTTPException: HTTP 错误响应
    - .runtime: 项目根目录解析
"""

import hashlib
import os
import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import HTTPException

from . import runtime

# 项目根目录：
# - 源码：仓库根
# - 打包：优先 cwd（看起来像项目根），否则可执行文件目录
# - 可通过环境变量覆盖
PROJECT_ROOT = str(runtime.get_project_root())
ALLOWED_CONFIG_EXTENSIONS = {".yaml", ".yml"}
_CONFIG_WRITE_LOCK = threading.Lock()


class ConfigRevisionConflictError(RuntimeError):
    """Raised when If-Match no longer matches the file on disk."""


def _validate_yaml_path(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_CONFIG_EXTENSIONS:
        raise PermissionError("only YAML config files are allowed")


def config_revision(path: str | Path) -> str:
    """Return the SHA-256 revision of the exact config bytes."""

    target = Path(path)
    _validate_yaml_path(target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_workspace_config(path: str | Path) -> tuple[str, str]:
    """Read one already-contained YAML file and return content plus revision."""

    target = Path(path)
    _validate_yaml_path(target)
    if not target.is_file():
        raise FileNotFoundError(str(target))
    encoded = target.read_bytes()
    content = encoded.decode("utf-8")
    return content, hashlib.sha256(encoded).hexdigest()


def write_workspace_config(
    path: str | Path,
    content: str,
    *,
    expected_revision: str,
) -> str:
    """Atomically replace one config after an optimistic revision check."""

    target = Path(path)
    _validate_yaml_path(target)
    if not expected_revision:
        raise ConfigRevisionConflictError("If-Match is required")
    target.parent.mkdir(parents=True, exist_ok=True)
    with _CONFIG_WRITE_LOCK:
        current_revision = config_revision(target) if target.is_file() else "missing"
        normalized = expected_revision.strip().strip('"')
        if normalized not in {current_revision, "*"}:
            raise ConfigRevisionConflictError(
                f"expected revision {normalized}, found {current_revision}"
            )

        encoded = content.encode("utf-8")
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, target)
            try:
                directory_fd = os.open(target.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temp_path.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _validate_path(path: str) -> str:
    """
    校验路径在项目目录内，返回绝对路径

    Args:
        path: 相对于项目根目录的路径

    Returns:
        str: 解析后的绝对路径

    Raises:
        HTTPException: 403 - 路径在项目目录外
    """
    # 解析符号链接和 .. 等
    real = os.path.realpath(os.path.join(PROJECT_ROOT, path))

    # 使用 commonpath 检查是否在项目目录内 (跨平台安全)
    try:
        common = os.path.commonpath([PROJECT_ROOT, real])
        if common != PROJECT_ROOT:
            raise HTTPException(403, "Path outside project directory")
    except ValueError:
        # Windows 上不同驱动器会抛出 ValueError
        raise HTTPException(403, "Path outside project directory")

    return real


def read_config(path: str) -> str:
    """
    读取配置文件内容

    Args:
        path: 相对于项目根目录的文件路径

    Returns:
        str: 文件内容 (raw text)

    Raises:
        HTTPException: 403 - 路径在项目目录外
        HTTPException: 404 - 文件不存在
    """
    real_path = _validate_path(path)
    _validate_read_target(path, real_path)

    if not os.path.isfile(real_path):
        raise HTTPException(404, f"File not found: {path}")

    with open(real_path, "r", encoding="utf-8") as f:
        return f.read()


def _validate_write_target(path: str, real_path: str) -> None:
    """
    校验写入目标是否属于允许的配置文件类型

    仅允许写入 YAML 配置文件，降低“任意文件改写”风险。
    """
    ext = os.path.splitext(real_path)[1].lower()
    if ext not in ALLOWED_CONFIG_EXTENSIONS:
        raise HTTPException(
            403,
            f"Write blocked: only YAML config files are allowed ({path})",
        )


def _validate_read_target(path: str, real_path: str) -> None:
    """
    校验读取目标是否属于允许的配置文件类型

    仅允许读取 YAML 配置文件，降低非必要文件暴露面。
    """
    ext = os.path.splitext(real_path)[1].lower()
    if ext not in ALLOWED_CONFIG_EXTENSIONS:
        raise HTTPException(
            403,
            f"Read blocked: only YAML config files are allowed ({path})",
        )


def write_config(path: str, content: str) -> dict:
    """
    写入配置文件

    写入前自动备份旧内容到 {path}.bak (仅保留最近一份)

    Args:
        path: 相对于项目根目录的文件路径
        content: 要写入的内容

    Returns:
        dict: {"success": True, "path": 实际路径, "backed_up": 是否创建了备份}

    Raises:
        HTTPException: 403 - 路径在项目目录外
    """
    real_path = _validate_path(path)
    _validate_write_target(path, real_path)
    backed_up = False

    # 如果文件已存在，创建备份
    if os.path.isfile(real_path):
        backup_path = real_path + ".bak"
        shutil.copy2(real_path, backup_path)
        backed_up = True

    # 原子写入: 先写临时文件，再 rename
    tmp_path = real_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, real_path)  # 原子操作
    except Exception as e:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(500, f"Write failed: {e}")

    return {
        "success": True,
        "path": real_path,
        "backed_up": backed_up,
    }


def get_project_root() -> str:
    """获取项目根目录"""
    return PROJECT_ROOT
