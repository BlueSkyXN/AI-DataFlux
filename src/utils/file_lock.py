"""跨进程本地文件锁；句柄关闭后由内核释放，锁文件不删除。"""

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import sys
import time
from typing import Iterator


@contextmanager
def exclusive_file_lock(
    path: Path,
    *,
    timeout_seconds: float = 5.0,
) -> Iterator[None]:
    """由操作系统释放的排他锁，不按文件年龄抢占或删除锁文件。"""

    deadline = time.monotonic() + timeout_seconds
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if sys.platform == "win32":
            import msvcrt

            def acquire() -> None:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)

            def release() -> None:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)

        else:
            import fcntl

            def acquire() -> None:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release() -> None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)

        while True:
            try:
                acquire()
                break
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out acquiring file lock: {path}"
                    ) from error
                time.sleep(0.01)
        try:
            yield
        finally:
            release()
    finally:
        os.close(descriptor)
