"""单 Gateway 进程内的 Responses 后端亲和表；丢失映射时必须拒绝续接。"""

from collections import OrderedDict
import time
from typing import Callable


class ResponseAffinity:
    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int,
        *,
        clock: Callable[[], float] = time.monotonic
    ):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self._entries: OrderedDict[str, tuple[str | None, float]] = OrderedDict()

    def _expire(self) -> None:
        now = self.clock()
        while self._entries:
            key, (_, expires) = next(iter(self._entries.items()))
            if expires > now:
                break
            self._entries.pop(key)

    def get(self, response_id: str) -> str | None:
        self._expire()
        entry = self._entries.get(response_id)
        return entry[0] if entry else None

    def remember(self, response_id: str, route_id: str) -> None:
        self._expire()
        current = self._entries.get(response_id)
        if current is not None:
            if current[0] != route_id:
                self._entries[response_id] = (None, current[1])
                raise ValueError("response id belongs to multiple routes")
            return
        self._entries[response_id] = (route_id, self.clock() + self.ttl_seconds)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
