"""Lightweight timing log used to compare upper-layer vs backend latency.

Records are written to the file pointed at by ``ASCEND_STORE_PERF_LOG``
(default ``/tmp/ascend_store_perf_<pid>.log``). The log is opened lazily and
guarded by a single lock so writes from multiple worker/sender/recver threads
stay intact. Pass ``ASCEND_STORE_PERF_LOG=`` (empty) to disable.
"""

from __future__ import annotations

import os
import threading
import time
from time import perf_counter
from typing import Any

_DEFAULT_PATH = f"/tmp/ascend_store_perf_{os.getpid()}.log"
_ENV_PATH = "ASCEND_STORE_PERF_LOG"

_lock = threading.Lock()
_fp = None
_enabled: bool | None = None
_resolved_path: str | None = None


def _ensure_open() -> None:
    global _fp, _enabled, _resolved_path
    if _enabled is not None:
        return
    raw = os.environ.get(_ENV_PATH, _DEFAULT_PATH)
    if raw == "":
        _enabled = False
        return
    try:
        _fp = open(raw, "a", buffering=1)  # line-buffered
        _resolved_path = raw
        _enabled = True
    except OSError:
        _enabled = False


def record(layer: str, op: str, num_keys: int, elapsed_ms: float, **extra: Any) -> None:
    """Append one timing record. ``layer`` is "upper" or "lower"."""
    _ensure_open()
    if not _enabled or _fp is None:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + f".{int((time.time() % 1) * 1000):03d}"
    parts = [
        ts,
        f"pid={os.getpid()}",
        f"tid={threading.get_ident()}",
        f"layer={layer}",
        f"op={op}",
        f"num_keys={num_keys}",
        f"elapsed_ms={elapsed_ms:.3f}",
    ]
    for key, value in extra.items():
        parts.append(f"{key}={value}")
    line = " ".join(parts) + "\n"
    with _lock:
        _fp.write(line)


class TimedSection:
    """Context manager that records elapsed wall time for ``op``."""

    __slots__ = ("layer", "op", "num_keys", "extra", "_start")

    def __init__(self, layer: str, op: str, num_keys: int, **extra: Any):
        self.layer = layer
        self.op = op
        self.num_keys = num_keys
        self.extra = extra
        self._start = 0.0

    def __enter__(self) -> "TimedSection":
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = (perf_counter() - self._start) * 1000.0
        extra = dict(self.extra)
        if exc_type is not None:
            extra["exc"] = exc_type.__name__
        record(self.layer, self.op, self.num_keys, elapsed_ms, **extra)
