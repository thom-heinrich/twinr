"""Own bounded glibc heap trimming for long-lived Pi MediaPipe gesture loops."""

from __future__ import annotations

from collections.abc import Callable
import ctypes
import threading
from time import monotonic


_MALLOC_TRIM_PAD = 0


def _load_malloc_trim_function() -> Callable[[int], int] | None:
    """Return one callable `malloc_trim(3)` wrapper when glibc exposes it."""

    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return None

    malloc_trim = getattr(libc, "malloc_trim", None)
    if malloc_trim is None:
        return None
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int

    def _call(pad: int, *, _malloc_trim=malloc_trim, _libc=libc) -> int:
        return int(_malloc_trim(pad))

    return _call


class NativeHeapTrimmer:
    """Rate-limit `malloc_trim(0)` so native gesture buffers do not balloon RSS."""

    def __init__(
        self,
        *,
        interval_s: float,
        trim_fn: Callable[[int], int] | None = None,
        monotonic_fn: Callable[[], float] | None = None,
    ) -> None:
        self._interval_s = max(0.0, float(interval_s))
        self._trim_fn = trim_fn if trim_fn is not None else _load_malloc_trim_function()
        self._monotonic = monotonic_fn or monotonic
        self._lock = threading.Lock()
        self._attempt_count = 0
        self._reclaimed_count = 0
        self._skipped_count = 0
        self._last_attempt_mono_s: float | None = None
        self._last_result: int | None = None

    def maybe_trim(self) -> bool:
        """Call `malloc_trim(0)` when the configured interval has elapsed."""

        if self._interval_s <= 0.0 or self._trim_fn is None:
            with self._lock:
                self._skipped_count += 1
            return False

        now = float(self._monotonic())
        with self._lock:
            if (
                self._last_attempt_mono_s is not None
                and (now - self._last_attempt_mono_s) < self._interval_s
            ):
                self._skipped_count += 1
                return False
            self._attempt_count += 1
            self._last_attempt_mono_s = now

        result = int(self._trim_fn(_MALLOC_TRIM_PAD))
        with self._lock:
            self._last_result = result
            if result != 0:
                self._reclaimed_count += 1
        return result != 0

    def snapshot(self) -> dict[str, object]:
        """Return one compact debug snapshot for pipeline/runtime forensics."""

        with self._lock:
            return {
                "native_heap_trim_interval_s": round(self._interval_s, 3),
                "native_heap_trim_supported": bool(self._trim_fn) and self._interval_s > 0.0,
                "native_heap_trim_attempt_count": self._attempt_count,
                "native_heap_trim_reclaimed_count": self._reclaimed_count,
                "native_heap_trim_skipped_count": self._skipped_count,
                "native_heap_trim_last_attempt_mono_s": (
                    None
                    if self._last_attempt_mono_s is None
                    else round(self._last_attempt_mono_s, 3)
                ),
                "native_heap_trim_last_result": self._last_result,
            }
