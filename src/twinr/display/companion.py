from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import math
from threading import Event, Thread
from typing import Protocol

from twinr.agent.base_agent.config import TwinrConfig


class DisplayLoopLike(Protocol):
    sleep: Callable[[float], object]

    def run(self, *, duration_s: float | None = None) -> int: ...


EmitFn = Callable[[str], None]
LockFactoryFn = Callable[[TwinrConfig, str], object]
LockOwnerFn = Callable[[TwinrConfig, str], int | None]
LoopFactoryFn = Callable[[TwinrConfig], DisplayLoopLike]


def _default_emit(line: str) -> None:
    print(line, flush=True)


def _default_lock_factory(config: TwinrConfig, loop_name: str) -> object:
    from twinr.ops import loop_instance_lock

    return loop_instance_lock(config, loop_name)


def _default_lock_owner(config: TwinrConfig, loop_name: str) -> int | None:
    from twinr.ops import loop_lock_owner

    return loop_lock_owner(config, loop_name)


def _default_loop_factory(config: TwinrConfig) -> DisplayLoopLike:
    from twinr.display.service import TwinrStatusDisplayLoop

    return TwinrStatusDisplayLoop.from_config(config)


def _join_timeout_seconds(config: TwinrConfig) -> float:
    interval_s = float(getattr(config, "display_poll_interval_s", 0.5) or 0.5)
    if not math.isfinite(interval_s) or interval_s <= 0:
        interval_s = 0.5
    return max(1.0, min(interval_s * 4.0, 5.0))


@contextmanager
def optional_display_companion(
    config: TwinrConfig,
    *,
    enabled: bool,
    emit: EmitFn = _default_emit,
    loop_factory: LoopFactoryFn = _default_loop_factory,
    lock_owner: LockOwnerFn = _default_lock_owner,
    lock_factory: LockFactoryFn = _default_lock_factory,
) -> Iterator[None]:
    if not enabled:
        yield
        return

    if lock_owner(config, "display-loop") is not None:
        yield
        return

    try:
        loop = loop_factory(config)
    except Exception as exc:
        emit(f"display_companion=unavailable:{exc}")
        yield
        return

    stop_event = Event()
    original_sleep = loop.sleep
    loop.sleep = stop_event.wait

    def _run() -> None:
        try:
            with lock_factory(config, "display-loop"):
                loop.run()
        except Exception as exc:
            emit(f"display_companion=failed:{exc}")

    thread = Thread(
        target=_run,
        name="twinr-display-companion",
        daemon=True,
    )
    thread.start()
    emit("display_companion=started")
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=_join_timeout_seconds(config))
        if thread.is_alive():
            emit("display_companion=stop-timeout")
        loop.sleep = original_sleep
