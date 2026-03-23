"""Run the ReSpeaker LED loop as an optional Pi-runtime companion thread."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import math
from threading import Event, Thread
from typing import Protocol

from twinr.agent.base_agent.config import TwinrConfig


class ReSpeakerLedLoopLike(Protocol):
    """Describe the minimal interface required by the LED companion runner."""

    sleep: Callable[[float], object]
    stop_requested: Callable[[], bool]

    def run(self, *, duration_s: float | None = None) -> int: ...


EmitFn = Callable[[str], None]
LockFactoryFn = Callable[[TwinrConfig, str], object]
LockOwnerFn = Callable[[TwinrConfig, str], int | None]
LoopFactoryFn = Callable[[TwinrConfig], ReSpeakerLedLoopLike]

_LOCK_NAME = "respeaker-led-loop"


def _default_emit(line: str) -> None:
    """Print one bounded telemetry line."""

    print(line, flush=True)


def _join_timeout_seconds(loop: ReSpeakerLedLoopLike) -> float:
    """Return a bounded shutdown wait that covers one in-flight LED write and off()."""

    controller = getattr(loop, "controller", None)
    transport = getattr(controller, "transport", None)
    raw_timeout_ms = getattr(transport, "read_timeout_ms", 1000)
    try:
        timeout_s = float(raw_timeout_ms) / 1000.0
    except (TypeError, ValueError):
        timeout_s = 1.0
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        timeout_s = 1.0
    return max(2.5, min((timeout_s * 2.0) + 0.5, 5.0))


def _default_lock_factory(config: TwinrConfig, loop_name: str) -> object:
    """Build the loop-instance lock context manager for the LED loop."""

    from twinr.ops import loop_instance_lock

    return loop_instance_lock(config, loop_name)


def _default_lock_owner(config: TwinrConfig, loop_name: str) -> int | None:
    """Return the PID that currently owns the LED companion lock."""

    from twinr.ops import loop_lock_owner

    return loop_lock_owner(config, loop_name)


def _default_loop_factory(config: TwinrConfig) -> ReSpeakerLedLoopLike:
    """Build the default ReSpeaker LED loop from configuration."""

    from twinr.hardware.respeaker.led_loop import ReSpeakerLedLoop

    return ReSpeakerLedLoop.from_config(config)


@contextmanager
def optional_respeaker_led_companion(
    config: TwinrConfig,
    *,
    enabled: bool,
    emit: EmitFn = _default_emit,
    loop_factory: LoopFactoryFn = _default_loop_factory,
    lock_owner: LockOwnerFn = _default_lock_owner,
    lock_factory: LockFactoryFn = _default_lock_factory,
) -> Iterator[None]:
    """Run the ReSpeaker LED loop in a background thread when enabled."""

    if not enabled:
        yield
        return

    if lock_owner(config, _LOCK_NAME) is not None:
        yield
        return

    try:
        loop = loop_factory(config)
    except Exception as exc:
        emit(f"respeaker_led_companion=unavailable:{exc}")
        yield
        return

    stop_event = Event()
    original_sleep = loop.sleep
    original_stop_requested = loop.stop_requested
    loop.sleep = stop_event.wait
    loop.stop_requested = stop_event.is_set

    def _run() -> None:
        try:
            with lock_factory(config, _LOCK_NAME):
                loop.run()
        except Exception as exc:
            emit(f"respeaker_led_companion=failed:{exc}")

    thread = Thread(
        target=_run,
        name="twinr-respeaker-led-companion",
        daemon=True,
    )
    thread.start()
    emit("respeaker_led_companion=started")
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=_join_timeout_seconds(loop))
        if thread.is_alive():
            emit("respeaker_led_companion=stop-timeout")
        loop.sleep = original_sleep
        loop.stop_requested = original_stop_requested


__all__ = [
    "optional_respeaker_led_companion",
]
