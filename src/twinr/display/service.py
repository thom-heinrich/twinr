from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
import socket
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime_state import RuntimeSnapshot, RuntimeSnapshotStore
from twinr.display.waveshare_v2 import WaveshareEPD4In2V2
from twinr.ops.health import TwinrSystemHealth, collect_system_health

_STATUS_ANIMATION_SPECS: dict[str, tuple[int, float]] = {
    "waiting": (6, 5.0),
    "listening": (4, 1.4),
    "processing": (4, 1.6),
    "answering": (4, 1.2),
    "printing": (4, 1.6),
    "error": (4, 2.0),
}

_HEALTH_FOOTER_REFRESH_S = 10.0
_INTERNET_FOOTER_REFRESH_S = 60.0


def _default_emit(line: str) -> None:
    print(line, flush=True)


def _default_clock() -> datetime:
    return datetime.now()


def _default_internet_probe() -> bool:
    endpoints = (
        ("1.1.1.1", 53),
        ("8.8.8.8", 53),
        ("api.openai.com", 443),
    )
    for host, port in endpoints:
        try:
            with socket.create_connection((host, port), timeout=0.7):
                return True
        except OSError:
            continue
    return False


@dataclass(slots=True)
class TwinrStatusDisplayLoop:
    config: TwinrConfig
    display: WaveshareEPD4In2V2
    snapshot_store: RuntimeSnapshotStore
    emit: callable = _default_emit
    sleep: callable = time.sleep
    health_collector: Callable[..., TwinrSystemHealth] = collect_system_health
    clock: Callable[[], datetime] = _default_clock
    internet_probe: Callable[[], bool] = _default_internet_probe
    _cached_health: TwinrSystemHealth | None = field(default=None, init=False, repr=False)
    _cached_health_error: str | None = field(default=None, init=False, repr=False)
    _cached_health_status: str | None = field(default=None, init=False, repr=False)
    _cached_health_at: float = field(default=0.0, init=False, repr=False)
    _cached_internet_ok: bool | None = field(default=None, init=False, repr=False)
    _cached_internet_at: float = field(default=0.0, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        emit=None,
        sleep=time.sleep,
    ) -> "TwinrStatusDisplayLoop":
        return cls(
            config=config,
            display=WaveshareEPD4In2V2.from_config(config),
            snapshot_store=RuntimeSnapshotStore(config.runtime_state_path),
            emit=emit or _default_emit,
            sleep=sleep,
        )

    def run(self, *, duration_s: float | None = None, max_cycles: int | None = None) -> int:
        started_at = time.monotonic()
        last_signature: tuple[str, int] | None = None
        cycles = 0
        try:
            while True:
                if duration_s is not None and (time.monotonic() - started_at) >= duration_s:
                    return 0
                if max_cycles is not None and cycles >= max_cycles:
                    return 0

                snapshot = self.snapshot_store.load()
                headline, details = self._build_status_content(snapshot)
                frame = self._animation_frame(snapshot.status)
                signature = (snapshot.status, frame, details)
                if signature != last_signature:
                    self.display.show_status(
                        snapshot.status,
                        headline=headline,
                        details=details,
                        animation_frame=frame,
                    )
                    self.emit(f"display_status={snapshot.status}")
                    last_signature = signature
                cycles += 1
                self.sleep(self.config.display_poll_interval_s)
        finally:
            close = getattr(self.display, "close", None)
            if callable(close):
                close()

    def _build_status_content(self, snapshot: RuntimeSnapshot) -> tuple[str, tuple[str, ...]]:
        status = snapshot.status.lower()
        health_footer = self._health_footer(snapshot)
        details = (health_footer,) if health_footer else ()
        if status == "waiting":
            return "Waiting", details
        if status == "listening":
            return "Listening", details
        if status == "processing":
            return "Processing", details
        if status == "answering":
            return "Answering", details
        if status == "printing":
            return "Printing", details
        if status == "error":
            return "Error", details
        return snapshot.status.title(), details

    def _detail_lines(self, *values: str | None) -> tuple[str, ...]:
        lines: list[str] = []
        for value in values:
            if not value:
                continue
            compact = " ".join(value.split())
            if not compact:
                continue
            while compact and len(lines) < 4:
                lines.append(compact[:32])
                compact = compact[32:].lstrip()
        return tuple(lines[:4])

    def _animation_frame(self, status: str) -> int:
        frame_count, frame_seconds = self._animation_spec(status)
        if frame_count <= 1:
            return 0
        return int(time.monotonic() / frame_seconds) % frame_count

    def _animation_spec(self, status: str) -> tuple[int, float]:
        return _STATUS_ANIMATION_SPECS.get(status.lower(), (1, 60.0))

    def _health_footer(self, snapshot: RuntimeSnapshot) -> str | None:
        health = self._current_health(snapshot)
        internet_ok = self._internet_ok()
        time_text = self.clock().strftime("%H:%M")
        base = " | ".join(
            (
                f"Internet {'ok' if internet_ok else 'down'}",
                self._ai_footer_label(snapshot, health),
                self._system_footer_label(snapshot, health),
            )
        )
        return f"{base} ({time_text})"

    def _current_health(self, snapshot: RuntimeSnapshot) -> TwinrSystemHealth:
        now = time.monotonic()
        snapshot_error = snapshot.error_message or ""
        if (
            self._cached_health is not None
            and (now - self._cached_health_at) < _HEALTH_FOOTER_REFRESH_S
            and self._cached_health_error == snapshot_error
            and self._cached_health_status == snapshot.status
        ):
            return self._cached_health

        health = self.health_collector(self.config, snapshot=snapshot)
        self._cached_health = health
        self._cached_health_error = snapshot_error
        self._cached_health_status = snapshot.status
        self._cached_health_at = now
        return health

    def _internet_ok(self) -> bool:
        now = time.monotonic()
        if self._cached_internet_ok is not None and (now - self._cached_internet_at) < _INTERNET_FOOTER_REFRESH_S:
            return self._cached_internet_ok
        self._cached_internet_ok = self.internet_probe()
        self._cached_internet_at = now
        return self._cached_internet_ok

    def _ai_footer_label(self, snapshot: RuntimeSnapshot, health: TwinrSystemHealth) -> str:
        if not self.config.openai_api_key:
            return "AI fehlt"
        error_text = " ".join(
            part.strip().lower()
            for part in (snapshot.error_message, health.runtime_error)
            if part and part.strip()
        )
        if any(token in error_text for token in ("openai", "api", "model", "auth", "token", "quota")):
            return "AI Achtung"
        return "AI ok"

    def _system_footer_label(self, snapshot: RuntimeSnapshot, health: TwinrSystemHealth) -> str:
        if snapshot.status.lower() == "error" or snapshot.error_message or health.runtime_error:
            return "System Fehler"
        if health.status == "fail":
            return "System Fehler"
        if not self._service_running(health, "conversation_loop"):
            return "System Achtung"
        if health.cpu_temperature_c is not None and health.cpu_temperature_c >= 72:
            return "System warm"
        if health.memory_used_percent is not None and health.memory_used_percent >= 80:
            return "System Achtung"
        if health.disk_used_percent is not None and health.disk_used_percent >= 85:
            return "System Achtung"
        if health.status == "warn":
            return "System Achtung"
        return "System ok"

    def _service_running(self, health: TwinrSystemHealth, key: str) -> bool:
        for service in health.services:
            if service.key == key:
                return service.running
        return True
