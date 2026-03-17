"""Drive the Twinr status display loop from runtime snapshots.

This module polls the runtime snapshot store, samples bounded health and
connectivity signals, translates them into short status frames for the e-paper
adapter, and persists a small display heartbeat so supervision can distinguish
an alive companion from a hung one. Hardware-specific rendering lives in
``waveshare_v2.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math
import os
import socket
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot, RuntimeSnapshotStore
from twinr.display.debug_log import LogSections, TwinrDisplayDebugLogBuilder
from twinr.display.heartbeat import DisplayHeartbeatStore, save_display_heartbeat
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
_DEGRADED_INTERNET_FOOTER_REFRESH_S = 10.0
_MIN_DISPLAY_POLL_INTERVAL_S = 0.05
_SNAPSHOT_RETRY_DELAY_S = 0.05
_STATUS_TEXT_MAX_LEN = 32
_EMIT_LINE_MAX_LEN = 160
_ERROR_LOG_THROTTLE_S = 30.0
_HEARTBEAT_IDLE_REFRESH_S = 5.0

_LOGGER = logging.getLogger(__name__)


def _default_emit(line: str) -> None:
    """Print a bounded telemetry line."""
    print(line, flush=True)


def _default_clock() -> datetime:
    """Return the current local wall clock as an aware datetime."""
    # AUDIT-FIX(#7): Use an aware local datetime to avoid ambiguous naive-time semantics around DST and injected clocks.
    return datetime.now().astimezone()


def _default_internet_probe() -> bool:
    """Probe internet reachability with bounded numeric endpoints."""
    # AUDIT-FIX(#4): Use numeric endpoints only so the probe remains bounded even when DNS resolution is unhealthy.
    endpoints = (
        ("1.1.1.1", 443),
        ("8.8.8.8", 53),
        ("9.9.9.9", 53),
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
    """Drive the status panel from runtime snapshots and health signals.

    The loop keeps the rendered state bounded and resilient: snapshot, health,
    or display faults degrade to last-known-good or unknown labels instead of
    terminating the loop.

    Attributes:
        config: Runtime configuration including display timings and API-key
            state.
        display: Display adapter that renders and uploads status images.
        snapshot_store: File-backed runtime snapshot source.
        emit: Best-effort telemetry sink for bounded status/error lines.
        sleep: Sleep primitive used between polling cycles.
        health_collector: Callable that samples system health for footer
            labels.
        clock: Callable that returns the current local time.
        internet_probe: Callable that samples outbound connectivity.
        heartbeat_store: File-backed display-progress heartbeat sink for
            supervisor and ops health consumers.
    """

    config: TwinrConfig
    display: WaveshareEPD4In2V2
    snapshot_store: RuntimeSnapshotStore
    # AUDIT-FIX(#8): Replace invalid `callable` annotations with explicit callable signatures for Python 3.11 type safety.
    emit: Callable[[str], None] = _default_emit
    sleep: Callable[[float], None] = time.sleep
    health_collector: Callable[..., TwinrSystemHealth] = collect_system_health
    clock: Callable[[], datetime] = _default_clock
    internet_probe: Callable[[], bool] = _default_internet_probe
    debug_log_builder: TwinrDisplayDebugLogBuilder | None = None
    heartbeat_store: DisplayHeartbeatStore | None = None
    _cached_health: TwinrSystemHealth | None = field(default=None, init=False, repr=False)
    _cached_health_error: str | None = field(default=None, init=False, repr=False)
    _cached_health_status: str | None = field(default=None, init=False, repr=False)
    _cached_health_at: float = field(default=0.0, init=False, repr=False)
    _cached_internet_ok: bool | None = field(default=None, init=False, repr=False)
    _cached_internet_at: float = field(default=0.0, init=False, repr=False)
    _last_snapshot: RuntimeSnapshot | None = field(default=None, init=False, repr=False)
    _last_error_key: str | None = field(default=None, init=False, repr=False)
    _last_error_at: float = field(default=0.0, init=False, repr=False)
    _heartbeat_seq: int = field(default=0, init=False, repr=False)
    _last_heartbeat_at: float = field(default=0.0, init=False, repr=False)
    _last_render_started_at: datetime | None = field(default=None, init=False, repr=False)
    _last_render_completed_at: datetime | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        # AUDIT-FIX(#8): Keep constructor injection but annotate optional emit correctly for 3.11 tooling and readability.
        emit: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "TwinrStatusDisplayLoop":
        """Build a display loop from Twinr configuration.

        Args:
            config: Runtime configuration with display and snapshot settings.
            emit: Optional telemetry sink for bounded status/error lines.
            sleep: Sleep primitive used between polling cycles.

        Returns:
            A configured ``TwinrStatusDisplayLoop`` instance.
        """
        return cls(
            config=config,
            display=WaveshareEPD4In2V2.from_config(config),
            snapshot_store=RuntimeSnapshotStore(config.runtime_state_path),
            emit=emit or _default_emit,
            sleep=sleep,
            debug_log_builder=TwinrDisplayDebugLogBuilder.from_config(config),
            heartbeat_store=DisplayHeartbeatStore.from_config(config),
        )

    def run(self, *, duration_s: float | None = None, max_cycles: int | None = None) -> int:
        """Run the display loop until a duration or cycle limit is reached.

        Args:
            duration_s: Optional maximum runtime in seconds.
            max_cycles: Optional maximum number of polling cycles.

        Returns:
            ``0`` when the loop exits cleanly.
        """
        started_at = time.monotonic()
        # AUDIT-FIX(#8): Track the full display signature with the correct tuple shape so static checks match runtime behavior.
        last_signature: tuple[str, str, tuple[str, ...], tuple[tuple[str, str], ...], LogSections, int] | None = None
        cycles = 0
        try:
            while True:
                if duration_s is not None and (time.monotonic() - started_at) >= duration_s:
                    return 0
                if max_cycles is not None and cycles >= max_cycles:
                    return 0

                # AUDIT-FIX(#1): Never let snapshot read/build/display failures terminate the status loop; degrade to last-known-good state.
                snapshot, snapshot_stale = self._load_snapshot()
                headline, details = self._build_status_content(snapshot, stale=snapshot_stale)
                state_fields = self._build_state_fields(snapshot, stale=snapshot_stale)
                log_sections = self._build_log_sections(snapshot, stale=snapshot_stale)
                status = self._snapshot_status(snapshot)
                frame = self._display_animation_frame(status)
                signature = (status, headline, details, state_fields, log_sections, frame)
                if signature != last_signature:
                    self._last_render_started_at = datetime.now(timezone.utc)
                    self._write_heartbeat(status, phase="rendering", force=True)
                    if self._show_status(
                        status,
                        headline=headline,
                        details=details,
                        state_fields=state_fields,
                        log_sections=log_sections,
                        animation_frame=frame,
                    ):
                        self._safe_emit(f"display_status={status}")
                        last_signature = signature
                        self._last_render_completed_at = datetime.now(timezone.utc)
                        self._write_heartbeat(status, phase="idle", force=True)
                    else:
                        self._write_heartbeat(status, phase="error", detail="display_show_failed", force=True)
                else:
                    self._write_heartbeat(status, phase="idle")
                cycles += 1
                self._sleep_once()
        finally:
            status = self._snapshot_status(self._last_snapshot)
            self._write_heartbeat(status, phase="stopping", detail="display_loop_stopping", force=True)
            # AUDIT-FIX(#10): Cleanup must never mask the real failure path.
            self._close_display()

    def _build_status_content(
        self,
        snapshot: RuntimeSnapshot | None,
        *,
        stale: bool = False,
    ) -> tuple[str, tuple[str, ...]]:
        # AUDIT-FIX(#1): Normalize snapshot access so corrupt/missing state cannot trigger AttributeError/None crashes.
        status = self._snapshot_status(snapshot)

        # AUDIT-FIX(#2): Make stale state explicit instead of silently presenting an old snapshot as current truth.
        note = "Status wird aktualisiert" if stale else None
        if snapshot is None and not stale:
            note = "Status nicht verfügbar"

        # AUDIT-FIX(#5): Build short, line-oriented footer details instead of one oversized string that overflows the eInk panel.
        details = self._detail_lines(
            note,
            *self._health_detail_values(snapshot),
        )
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
        # AUDIT-FIX(#9): Sanitize unexpected status values before they reach the display surface.
        return status.replace("_", " ").title(), details

    def _detail_lines(self, *values: str | None) -> tuple[str, ...]:
        lines: list[str] = []
        for value in values:
            if not value:
                continue
            compact = self._compact_text(value)
            if not compact:
                continue
            while compact and len(lines) < 4:
                lines.append(compact[:32])
                compact = compact[32:].lstrip()
        return tuple(lines[:4])

    def _animation_frame(self, status: str) -> int:
        frame_count, frame_seconds = self._animation_spec(status)
        if frame_count <= 1 or frame_seconds <= 0 or not math.isfinite(frame_seconds):
            return 0
        return int(time.monotonic() / frame_seconds) % frame_count

    def _animation_spec(self, status: str) -> tuple[int, float]:
        return _STATUS_ANIMATION_SPECS.get(self._compact_text(status).lower(), (1, 60.0))

    def _health_detail_values(self, snapshot: RuntimeSnapshot | None) -> tuple[str, str, str, str]:
        # AUDIT-FIX(#1): Footer generation must survive collector/probe/clock failures and fall back to cached/unknown values.
        health = self._current_health(snapshot)
        internet_ok = self._internet_ok()
        return (
            self._internet_footer_label(internet_ok),
            self._ai_footer_label(snapshot, health, internet_ok),
            self._system_footer_label(snapshot, health),
            f"Zeit {self._clock_text()}",
        )

    def _build_state_fields(
        self,
        snapshot: RuntimeSnapshot | None,
        *,
        stale: bool = False,
    ) -> tuple[tuple[str, str], ...]:
        note = "Status wird aktualisiert" if stale else None
        if snapshot is None and not stale:
            note = "Status nicht verfügbar"
        health = self._current_health(snapshot)
        internet_ok = self._internet_ok()
        state_fields = [
            ("Status", self._runtime_state_value(snapshot)),
            ("Internet", self._internet_state_value(internet_ok)),
            ("AI", self._ai_state_value(snapshot, health, internet_ok)),
            ("System", self._system_state_value(snapshot, health)),
            ("Zeit", self._clock_text()),
        ]
        if note:
            state_fields.append(("Hinweis", note))
        return tuple(state_fields)

    def _build_log_sections(
        self,
        snapshot: RuntimeSnapshot | None,
        *,
        stale: bool = False,
    ) -> LogSections:
        if self._display_layout() != "debug_log":
            return ()
        health = self._current_health(snapshot)
        internet_ok = self._internet_ok()
        return self._debug_log_builder().build_sections(
            snapshot=snapshot,
            runtime_status=self._runtime_state_value(snapshot),
            internet_state=self._internet_state_value(internet_ok),
            ai_state=self._ai_state_value(snapshot, health, internet_ok),
            system_state=self._system_state_value(snapshot, health),
            clock_text=self._clock_text(),
            health=health,
            stale=stale,
        )

    def _current_health(self, snapshot: RuntimeSnapshot | None) -> TwinrSystemHealth | None:
        now = time.monotonic()
        snapshot_error = self._snapshot_error_message(snapshot)
        snapshot_status = self._snapshot_status(snapshot)

        if (
            (now - self._cached_health_at) < _HEALTH_FOOTER_REFRESH_S
            and self._cached_health_error == snapshot_error
            and self._cached_health_status == snapshot_status
        ):
            return self._cached_health

        if snapshot is None:
            self._cached_health_error = snapshot_error
            self._cached_health_status = snapshot_status
            self._cached_health_at = now
            return self._cached_health

        try:
            # AUDIT-FIX(#1): Guard the external health collector; reuse the last cached health on transient failures.
            health = self.health_collector(self.config, snapshot=snapshot)
        except Exception as exc:
            self._emit_error("health_collect_failed", exc)
            self._cached_health_error = snapshot_error
            self._cached_health_status = snapshot_status
            self._cached_health_at = now
            return self._cached_health

        self._cached_health = health
        self._cached_health_error = snapshot_error
        self._cached_health_status = snapshot_status
        self._cached_health_at = now
        return health

    def _internet_ok(self) -> bool | None:
        now = time.monotonic()
        refresh_s = _INTERNET_FOOTER_REFRESH_S if self._cached_internet_ok is True else _DEGRADED_INTERNET_FOOTER_REFRESH_S
        if (now - self._cached_internet_at) < refresh_s:
            return self._cached_internet_ok
        try:
            # AUDIT-FIX(#4): Treat probe failure as an unknown/down state instead of letting it abort the footer path.
            self._cached_internet_ok = bool(self.internet_probe())
        except Exception as exc:
            self._emit_error("internet_probe_failed", exc)
            self._cached_internet_ok = None
        self._cached_internet_at = now
        return self._cached_internet_ok

    def _ai_footer_label(
        self,
        snapshot: RuntimeSnapshot | None,
        health: TwinrSystemHealth | None,
        internet_ok: bool | None,
    ) -> str:
        return f"AI {self._ai_state_value(snapshot, health, internet_ok)}"

    def _ai_state_value(
        self,
        snapshot: RuntimeSnapshot | None,
        health: TwinrSystemHealth | None,
        internet_ok: bool | None,
    ) -> str:
        if not self.config.openai_api_key:
            return "fehlt"
        # AUDIT-FIX(#6): Do not claim "AI ok" while the network is down or unknown.
        if internet_ok is False:
            return "wartet"
        if internet_ok is None:
            return "?"
        health_error = self._compact_text(getattr(health, "runtime_error", None))
        error_text = " ".join(
            part
            for part in (self._snapshot_error_message(snapshot), health_error)
            if part
        ).lower()
        # AUDIT-FIX(#6): Tighten keyword matching to avoid false positives from the overly broad `api` substring.
        if any(
            token in error_text
            for token in (
                "openai",
                "api key",
                "authentication",
                "auth",
                "token",
                "quota",
                "rate limit",
                "model",
            )
        ):
            return "Achtung"
        return "ok"

    def _system_footer_label(self, snapshot: RuntimeSnapshot | None, health: TwinrSystemHealth | None) -> str:
        return f"System {self._system_state_value(snapshot, health)}"

    def _system_state_value(self, snapshot: RuntimeSnapshot | None, health: TwinrSystemHealth | None) -> str:
        snapshot_status = self._snapshot_status(snapshot)
        snapshot_error = self._snapshot_error_message(snapshot)
        runtime_error = self._compact_text(getattr(health, "runtime_error", None))
        if snapshot_status == "error" or snapshot_error or runtime_error:
            return "Fehler"
        if health is None:
            return "?"

        health_status = self._compact_text(getattr(health, "status", None)).lower()
        if health_status == "fail":
            return "Fehler"
        if not self._service_running(health, "conversation_loop"):
            return "Achtung"
        if getattr(health, "cpu_temperature_c", None) is not None and health.cpu_temperature_c >= 72:
            return "warm"
        if getattr(health, "memory_used_percent", None) is not None and health.memory_used_percent >= 80:
            return "Achtung"
        if getattr(health, "disk_used_percent", None) is not None and health.disk_used_percent >= 85:
            return "Achtung"
        if health_status == "warn":
            return "Achtung"
        return "ok"

    def _internet_footer_label(self, internet_ok: bool | None) -> str:
        return f"Internet {self._internet_state_value(internet_ok)}"

    def _internet_state_value(self, internet_ok: bool | None) -> str:
        if internet_ok is True:
            return "ok"
        if internet_ok is False:
            return "fehlt"
        return "?"

    def _runtime_state_value(self, snapshot: RuntimeSnapshot | None) -> str:
        status = self._snapshot_status(snapshot)
        if status == "waiting":
            return "Waiting"
        if status == "listening":
            return "Listening"
        if status == "processing":
            return "Processing"
        if status == "answering":
            return "Answering"
        if status == "printing":
            return "Printing"
        if status == "error":
            return "Error"
        return status.replace("_", " ").title()

    def _service_running(self, health: TwinrSystemHealth | None, key: str) -> bool:
        if health is None:
            return False
        services = getattr(health, "services", ()) or ()
        for service in services:
            if getattr(service, "key", None) == key:
                return bool(getattr(service, "running", False))
        return True

    def _load_snapshot(self) -> tuple[RuntimeSnapshot | None, bool]:
        try:
            snapshot = self.snapshot_store.load()
        except Exception as exc:
            self._emit_error("snapshot_load_failed", exc)
            # AUDIT-FIX(#2): Retry once after a very short pause to ride out partial file writes in the file-backed store.
            time.sleep(_SNAPSHOT_RETRY_DELAY_S)
            try:
                snapshot = self.snapshot_store.load()
            except Exception as retry_exc:
                self._emit_error("snapshot_load_retry_failed", retry_exc)
                return self._last_snapshot, self._last_snapshot is not None

        self._last_snapshot = snapshot
        return snapshot, False

    def _show_status(
        self,
        status: str,
        *,
        headline: str,
        details: tuple[str, ...],
        state_fields: tuple[tuple[str, str], ...],
        log_sections: LogSections,
        animation_frame: int,
    ) -> bool:
        try:
            self.display.show_status(
                status,
                headline=headline,
                details=details,
                state_fields=state_fields,
                log_sections=log_sections,
                animation_frame=animation_frame,
            )
            return True
        except Exception as exc:
            self._emit_error("display_show_failed", exc)

        # AUDIT-FIX(#1): Attempt one display re-open before giving up on the frame update.
        if self._reopen_display():
            try:
                self.display.show_status(
                    status,
                    headline=headline,
                    details=details,
                    state_fields=state_fields,
                    log_sections=log_sections,
                    animation_frame=animation_frame,
                )
                return True
            except Exception as retry_exc:
                self._emit_error("display_show_retry_failed", retry_exc)
        return False

    def _write_heartbeat(
        self,
        runtime_status: str,
        *,
        phase: str,
        detail: str | None = None,
        force: bool = False,
    ) -> None:
        store = self.heartbeat_store
        if store is None:
            return
        now_monotonic = time.monotonic()
        if not force and (now_monotonic - self._last_heartbeat_at) < _HEARTBEAT_IDLE_REFRESH_S:
            return
        self._heartbeat_seq += 1
        try:
            save_display_heartbeat(
                store,
                runtime_status=runtime_status,
                phase=phase,
                seq=self._heartbeat_seq,
                detail=self._compact_text(detail, max_len=_STATUS_TEXT_MAX_LEN) or None,
                pid=os.getpid(),
                updated_at=datetime.now(timezone.utc),
                last_render_started_at=self._last_render_started_at,
                last_render_completed_at=self._last_render_completed_at,
            )
        except Exception as exc:
            self._emit_error("display_heartbeat_save_failed", exc)
            return
        self._last_heartbeat_at = now_monotonic

    def _reopen_display(self) -> bool:
        self._close_display()
        factories: list[Callable[[TwinrConfig], WaveshareEPD4In2V2]] = []
        display_factory = getattr(type(self.display), "from_config", None)
        if callable(display_factory):
            factories.append(display_factory)
        default_factory = getattr(WaveshareEPD4In2V2, "from_config", None)
        if callable(default_factory) and default_factory not in factories:
            factories.append(default_factory)

        for factory in factories:
            try:
                self.display = factory(self.config)
                return True
            except Exception as exc:
                self._emit_error("display_reopen_failed", exc)
        return False

    def _close_display(self) -> None:
        close = getattr(self.display, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                self._emit_error("display_close_failed", exc)

    def _sleep_once(self) -> None:
        poll_interval_s = self._poll_interval_s()
        try:
            self.sleep(poll_interval_s)
        except Exception as exc:
            # AUDIT-FIX(#1): A broken injected sleeper must not terminate the loop.
            self._emit_error("display_sleep_failed", exc)
            time.sleep(poll_interval_s)

    def _poll_interval_s(self) -> float:
        raw_interval = getattr(self.config, "display_poll_interval_s", _MIN_DISPLAY_POLL_INTERVAL_S)
        try:
            interval_s = float(raw_interval)
        except (TypeError, ValueError):
            self._emit_error("display_poll_interval_invalid")
            return _MIN_DISPLAY_POLL_INTERVAL_S
        # AUDIT-FIX(#3): Clamp invalid/degenerate intervals so config mistakes do not crash or spin the process.
        if not math.isfinite(interval_s) or interval_s < _MIN_DISPLAY_POLL_INTERVAL_S:
            self._emit_error("display_poll_interval_clamped")
            return _MIN_DISPLAY_POLL_INTERVAL_S
        return interval_s

    def _clock_text(self) -> str:
        try:
            current = self.clock()
            if not isinstance(current, datetime):
                raise TypeError("clock() must return datetime")
            return current.strftime("%H:%M")
        except Exception as exc:
            self._emit_error("clock_failed", exc)
            return "--:--"

    def _display_animation_frame(self, status: str) -> int:
        if self._display_layout() != "default":
            return 0
        return self._animation_frame(status)

    def _display_layout(self) -> str:
        return self._compact_text(getattr(self.config, "display_layout", "default")).lower() or "default"

    def _debug_log_builder(self) -> TwinrDisplayDebugLogBuilder:
        if self.debug_log_builder is None:
            self.debug_log_builder = TwinrDisplayDebugLogBuilder.from_config(self.config)
        return self.debug_log_builder

    def _snapshot_status(self, snapshot: RuntimeSnapshot | None) -> str:
        if snapshot is None:
            return "error"
        # AUDIT-FIX(#9): Normalize status values from the file-backed store before they reach logs/UI logic.
        compact = self._compact_text(getattr(snapshot, "status", None), max_len=_STATUS_TEXT_MAX_LEN).lower()
        return compact or "error"

    def _snapshot_error_message(self, snapshot: RuntimeSnapshot | None) -> str:
        if snapshot is None:
            return ""
        return self._compact_text(getattr(snapshot, "error_message", None), max_len=_EMIT_LINE_MAX_LEN)

    def _compact_text(self, value: object | None, *, max_len: int | None = None) -> str:
        if value is None:
            return ""
        text = "".join(ch if ch.isprintable() else " " for ch in str(value))
        text = " ".join(text.split())
        if max_len is not None:
            return text[:max_len].rstrip()
        return text

    def _emit_error(self, key: str, exc: Exception | None = None) -> None:
        now = time.monotonic()
        exc_name = type(exc).__name__ if exc is not None else "Error"
        error_key = f"{key}:{exc_name}"
        if self._last_error_key == error_key and (now - self._last_error_at) < _ERROR_LOG_THROTTLE_S:
            return
        self._last_error_key = error_key
        self._last_error_at = now
        self._safe_emit(f"{key}={exc_name}")

    def _safe_emit(self, line: str) -> None:
        compact = self._compact_text(line, max_len=_EMIT_LINE_MAX_LEN)
        if not compact:
            return
        try:
            self.emit(compact)
        except Exception:
            _LOGGER.warning("Display emit sink failed.", exc_info=True)
