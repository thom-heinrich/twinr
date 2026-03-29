# CHANGELOG: 2026-03-28
# BUG-1: Detect stale active runtime snapshots and stop silently presenting old listening/processing/answering states as current truth.
# BUG-2: Force a repaint after display reopen/tick recovery; the previous loop could reopen a fresh backend and then skip rendering forever.
# BUG-3: Let debug-log layouts repaint immediately on semantic changes while still allowing periodic refreshes for identical content.
# SEC-1: Sanitize emitted presentation telemetry tokens so file-backed cue payloads cannot forge key=value operator logs.
# IMP-1: Move internet and health sampling off the hot render path with bounded async workers and cached fallback values.
# IMP-2: Add adaptive idle polling, optional systemd sd_notify status/watchdog integration, and periodic maintenance refresh for unchanged Waveshare frames.
# IMP-3: Persist the last successfully rendered state fields plus health verdict so display-side ERROR claims
#        can be proven from one authoritative ops artifact.

"""Drive the Twinr status display loop from runtime snapshots.

This module polls the runtime snapshot store, samples bounded health and
connectivity signals, translates them into short status frames for the active
display backend, and persists a small display heartbeat so supervision can
distinguish an alive companion from a hung one.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
import logging
import math
import os
import socket
import time
from typing import cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot, RuntimeSnapshotStore
from twinr.display.ambient_impulse_cues import (
    DisplayAmbientImpulseCue,
    DisplayAmbientImpulseCueStore,
)
from twinr.display.contracts import TwinrDisplayAdapter
from twinr.display.debug_signals import DisplayDebugSignal, DisplayDebugSignalStore
from twinr.display.debug_log import LogSections, TwinrDisplayDebugLogBuilder
from twinr.display.emoji_cues import DisplayEmojiCue, DisplayEmojiCueStore
from twinr.display.face_cues import DisplayFaceCue, DisplayFaceCueStore
from twinr.display.factory import create_display_adapter
from twinr.display.heartbeat import DisplayHeartbeatStore, save_display_heartbeat
from twinr.display.news_ticker import DisplayNewsTickerRuntime
from twinr.display.presentation_cues import DisplayPresentationCue, DisplayPresentationStore
from twinr.display.render_state import DisplayRenderStateStore
from twinr.display.reserve_bus import resolve_display_reserve_bus
from twinr.display.respeaker_hci import DisplayReSpeakerHciStore
from twinr.display.service_connect_cues import DisplayServiceConnectCue, DisplayServiceConnectCueStore
from twinr.ops.health import (
    TwinrSystemHealth,
    assess_memory_pressure_status,
    collect_system_health,
)

_STATUS_ANIMATION_SPECS: dict[str, tuple[int, float]] = {
    "waiting": (12, 0.75),
    "listening": (6, 0.45),
    "processing": (6, 0.55),
    "answering": (6, 0.40),
    "printing": (6, 0.55),
    "error": (6, 0.65),
}

_HEALTH_FOOTER_REFRESH_S = 10.0
_INTERNET_FOOTER_REFRESH_S = 60.0
_DEGRADED_INTERNET_FOOTER_REFRESH_S = 10.0
_DEBUG_LOG_MIN_REFRESH_S = 30.0
_MIN_DISPLAY_POLL_INTERVAL_S = 0.05
_SNAPSHOT_RETRY_DELAY_S = 0.05
_STATUS_TEXT_MAX_LEN = 32
_EMIT_LINE_MAX_LEN = 160
_ERROR_LOG_THROTTLE_S = 30.0
_HEARTBEAT_IDLE_REFRESH_S = 5.0
_BACKGROUND_WORKERS = 2
_IDLE_POLL_MULTIPLIER = 4.0
_IDLE_POLL_MAX_S = 1.0
_ACTIVE_STATUS_SNAPSHOT_STALE_AFTER_S = 20.0
_WAVESHARE_MAINTENANCE_REFRESH_S = 23 * 60 * 60
_TELEMETRY_TOKEN_MAX_LEN = 48
_UNSET = object()

_LOGGER = logging.getLogger(__name__)
_VOICE_QUIET_FACE_CUE = DisplayFaceCue(
    source="runtime_voice_quiet",
    head_dy=1,
    mouth="neutral",
    brows="soft",
    blink=True,
)
_ACTIVE_RUNTIME_STATES = frozenset({"listening", "processing", "answering", "printing"})


def _default_emit(line: str) -> None:
    """Print a bounded telemetry line."""
    print(line, flush=True)


def _default_clock() -> datetime:
    """Return the current local wall clock as an aware datetime."""
    return datetime.now().astimezone()


def _default_internet_probe() -> bool:
    """Probe internet reachability with bounded numeric endpoints."""
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


def _never_stop() -> bool:
    """Return the default stop signal for standalone display loops."""
    return False


@dataclass(slots=True)
class TwinrStatusDisplayLoop:
    """Drive the status panel from runtime snapshots and health signals."""

    config: TwinrConfig
    display: TwinrDisplayAdapter
    snapshot_store: RuntimeSnapshotStore
    emit: Callable[[str], None] = _default_emit
    sleep: Callable[[float], None] = time.sleep
    stop_requested: Callable[[], bool] = _never_stop
    health_collector: Callable[..., TwinrSystemHealth] = collect_system_health
    clock: Callable[[], datetime] = _default_clock
    internet_probe: Callable[[], bool] = _default_internet_probe
    debug_log_builder: TwinrDisplayDebugLogBuilder | None = None
    heartbeat_store: DisplayHeartbeatStore | None = None
    face_cue_store: DisplayFaceCueStore | None = None
    emoji_cue_store: DisplayEmojiCueStore | None = None
    ambient_impulse_cue_store: DisplayAmbientImpulseCueStore | None = None
    service_connect_cue_store: DisplayServiceConnectCueStore | None = None
    presentation_cue_store: DisplayPresentationStore | None = None
    news_ticker_runtime: DisplayNewsTickerRuntime | None = None
    respeaker_hci_store: DisplayReSpeakerHciStore | None = None
    debug_signal_store: DisplayDebugSignalStore | None = None
    render_state_store: DisplayRenderStateStore | None = None
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
    _last_render_status: str | None = field(default=None, init=False, repr=False)
    _last_render_monotonic_s: float = field(default=0.0, init=False, repr=False)
    _last_display_telemetry_signature: tuple[object, ...] | None = field(default=None, init=False, repr=False)
    _background_pool: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _pending_health_future: Future[TwinrSystemHealth] | None = field(default=None, init=False, repr=False)
    _pending_health_error: str | None = field(default=None, init=False, repr=False)
    _pending_health_status: str | None = field(default=None, init=False, repr=False)
    _pending_internet_future: Future[bool] | None = field(default=None, init=False, repr=False)
    _force_render_next_cycle: bool = field(default=False, init=False, repr=False)
    _systemd_notify_socket: str | None = field(default=None, init=False, repr=False)
    _systemd_watchdog_interval_s: float | None = field(default=None, init=False, repr=False)
    _last_systemd_watchdog_at: float = field(default=0.0, init=False, repr=False)
    _systemd_ready_sent: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._systemd_notify_socket = self._resolve_systemd_notify_socket()
        self._systemd_watchdog_interval_s = self._resolve_systemd_watchdog_interval_s()

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        emit: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "TwinrStatusDisplayLoop":
        """Build a display loop from Twinr configuration."""
        return cls(
            config=config,
            display=create_display_adapter(config, emit=emit or _default_emit),
            snapshot_store=RuntimeSnapshotStore(config.runtime_state_path),
            emit=emit or _default_emit,
            sleep=sleep,
            debug_log_builder=TwinrDisplayDebugLogBuilder.from_config(config),
            heartbeat_store=DisplayHeartbeatStore.from_config(config),
            face_cue_store=DisplayFaceCueStore.from_config(config),
            emoji_cue_store=DisplayEmojiCueStore.from_config(config),
            ambient_impulse_cue_store=DisplayAmbientImpulseCueStore.from_config(config),
            service_connect_cue_store=DisplayServiceConnectCueStore.from_config(config),
            presentation_cue_store=DisplayPresentationStore.from_config(config),
            news_ticker_runtime=DisplayNewsTickerRuntime.from_config(config, emit=emit or _default_emit),
            respeaker_hci_store=DisplayReSpeakerHciStore.from_config(config),
            debug_signal_store=DisplayDebugSignalStore.from_config(config),
            render_state_store=DisplayRenderStateStore.from_config(config),
        )

    def run(self, *, duration_s: float | None = None, max_cycles: int | None = None) -> int:
        """Run the display loop until a duration or cycle limit is reached."""
        started_at = time.monotonic()
        last_signature: tuple[object, ...] | None = None
        cycles = 0
        self._systemd_notify_status("starting", phase="starting", detail="display_loop_starting")
        try:
            while True:
                if self.stop_requested():
                    return 0
                if duration_s is not None and (time.monotonic() - started_at) >= duration_s:
                    return 0
                if max_cycles is not None and cycles >= max_cycles:
                    return 0

                snapshot, snapshot_stale = self._load_snapshot()
                status = self._effective_runtime_status(snapshot, stale=snapshot_stale)
                health = self._current_health(snapshot)
                internet_ok = self._internet_ok()
                headline, details = self._build_status_content(
                    snapshot,
                    status=status,
                    stale=snapshot_stale,
                    health=health,
                    internet_ok=internet_ok,
                )
                state_fields = self._build_state_fields(
                    snapshot,
                    status=status,
                    stale=snapshot_stale,
                    health=health,
                    internet_ok=internet_ok,
                )
                log_sections = self._build_log_sections(
                    snapshot,
                    status=status,
                    stale=snapshot_stale,
                    health=health,
                    internet_ok=internet_ok,
                )
                frame = self._display_animation_frame(status)
                face_cue = self._active_face_cue(snapshot=snapshot)
                emoji_cue = self._active_emoji_cue()
                ambient_impulse_cue = self._active_ambient_impulse_cue()
                service_connect_cue = self._active_service_connect_cue()
                presentation_cue = self._active_presentation_cue()
                debug_signals = self._active_debug_signals()
                ticker_text = self._ticker_text()
                signature = self._render_signature(
                    status=status,
                    headline=headline,
                    ticker_text=ticker_text,
                    details=details,
                    state_fields=state_fields,
                    log_sections=log_sections,
                    animation_frame=frame,
                    face_cue=face_cue,
                    emoji_cue=emoji_cue,
                    ambient_impulse_cue=ambient_impulse_cue,
                    service_connect_cue=service_connect_cue,
                    presentation_cue=presentation_cue,
                    debug_signals=debug_signals,
                )

                if self._should_render_signature(status=status, signature=signature, last_signature=last_signature):
                    self._last_render_started_at = datetime.now(timezone.utc)
                    self._write_heartbeat(status, phase="rendering", force=True)
                    if self._show_status(
                        status,
                        headline=headline,
                        ticker_text=ticker_text,
                        details=details,
                        state_fields=state_fields,
                        log_sections=log_sections,
                        animation_frame=frame,
                        face_cue=face_cue,
                        emoji_cue=emoji_cue,
                        ambient_impulse_cue=ambient_impulse_cue,
                        service_connect_cue=service_connect_cue,
                        presentation_cue=presentation_cue,
                        debug_signals=debug_signals,
                    ):
                        self._emit_display_status(status=status, presentation_cue=presentation_cue)
                        last_signature = signature
                        self._last_render_completed_at = datetime.now(timezone.utc)
                        self._last_render_status = status
                        self._last_render_monotonic_s = time.monotonic()
                        self._force_render_next_cycle = False
                        self._save_render_state(
                            status=status,
                            headline=headline,
                            details=details,
                            state_fields=state_fields,
                            snapshot=snapshot,
                            snapshot_stale=snapshot_stale,
                            health=health,
                        )
                        self._write_heartbeat(status, phase="idle", force=True)
                        if not self._systemd_ready_sent:
                            self._systemd_notify("READY=1")
                            self._systemd_ready_sent = True
                    else:
                        self._write_heartbeat(status, phase="error", detail="display_show_failed", force=True)
                else:
                    self._write_heartbeat(status, phase="idle")

                self._tick_display()
                self._maybe_systemd_watchdog(status=status, phase="idle")
                cycles += 1
                self._sleep_once(
                    status=status,
                    ticker_text=ticker_text,
                    presentation_cue=presentation_cue,
                )
        finally:
            status = self._effective_runtime_status(
                self._last_snapshot,
                stale=self._snapshot_is_stale(self._last_snapshot),
            )
            self._write_heartbeat(status, phase="stopping", detail="display_loop_stopping", force=True)
            self._systemd_notify("STOPPING=1", f"STATUS={self._systemd_status_text(status, 'stopping', 'display_loop_stopping')}")
            self._shutdown_background_pool()
            self._close_display()

    def _build_status_content(
        self,
        snapshot: RuntimeSnapshot | None,
        *,
        status: str | None = None,
        stale: bool = False,
        health: TwinrSystemHealth | None | object = _UNSET,
        internet_ok: bool | None | object = _UNSET,
    ) -> tuple[str, tuple[str, ...]]:
        status = status or self._snapshot_status(snapshot)
        note = "Status veraltet" if stale else None
        if snapshot is None and not stale:
            note = "Status nicht verfügbar"
        health, internet_ok = self._resolve_health_context(
            snapshot,
            health=health,
            internet_ok=internet_ok,
        )
        details = self._detail_lines(
            note,
            *self._health_detail_values(snapshot, health=health, internet_ok=internet_ok),
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
                chunk = compact[:32]
                if len(compact) > 32:
                    split_at = chunk.rfind(" ")
                    if split_at >= 12:
                        chunk = chunk[:split_at]
                lines.append(chunk.rstrip())
                compact = compact[len(chunk):].lstrip()
        return tuple(lines[:4])

    def _animation_frame(self, status: str) -> int:
        frame_count, frame_seconds = self._animation_spec(status)
        if frame_count <= 1 or frame_seconds <= 0 or not math.isfinite(frame_seconds):
            return 0
        return int(time.monotonic() / frame_seconds) % frame_count

    def _animation_spec(self, status: str) -> tuple[int, float]:
        return _STATUS_ANIMATION_SPECS.get(self._compact_text(status).lower(), (1, 60.0))

    def _health_detail_values(
        self,
        snapshot: RuntimeSnapshot | None,
        *,
        health: TwinrSystemHealth | None,
        internet_ok: bool | None,
    ) -> tuple[str, str, str, str]:
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
        status: str | None = None,
        stale: bool = False,
        health: TwinrSystemHealth | None | object = _UNSET,
        internet_ok: bool | None | object = _UNSET,
    ) -> tuple[tuple[str, str], ...]:
        status = status or self._snapshot_status(snapshot)
        note = "Status veraltet" if stale else None
        if snapshot is None and not stale:
            note = "Status nicht verfügbar"
        health, internet_ok = self._resolve_health_context(
            snapshot,
            health=health,
            internet_ok=internet_ok,
        )
        state_fields = [
            ("Status", self._runtime_state_value(snapshot, status=status, stale=stale)),
            ("Internet", self._internet_state_value(internet_ok)),
            ("AI", self._ai_state_value(snapshot, health, internet_ok)),
            ("System", self._system_state_value(snapshot, health)),
        ]
        state_fields.extend(self._respeaker_state_fields())
        state_fields.append(("Zeit", self._clock_text()))
        if note:
            state_fields.append(("Hinweis", note))
        return tuple(state_fields)

    def _respeaker_state_fields(self) -> list[tuple[str, str]]:
        """Return calm operator-visible ReSpeaker HCI fields when relevant."""
        store = self.respeaker_hci_store
        if store is None:
            return []
        try:
            state = store.load()
        except Exception as exc:
            self._emit_error("display_respeaker_hci_load_failed", exc)
            return []
        if state is None:
            return []
        return list(state.state_fields())

    def _build_log_sections(
        self,
        snapshot: RuntimeSnapshot | None,
        *,
        status: str,
        stale: bool = False,
        health: TwinrSystemHealth | None | object = _UNSET,
        internet_ok: bool | None | object = _UNSET,
    ) -> LogSections:
        if self._display_layout() != "debug_log":
            return ()
        health, internet_ok = self._resolve_health_context(
            snapshot,
            health=health,
            internet_ok=internet_ok,
        )
        return self._debug_log_builder().build_sections(
            snapshot=snapshot,
            runtime_status=self._runtime_state_value(snapshot, status=status, stale=stale),
            internet_state=self._internet_state_value(internet_ok),
            ai_state=self._ai_state_value(snapshot, health, internet_ok),
            system_state=self._system_state_value(snapshot, health),
            clock_text=self._clock_text(),
            health=health,
            stale=stale,
        )

    def _resolve_health_context(
        self,
        snapshot: RuntimeSnapshot | None,
        *,
        health: TwinrSystemHealth | None | object = _UNSET,
        internet_ok: bool | None | object = _UNSET,
    ) -> tuple[TwinrSystemHealth | None, bool | None]:
        resolved_health = (
            self._current_health(snapshot)
            if health is _UNSET
            else cast(TwinrSystemHealth | None, health)
        )
        resolved_internet_ok = (
            self._internet_ok()
            if internet_ok is _UNSET
            else cast(bool | None, internet_ok)
        )
        return resolved_health, resolved_internet_ok

    def _current_health(self, snapshot: RuntimeSnapshot | None) -> TwinrSystemHealth | None:
        now = time.monotonic()
        snapshot_error = self._snapshot_error_message(snapshot)
        snapshot_status = self._snapshot_status(snapshot)
        self._drain_health_future()

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

        if self._cached_health is None:
            try:
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

        if self._pending_health_future is None:
            try:
                self._pending_health_future = self._background_executor().submit(
                    self.health_collector,
                    self.config,
                    snapshot=snapshot,
                )
                self._pending_health_error = snapshot_error
                self._pending_health_status = snapshot_status
            except Exception as exc:
                self._emit_error("health_collect_submit_failed", exc)
                self._cached_health_error = snapshot_error
                self._cached_health_status = snapshot_status
                self._cached_health_at = now
        return self._cached_health

    def _drain_health_future(self) -> None:
        future = self._pending_health_future
        if future is None or not future.done():
            return
        self._pending_health_future = None
        pending_error = self._pending_health_error
        pending_status = self._pending_health_status
        self._pending_health_error = None
        self._pending_health_status = None
        try:
            health = future.result()
        except Exception as exc:
            self._emit_error("health_collect_async_failed", exc)
            self._cached_health_error = pending_error
            self._cached_health_status = pending_status
            self._cached_health_at = time.monotonic()
            return
        self._cached_health = health
        self._cached_health_error = pending_error
        self._cached_health_status = pending_status
        self._cached_health_at = time.monotonic()

    def _internet_ok(self) -> bool | None:
        now = time.monotonic()
        self._drain_internet_future()
        refresh_s = _INTERNET_FOOTER_REFRESH_S if self._cached_internet_ok is True else _DEGRADED_INTERNET_FOOTER_REFRESH_S
        if (now - self._cached_internet_at) < refresh_s:
            return self._cached_internet_ok

        if self._cached_internet_at <= 0.0:
            try:
                self._cached_internet_ok = bool(self.internet_probe())
            except Exception as exc:
                self._emit_error("internet_probe_failed", exc)
                self._cached_internet_ok = None
            self._cached_internet_at = now
            return self._cached_internet_ok

        if self._pending_internet_future is None:
            try:
                self._pending_internet_future = self._background_executor().submit(self.internet_probe)
            except Exception as exc:
                self._emit_error("internet_probe_submit_failed", exc)
                self._cached_internet_ok = None
                self._cached_internet_at = now
        return self._cached_internet_ok

    def _drain_internet_future(self) -> None:
        future = self._pending_internet_future
        if future is None or not future.done():
            return
        self._pending_internet_future = None
        try:
            self._cached_internet_ok = bool(future.result())
        except Exception as exc:
            self._emit_error("internet_probe_failed", exc)
            self._cached_internet_ok = None
        self._cached_internet_at = time.monotonic()

    def _background_executor(self) -> ThreadPoolExecutor:
        if self._background_pool is None:
            self._background_pool = ThreadPoolExecutor(
                max_workers=_BACKGROUND_WORKERS,
                thread_name_prefix="twinr-display",
            )
        return self._background_pool

    def _shutdown_background_pool(self) -> None:
        pool = self._background_pool
        if pool is None:
            return
        self._background_pool = None
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            self._emit_error("display_background_pool_shutdown_failed", exc)

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
        cpu_temperature_c = getattr(health, "cpu_temperature_c", None)
        temperature_warn = cpu_temperature_c is not None and cpu_temperature_c >= 72
        if health_status == "fail":
            return "Fehler"
        if not self._service_running(health, "conversation_loop"):
            return "Achtung"
        memory_status = assess_memory_pressure_status(
            memory_available_mb=getattr(health, "memory_available_mb", None),
            memory_used_percent=getattr(health, "memory_used_percent", None),
        )
        if memory_status in {"warn", "fail"}:
            return "Achtung"
        if getattr(health, "disk_used_percent", None) is not None and health.disk_used_percent >= 85:
            return "Achtung"
        if health_status == "warn" and not temperature_warn:
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

    def _runtime_state_value(
        self,
        snapshot: RuntimeSnapshot | None,
        *,
        status: str | None = None,
        stale: bool = False,
    ) -> str:
        effective_status = status or self._effective_runtime_status(snapshot, stale=stale)
        if effective_status == "waiting":
            base = "Waiting"
        elif effective_status == "listening":
            base = "Listening"
        elif effective_status == "processing":
            base = "Processing"
        elif effective_status == "answering":
            base = "Answering"
        elif effective_status == "printing":
            base = "Printing"
        elif effective_status == "error":
            base = "Error"
        else:
            base = effective_status.replace("_", " ").title()
        raw_status = self._snapshot_status(snapshot)
        if stale and raw_status in _ACTIVE_RUNTIME_STATES:
            return f"{raw_status.title()} (stale)"
        return base

    def _service_running(self, health: TwinrSystemHealth | None, key: str) -> bool:
        if health is None:
            return False
        services = getattr(health, "services", ()) or ()
        for service in services:
            if getattr(service, "key", None) == key:
                return bool(getattr(service, "running", False))
        return True

    def _save_render_state(
        self,
        *,
        status: str,
        headline: str,
        details: tuple[str, ...],
        state_fields: tuple[tuple[str, str], ...],
        snapshot: RuntimeSnapshot | None,
        snapshot_stale: bool,
        health: TwinrSystemHealth | None,
    ) -> None:
        store = self.render_state_store
        if store is None:
            return
        try:
            store.save(
                layout=self._display_layout(),
                runtime_status=status,
                headline=headline,
                details=details,
                state_fields=state_fields,
                health_status=self._compact_text(getattr(health, "status", None)).lower() or None,
                snapshot_status=self._snapshot_status(snapshot) if snapshot is not None else None,
                snapshot_stale=snapshot_stale,
                snapshot_error=self._snapshot_error_message(snapshot),
                runtime_error=self._compact_text(getattr(health, "runtime_error", None)),
            )
        except Exception as exc:
            self._emit_error("display_render_state_save_failed", exc)

    def _load_snapshot(self) -> tuple[RuntimeSnapshot | None, bool]:
        try:
            snapshot = self.snapshot_store.load()
        except Exception as exc:
            self._emit_error("snapshot_load_failed", exc)
            self._pause(_SNAPSHOT_RETRY_DELAY_S)
            try:
                snapshot = self.snapshot_store.load()
            except Exception as retry_exc:
                self._emit_error("snapshot_load_retry_failed", retry_exc)
                return self._last_snapshot, self._last_snapshot is not None
        self._last_snapshot = snapshot
        return snapshot, self._snapshot_is_stale(snapshot)

    def _show_status(
        self,
        status: str,
        *,
        headline: str,
        ticker_text: str | None,
        details: tuple[str, ...],
        state_fields: tuple[tuple[str, str], ...],
        log_sections: LogSections,
        animation_frame: int,
        face_cue: DisplayFaceCue | None,
        emoji_cue: DisplayEmojiCue | None,
        ambient_impulse_cue: DisplayAmbientImpulseCue | None,
        service_connect_cue: DisplayServiceConnectCue | None,
        presentation_cue: DisplayPresentationCue | None,
        debug_signals: tuple[DisplayDebugSignal, ...],
    ) -> bool:
        try:
            self.display.show_status(
                status,
                headline=headline,
                ticker_text=ticker_text,
                details=details,
                state_fields=state_fields,
                log_sections=log_sections,
                animation_frame=animation_frame,
                face_cue=face_cue,
                emoji_cue=emoji_cue,
                ambient_impulse_cue=ambient_impulse_cue,
                service_connect_cue=service_connect_cue,
                presentation_cue=presentation_cue,
                debug_signals=debug_signals,
            )
            return True
        except Exception as exc:
            self._emit_error("display_show_failed", exc)

        if self._reopen_display():
            try:
                self.display.show_status(
                    status,
                    headline=headline,
                    ticker_text=ticker_text,
                    details=details,
                    state_fields=state_fields,
                    log_sections=log_sections,
                    animation_frame=animation_frame,
                    face_cue=face_cue,
                    emoji_cue=emoji_cue,
                    ambient_impulse_cue=ambient_impulse_cue,
                    service_connect_cue=service_connect_cue,
                    presentation_cue=presentation_cue,
                    debug_signals=debug_signals,
                )
                self._force_render_next_cycle = False
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
        now_monotonic = time.monotonic()
        if not force and (now_monotonic - self._last_heartbeat_at) < _HEARTBEAT_IDLE_REFRESH_S:
            self._maybe_systemd_watchdog(status=runtime_status, phase=phase, detail=detail)
            return
        self._heartbeat_seq += 1
        if store is not None:
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
            else:
                self._last_heartbeat_at = now_monotonic
        else:
            self._last_heartbeat_at = now_monotonic
        self._systemd_notify_status(runtime_status, phase=phase, detail=detail)
        self._maybe_systemd_watchdog(status=runtime_status, phase=phase, detail=detail)

    def _reopen_display(self) -> bool:
        last_rendered_status = getattr(self.display, "_last_rendered_status", None)
        self._close_display()
        factories: list[Callable[[TwinrConfig], TwinrDisplayAdapter]] = []
        display_factory = getattr(type(self.display), "from_config", None)
        if callable(display_factory):
            factories.append(display_factory)
        if create_display_adapter not in factories:
            factories.append(create_display_adapter)

        for factory in factories:
            try:
                reopened = self._build_display_from_factory(factory)
                if last_rendered_status is not None and hasattr(reopened, "_last_rendered_status"):
                    setattr(reopened, "_last_rendered_status", last_rendered_status)
                self.display = reopened
                self._last_display_telemetry_signature = None
                self._force_render_next_cycle = True
                return True
            except Exception as exc:
                self._emit_error("display_reopen_failed", exc)
        return False

    def _build_display_from_factory(
        self,
        factory: Callable[[TwinrConfig], TwinrDisplayAdapter],
    ) -> TwinrDisplayAdapter:
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            return factory(self.config)
        accepts_emit = False
        for name, parameter in signature.parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_emit = True
                break
            if name == "emit":
                accepts_emit = True
                break
        if accepts_emit:
            return factory(self.config, emit=self.emit)
        return factory(self.config)

    def _close_display(self) -> None:
        close = getattr(self.display, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                self._emit_error("display_close_failed", exc)

    def _tick_display(self) -> None:
        tick = getattr(self.display, "tick", None)
        if not callable(tick):
            return
        tick_callback = cast(Callable[[], None], tick)
        try:
            tick_callback()  # pylint: disable=not-callable
        except Exception as exc:
            self._emit_error("display_tick_failed", exc)
            self._reopen_display()

    def _sleep_once(
        self,
        *,
        status: str,
        ticker_text: str | None,
        presentation_cue: DisplayPresentationCue | None,
    ) -> None:
        poll_interval_s = self._effective_poll_interval_s(
            status=status,
            ticker_text=ticker_text,
            presentation_cue=presentation_cue,
        )
        self._pause(poll_interval_s)

    def _pause(self, duration_s: float) -> None:
        try:
            self.sleep(duration_s)
        except Exception as exc:
            self._emit_error("display_sleep_failed", exc)
            time.sleep(duration_s)

    def _effective_poll_interval_s(
        self,
        *,
        status: str,
        ticker_text: str | None,
        presentation_cue: DisplayPresentationCue | None,
    ) -> float:
        base_interval = self._poll_interval_s()
        if self._display_layout() != "default":
            return max(base_interval, 0.25)
        normalized_status = self._compact_text(status).lower()
        if normalized_status in _ACTIVE_RUNTIME_STATES:
            return base_interval
        if presentation_cue is not None:
            return base_interval
        if ticker_text:
            return base_interval
        if normalized_status == "waiting" and self._supports_idle_waiting_animation():
            return base_interval
        idle_interval = min(max(base_interval * _IDLE_POLL_MULTIPLIER, 0.25), _IDLE_POLL_MAX_S)
        return max(base_interval, idle_interval)

    def _poll_interval_s(self) -> float:
        raw_interval = getattr(self.config, "display_poll_interval_s", _MIN_DISPLAY_POLL_INTERVAL_S)
        try:
            interval_s = float(raw_interval)
        except (TypeError, ValueError):
            self._emit_error("display_poll_interval_invalid")
            return _MIN_DISPLAY_POLL_INTERVAL_S
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
        if self._compact_text(status).lower() == "waiting" and not self._supports_idle_waiting_animation():
            return 0
        return self._animation_frame(status)

    def _ticker_text(self) -> str | None:
        runtime = self.news_ticker_runtime
        if runtime is None:
            return None
        try:
            return runtime.current_text(now=self.clock())
        except Exception as exc:
            self._emit_error("display_news_ticker_failed", exc)
            return None

    def _supports_idle_waiting_animation(self) -> bool:
        capability = getattr(self.display, "supports_idle_waiting_animation", None)
        if not callable(capability):
            return False
        try:
            return bool(capability())
        except Exception as exc:
            self._emit_error("display_idle_waiting_animation_capability_failed", exc)
            return False

    def _render_signature(
        self,
        *,
        status: str,
        headline: str,
        ticker_text: str | None,
        details: tuple[str, ...],
        state_fields: tuple[tuple[str, str], ...],
        log_sections: LogSections,
        animation_frame: int,
        face_cue: DisplayFaceCue | None,
        emoji_cue: DisplayEmojiCue | None,
        ambient_impulse_cue: DisplayAmbientImpulseCue | None,
        service_connect_cue: DisplayServiceConnectCue | None,
        presentation_cue: DisplayPresentationCue | None,
        debug_signals: tuple[DisplayDebugSignal, ...] = (),
    ) -> tuple[object, ...]:
        layout_mode = self._display_layout()
        if layout_mode == "debug_log":
            return (layout_mode, status, headline, log_sections)
        cue_signature = face_cue.signature() if face_cue is not None else None
        reserve_signature = resolve_display_reserve_bus(
            service_connect_cue=service_connect_cue,
            emoji_cue=emoji_cue,
            ambient_impulse_cue=ambient_impulse_cue,
        ).signature()
        presentation_signature = presentation_cue.signature() if presentation_cue is not None else None
        presentation_bucket = presentation_cue.transition_bucket() if presentation_cue is not None else None
        return (
            layout_mode,
            status,
            headline,
            ticker_text,
            details,
            state_fields,
            log_sections,
            animation_frame,
            cue_signature,
            reserve_signature,
            presentation_signature,
            presentation_bucket,
            tuple(signal.signature() for signal in debug_signals),
        )

    def _should_render_signature(
        self,
        *,
        status: str,
        signature: tuple[object, ...],
        last_signature: tuple[object, ...] | None,
    ) -> bool:
        if self._force_render_next_cycle:
            return True
        if signature == last_signature:
            return self._maintenance_refresh_due()
        if self._display_layout() != "debug_log":
            return True
        if last_signature is None or status != self._last_render_status:
            return True
        if self._maintenance_refresh_due():
            return True
        return (time.monotonic() - self._last_render_monotonic_s) >= _DEBUG_LOG_MIN_REFRESH_S

    def _maintenance_refresh_due(self) -> bool:
        interval_s = self._maintenance_refresh_interval_s()
        if interval_s <= 0 or self._last_render_monotonic_s <= 0:
            return False
        return (time.monotonic() - self._last_render_monotonic_s) >= interval_s

    def _maintenance_refresh_interval_s(self) -> float:
        configured = getattr(self.config, "display_maintenance_refresh_s", None)
        if configured is not None:
            try:
                interval_s = float(configured)
            except (TypeError, ValueError):
                self._emit_error("display_maintenance_refresh_invalid")
                interval_s = 0.0
            if math.isfinite(interval_s) and interval_s > 0:
                return interval_s
        driver = self._compact_text(getattr(self.config, "display_driver", None)).lower()
        if driver.startswith("waveshare"):
            return float(_WAVESHARE_MAINTENANCE_REFRESH_S)
        return 0.0

    def _display_layout(self) -> str:
        return self._compact_text(getattr(self.config, "display_layout", "default")).lower() or "default"

    def _emit_display_status(
        self,
        *,
        status: str,
        presentation_cue: DisplayPresentationCue | None,
    ) -> None:
        signature = self._display_telemetry_signature(status=status, presentation_cue=presentation_cue)
        if signature == self._last_display_telemetry_signature:
            return
        self._last_display_telemetry_signature = signature
        safe_status = self._telemetry_token(status, fallback="error")
        parts = [f"display_status={safe_status}"]
        layout_mode = self._display_layout()
        if layout_mode != "default":
            parts.append(f"layout={self._telemetry_token(layout_mode, fallback='default')}")
        if presentation_cue is not None:
            presentation_signature = presentation_cue.telemetry_signature()
            active_card = presentation_cue.active_card()
            if presentation_signature is not None and active_card is not None:
                parts.append(
                    "presentation="
                    f"{self._telemetry_token(active_card.kind, fallback='card')}:"
                    f"{self._telemetry_token(presentation_cue.transition_stage(), fallback='idle')}"
                )
                parts.append(
                    "presentation_key="
                    f"{self._telemetry_token(active_card.key, fallback='unknown')}"
                )
                queued_count = len(presentation_cue.queued_cards())
                if queued_count > 0:
                    parts.append(f"presentation_queue={queued_count}")
        self._safe_emit(" ".join(parts))

    def _telemetry_token(
        self,
        value: object | None,
        *,
        max_len: int = _TELEMETRY_TOKEN_MAX_LEN,
        fallback: str | None = None,
    ) -> str | None:
        text = self._compact_text(value, max_len=max_len * 2)
        if not text:
            return fallback
        token = "".join(
            char if char.isalnum() or char in "-_.:/" else "_"
            for char in text
        )
        token = token[:max_len].strip("_.")
        return token or fallback

    def _display_telemetry_signature(
        self,
        *,
        status: str,
        presentation_cue: DisplayPresentationCue | None,
    ) -> tuple[object, ...]:
        layout_mode = self._display_layout()
        presentation_signature = presentation_cue.telemetry_signature() if presentation_cue is not None else None
        return (layout_mode, status, presentation_signature)

    def _active_face_cue(self, *, snapshot: RuntimeSnapshot | None = None) -> DisplayFaceCue | None:
        if self._display_layout() != "default":
            return None
        if self._snapshot_voice_quiet_active(snapshot):
            return _VOICE_QUIET_FACE_CUE
        store = self.face_cue_store
        if store is None:
            return None
        try:
            return store.load_active()
        except Exception as exc:
            self._emit_error("display_face_cue_load_failed", exc)
            return None

    def _active_emoji_cue(self) -> DisplayEmojiCue | None:
        if self._display_layout() != "default":
            return None
        store = self.emoji_cue_store
        if store is None:
            return None
        try:
            return store.load_active()
        except Exception as exc:
            self._emit_error("display_emoji_cue_load_failed", exc)
            return None

    def _active_service_connect_cue(self) -> DisplayServiceConnectCue | None:
        if self._display_layout() != "default":
            return None
        store = self.service_connect_cue_store
        if store is None:
            return None
        try:
            return store.load_active()
        except Exception as exc:
            self._emit_error("display_service_connect_cue_load_failed", exc)
            return None

    def _active_presentation_cue(self) -> DisplayPresentationCue | None:
        if self._display_layout() != "default":
            return None
        store = self.presentation_cue_store
        if store is None:
            return None
        try:
            return store.load_active()
        except Exception as exc:
            self._emit_error("display_presentation_cue_load_failed", exc)
            return None

    def _active_ambient_impulse_cue(self) -> DisplayAmbientImpulseCue | None:
        if self._display_layout() != "default":
            return None
        store = self.ambient_impulse_cue_store
        if store is None:
            return None
        try:
            return store.load_active()
        except Exception as exc:
            self._emit_error("display_ambient_impulse_cue_load_failed", exc)
            return None

    def _active_debug_signals(self) -> tuple[DisplayDebugSignal, ...]:
        if self._display_layout() != "default":
            return ()
        store = self.debug_signal_store
        if store is None:
            return ()
        try:
            snapshot = store.load_active()
        except Exception as exc:
            self._emit_error("display_debug_signal_load_failed", exc)
            return ()
        if snapshot is None:
            return ()
        return snapshot.signals

    def _debug_log_builder(self) -> TwinrDisplayDebugLogBuilder:
        if self.debug_log_builder is None:
            self.debug_log_builder = TwinrDisplayDebugLogBuilder.from_config(self.config)
        return self.debug_log_builder

    def _snapshot_status(self, snapshot: RuntimeSnapshot | None) -> str:
        if snapshot is None:
            return "error"
        compact = self._compact_text(getattr(snapshot, "status", None), max_len=_STATUS_TEXT_MAX_LEN).lower()
        return compact or "error"

    def _effective_runtime_status(self, snapshot: RuntimeSnapshot | None, *, stale: bool) -> str:
        status = self._snapshot_status(snapshot)
        if stale and status in _ACTIVE_RUNTIME_STATES:
            return "error"
        return status

    def _snapshot_error_message(self, snapshot: RuntimeSnapshot | None) -> str:
        if snapshot is None:
            return ""
        return self._compact_text(getattr(snapshot, "error_message", None), max_len=_EMIT_LINE_MAX_LEN)

    def _snapshot_updated_at_utc(self, snapshot: RuntimeSnapshot | None) -> datetime | None:
        if snapshot is None:
            return None
        raw_updated_at = self._compact_text(getattr(snapshot, "updated_at", None), max_len=64)
        if not raw_updated_at:
            return None
        normalized = raw_updated_at[:-1] + "+00:00" if raw_updated_at.endswith("Z") else raw_updated_at
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _snapshot_is_stale(self, snapshot: RuntimeSnapshot | None) -> bool:
        if snapshot is None:
            return False
        status = self._snapshot_status(snapshot)
        if status not in _ACTIVE_RUNTIME_STATES:
            return False
        updated_at = self._snapshot_updated_at_utc(snapshot)
        if updated_at is None:
            return True
        age_s = max(0.0, (self._current_time_utc() - updated_at).total_seconds())
        return age_s > self._active_snapshot_stale_after_s()

    def _active_snapshot_stale_after_s(self) -> float:
        configured = getattr(self.config, "display_active_snapshot_stale_after_s", None)
        if configured is not None:
            try:
                interval_s = float(configured)
            except (TypeError, ValueError):
                self._emit_error("display_active_snapshot_stale_after_invalid")
            else:
                if math.isfinite(interval_s) and interval_s > 0:
                    return interval_s
        return max(_ACTIVE_STATUS_SNAPSHOT_STALE_AFTER_S, self._poll_interval_s() * 20.0)

    def _snapshot_voice_quiet_active(self, snapshot: RuntimeSnapshot | None) -> bool:
        if self._snapshot_status(snapshot) != "waiting":
            return False
        deadline = self._snapshot_voice_quiet_deadline_utc(snapshot)
        if deadline is None:
            return False
        return deadline > self._current_time_utc()

    def _snapshot_voice_quiet_deadline_utc(self, snapshot: RuntimeSnapshot | None) -> datetime | None:
        if snapshot is None:
            return None
        raw_deadline = self._compact_text(getattr(snapshot, "voice_quiet_until_utc", None), max_len=64)
        if not raw_deadline:
            return None
        normalized = raw_deadline[:-1] + "+00:00" if raw_deadline.endswith("Z") else raw_deadline
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _current_time_utc(self) -> datetime:
        try:
            current = self.clock()
        except Exception:
            return datetime.now(timezone.utc)
        if not isinstance(current, datetime):
            return datetime.now(timezone.utc)
        if current.tzinfo is None:
            return current.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc)

    def _resolve_systemd_notify_socket(self) -> str | None:
        return self._compact_text(os.environ.get("NOTIFY_SOCKET"), max_len=256) or None

    def _resolve_systemd_watchdog_interval_s(self) -> float | None:
        if self._systemd_notify_socket is None:
            return None
        raw_value = self._compact_text(os.environ.get("WATCHDOG_USEC"), max_len=32)
        if not raw_value or not raw_value.isdigit():
            return None
        watchdog_usec = int(raw_value)
        if watchdog_usec <= 0:
            return None
        return max(1.0, watchdog_usec / 2_000_000.0)

    def _systemd_notify(self, *fields: str) -> None:
        notify_socket = self._systemd_notify_socket
        if not notify_socket:
            return
        payload = "\n".join(field for field in fields if field).encode("utf-8", errors="replace")
        if not payload:
            return
        address: str | bytes = notify_socket
        if notify_socket.startswith("@"):
            address = b"\0" + notify_socket[1:].encode("utf-8", errors="replace")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify_sock:
                notify_sock.connect(address)
                notify_sock.sendall(payload)
        except Exception as exc:
            self._emit_error("systemd_notify_failed", exc)

    def _systemd_status_text(self, status: str, phase: str, detail: str | None = None) -> str:
        parts = [
            "display-loop",
            f"phase={self._telemetry_token(phase, fallback='idle')}",
            f"status={self._telemetry_token(status, fallback='error')}",
        ]
        safe_detail = self._telemetry_token(detail, max_len=64)
        if safe_detail:
            parts.append(f"detail={safe_detail}")
        return self._compact_text(" ".join(parts), max_len=_EMIT_LINE_MAX_LEN)

    def _systemd_notify_status(self, status: str, *, phase: str, detail: str | None = None) -> None:
        self._systemd_notify(f"STATUS={self._systemd_status_text(status, phase, detail)}")

    def _maybe_systemd_watchdog(self, *, status: str, phase: str, detail: str | None = None) -> None:
        interval_s = self._systemd_watchdog_interval_s
        if interval_s is None:
            return
        now = time.monotonic()
        if self._last_systemd_watchdog_at and (now - self._last_systemd_watchdog_at) < interval_s:
            return
        self._last_systemd_watchdog_at = now
        self._systemd_notify(
            "WATCHDOG=1",
            f"STATUS={self._systemd_status_text(status, phase, detail)}",
        )

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
