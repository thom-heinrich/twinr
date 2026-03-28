# CHANGELOG: 2026-03-27
# BUG-1: Respected check_required_remote_dependency(force_sync=True); startup no longer ignores the caller's explicit request for a direct required-remote readiness probe.
# BUG-2: Made degraded-startup telemetry best-effort; a broken ops event sink no longer crashes an otherwise recoverable boot.
# BUG-3: Serialized shutdown snapshot capture against the runtime flow lock when available; this closes a real race with in-flight turn/state mutations.
# SEC-1: Hardened file-backed state path preflight to reject symlinked parent chains and non-regular targets before bootstrap touches runtime-owned storage paths.
# IMP-1: Added supervisor-grade lifecycle metadata plus a redacted health_snapshot() surface that separates conversation state from runtime readiness/degraded/stopping phases.
# IMP-2: Added optional native systemd sd_notify READY/STATUS/STOPPING/WATCHDOG support for modern Raspberry Pi service deployments without introducing a new dependency.
# IMP-3: Upgraded shutdown to bounded fan-out with isolated daemon worker threads so one hung component does not starve every later flush/close step.

"""Bootstrap runtime-owned services and coordinate shutdown cleanup."""

from __future__ import annotations

import logging
import math
import os
import socket
import stat
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from twinr.agent.base_agent.conversation.adaptive_timing import AdaptiveTimingStore
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshotStore
from twinr.agent.base_agent.state.machine import TwinrStateMachine, TwinrStatus
from twinr.automations import AutomationStore
from twinr.memory import LongTermMemoryService, OnDeviceMemory, TwinrPersonalGraphStore
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.memory.reminders import ReminderStore
from twinr.ops.events import TwinrOpsEventStore
from twinr.proactive import ProactiveGovernor


_LOGGER = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_S = 2.0
_STARTUP_CLEANUP_TIMEOUT_S = 1.0
_MAX_STATUS_TEXT_LEN = 240


class TwinrLifecyclePhase(StrEnum):
    """Describe coarse runtime lifecycle phases distinct from conversation state."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(slots=True)
class TwinrRuntimeBase:
    """Own runtime bootstrap, shared stores, and orderly shutdown."""

    config: TwinrConfig = field(repr=False)  # AUDIT-FIX(#2): Kein Secret-/PII-Leak via auto-generated dataclass repr.
    state_machine: TwinrStateMachine = field(default_factory=TwinrStateMachine)
    memory: OnDeviceMemory = field(init=False, repr=False)  # AUDIT-FIX(#2): Runtime-internes Memory nicht in Logs repräsentieren.
    graph_memory: TwinrPersonalGraphStore = field(init=False, repr=False)  # AUDIT-FIX(#2): Memory-Backends tragen potenziell PII.
    long_term_memory: LongTermMemoryService = field(init=False, repr=False)  # AUDIT-FIX(#2): Runtime-Service nicht versehentlich ausloggen.
    reminder_store: ReminderStore = field(init=False, repr=False)  # AUDIT-FIX(#2): Store-Objekte nicht in repr exponieren.
    automation_store: AutomationStore = field(init=False, repr=False)  # AUDIT-FIX(#2): Store-Objekte nicht in repr exponieren.
    adaptive_timing_store: AdaptiveTimingStore = field(init=False, repr=False)  # AUDIT-FIX(#2): Interner Store bleibt privat.
    snapshot_store: RuntimeSnapshotStore = field(init=False, repr=False)  # AUDIT-FIX(#2): Snapshot-Store bleibt privat.
    ops_events: TwinrOpsEventStore = field(init=False, repr=False)  # AUDIT-FIX(#2): Ops-Events nicht über repr leaken.
    proactive_governor: ProactiveGovernor = field(init=False, repr=False)  # AUDIT-FIX(#2): Interne Governor-Instanz privat halten.
    last_transcript: str | None = field(default=None, repr=False)  # AUDIT-FIX(#2): Letzte Nutzeraussage nicht via repr leaken.
    last_response: str | None = field(default=None, repr=False)  # AUDIT-FIX(#2): Letzte Systemantwort nicht via repr leaken.
    user_voice_status: str | None = field(default=None, repr=False)  # AUDIT-FIX(#2): Voice-Metadaten sind privat.
    user_voice_confidence: float | None = field(default=None, repr=False)  # AUDIT-FIX(#2): Voice-Metadaten sind privat.
    user_voice_checked_at: str | None = field(default=None, repr=False)  # AUDIT-FIX(#2): Zeitstempel sind privat.
    user_voice_user_id: str | None = field(default=None, repr=False)  # AUDIT-FIX(#2): Enrolled local household match ids stay private.
    user_voice_user_display_name: str | None = field(default=None, repr=False)  # AUDIT-FIX(#2): Household display names are private identity data.
    user_voice_match_source: str | None = field(default=None, repr=False)  # AUDIT-FIX(#2): Internal match-source metadata stays private.
    lifecycle_phase: TwinrLifecyclePhase = field(
        init=False,
        repr=False,
        default=TwinrLifecyclePhase.INITIALIZING,
    )  # IMP-1: Runtime-Lifecycle separat vom Gesprächsstatus exponieren.
    startup_degraded_reason: str | None = field(
        init=False,
        repr=False,
        default=None,
    )  # IMP-1: Betreiberfreundliche Degraded-Ursache ohne PII-Content.
    startup_failed_reason: str | None = field(
        init=False,
        repr=False,
        default=None,
    )  # IMP-1: Betreiberfreundliche Failure-Ursache ohne PII-Content.
    boot_started_monotonic: float = field(
        init=False,
        repr=False,
        default_factory=time.monotonic,
    )  # IMP-1: Spätere Health/Shutdown-Metriken ohne Wanduhr-Abhängigkeit.
    last_shutdown_duration_s: float | None = field(
        init=False,
        repr=False,
        default=None,
    )  # IMP-3: Bounded-Shutdown-Metrik für Supervisor und Ops.
    _shutdown_lock: threading.RLock = field(
        init=False,
        repr=False,
        default_factory=threading.RLock,
    )  # AUDIT-FIX(#6): Shutdown idempotent und thread-safe machen.
    _shutdown_started: bool = field(
        init=False,
        repr=False,
        default=False,
    )  # AUDIT-FIX(#6): Doppelte Shutdown-Aufrufe sauber abfangen.
    _systemd_watchdog_stop: threading.Event = field(
        init=False,
        repr=False,
        default_factory=threading.Event,
    )  # IMP-2: Watchdog-Loop kontrolliert stoppen.
    _systemd_watchdog_thread: threading.Thread | None = field(
        init=False,
        repr=False,
        default=None,
    )  # IMP-2: Optionaler systemd-Watchdog-Pinger.

    def __post_init__(self) -> None:
        long_term_startup_warning: str | None = None
        long_term_startup_error: str | None = None
        self._notify_systemd_status("Bootstrapping Twinr runtime.")
        try:
            reminder_store_path = self._resolve_runtime_owned_path(self.config.reminder_store_path)
            automation_store_path = self._resolve_runtime_owned_path(self.config.automation_store_path)
            adaptive_timing_store_path = self._resolve_runtime_owned_path(self.config.adaptive_timing_store_path)
            runtime_state_path = self._resolve_runtime_owned_path(self.config.runtime_state_path)

            # SEC-1: Zentrale Safe-Path-Preflight für alle file-backed Runtime-Artefakte.
            self._ensure_file_backed_target(reminder_store_path)
            self._ensure_file_backed_target(automation_store_path)
            self._ensure_file_backed_target(adaptive_timing_store_path)
            self._ensure_file_backed_target(runtime_state_path)

            self.memory = OnDeviceMemory(
                max_turns=self.config.memory_max_turns,
                keep_recent=self.config.memory_keep_recent,
            )
            self.graph_memory = TwinrPersonalGraphStore.from_config(self.config)
            self.long_term_memory = LongTermMemoryService.from_config(
                self.config,
                graph_store=self.graph_memory,
            )
            try:
                self.check_required_remote_dependency(force_sync=True)
            except LongTermRemoteUnavailableError as exc:
                detail = self._sanitize_status_text(exc, default="Required remote long-term memory is unavailable.")
                if self.remote_dependency_required():
                    long_term_startup_error = detail
                    _LOGGER.error(
                        "Twinr runtime startup entered error because required remote long-term memory is unavailable: %s",
                        long_term_startup_error,
                    )
                else:
                    long_term_startup_warning = detail
                    _LOGGER.warning(
                        "Twinr runtime startup degraded because remote long-term memory is unavailable: %s",
                        long_term_startup_warning,
                    )
            except Exception as exc:
                detail = self._sanitize_status_text(
                    exc,
                    default="Remote long-term memory readiness check failed unexpectedly.",
                )
                if self.remote_dependency_required():
                    long_term_startup_error = detail
                    _LOGGER.exception(
                        "Twinr runtime startup entered error because the required remote long-term memory check crashed.",
                    )
                else:
                    long_term_startup_warning = detail
                    _LOGGER.exception(
                        "Twinr runtime startup degraded because the optional remote long-term memory check crashed.",
                    )

            self.reminder_store = ReminderStore(
                reminder_store_path,
                timezone_name=self.config.local_timezone_name,
                retry_delay_s=self.config.reminder_retry_delay_s,
                max_entries=self.config.reminder_max_entries,
            )
            self.automation_store = AutomationStore(
                automation_store_path,
                timezone_name=self.config.local_timezone_name,
                max_entries=self.config.automation_max_entries,
            )
            self.adaptive_timing_store = AdaptiveTimingStore(
                adaptive_timing_store_path,
                config=self.config,
            )
            if self.config.adaptive_timing_enabled:
                try:
                    # AUDIT-FIX(#4): Fehler beim optionalen Persistieren von Adaptive Timing dürfen den Boot nicht abbrechen.
                    self.adaptive_timing_store.ensure_saved()
                except Exception:
                    _LOGGER.exception(
                        "Adaptive timing state could not be written during startup; continuing without persisted adaptive timing data.",
                    )

            self.snapshot_store = RuntimeSnapshotStore(runtime_state_path)
            self.ops_events = TwinrOpsEventStore.from_config(self.config)
            self.proactive_governor = ProactiveGovernor.from_config(self.config)

            if long_term_startup_warning:
                self.startup_degraded_reason = long_term_startup_warning
                self._append_ops_event_safe(
                    event="longterm_startup_degraded",
                    message="Twinr started without remote long-term memory because the remote snapshot was unavailable.",
                    data={"detail": long_term_startup_warning},
                )

            if self.config.restore_runtime_state_on_startup:
                restore_snapshot_context = getattr(self, "_restore_snapshot_context", None)
                if callable(restore_snapshot_context):
                    restore_snapshot_context()  # pylint: disable=not-callable

            if long_term_startup_error:
                self.startup_failed_reason = long_term_startup_error
                self.lifecycle_phase = TwinrLifecyclePhase.FAILED
                fail = getattr(self, "fail", None)
                if callable(fail):
                    fail_fn = fail
                    fail_fn(long_term_startup_error)  # pylint: disable=not-callable
                else:
                    self.state_machine.fail(long_term_startup_error)
                    persist_snapshot = getattr(self, "_persist_snapshot", None)
                    if callable(persist_snapshot):
                        persist_snapshot()  # pylint: disable=not-callable
            else:
                self.lifecycle_phase = (
                    TwinrLifecyclePhase.DEGRADED
                    if long_term_startup_warning
                    else TwinrLifecyclePhase.RUNNING
                )
                persist_snapshot = getattr(self, "_persist_snapshot", None)
                if callable(persist_snapshot):
                    persist_snapshot()  # pylint: disable=not-callable

            self._start_systemd_watchdog_if_enabled()
            self._notify_systemd_ready()
        except Exception as exc:
            self.startup_failed_reason = self._sanitize_status_text(exc, default="Twinr bootstrap failed.")
            self.lifecycle_phase = TwinrLifecyclePhase.FAILED
            self._notify_systemd_status(self._build_systemd_status_line())
            _LOGGER.exception(
                "Failed to initialize TwinrRuntimeBase; cleaning up partially initialized components.",
            )
            self._shutdown_components(timeout_s=_STARTUP_CLEANUP_TIMEOUT_S)  # AUDIT-FIX(#3): Partielle Initialisierung sauber zurückbauen.
            self._stop_systemd_watchdog_thread(join_timeout_s=0.0)
            raise

    @property
    def status(self) -> TwinrStatus:
        """Return the current runtime status from the state machine."""

        status_override = getattr(self, "_runtime_visible_status_override", None)
        if callable(status_override):
            status_override_fn = status_override
            override = status_override_fn()  # pylint: disable=not-callable
            if isinstance(override, TwinrStatus):
                return override
        return self.state_machine.status

    @property
    def ready(self) -> bool:
        """Return whether the runtime finished boot and is able to serve requests."""

        return self.lifecycle_phase in {TwinrLifecyclePhase.RUNNING, TwinrLifecyclePhase.DEGRADED}

    @property
    def is_shutting_down(self) -> bool:
        """Return whether shutdown has started."""

        return self.lifecycle_phase in {TwinrLifecyclePhase.STOPPING, TwinrLifecyclePhase.STOPPED}

    def health_snapshot(self) -> dict[str, object]:
        """Expose one redacted, JSON-safe runtime health snapshot for dashboards/supervisors."""

        state_machine = getattr(self, "state_machine", None)
        lifecycle_phase = getattr(self, "lifecycle_phase", TwinrLifecyclePhase.INITIALIZING)
        return {
            "status": getattr(getattr(self, "status", None), "value", None),
            "lifecycle_phase": getattr(lifecycle_phase, "value", str(lifecycle_phase)),
            "ready": self.ready,
            "shutdown_started": bool(getattr(self, "_shutdown_started", False)),
            "startup_degraded_reason": self.startup_degraded_reason,
            "startup_failed_reason": self.startup_failed_reason,
            "remote_dependency_required": self.remote_dependency_required(),
            "last_error": self._sanitize_status_text(
                getattr(state_machine, "last_error", None),
                default="",
                max_len=200,
            )
            or None,
            "has_last_transcript": bool(getattr(self, "last_transcript", None)),
            "has_last_response": bool(getattr(self, "last_response", None)),
            "user_voice_status": self.user_voice_status,
            "user_voice_confidence": self.user_voice_confidence,
            "watchdog_enabled": self._systemd_watchdog_interval_s() is not None,
            "uptime_s": round(max(0.0, time.monotonic() - self.boot_started_monotonic), 3),
            "last_shutdown_duration_s": self.last_shutdown_duration_s,
        }

    def remote_dependency_required(self) -> bool:
        long_term_memory = getattr(self, "long_term_memory", None)
        remote_required = getattr(long_term_memory, "remote_required", None)
        if callable(remote_required):
            try:
                return bool(remote_required())
            except Exception:
                _LOGGER.exception("Failed to determine whether remote long-term memory is required.")
        return bool(
            self.config.long_term_memory_enabled
            and self.config.long_term_memory_mode == "remote_primary"
            and self.config.long_term_memory_remote_required
        )

    def _required_remote_dependency_uses_watchdog_artifact(self) -> bool:
        """Return whether required-remote checks must trust the external watchdog artifact."""

        mode = str(
            getattr(self.config, "long_term_memory_remote_runtime_check_mode", "direct") or "direct"
        ).strip().lower()
        return mode == "watchdog_artifact"

    def check_required_remote_dependency(self, *, force_sync: bool = False) -> None:
        if not self.remote_dependency_required():
            return

        long_term_memory = getattr(self, "long_term_memory", None)

        # BREAKING: force_sync=True now truly means "probe the live remote backend now"
        # even when the runtime-wide mode prefers watchdog artifacts for hot-path checks.
        # Callers that depended on the previous no-op force_sync behavior will now get the
        # fresher direct-read semantics they explicitly requested.
        if force_sync:
            if long_term_memory is None:
                raise LongTermRemoteUnavailableError("Required remote long-term memory is not initialized.")
            long_term_memory.ensure_remote_ready()
            return

        if self._required_remote_dependency_uses_watchdog_artifact():
            from twinr.agent.workflows.required_remote_snapshot import (
                ensure_required_remote_watchdog_snapshot_ready,
            )

            ensure_required_remote_watchdog_snapshot_ready(self.config)
            return

        if long_term_memory is None:
            raise LongTermRemoteUnavailableError("Required remote long-term memory is not initialized.")
        long_term_memory.ensure_remote_ready()

    def shutdown(self, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        """Persist runtime state and stop owned components best-effort."""

        shutdown_started = time.monotonic()
        timeout_s = self._coerce_timeout(timeout_s)  # AUDIT-FIX(#6): Negative/NaN-Timeouts hart normalisieren.
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
            self.lifecycle_phase = TwinrLifecyclePhase.STOPPING

        self._notify_systemd_stopping()

        # BUG-3: Snapshot shutdown state under the same flow lock used by turn mutations
        # so active state transitions cannot race with the final persisted context.
        with self._runtime_mutation_guard(timeout_s=min(0.5, timeout_s)):
            persist_snapshot = getattr(self, "_persist_snapshot", None)
            if callable(persist_snapshot):
                persist_snapshot()  # pylint: disable=not-callable

        self._shutdown_components(timeout_s=timeout_s)  # IMP-3: Alle bekannten Lifecycle-Komponenten bounded fan-out beenden.
        self._stop_systemd_watchdog_thread(join_timeout_s=min(0.25, timeout_s))
        self.last_shutdown_duration_s = round(max(0.0, time.monotonic() - shutdown_started), 6)
        self.lifecycle_phase = TwinrLifecyclePhase.STOPPED
        self._notify_systemd_status(self._build_systemd_status_line())

    @staticmethod
    def _coerce_path(value: object) -> Path:
        return Path(str(value)).expanduser()

    def _resolve_runtime_owned_path(self, value: object) -> Path:
        path = self._coerce_path(value)
        if path.is_absolute():
            return path.resolve(strict=False)
        project_root = self._coerce_path(self.config.project_root)
        return (project_root / path).resolve(strict=False)

    @classmethod
    def _ensure_file_backed_target(cls, value: object) -> Path:
        # BREAKING: symlinked runtime-state paths and parent directories are now
        # rejected during bootstrap instead of being tolerated implicitly.
        path = cls._coerce_path(value)
        cls._ensure_safe_directory_chain(path.parent)
        cls._assert_safe_file_target(path)
        return path

    @classmethod
    def _ensure_safe_directory_chain(cls, directory: Path) -> None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Unable to prepare runtime storage directory: {directory}") from exc

        current = directory
        while True:
            try:
                info = current.lstat()
            except OSError as exc:
                raise RuntimeError(f"Unable to inspect runtime storage directory: {current}") from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise NotADirectoryError(f"Runtime storage directory is not a safe directory: {current}")
            if current.parent == current:
                break
            current = current.parent

    @staticmethod
    def _assert_safe_file_target(path: Path) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise RuntimeError(f"Unable to inspect runtime storage target: {path}") from exc

        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"Runtime storage target is not a regular file: {path}")

    @staticmethod
    def _coerce_timeout(timeout_s: float) -> float:
        try:
            timeout = float(timeout_s)
        except (TypeError, ValueError):
            return _DEFAULT_TIMEOUT_S
        if not math.isfinite(timeout) or timeout <= 0.0:
            return _DEFAULT_TIMEOUT_S
        return timeout

    @staticmethod
    def _sanitize_status_text(
        value: object,
        *,
        default: str = "Runtime status unavailable.",
        max_len: int = _MAX_STATUS_TEXT_LEN,
    ) -> str:
        if value is None:
            text = default
        elif isinstance(value, str):
            text = value
        else:
            text = str(value)

        printable = "".join(character if character.isprintable() else " " for character in text)
        cleaned = " ".join(printable.split())
        if not cleaned:
            cleaned = default
        if len(cleaned) > max_len:
            cleaned = f"{cleaned[: max_len - 3].rstrip()}..."
        return cleaned

    @contextmanager
    def _runtime_mutation_guard(self, *, timeout_s: float | None = None):
        lock_factory = getattr(self, "_runtime_flow_lock", None)
        if not callable(lock_factory):
            yield
            return

        try:
            lock_factory_fn = lock_factory
            lock = lock_factory_fn()  # pylint: disable=not-callable
        except Exception:
            _LOGGER.exception("Failed to obtain the runtime flow lock; continuing without it.")
            yield
            return

        acquired = False
        acquire = getattr(lock, "acquire", None)
        release = getattr(lock, "release", None)
        if callable(acquire) and callable(release):
            try:
                if timeout_s is None:
                    acquired = bool(acquire())
                else:
                    acquired = bool(acquire(timeout=max(0.0, float(timeout_s))))
            except Exception:
                _LOGGER.exception("Failed while acquiring the runtime flow lock; continuing without it.")
            if not acquired:
                _LOGGER.warning(
                    "Timed out while waiting for the runtime flow lock during lifecycle work; continuing without the lock.",
                )
                yield
                return
            try:
                yield
            finally:
                try:
                    release()
                except Exception:
                    _LOGGER.exception("Failed while releasing the runtime flow lock.")
            return

        with nullcontext():
            yield

    def _append_ops_event_safe(self, **kwargs: object) -> None:
        ops_events = getattr(self, "ops_events", None)
        append = getattr(ops_events, "append", None)
        if not callable(append):
            return
        try:
            append(**kwargs)
        except Exception:
            _LOGGER.exception("Failed to append Twinr ops event %r during runtime bootstrap.", kwargs.get("event"))

    def _shutdown_components(self, *, timeout_s: float) -> None:
        adaptive_timing_store = getattr(self, "adaptive_timing_store", None)
        if adaptive_timing_store is not None and getattr(self.config, "adaptive_timing_enabled", False):
            ensure_saved = getattr(adaptive_timing_store, "ensure_saved", None)
            if callable(ensure_saved):
                try:
                    ensure_saved()  # AUDIT-FIX(#6): Adaptives Timing vor Exit explizit flushen.
                except Exception:
                    _LOGGER.exception("Failed to flush adaptive timing state during shutdown.")

        components: list[tuple[str, object]] = []
        for component_name in (
            "proactive_governor",
            "ops_events",
            "long_term_memory",
            "automation_store",
            "reminder_store",
            "graph_memory",
            "memory",
        ):
            component = getattr(self, component_name, None)
            if component is None:
                continue
            components.append((component_name, component))

        if not components:
            return

        deadline = time.monotonic() + timeout_s
        workers: list[tuple[str, threading.Thread]] = []
        for component_name, component in components:
            remaining = max(0.05, deadline - time.monotonic())
            worker = threading.Thread(
                target=self._shutdown_component,
                kwargs={
                    "component_name": component_name,
                    "component": component,
                    "timeout_s": remaining,
                },
                name=f"twinr-shutdown-{component_name}",
                daemon=True,
            )
            worker.start()
            workers.append((component_name, worker))

        for component_name, worker in workers:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0.0:
                _LOGGER.error(
                    "Runtime shutdown budget was exhausted before component '%s' finished.",
                    component_name,
                )
                continue
            try:
                worker.join(remaining)
            except Exception:
                _LOGGER.exception(
                    "Failed while waiting for component '%s' to finish shutdown.",
                    component_name,
                )
                continue
            if worker.is_alive():
                _LOGGER.error(
                    "Component '%s' did not finish shutdown inside the %.3fs global budget.",
                    component_name,
                    timeout_s,
                )

    def _shutdown_component(self, component_name: str, component: object, *, timeout_s: float) -> None:
        for method_name in ("shutdown", "close", "stop"):
            method = getattr(component, method_name, None)
            if not callable(method):
                continue
            try:
                if method_name == "shutdown":
                    try:
                        method(timeout_s=timeout_s)
                    except TypeError:
                        try:
                            method(timeout_s)
                        except TypeError:
                            method()
                else:
                    method()
            except Exception:
                _LOGGER.exception(
                    "Failed while shutting down component '%s' via %s().",
                    component_name,
                    method_name,
                )
            return

    @staticmethod
    def _systemd_notify_socket() -> str | None:
        raw_value = os.environ.get("NOTIFY_SOCKET", "").strip()
        return raw_value or None

    def _sd_notify(self, *assignments: str) -> bool:
        notify_socket = self._systemd_notify_socket()
        if not notify_socket:
            return False

        payload = "\n".join(
            str(assignment).strip()
            for assignment in assignments
            if isinstance(assignment, str) and assignment.strip()
        )
        if not payload:
            return False

        address = notify_socket
        if notify_socket.startswith("@"):
            address = "\0" + notify_socket[1:]

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
                client.connect(address)
                client.sendall(payload.encode("utf-8"))
            return True
        except Exception:
            _LOGGER.debug("Failed to send systemd notification payload.", exc_info=True)
            return False

    def _systemd_watchdog_interval_s(self) -> float | None:
        if self._systemd_notify_socket() is None:
            return None

        raw_pid = os.environ.get("WATCHDOG_PID", "").strip()
        if raw_pid:
            try:
                watchdog_pid = int(raw_pid)
            except ValueError:
                return None
            if watchdog_pid not in (0, os.getpid()):
                return None

        raw_usec = os.environ.get("WATCHDOG_USEC", "").strip()
        if not raw_usec:
            return None
        try:
            watchdog_usec = int(raw_usec)
        except ValueError:
            return None
        if watchdog_usec <= 0:
            return None

        return max(0.25, watchdog_usec / 2_000_000.0)

    def _build_systemd_status_line(self) -> str:
        phase = getattr(self, "lifecycle_phase", TwinrLifecyclePhase.INITIALIZING)
        status_value = getattr(getattr(self, "status", None), "value", "unknown")
        detail: str | None = None

        if phase == TwinrLifecyclePhase.DEGRADED:
            detail = self.startup_degraded_reason
        elif phase == TwinrLifecyclePhase.FAILED:
            detail = self.startup_failed_reason or getattr(self.state_machine, "last_error", None)

        if detail:
            return self._sanitize_status_text(
                f"Twinr {phase.value}; state={status_value}; detail={detail}",
                default="Twinr runtime state unavailable.",
            )
        return self._sanitize_status_text(
            f"Twinr {phase.value}; state={status_value}",
            default="Twinr runtime state unavailable.",
        )

    def _notify_systemd_status(self, text: object) -> None:
        sanitized = self._sanitize_status_text(text)
        self._sd_notify(
            f"STATUS={sanitized}",
            f"MAINPID={os.getpid()}",
        )

    def _notify_systemd_ready(self) -> None:
        self._sd_notify(
            "READY=1",
            f"STATUS={self._build_systemd_status_line()}",
            f"MAINPID={os.getpid()}",
        )

    def _notify_systemd_stopping(self) -> None:
        self._sd_notify(
            "STOPPING=1",
            f"STATUS={self._build_systemd_status_line()}",
            f"MAINPID={os.getpid()}",
        )

    def _start_systemd_watchdog_if_enabled(self) -> None:
        interval_s = self._systemd_watchdog_interval_s()
        if interval_s is None:
            return

        with self._shutdown_lock:
            existing = self._systemd_watchdog_thread
            if existing is not None and existing.is_alive():
                return
            self._systemd_watchdog_stop.clear()
            self._systemd_watchdog_thread = threading.Thread(
                target=self._systemd_watchdog_loop,
                args=(interval_s,),
                name="twinr-systemd-watchdog",
                daemon=True,
            )
            self._systemd_watchdog_thread.start()

    def _systemd_watchdog_loop(self, interval_s: float) -> None:
        # IMP-2: Native systemd watchdog support keeps the service supervisor-aware
        # even when the higher-level runtime is busy with I/O or edge-network stalls.
        self._sd_notify(
            "WATCHDOG=1",
            f"STATUS={self._build_systemd_status_line()}",
            f"MAINPID={os.getpid()}",
        )
        while not self._systemd_watchdog_stop.wait(interval_s):
            self._sd_notify(
                "WATCHDOG=1",
                f"STATUS={self._build_systemd_status_line()}",
                f"MAINPID={os.getpid()}",
            )

    def _stop_systemd_watchdog_thread(self, *, join_timeout_s: float) -> None:
        self._systemd_watchdog_stop.set()
        worker = self._systemd_watchdog_thread
        if worker is None:
            return
        try:
            if join_timeout_s > 0.0:
                worker.join(join_timeout_s)
        except Exception:
            _LOGGER.debug("Failed while joining the systemd watchdog thread.", exc_info=True)
        finally:
            self._systemd_watchdog_thread = None


__all__ = ["TwinrLifecyclePhase", "TwinrRuntimeBase"]
