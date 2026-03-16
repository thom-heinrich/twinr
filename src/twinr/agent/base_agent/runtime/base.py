"""Bootstrap runtime-owned services and coordinate shutdown cleanup."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
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

    def __post_init__(self) -> None:
        long_term_startup_warning: str | None = None
        long_term_startup_error: str | None = None
        try:
            # AUDIT-FIX(#3): Elternverzeichnisse der file-backed Stores vor Initialisierung sicherstellen.
            self._ensure_parent_directory(self.config.reminder_store_path)
            self._ensure_parent_directory(self.config.automation_store_path)
            self._ensure_parent_directory(self.config.adaptive_timing_store_path)
            self._ensure_parent_directory(self.config.runtime_state_path)

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
                if self.remote_dependency_required():
                    long_term_startup_error = str(exc)
                    _LOGGER.error(
                        "Twinr runtime startup entered error because required remote long-term memory is unavailable: %s",
                        long_term_startup_error,
                    )
                else:
                    long_term_startup_warning = str(exc)
                    _LOGGER.warning(
                        "Twinr runtime startup degraded because remote long-term memory is unavailable: %s",
                        long_term_startup_warning,
                    )
            self.reminder_store = ReminderStore(
                self.config.reminder_store_path,
                timezone_name=self.config.local_timezone_name,
                retry_delay_s=self.config.reminder_retry_delay_s,
                max_entries=self.config.reminder_max_entries,
            )
            self.automation_store = AutomationStore(
                self.config.automation_store_path,
                timezone_name=self.config.local_timezone_name,
                max_entries=self.config.automation_max_entries,
            )
            self.adaptive_timing_store = AdaptiveTimingStore(
                self.config.adaptive_timing_store_path,
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

            self.snapshot_store = RuntimeSnapshotStore(self.config.runtime_state_path)
            self.ops_events = TwinrOpsEventStore.from_config(self.config)
            self.proactive_governor = ProactiveGovernor.from_config(self.config)
            if long_term_startup_warning:
                self.ops_events.append(
                    event="longterm_startup_degraded",
                    message="Twinr started without remote long-term memory because the remote snapshot was unavailable.",
                    data={"detail": long_term_startup_warning},
                )

            if self.config.restore_runtime_state_on_startup:
                self._restore_snapshot_context()  # AUDIT-FIX(#1,#5): Snapshot-Restore implementiert und fehlertolerant.
            if long_term_startup_error:
                self.fail(long_term_startup_error)
            else:
                self._persist_snapshot()  # AUDIT-FIX(#1,#5): Snapshot-Persistierung implementiert und atomar.
        except Exception:
            _LOGGER.exception(
                "Failed to initialize TwinrRuntimeBase; cleaning up partially initialized components.",
            )
            self._shutdown_components(timeout_s=1.0)  # AUDIT-FIX(#3): Partielle Initialisierung sauber zurückbauen.
            raise

    @property
    def status(self) -> TwinrStatus:
        """Return the current runtime status from the state machine."""

        return self.state_machine.status

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

    def check_required_remote_dependency(self, *, force_sync: bool = False) -> None:
        if not self.remote_dependency_required():
            return
        long_term_memory = getattr(self, "long_term_memory", None)
        if long_term_memory is None:
            raise LongTermRemoteUnavailableError("Required remote long-term memory is not initialized.")
        long_term_memory.ensure_remote_ready()

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        """Persist runtime state and stop owned components best-effort."""

        timeout_s = self._coerce_timeout(timeout_s)  # AUDIT-FIX(#6): Negative/NaN-Timeouts hart normalisieren.
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True

        self._persist_snapshot()  # AUDIT-FIX(#6): Letzten Runtime-Kontext vor Shutdown atomar sichern.
        self._shutdown_components(timeout_s=timeout_s)  # AUDIT-FIX(#6): Alle bekannten Lifecycle-Komponenten best-effort beenden.

    @staticmethod
    def _coerce_path(value: object) -> Path:
        return Path(str(value)).expanduser()

    @classmethod
    def _ensure_parent_directory(cls, value: object) -> None:
        path = cls._coerce_path(value)
        parent = path.parent
        if parent.exists() and not parent.is_dir():
            raise NotADirectoryError(f"Parent path is not a directory: {parent}")
        parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _coerce_timeout(timeout_s: float) -> float:
        try:
            timeout = float(timeout_s)
        except (TypeError, ValueError):
            return 2.0
        if not math.isfinite(timeout) or timeout <= 0.0:
            return 2.0
        return timeout

    def _shutdown_components(self, *, timeout_s: float) -> None:
        adaptive_timing_store = getattr(self, "adaptive_timing_store", None)
        if adaptive_timing_store is not None and getattr(self.config, "adaptive_timing_enabled", False):
            ensure_saved = getattr(adaptive_timing_store, "ensure_saved", None)
            if callable(ensure_saved):
                try:
                    ensure_saved()  # AUDIT-FIX(#6): Adaptives Timing vor Exit explizit flushen.
                except Exception:
                    _LOGGER.exception("Failed to flush adaptive timing state during shutdown.")

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
            self._shutdown_component(component_name, component, timeout_s=timeout_s)

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
