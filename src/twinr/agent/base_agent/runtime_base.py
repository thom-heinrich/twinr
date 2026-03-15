from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.base_agent.adaptive_timing import AdaptiveTimingStore
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime_state import RuntimeSnapshotStore
from twinr.agent.base_agent.state_machine import TwinrStateMachine, TwinrStatus
from twinr.automations import AutomationStore
from twinr.memory import LongTermMemoryService, OnDeviceMemory, TwinrPersonalGraphStore
from twinr.memory.reminders import ReminderStore
from twinr.ops.events import TwinrOpsEventStore
from twinr.proactive import ProactiveGovernor


_LOGGER = logging.getLogger(__name__)
_MAX_RUNTIME_SNAPSHOT_BYTES = 128 * 1024
_QUIESCENT_RESTORABLE_STATUSES = frozenset(
    {"IDLE", "READY", "STANDBY", "WAITING", "SLEEP", "SLEEPING"}
)


@dataclass(slots=True)
class TwinrRuntimeBase:
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
    _snapshot_lock: threading.RLock = field(
        init=False,
        repr=False,
        default_factory=threading.RLock,
    )  # AUDIT-FIX(#5): Snapshot-I/O serialisieren, um Dateikorruption zu vermeiden.
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

            if self.config.restore_runtime_state_on_startup:
                self._restore_snapshot_context()  # AUDIT-FIX(#1,#5): Snapshot-Restore implementiert und fehlertolerant.
            self._persist_snapshot()  # AUDIT-FIX(#1,#5): Snapshot-Persistierung implementiert und atomar.
        except Exception:
            _LOGGER.exception(
                "Failed to initialize TwinrRuntimeBase; cleaning up partially initialized components.",
            )
            self._shutdown_components(timeout_s=1.0)  # AUDIT-FIX(#3): Partielle Initialisierung sauber zurückbauen.
            raise

    @property
    def status(self) -> TwinrStatus:
        return self.state_machine.status

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
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

    def _snapshot_path(self) -> Path:
        return self._coerce_path(self.config.runtime_state_path)

    def _local_timezone(self) -> timezone | ZoneInfo:
        timezone_name = getattr(self.config, "local_timezone_name", None)
        if not timezone_name:
            return timezone.utc
        try:
            return ZoneInfo(str(timezone_name))
        except ZoneInfoNotFoundError:
            _LOGGER.warning(
                "Unknown local timezone '%s'; falling back to UTC for runtime snapshot normalization.",
                timezone_name,
            )
            return timezone.utc

    @staticmethod
    def _coerce_optional_text(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)

    def _normalise_timestamp(self, value: object) -> str | None:
        text_value = self._coerce_optional_text(value)
        if text_value is None:
            return None
        candidate = text_value.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = f"{candidate[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return text_value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=self._local_timezone())  # AUDIT-FIX(#8): Naive Zeitstempel in lokale Zone einbetten.
        return parsed.isoformat()

    @staticmethod
    def _normalise_confidence(value: object) -> float | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        if numeric < 0.0:
            return 0.0
        if numeric > 1.0:
            return 1.0
        return numeric  # AUDIT-FIX(#7): Confidence auf finite [0.0, 1.0] begrenzen.

    def _serialise_status(self) -> str | None:
        try:
            current_status = self.status
        except Exception:
            _LOGGER.exception("Failed to read runtime status for snapshot persistence.")
            return None
        status_name = getattr(current_status, "name", None)
        if isinstance(status_name, str) and status_name:
            return status_name
        status_value = getattr(current_status, "value", None)
        if isinstance(status_value, str) and status_value:
            return status_value
        status_text = str(current_status)
        return status_text or None

    def _build_snapshot_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "status": self._serialise_status(),
            "last_transcript": self._coerce_optional_text(self.last_transcript),
            "last_response": self._coerce_optional_text(self.last_response),
            "user_voice_status": self._coerce_optional_text(self.user_voice_status),
            "user_voice_confidence": self._normalise_confidence(self.user_voice_confidence),
            "user_voice_checked_at": self._normalise_timestamp(self.user_voice_checked_at),
        }

    def _read_snapshot_file(self) -> dict[str, object] | None:
        path = self._snapshot_path()
        if path.is_symlink():
            raise OSError(f"Refusing to read runtime snapshot through symlink: {path}")  # AUDIT-FIX(#5): Symlink-Angriff blockieren.
        if not path.exists():
            return None
        if not path.is_file():
            raise OSError(f"Runtime snapshot path is not a regular file: {path}")
        if path.stat().st_size > _MAX_RUNTIME_SNAPSHOT_BYTES:
            raise OSError(
                f"Runtime snapshot file exceeds {_MAX_RUNTIME_SNAPSHOT_BYTES} bytes and is treated as corrupt.",
            )

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("Runtime snapshot payload must be a JSON object.")
        return payload

    def _write_snapshot_file(self, payload: dict[str, object]) -> None:
        path = self._snapshot_path()
        self._ensure_parent_directory(path)
        if path.is_symlink():
            raise OSError(f"Refusing to write runtime snapshot through symlink: {path}")  # AUDIT-FIX(#5): Symlink-Angriff blockieren.

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = handle.name
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if os.name != "nt":
                os.chmod(temp_path, 0o600)  # AUDIT-FIX(#5): Snapshot-Datei mit restriktiven Rechten ablegen.
            os.replace(temp_path, path)  # AUDIT-FIX(#5): Atomarer Replace gegen Power-Loss-Korruption/TOCTOU.
            if os.name != "nt":
                os.chmod(path, 0o600)
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _persist_snapshot(self) -> None:
        payload = self._build_snapshot_payload()
        with self._snapshot_lock:
            try:
                self._write_snapshot_file(payload)
            except Exception:
                _LOGGER.exception("Runtime snapshot persist failed; continuing with in-memory state only.")

    def _restore_snapshot_context(self) -> None:
        with self._snapshot_lock:
            try:
                payload = self._read_snapshot_file()
            except Exception:
                _LOGGER.exception("Runtime snapshot restore failed; continuing with clean runtime state.")
                return

        if not payload:
            return

        self.last_transcript = self._coerce_optional_text(payload.get("last_transcript"))
        self.last_response = self._coerce_optional_text(payload.get("last_response"))
        self.user_voice_status = self._coerce_optional_text(payload.get("user_voice_status"))
        self.user_voice_confidence = self._normalise_confidence(payload.get("user_voice_confidence"))
        self.user_voice_checked_at = self._normalise_timestamp(payload.get("user_voice_checked_at"))

        restored_status = self._coerce_status(payload.get("status"))
        if restored_status is not None:
            self._restore_status(restored_status)  # AUDIT-FIX(#1): Status-Restore implementieren, aber nur für ruhige Zustände.

    @staticmethod
    def _coerce_status(value: object) -> TwinrStatus | None:
        if value is None:
            return None
        if isinstance(value, TwinrStatus):
            return value
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                return TwinrStatus[candidate]
            except Exception:
                pass
            try:
                return TwinrStatus(candidate)
            except Exception:
                return None
        return None

    def _restore_status(self, status: TwinrStatus) -> None:
        status_name = getattr(status, "name", str(status)).upper()
        if status_name not in _QUIESCENT_RESTORABLE_STATUSES:
            _LOGGER.info(
                "Skipping restore of non-quiescent runtime status '%s' after restart.",
                status_name,
            )
            return

        set_status = getattr(self.state_machine, "set_status", None)
        if callable(set_status):
            try:
                set_status(status)
                return
            except Exception:
                _LOGGER.exception("State-machine restore via set_status() failed; keeping default status.")
                return

        transition_to = getattr(self.state_machine, "transition_to", None)
        if callable(transition_to):
            try:
                transition_to(status)
                return
            except Exception:
                _LOGGER.exception("State-machine restore via transition_to() failed; keeping default status.")
                return

        try:
            setattr(self.state_machine, "status", status)
        except Exception:
            _LOGGER.exception("Could not restore runtime status; keeping default state-machine status.")