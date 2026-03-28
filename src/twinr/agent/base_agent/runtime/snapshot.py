# CHANGELOG: 2026-03-27
# BUG-1: Persist now snapshots a deep, deterministic, JSON-safe copy of runtime memory/state; nested mutable objects can no longer race with serialization and corrupt snapshots.
# BUG-2: Restore now reinstates persisted runtime status, reuses frozen collections for metrics, hardens voice-quiet handling, and makes telemetry failures non-fatal.
# BUG-3: Startup restore no longer resurrects persisted `error` status across fresh process boots; live blockers must assert a current runtime error themselves instead of leaving Twinr stuck in a stale operator error from an older outage.
# SEC-1: Restore is now size-bounded, control-character-sanitized, optionally HMAC-verified, descriptor-safe, and lock-guarded to mitigate practical snapshot poisoning/tampering and snapshot-based DoS on Raspberry Pi deployments.
# IMP-1: Added schema-versioned snapshot envelope metadata, in-process locking, and optional cross-process file locking via portalocker when available.
# IMP-2: Added bounded normalization for text, metadata, timestamps, and collections to align the snapshot boundary with 2026 typed, deterministic, policy-mediated agent-memory patterns.

"""Persist and restore the structured runtime snapshot representation."""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import logging
import math
import os
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from types import TracebackType
from typing import TYPE_CHECKING, Any, Iterator, Protocol, cast

from twinr.agent.workflows.forensics import workflow_event
from twinr.memory import ConversationTurn, MemoryLedgerItem, MemoryState, OnDeviceMemory, SearchMemoryEntry

try:  # Optional fast path for canonical JSON encoding on ARM/aarch64 as well.
    import orjson as _orjson  # type: ignore[import-not-found]  # pylint: disable=import-error
except Exception:  # pragma: no cover - optional dependency
    _orjson = None

try:  # Optional cross-process lock if the deployment already ships it.
    import portalocker as _portalocker  # type: ignore[import-not-found]  # pylint: disable=import-error
except Exception:  # pragma: no cover - optional dependency
    _portalocker = None

if TYPE_CHECKING:
    from twinr.agent.base_agent.state.snapshot import RuntimeSnapshotStore


LOGGER = logging.getLogger(__name__)
_SNAPSHOT_FALLBACK_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)
_SNAPSHOT_SCHEMA_VERSION = 2


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        LOGGER.warning("Invalid integer for %s=%r; using default %s", name, raw, default)
        return default
    return parsed if parsed > 0 else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        LOGGER.warning("Invalid float for %s=%r; using default %s", name, raw, default)
        return default
    return parsed if parsed > 0 else default


# Conservative defaults for Pi 4-class deployments: bounded boot latency and bounded RAM.
_MAX_TEXT_CHARS = _env_int("TWINR_SNAPSHOT_MAX_TEXT_CHARS", 16384)
_MAX_SHORT_TEXT_CHARS = _env_int("TWINR_SNAPSHOT_MAX_SHORT_TEXT_CHARS", 512)
_MAX_ROLE_CHARS = _env_int("TWINR_SNAPSHOT_MAX_ROLE_CHARS", 32)
_MAX_SOURCE_CHARS = _env_int("TWINR_SNAPSHOT_MAX_SOURCE_CHARS", 512)
_MAX_SOURCES = _env_int("TWINR_SNAPSHOT_MAX_SOURCES", 16)
_MAX_OPEN_LOOPS = _env_int("TWINR_SNAPSHOT_MAX_OPEN_LOOPS", 64)
_MAX_METADATA_ITEMS = _env_int("TWINR_SNAPSHOT_MAX_METADATA_ITEMS", 64)
_MAX_METADATA_DEPTH = _env_int("TWINR_SNAPSHOT_MAX_METADATA_DEPTH", 4)
_MAX_MEMORY_TURNS = _env_int("TWINR_SNAPSHOT_MAX_MEMORY_TURNS", 256)
_MAX_MEMORY_RAW_TAIL = _env_int("TWINR_SNAPSHOT_MAX_MEMORY_RAW_TAIL", 256)
_MAX_MEMORY_LEDGER = _env_int("TWINR_SNAPSHOT_MAX_MEMORY_LEDGER", 512)
_MAX_MEMORY_SEARCH_RESULTS = _env_int("TWINR_SNAPSHOT_MAX_MEMORY_SEARCH_RESULTS", 128)
_FILE_LOCK_TIMEOUT_SECONDS = _env_float("TWINR_SNAPSHOT_FILE_LOCK_TIMEOUT_SECONDS", 2.0)
_REQUIRE_INTEGRITY = os.getenv("TWINR_SNAPSHOT_REQUIRE_INTEGRITY", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class _RuntimeLockLike(Protocol):
    """Describe the minimal lock API used by the snapshot guard."""

    def __enter__(self) -> object: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> object: ...
    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool: ...
    def release(self) -> None: ...


class TwinrRuntimeSnapshotMixin:
    """Provide the canonical runtime snapshot save and restore path."""

    memory: OnDeviceMemory
    snapshot_store: RuntimeSnapshotStore

    def _persist_snapshot(self, *, error_message: str | None = None) -> None:
        """Save the current runtime state through RuntimeSnapshotStore."""

        caller = self._snapshot_caller_name()
        persist_started = time.monotonic()
        with self._snapshot_operation_guard():
            try:
                payload, metrics = self._build_snapshot_save_payload(error_message=error_message)
                self._snapshot_store_save(payload)
                self._emit_workflow_event_safely(
                    kind="metric",
                    msg="runtime_snapshot_persist_completed",
                    details={
                        "caller": caller,
                        "status": payload.get("status"),
                        "error_message_present": bool(error_message),
                        **metrics,
                    },
                    kpi={"duration_ms": round((time.monotonic() - persist_started) * 1000.0, 3)},
                )
            except Exception:
                self._emit_workflow_event_safely(
                    kind="warning",
                    level="WARN",
                    msg="runtime_snapshot_persist_failed",
                    details={
                        "caller": caller,
                        "status": getattr(getattr(self, "status", None), "value", None),
                        "error_message_present": bool(error_message),
                    },
                    kpi={"duration_ms": round((time.monotonic() - persist_started) * 1000.0, 3)},
                )
                LOGGER.exception("Failed to persist runtime snapshot")

    def _restore_snapshot_context(self) -> None:
        """Restore runtime state from the structured snapshot store."""

        restore_started = time.monotonic()
        with self._snapshot_operation_guard():
            try:
                snapshot = self.snapshot_store.load()
            except Exception:
                LOGGER.exception("Failed to load runtime snapshot")
                self._emit_workflow_event_safely(
                    kind="warning",
                    level="WARN",
                    msg="runtime_snapshot_load_failed",
                    details={},
                    kpi={"duration_ms": round((time.monotonic() - restore_started) * 1000.0, 3)},
                )
                self._reset_runtime_snapshot_context()
                return

            if snapshot is None:
                self._reset_runtime_snapshot_context()
                self._emit_workflow_event_safely(
                    kind="metric",
                    msg="runtime_snapshot_restore_skipped",
                    details={"reason": "missing_snapshot"},
                    kpi={"duration_ms": round((time.monotonic() - restore_started) * 1000.0, 3)},
                )
                return

            integrity_state = "not_checked"
            try:
                integrity_state = self._verify_snapshot_integrity(snapshot)
                restored_status = self._coerce_optional_text(
                    self._snapshot_get(snapshot, "status"),
                    empty_to_none=False,
                )
                restored_printing_active = bool(self._snapshot_get(snapshot, "printing_active"))
                if self._should_restore_status(restored_status):
                    self._restore_status(restored_status, printing_active=restored_printing_active)
                self.last_transcript = self._coerce_optional_text(
                    self._snapshot_get(snapshot, "last_transcript"),
                    empty_to_none=False,
                    max_chars=_MAX_TEXT_CHARS,
                )
                self.last_response = self._coerce_optional_text(
                    self._snapshot_get(snapshot, "last_response"),
                    max_chars=_MAX_TEXT_CHARS,
                )
                self.user_voice_status = self._coerce_optional_text(
                    self._snapshot_get(snapshot, "user_voice_status"),
                    max_chars=_MAX_SHORT_TEXT_CHARS,
                )
                self.user_voice_confidence = self._coerce_optional_float(
                    self._snapshot_get(snapshot, "user_voice_confidence")
                )
                restored_voice_checked_at = self._parse_optional_snapshot_timestamp(
                    self._snapshot_get(snapshot, "user_voice_checked_at")
                )
                self.user_voice_checked_at = (
                    self._format_snapshot_timestamp(restored_voice_checked_at)
                    if restored_voice_checked_at is not None
                    else None
                )
                self.user_voice_user_id = self._coerce_optional_text(
                    self._snapshot_get(snapshot, "user_voice_user_id"),
                    max_chars=_MAX_SHORT_TEXT_CHARS,
                )
                self.user_voice_user_display_name = self._coerce_optional_text(
                    self._snapshot_get(snapshot, "user_voice_user_display_name"),
                    max_chars=_MAX_SHORT_TEXT_CHARS,
                )
                self.user_voice_match_source = self._coerce_optional_text(
                    self._snapshot_get(snapshot, "user_voice_match_source"),
                    max_chars=_MAX_SHORT_TEXT_CHARS,
                )
                self._restore_voice_quiet_safely(
                    until_utc=self._serialize_optional_snapshot_timestamp(
                        self._snapshot_get(snapshot, "voice_quiet_until_utc")
                    ),
                    reason=self._coerce_optional_text(
                        self._snapshot_get(snapshot, "voice_quiet_reason"),
                        max_chars=_MAX_SHORT_TEXT_CHARS,
                    ),
                )

                legacy_turns = self._restore_legacy_turns(
                    self._snapshot_get(snapshot, "memory_turns", ()),
                    limit=_MAX_MEMORY_TURNS,
                )
                raw_tail = self._restore_raw_tail(
                    self._snapshot_get(snapshot, "memory_raw_tail", ()),
                    limit=_MAX_MEMORY_RAW_TAIL,
                )
                ledger = self._restore_ledger(
                    self._snapshot_get(snapshot, "memory_ledger", ()),
                    limit=_MAX_MEMORY_LEDGER,
                )
                search_results = self._restore_search_results(
                    self._snapshot_get(snapshot, "memory_search_results", ()),
                    limit=_MAX_MEMORY_SEARCH_RESULTS,
                )
                state = self._build_memory_state(self._snapshot_get(snapshot, "memory_state"))

                if self._should_restore_structured(
                    raw_tail=raw_tail,
                    ledger=ledger,
                    search_results=search_results,
                    state=state,
                ):
                    if not raw_tail and not ledger and not search_results and legacy_turns:
                        raw_tail = legacy_turns
                    self.memory.restore_structured(
                        raw_tail=raw_tail,
                        ledger=ledger,
                        search_results=search_results,
                        state=state,
                    )
                else:
                    self.memory.restore(legacy_turns)
            except Exception:
                LOGGER.exception("Failed to restore runtime snapshot; resetting runtime context")
                self._emit_workflow_event_safely(
                    kind="warning",
                    level="WARN",
                    msg="runtime_snapshot_restore_failed",
                    details={"integrity_state": integrity_state},
                    kpi={"duration_ms": round((time.monotonic() - restore_started) * 1000.0, 3)},
                )
                self._reset_runtime_snapshot_context()
                return

            if not self.last_response:
                self.last_response = self._safe_last_assistant_message()

            self._emit_workflow_event_safely(
                kind="metric",
                msg="runtime_snapshot_restore_completed",
                details={
                    "integrity_state": integrity_state,
                    "schema_version": self._snapshot_get(snapshot, "snapshot_schema_version"),
                    "status": getattr(getattr(self, "status", None), "value", None),
                    "memory_turns": len(legacy_turns),
                    "memory_raw_tail": len(raw_tail),
                    "memory_ledger": len(ledger),
                    "memory_search_results": len(search_results),
                },
                kpi={"duration_ms": round((time.monotonic() - restore_started) * 1000.0, 3)},
            )

    def _reset_runtime_snapshot_context(self) -> None:
        self.last_transcript = None
        self.last_response = None
        self.user_voice_status = None
        self.user_voice_confidence = None
        self.user_voice_checked_at = None
        self.user_voice_user_id = None
        self.user_voice_user_display_name = None
        self.user_voice_match_source = None
        reset_voice_quiet = getattr(self, "reset_voice_quiet", None)
        if callable(reset_voice_quiet):
            try:
                reset_voice_quiet()  # pylint: disable=not-callable
            except Exception:
                LOGGER.exception("Failed to reset voice quiet state after snapshot failure")
        self._reset_memory_context()

    def _reset_memory_context(self) -> None:
        empty_state = self._build_memory_state(None)
        try:
            self.memory.restore_structured(
                raw_tail=(),
                ledger=(),
                search_results=(),
                state=empty_state,
            )
        except Exception:
            try:
                self.memory.restore(())
            except Exception:
                LOGGER.exception("Failed to reset memory context after snapshot failure")

    def _restore_legacy_turns(self, value: Any, *, limit: int) -> tuple[ConversationTurn, ...]:
        return self._restore_turn_collection(value, limit=limit, label="legacy snapshot turn")

    def _restore_raw_tail(self, value: Any, *, limit: int) -> tuple[ConversationTurn, ...]:
        return self._restore_turn_collection(value, limit=limit, label="snapshot raw_tail entry")

    def _restore_ledger(self, value: Any, *, limit: int) -> tuple[MemoryLedgerItem, ...]:
        ledger: list[MemoryLedgerItem] = []
        for index, item in enumerate(self._iter_bounded_tail(value, limit=limit)):
            try:
                ledger.append(
                    MemoryLedgerItem(
                        kind=self._coerce_required_text(
                            self._snapshot_get(item, "kind"),
                            field_name="ledger.kind",
                            max_chars=_MAX_SHORT_TEXT_CHARS,
                        ),
                        content=self._coerce_required_text(
                            self._snapshot_get(item, "content"),
                            field_name="ledger.content",
                            max_chars=_MAX_TEXT_CHARS,
                        ),
                        created_at=self._parse_snapshot_timestamp(self._snapshot_get(item, "created_at")),
                        source=self._coerce_optional_text(
                            self._snapshot_get(item, "source"),
                            max_chars=_MAX_SOURCE_CHARS,
                        ),
                        metadata=self._coerce_mapping(self._snapshot_get(item, "metadata")),
                    )
                )
            except Exception as exc:
                LOGGER.warning("Skipping invalid snapshot ledger entry at index %s: %s", index, exc)
        return tuple(ledger)

    def _restore_search_results(self, value: Any, *, limit: int) -> tuple[SearchMemoryEntry, ...]:
        search_results: list[SearchMemoryEntry] = []
        for index, item in enumerate(self._iter_bounded_tail(value, limit=limit)):
            try:
                search_results.append(
                    SearchMemoryEntry(
                        question=self._coerce_required_text(
                            self._snapshot_get(item, "question"),
                            field_name="search_results.question",
                            max_chars=_MAX_TEXT_CHARS,
                        ),
                        answer=self._coerce_required_text(
                            self._snapshot_get(item, "answer"),
                            field_name="search_results.answer",
                            max_chars=_MAX_TEXT_CHARS,
                        ),
                        sources=self._coerce_text_tuple(
                            self._snapshot_get(item, "sources"),
                            max_items=_MAX_SOURCES,
                            max_chars=_MAX_SOURCE_CHARS,
                        ),
                        created_at=self._parse_snapshot_timestamp(self._snapshot_get(item, "created_at")),
                        location_hint=self._coerce_optional_text(
                            self._snapshot_get(item, "location_hint"),
                            max_chars=_MAX_SHORT_TEXT_CHARS,
                        ),
                        date_context=self._coerce_optional_text(
                            self._snapshot_get(item, "date_context"),
                            max_chars=_MAX_SHORT_TEXT_CHARS,
                        ),
                    )
                )
            except Exception as exc:
                LOGGER.warning("Skipping invalid snapshot search result at index %s: %s", index, exc)
        return tuple(search_results)

    @classmethod
    def _build_memory_state(cls, value: Any) -> MemoryState:
        return MemoryState(
            active_topic=cls._coerce_optional_text(cls._snapshot_get(value, "active_topic"), max_chars=_MAX_SHORT_TEXT_CHARS),
            last_user_goal=cls._coerce_optional_text(
                cls._snapshot_get(value, "last_user_goal"),
                max_chars=_MAX_SHORT_TEXT_CHARS,
            ),
            pending_printable=cls._sanitize_jsonish(cls._snapshot_get(value, "pending_printable"), depth=1),
            last_search_summary=cls._coerce_optional_text(
                cls._snapshot_get(value, "last_search_summary"),
                max_chars=_MAX_TEXT_CHARS,
            ),
            open_loops=cls._coerce_text_tuple(
                cls._snapshot_get(value, "open_loops", ()),
                max_items=_MAX_OPEN_LOOPS,
                max_chars=_MAX_SHORT_TEXT_CHARS,
            ),
        )

    @staticmethod
    def _should_restore_structured(
        *,
        raw_tail: tuple[ConversationTurn, ...],
        ledger: tuple[MemoryLedgerItem, ...],
        search_results: tuple[SearchMemoryEntry, ...],
        state: MemoryState,
    ) -> bool:
        return bool(
            raw_tail
            or ledger
            or search_results
            or TwinrRuntimeSnapshotMixin._memory_state_has_data(state)
        )

    @staticmethod
    def _memory_state_has_data(state: MemoryState) -> bool:
        return (
            state.active_topic is not None
            or state.last_user_goal is not None
            or state.pending_printable is not None
            or state.last_search_summary is not None
            or bool(state.open_loops)
        )

    @staticmethod
    def _snapshot_get(value: Any, name: str, default: Any = None) -> Any:
        if value is None:
            return default
        if isinstance(value, Mapping):
            return value.get(name, default)
        try:
            value_dict = vars(value)
        except TypeError:
            value_dict = None
        except Exception:
            value_dict = None
        if isinstance(value_dict, dict):
            return value_dict.get(name, default)
        try:
            return getattr(value, name, default)
        except Exception:
            return default

    @staticmethod
    def _iter_sequence(value: Any) -> Iterator[Any]:
        if value is None:
            return
        if isinstance(value, (str, bytes, bytearray)):
            yield value
            return
        if isinstance(value, Mapping):
            yield value
            return
        try:
            iterator = iter(value)
        except TypeError:
            yield value
            return
        for item in iterator:
            yield item

    @classmethod
    def _iter_bounded_tail(cls, value: Any, *, limit: int) -> Iterator[Any]:
        if limit <= 0:
            return
        if isinstance(value, list):
            yield from value[-limit:]
            return
        if isinstance(value, tuple):
            yield from value[-limit:]
            return
        window: deque[Any] = deque(maxlen=limit)
        for item in cls._iter_sequence(value):
            window.append(item)
        for item in window:
            yield item

    @classmethod
    def _coerce_mapping_key(cls, value: Any) -> str | None:
        if isinstance(value, (str, bytes, bytearray, memoryview)):
            return cls._coerce_optional_text(value, empty_to_none=False, max_chars=_MAX_SHORT_TEXT_CHARS)
        if value is None:
            return None
        return cls._coerce_optional_text(str(value), empty_to_none=False, max_chars=_MAX_SHORT_TEXT_CHARS)

    @classmethod
    def _coerce_mapping(cls, value: Any) -> dict[str, Any]:
        sanitized = cls._sanitize_jsonish(value, depth=0)
        if sanitized is None:
            return {}
        if isinstance(sanitized, dict):
            return sanitized
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= _MAX_METADATA_ITEMS:
                    break
                sanitized_key = cls._coerce_mapping_key(key)
                if not sanitized_key:
                    continue
                result[sanitized_key] = cls._sanitize_jsonish(item, depth=1)
            return result
        return {}

    @classmethod
    def _coerce_text_tuple(
        cls,
        value: Any,
        *,
        max_items: int = _MAX_METADATA_ITEMS,
        max_chars: int = _MAX_SHORT_TEXT_CHARS,
    ) -> tuple[str, ...]:
        items: list[str] = []
        seen: set[str] = set()
        for item in cls._iter_sequence(value):
            if len(items) >= max_items:
                break
            text = cls._coerce_optional_text(item, empty_to_none=True, max_chars=max_chars)
            if text is None or text in seen:
                continue
            seen.add(text)
            items.append(text)
        return tuple(items)

    @staticmethod
    def _coerce_optional_text(
        value: Any,
        *,
        empty_to_none: bool = True,
        max_chars: int = _MAX_TEXT_CHARS,
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = bytes(value).decode("utf-8", errors="replace")
        elif not isinstance(value, str):
            return None
        if value == "" and empty_to_none:
            return None
        value = TwinrRuntimeSnapshotMixin._normalize_snapshot_text(value, max_chars=max_chars)
        if value == "" and empty_to_none:
            return None
        return value

    @classmethod
    def _coerce_required_text(cls, value: Any, *, field_name: str, max_chars: int) -> str:
        text = cls._coerce_optional_text(value, empty_to_none=False, max_chars=max_chars)
        if text is None or text == "":
            raise ValueError(f"missing or invalid {field_name}")
        return text

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        if value is None or value == "" or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    def _safe_last_assistant_message(self) -> str | None:
        try:
            return self._coerce_optional_text(self.memory.last_assistant_message())
        except Exception:
            LOGGER.exception("Failed to read last assistant message from restored memory")
            return None

    @classmethod
    def _parse_optional_snapshot_timestamp(cls, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        return cls._parse_snapshot_timestamp(value)

    @staticmethod
    def _parse_snapshot_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, (bytes, bytearray, memoryview)):
            return TwinrRuntimeSnapshotMixin._parse_snapshot_timestamp(bytes(value).decode("utf-8", errors="replace"))
        elif isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return _SNAPSHOT_FALLBACK_AT
            if candidate.endswith(("Z", "z")):
                candidate = f"{candidate[:-1]}+00:00"
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                return _SNAPSHOT_FALLBACK_AT
        else:
            return _SNAPSHOT_FALLBACK_AT

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _format_snapshot_timestamp(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @classmethod
    def _serialize_optional_snapshot_timestamp(cls, value: Any) -> str | None:
        parsed = cls._parse_optional_snapshot_timestamp(value)
        if parsed is None:
            return None
        return cls._format_snapshot_timestamp(parsed)

    @staticmethod
    def _normalize_snapshot_text(value: str, *, max_chars: int) -> str:
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        normalized = "".join(
            character for character in normalized if character == "\n" or character == "\t" or character >= " "
        )
        if len(normalized) > max_chars:
            # BREAKING: Oversized snapshot text is truncated to keep restore/persist bounded on Pi 4-class devices.
            normalized = normalized[:max_chars]
        return normalized

    @classmethod
    def _sanitize_jsonish(cls, value: Any, *, depth: int) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, datetime):
            return cls._format_snapshot_timestamp(cls._parse_snapshot_timestamp(value))
        if isinstance(value, (str, bytes, bytearray, memoryview)):
            return cls._coerce_optional_text(value, empty_to_none=False, max_chars=_MAX_TEXT_CHARS)
        if depth >= _MAX_METADATA_DEPTH:
            return cls._coerce_optional_text(str(value), empty_to_none=False, max_chars=_MAX_TEXT_CHARS)
        if isinstance(value, Mapping):
            sanitized: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= _MAX_METADATA_ITEMS:
                    break
                sanitized_key = cls._coerce_mapping_key(key)
                if not sanitized_key:
                    continue
                sanitized[sanitized_key] = cls._sanitize_jsonish(item, depth=depth + 1)
            return sanitized
        if isinstance(value, Iterable):
            items: list[Any] = []
            for index, item in enumerate(value):
                if index >= _MAX_METADATA_ITEMS:
                    break
                items.append(cls._sanitize_jsonish(item, depth=depth + 1))
            return tuple(items)
        return cls._coerce_optional_text(str(value), empty_to_none=False, max_chars=_MAX_TEXT_CHARS)

    def _snapshot_caller_name(self) -> str:
        caller = "unknown"
        frame = inspect.currentframe()
        try:
            if frame is not None and frame.f_back is not None:
                caller = str(frame.f_back.f_code.co_name or "unknown")
        finally:
            del frame
        return caller

    def _build_snapshot_save_payload(self, *, error_message: str | None) -> tuple[dict[str, Any], dict[str, int]]:
        memory_turns = self._snapshot_turn_payloads(getattr(self.memory, "turns", ()), limit=_MAX_MEMORY_TURNS)
        memory_raw_tail = self._snapshot_turn_payloads(getattr(self.memory, "raw_tail", ()), limit=_MAX_MEMORY_RAW_TAIL)
        memory_ledger = self._snapshot_ledger_payloads(getattr(self.memory, "ledger", ()), limit=_MAX_MEMORY_LEDGER)
        memory_search_results = self._snapshot_search_result_payloads(
            getattr(self.memory, "search_results", ()),
            limit=_MAX_MEMORY_SEARCH_RESULTS,
        )
        memory_state = self._snapshot_memory_state_payload(getattr(self.memory, "state", None))

        quiet_until_raw = getattr(self, "voice_quiet_until_utc", None)
        quiet_until_getter = quiet_until_raw if callable(quiet_until_raw) else None
        quiet_until = quiet_until_getter() if quiet_until_getter is not None else quiet_until_raw

        payload: dict[str, Any] = {
            "status": self._coerce_optional_text(
                getattr(getattr(self, "status", None), "value", None),
                empty_to_none=False,
                max_chars=_MAX_SHORT_TEXT_CHARS,
            ),
            "printing_active": bool(getattr(getattr(self, "state_machine", None), "printing_active", False)),
            # BREAKING: New snapshots are persisted as builtin containers + RFC3339/UTC strings instead of live model objects.
            "memory_turns": memory_turns,
            "memory_raw_tail": memory_raw_tail,
            "memory_ledger": memory_ledger,
            "memory_search_results": memory_search_results,
            "memory_state": memory_state,
            "last_transcript": self._coerce_optional_text(
                getattr(self, "last_transcript", None),
                empty_to_none=False,
                max_chars=_MAX_TEXT_CHARS,
            ),
            "last_response": self._coerce_optional_text(
                getattr(self, "last_response", None),
                max_chars=_MAX_TEXT_CHARS,
            ),
            "error_message": self._coerce_optional_text(error_message, max_chars=_MAX_TEXT_CHARS),
            "user_voice_status": self._coerce_optional_text(
                getattr(self, "user_voice_status", None),
                max_chars=_MAX_SHORT_TEXT_CHARS,
            ),
            "user_voice_confidence": self._coerce_optional_float(getattr(self, "user_voice_confidence", None)),
            "user_voice_checked_at": self._serialize_optional_snapshot_timestamp(getattr(self, "user_voice_checked_at", None)),
            "user_voice_user_id": self._coerce_optional_text(
                getattr(self, "user_voice_user_id", None),
                max_chars=_MAX_SHORT_TEXT_CHARS,
            ),
            "user_voice_user_display_name": self._coerce_optional_text(
                getattr(self, "user_voice_user_display_name", None),
                max_chars=_MAX_SHORT_TEXT_CHARS,
            ),
            "user_voice_match_source": self._coerce_optional_text(
                getattr(self, "user_voice_match_source", None),
                max_chars=_MAX_SHORT_TEXT_CHARS,
            ),
            "voice_quiet_until_utc": self._serialize_optional_snapshot_timestamp(quiet_until),
            "voice_quiet_reason": self._coerce_optional_text(
                getattr(self, "_voice_quiet_reason", None),
                max_chars=_MAX_SHORT_TEXT_CHARS,
            ),
        }

        payload.update(self._snapshot_envelope(payload))
        metrics = {
            "memory_turns": len(memory_turns),
            "memory_raw_tail": len(memory_raw_tail),
            "memory_ledger": len(memory_ledger),
            "memory_search_results": len(memory_search_results),
        }
        return payload, metrics

    def _snapshot_store_save(self, payload: dict[str, Any]) -> None:
        save_callable = self.snapshot_store.save
        supported_payload = self._filter_payload_for_callable(save_callable, payload)
        if set(supported_payload) != set(payload):
            missing = sorted(set(payload) - set(supported_payload))
            LOGGER.debug("Snapshot store.save does not accept fields: %s", missing)
        save_callable(**supported_payload)

    def _snapshot_envelope(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        extras: dict[str, Any] = {
            "snapshot_schema_version": _SNAPSHOT_SCHEMA_VERSION,
            "snapshot_saved_at": self._format_snapshot_timestamp(datetime.now(timezone.utc)),
        }
        integrity = self._snapshot_integrity_for_payload(payload)
        if integrity is not None:
            extras["snapshot_integrity_alg"] = integrity["alg"]
            extras["snapshot_integrity"] = integrity["digest"]
        return extras

    @staticmethod
    def _filter_payload_for_callable(func: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return dict(payload)
        parameters = signature.parameters
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return dict(payload)
        accepted = {
            name
            for name, parameter in parameters.items()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return {name: value for name, value in payload.items() if name in accepted}

    def _snapshot_integrity_for_payload(self, payload: Mapping[str, Any]) -> dict[str, str] | None:
        key = os.getenv("TWINR_SNAPSHOT_HMAC_KEY", "").encode("utf-8")
        if not key:
            return None
        digest = hmac.new(key, self._stable_payload_bytes(payload), hashlib.sha256).hexdigest()
        return {"alg": "hmac-sha256", "digest": digest}

    def _verify_snapshot_integrity(self, snapshot: Any) -> str:
        digest = self._coerce_optional_text(self._snapshot_get(snapshot, "snapshot_integrity"), max_chars=128)
        algorithm = self._coerce_optional_text(self._snapshot_get(snapshot, "snapshot_integrity_alg"), max_chars=64)
        key = os.getenv("TWINR_SNAPSHOT_HMAC_KEY", "").encode("utf-8")

        if not digest or not algorithm:
            if _REQUIRE_INTEGRITY and key:
                raise ValueError("snapshot integrity metadata missing while integrity is required")
            return "missing"

        if algorithm != "hmac-sha256":
            raise ValueError(f"unsupported snapshot integrity algorithm: {algorithm}")

        if not key:
            if _REQUIRE_INTEGRITY:
                raise ValueError("snapshot integrity required but TWINR_SNAPSHOT_HMAC_KEY is not configured")
            return "present_but_unverifiable"

        payload = {
            "status": self._snapshot_get(snapshot, "status"),
            "printing_active": self._snapshot_get(snapshot, "printing_active"),
            "memory_turns": self._snapshot_get(snapshot, "memory_turns"),
            "memory_raw_tail": self._snapshot_get(snapshot, "memory_raw_tail"),
            "memory_ledger": self._snapshot_get(snapshot, "memory_ledger"),
            "memory_search_results": self._snapshot_get(snapshot, "memory_search_results"),
            "memory_state": self._snapshot_get(snapshot, "memory_state"),
            "last_transcript": self._snapshot_get(snapshot, "last_transcript"),
            "last_response": self._snapshot_get(snapshot, "last_response"),
            "error_message": self._snapshot_get(snapshot, "error_message"),
            "user_voice_status": self._snapshot_get(snapshot, "user_voice_status"),
            "user_voice_confidence": self._snapshot_get(snapshot, "user_voice_confidence"),
            "user_voice_checked_at": self._snapshot_get(snapshot, "user_voice_checked_at"),
            "user_voice_user_id": self._snapshot_get(snapshot, "user_voice_user_id"),
            "user_voice_user_display_name": self._snapshot_get(snapshot, "user_voice_user_display_name"),
            "user_voice_match_source": self._snapshot_get(snapshot, "user_voice_match_source"),
            "voice_quiet_until_utc": self._snapshot_get(snapshot, "voice_quiet_until_utc"),
            "voice_quiet_reason": self._snapshot_get(snapshot, "voice_quiet_reason"),
        }
        expected = hmac.new(key, self._stable_payload_bytes(payload), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(digest, expected):
            raise ValueError("snapshot integrity verification failed")
        return "verified"

    @staticmethod
    def _stable_payload_bytes(payload: Mapping[str, Any]) -> bytes:
        if _orjson is not None:
            return _orjson.dumps(payload, option=_orjson.OPT_SORT_KEYS)
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

    @classmethod
    def _snapshot_turn_payloads(cls, value: Any, *, limit: int) -> tuple[dict[str, Any], ...]:
        items: list[dict[str, Any]] = []
        for item in cls._iter_bounded_tail(value, limit=limit):
            try:
                items.append(cls._snapshot_turn_payload(item))
            except Exception as exc:
                LOGGER.warning("Skipping invalid runtime turn during snapshot persist: %s", exc)
        return tuple(items)

    @classmethod
    def _snapshot_turn_payload(cls, value: Any) -> dict[str, Any]:
        return {
            "role": cls._coerce_required_text(
                cls._snapshot_get(value, "role"),
                field_name="turn.role",
                max_chars=_MAX_ROLE_CHARS,
            ),
            "content": cls._coerce_required_text(
                cls._snapshot_get(value, "content"),
                field_name="turn.content",
                max_chars=_MAX_TEXT_CHARS,
            ),
            "created_at": cls._serialize_optional_snapshot_timestamp(cls._snapshot_get(value, "created_at"))
            or cls._format_snapshot_timestamp(_SNAPSHOT_FALLBACK_AT),
        }

    @classmethod
    def _snapshot_ledger_payloads(cls, value: Any, *, limit: int) -> tuple[dict[str, Any], ...]:
        items: list[dict[str, Any]] = []
        for item in cls._iter_bounded_tail(value, limit=limit):
            try:
                items.append(
                    {
                        "kind": cls._coerce_required_text(
                            cls._snapshot_get(item, "kind"),
                            field_name="ledger.kind",
                            max_chars=_MAX_SHORT_TEXT_CHARS,
                        ),
                        "content": cls._coerce_required_text(
                            cls._snapshot_get(item, "content"),
                            field_name="ledger.content",
                            max_chars=_MAX_TEXT_CHARS,
                        ),
                        "created_at": cls._serialize_optional_snapshot_timestamp(cls._snapshot_get(item, "created_at"))
                        or cls._format_snapshot_timestamp(_SNAPSHOT_FALLBACK_AT),
                        "source": cls._coerce_optional_text(
                            cls._snapshot_get(item, "source"),
                            max_chars=_MAX_SOURCE_CHARS,
                        ),
                        "metadata": cls._coerce_mapping(cls._snapshot_get(item, "metadata")),
                    }
                )
            except Exception as exc:
                LOGGER.warning("Skipping invalid runtime ledger item during snapshot persist: %s", exc)
        return tuple(items)

    @classmethod
    def _snapshot_search_result_payloads(cls, value: Any, *, limit: int) -> tuple[dict[str, Any], ...]:
        items: list[dict[str, Any]] = []
        for item in cls._iter_bounded_tail(value, limit=limit):
            try:
                items.append(
                    {
                        "question": cls._coerce_required_text(
                            cls._snapshot_get(item, "question"),
                            field_name="search_results.question",
                            max_chars=_MAX_TEXT_CHARS,
                        ),
                        "answer": cls._coerce_required_text(
                            cls._snapshot_get(item, "answer"),
                            field_name="search_results.answer",
                            max_chars=_MAX_TEXT_CHARS,
                        ),
                        "sources": cls._coerce_text_tuple(
                            cls._snapshot_get(item, "sources"),
                            max_items=_MAX_SOURCES,
                            max_chars=_MAX_SOURCE_CHARS,
                        ),
                        "created_at": cls._serialize_optional_snapshot_timestamp(cls._snapshot_get(item, "created_at"))
                        or cls._format_snapshot_timestamp(_SNAPSHOT_FALLBACK_AT),
                        "location_hint": cls._coerce_optional_text(
                            cls._snapshot_get(item, "location_hint"),
                            max_chars=_MAX_SHORT_TEXT_CHARS,
                        ),
                        "date_context": cls._coerce_optional_text(
                            cls._snapshot_get(item, "date_context"),
                            max_chars=_MAX_SHORT_TEXT_CHARS,
                        ),
                    }
                )
            except Exception as exc:
                LOGGER.warning("Skipping invalid runtime search result during snapshot persist: %s", exc)
        return tuple(items)

    @classmethod
    def _snapshot_memory_state_payload(cls, value: Any) -> dict[str, Any]:
        return {
            "active_topic": cls._coerce_optional_text(
                cls._snapshot_get(value, "active_topic"),
                max_chars=_MAX_SHORT_TEXT_CHARS,
            ),
            "last_user_goal": cls._coerce_optional_text(
                cls._snapshot_get(value, "last_user_goal"),
                max_chars=_MAX_SHORT_TEXT_CHARS,
            ),
            "pending_printable": cls._sanitize_jsonish(cls._snapshot_get(value, "pending_printable"), depth=1),
            "last_search_summary": cls._coerce_optional_text(
                cls._snapshot_get(value, "last_search_summary"),
                max_chars=_MAX_TEXT_CHARS,
            ),
            "open_loops": cls._coerce_text_tuple(
                cls._snapshot_get(value, "open_loops", ()),
                max_items=_MAX_OPEN_LOOPS,
                max_chars=_MAX_SHORT_TEXT_CHARS,
            ),
        }

    @classmethod
    def _restore_turn_collection(
        cls,
        value: Any,
        *,
        limit: int,
        label: str,
    ) -> tuple[ConversationTurn, ...]:
        turns: list[ConversationTurn] = []
        for index, turn in enumerate(cls._iter_bounded_tail(value, limit=limit)):
            try:
                turns.append(
                    ConversationTurn(
                        role=cls._coerce_required_text(
                            cls._snapshot_get(turn, "role"),
                            field_name=f"{label}.role",
                            max_chars=_MAX_ROLE_CHARS,
                        ),
                        content=cls._coerce_required_text(
                            cls._snapshot_get(turn, "content"),
                            field_name=f"{label}.content",
                            max_chars=_MAX_TEXT_CHARS,
                        ),
                        created_at=cls._parse_snapshot_timestamp(cls._snapshot_get(turn, "created_at")),
                    )
                )
            except Exception as exc:
                LOGGER.warning("Skipping invalid %s at index %s: %s", label, index, exc)
        return tuple(turns)

    @staticmethod
    def _should_restore_status(value: str | None) -> bool:
        normalized = str(value or "").strip().lower()
        # Do not carry a previously persisted operator error across a fresh
        # process boot. The current runtime/supervisor startup path must prove
        # and re-assert any real blocker instead of inheriting stale error.
        return normalized != "error"

    @staticmethod
    def _restore_status_hook_accepts_printing_active(hook: Any) -> bool:
        try:
            parameters = inspect.signature(hook).parameters.values()
        except (TypeError, ValueError):
            return False

        for parameter in parameters:
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                return True
            if parameter.name == "printing_active" and parameter.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                return True
        return False

    def _restore_status(self, value: str | None, *, printing_active: bool = False) -> None:
        if value is None:
            return
        hook = getattr(self, "restore_runtime_status", None)
        if callable(hook):
            try:
                if self._restore_status_hook_accepts_printing_active(hook):
                    hook(value, printing_active=printing_active)  # pylint: disable=not-callable
                else:
                    hook(value)  # pylint: disable=not-callable
                return
            except Exception:
                LOGGER.exception("Failed to restore runtime status via restore_runtime_status hook")
                return

        state_machine = getattr(self, "state_machine", None)
        restore_snapshot_state = getattr(state_machine, "restore_snapshot_state", None)
        if callable(restore_snapshot_state):
            try:
                restore_snapshot_state(status=value, printing_active=printing_active)  # pylint: disable=not-callable
                return
            except Exception:
                LOGGER.exception("Failed to restore runtime status via state_machine.restore_snapshot_state")
                return

        current_status = getattr(self, "status", None)
        status_type = getattr(current_status, "__class__", None)
        if status_type is None:
            return

        try:
            self.status = status_type(value)
            return
        except Exception:
            pass

        try:
            self.status = status_type[value]
        except Exception:
            LOGGER.warning("Unable to restore runtime status from snapshot value %r", value)

    def _restore_voice_quiet_safely(self, *, until_utc: str | None, reason: str | None) -> None:
        restore_voice_quiet = getattr(self, "restore_voice_quiet", None)
        if not callable(restore_voice_quiet):
            return
        try:
            restore_voice_quiet(until_utc=until_utc, reason=reason)  # pylint: disable=not-callable
        except Exception:
            LOGGER.exception("Failed to restore voice quiet state from snapshot")

    def _emit_workflow_event_safely(self, **payload: Any) -> None:
        try:
            workflow_event(**payload)
        except Exception:
            LOGGER.exception("workflow_event failed while handling runtime snapshot telemetry")

    @contextmanager
    def _snapshot_operation_guard(self) -> Iterator[None]:
        runtime_lock = self._snapshot_runtime_lock()
        with runtime_lock:
            with self._snapshot_file_lock():
                yield

    def _snapshot_runtime_lock(self) -> _RuntimeLockLike:
        lock = getattr(self, "_runtime_snapshot_lock", None)
        if getattr(lock, "acquire", None) is not None and getattr(lock, "release", None) is not None:
            return cast(_RuntimeLockLike, lock)
        created = threading.RLock()
        try:
            setattr(self, "_runtime_snapshot_lock", created)
            return created
        except Exception:
            return created

    @contextmanager
    def _snapshot_file_lock(self) -> Iterator[None]:
        if self._snapshot_store_has_authoritative_file_lock():
            yield
            return
        if _portalocker is None:
            yield
            return
        lock_path = self._snapshot_lock_path()
        if lock_path is None:
            yield
            return
        try:
            with _portalocker.Lock(str(lock_path), mode="a", timeout=_FILE_LOCK_TIMEOUT_SECONDS):
                yield
        except Exception:
            LOGGER.debug(
                "Proceeding without optional snapshot portalocker guard for %s; store-level locking remains authoritative.",
                lock_path,
            )
            yield

    def _snapshot_store_has_authoritative_file_lock(self) -> bool:
        """Return whether the snapshot store already owns the canonical file lock.

        ``RuntimeSnapshotStore`` already acquires ``fcntl.flock()`` on its hidden
        lock file for every load/save. Re-entering the same path through
        ``portalocker`` on Pi deployments can self-deadlock the restore path when
        the process later opens the store lock file again through a fresh file
        descriptor. In that case the store-level lock stays authoritative and the
        optional outer portalocker guard must stay disabled.
        """

        store = getattr(self, "snapshot_store", None)
        if store is None:
            return False
        if not callable(getattr(store, "_locked", None)):
            return False
        return self._snapshot_lock_path() is not None

    def _snapshot_lock_path(self) -> str | None:
        store = getattr(self, "snapshot_store", None)
        if store is None:
            return None
        for attribute_name in ("lock_path", "_lock_path"):
            direct = getattr(store, attribute_name, None)
            if isinstance(direct, os.PathLike):
                return os.fspath(direct)
            if isinstance(direct, str) and direct.strip():
                return direct
        for attribute_name in ("path", "filepath", "file_path", "snapshot_path"):
            candidate = getattr(store, attribute_name, None)
            if isinstance(candidate, os.PathLike):
                candidate = os.fspath(candidate)
            if isinstance(candidate, str) and candidate.strip():
                return f"{candidate}.lock"
        return None
